import os
import sys
import datetime
import time
import argparse
from pathlib import Path
import matplotlib

import colossalai
import timm.optim.optim_factory as optim_factory
import torch
from torch.utils.tensorboard import SummaryWriter
from colossalai.context import Config
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn.lr_scheduler import CosineAnnealingLR
from colossalai.utils import get_dataloader
from colossalai.utils.checkpointing import load_checkpoint, save_checkpoint
from tqdm import tqdm

from baseline.ResidualNet import load_model
from baseline.plot_confusion import get_confusion_matrix, add_confusion_matrix
from dataIO.load_dataset import load_noise_dataset

matplotlib.use('Agg')
# DATA_ROOT = Path(os.environ.get('DATA', './data'))


LOGGER = get_dist_logger()
VERBOSE = True
DEBUG = False
CLASS_NAMES = [0, 1, 2, 3]


def model(norm_pix_loss):
    m = load_model('resnet18')
    if VERBOSE:
        LOGGER.info("Use model Resnet18")
    return m


def criterion():
    c = torch.nn.CrossEntropyLoss()
    if VERBOSE:
        LOGGER.info(f"Criterion:\n{c}")
    return c


def optimizer(model, learning_rate, weight_decay):
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model, weight_decay=weight_decay)
    o = torch.optim.AdamW(param_groups, lr=learning_rate, betas=(0.9, 0.95))
    if VERBOSE:
        LOGGER.info(f"Optimizer:\n{o}")
    return o


def pretrain_dataloaders(
        data_path,
):
    if VERBOSE:
        LOGGER.info(f"DATA_PATH: {data_path}")

    train_dataset, valid_dataset, _ = load_noise_dataset(data_path)
    if VERBOSE:
        LOGGER.info(f"Train dataset:\n{train_dataset}")
        LOGGER.info(f"Valid dataset:\n{valid_dataset}")

    train_dataloader = get_dataloader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=gpc.config.BATCH_SIZE,
        num_workers=5,
        pin_memory=True,
        drop_last=True,
    )

    test_dataloader = get_dataloader(
        dataset=valid_dataset,
        shuffle=False,
        batch_size=gpc.config.BATCH_SIZE,
        num_workers=5,
        pin_memory=True,
        drop_last=False,
    )

    return train_dataloader, test_dataloader


def init_global_states(config: Config):
    global VERBOSE, DEBUG
    VERBOSE = config.VERBOSE
    DEBUG = config.DEBUG


def init_engine(config: Config):
    _model = model(config.NORM_PIX_LOSS)
    _optimizer = optimizer(_model, config.LEARNING_RATE, config.WEIGHT_DECAY)
    _criterion = criterion()
    train_dataloader, test_dataloader = pretrain_dataloaders(
        config.DATA_PATH
    )
    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(
        _model,
        _optimizer,
        _criterion,
        train_dataloader,
        test_dataloader,
    )
    return engine, train_dataloader, test_dataloader, _model


def scale_loss(engine, loss, loss_scaler, data_iter_step, config):
    loss /= config.ACCUM_ITER
    loss_scaler(
        loss,
        engine.optimizer,
        parameters=engine.model.parameters(),
        update_grad=(data_iter_step + 1) % config.ACCUM_ITER == 0,
    )


def save_model(model, output_dir, epoch):
    checkpoint_path = output_dir / (f"checkpoint-{epoch}.pth")
    save_checkpoint(checkpoint_path, epoch, model, None, None)


def main(args):
    colossalai.launch(args.config,
                      rank=args.rank,
                      world_size=args.world_size,
                      host=args.host,
                      port=args.port,
                      backend=args.backend)
    config = gpc.config
    init_global_states(config)
    LOGGER.info("START")
    LOGGER.info("--------------------------------------------------------------------------")
    engine, train_dataloader, test_dataloader, m = init_engine(config)
    lr_scheduler = CosineAnnealingLR(engine.optimizer, total_steps=config.NUM_EPOCHS)

    start_epoch = 0
    if config.RESUME:
        # WARNING: `load_checkpoint()` and `save_checkpoint()`
        #          won't touch optimizer and lr_scheduler!
        start_epoch = 1 + load_checkpoint(
            config.RESUME_DIR, engine.model, engine.optimizer, lr_scheduler, strict=False
        )
        LOGGER.info(
            f"Resume from checkpoint {config.RESUME_DIR}, start epoch {start_epoch}"
        )

    writer = SummaryWriter()
    LOGGER.info(f"Start training for {config.NUM_EPOCHS} epochs")
    start_time = time.time()
    for epoch in range(start_epoch, config.NUM_EPOCHS):

        engine.train()
        # TODO: This part could be more "colossal-native", like construct a correct `engine.criterion`.
        for idx, (sig, labels) in enumerate(tqdm(train_dataloader, desc=f"epoch {epoch}")):
            # we use a per iteration (instead of per epoch) lr scheduler
            labels = labels.reshape(-1)
            sig = sig.cuda()
            labels = labels.cuda()

            engine.zero_grad()
            pred = engine.model(sig, 16)
            loss = engine.criterion(pred, labels)
            engine.backward(loss)
            engine.step()
            lr_scheduler.step()
            # add loss into tensorboard
            writer.add_scalar(tag='loss/train', scalar_value=loss.cpu(), global_step=epoch * idx + idx)

            if idx % 1000 == 0 and idx != 0:  # test
                engine.eval()
                cmtx = 0
                test_loss = 0
                for i, (sig_test, labels_test) in enumerate(tqdm(test_dataloader, desc=f"epoch test{epoch}")):
                    labels_test = labels_test.reshape(-1)
                    sig_test = sig_test.cuda()
                    labels_test = labels_test.cuda()

                    with torch.no_grad():
                        pred = engine.model(sig_test, 16)
                        test_loss += engine.criterion(pred, labels_test)
                    cmtx += get_confusion_matrix(pred.cpu(), labels_test.cpu(), len(CLASS_NAMES))
                    # if i == 100: break
                test_loss /= i
                add_confusion_matrix(writer, cmtx, num_classes=len(CLASS_NAMES), class_names=CLASS_NAMES,
                                     tag=f"epoch {epoch} Train Confusion Matrix", figsize=[10, 8])
                writer.add_scalar(tag='loss/test', scalar_value=test_loss, global_step=epoch * idx + idx)

            if DEBUG:
                LOGGER.info(
                    f"iteration {idx}, first 10 elements of param: {next(engine.model.parameters()).flatten()[:10]}"
                )

        if config.OUTPUT_DIR and (
                epoch % config.CHECKPOINT_INTERVAL == 0 or epoch + 1 == config.NUM_EPOCHS
        ):
            save_model(
                engine.model,
                config.OUTPUT_DIR,
                epoch,
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    writer.close()
    LOGGER.info(f"Training time {total_time_str}")
    LOGGER.info("--------------------------------------------------------------------------")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=Path(__file__).parent / 'config_resnet18.py',
                        help='path to the config file')
    parser.add_argument('--host', default='localhost', type=str, help='the master address for distributed training')
    parser.add_argument('--port', type=int, default='8888', help='the master port for distributed training')
    parser.add_argument('--world_size', type=int, default=int(os.environ['WORLD_SIZE']),
                        help='world size for distributed training')
    parser.add_argument('--rank', type=int, default=int(os.environ['RANK']), help='rank for the default process group')
    parser.add_argument('--local_rank', type=int, default=int(os.environ['LOCAL_RANK']), help='local rank on the node')
    parser.add_argument('--backend', type=str, default='nccl', help='backend for distributed communication')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
