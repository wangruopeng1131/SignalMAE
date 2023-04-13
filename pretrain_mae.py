import os
import sys
import datetime
import time
import argparse
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import colossalai
import timm.optim.optim_factory as optim_factory
import torch
from torch.utils.tensorboard import SummaryWriter
from colossalai.context import Config
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn.lr_scheduler import CosineAnnealingLR
from dataIO.data_parallel_sampler import get_dataloader
from colossalai.utils.checkpointing import load_checkpoint, save_checkpoint
from tqdm import tqdm

from SignalMAE.mae import mae_vit_base_patch25
from dataIO.load_dataset import PreTrainedDataset

matplotlib.use('Agg')
# DATA_ROOT = Path(os.environ.get('DATA', './data'))


LOGGER = get_dist_logger()
VERBOSE = True
DEBUG = False


def model(norm_pix_loss):
    m = mae_vit_base_patch25(norm_pix_loss=norm_pix_loss)
    if VERBOSE:
        LOGGER.info("Use model vit_large_patch16")
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
        ecg_data_path,
        eeg_data_path
):
    if VERBOSE:
        LOGGER.info(f"ECG_DATA_PATH: {ecg_data_path}")
        LOGGER.info(f"EEG_DATA_PATH: {eeg_data_path}")

    train_dataset = PreTrainedDataset(train=True, DATAPATH=ecg_data_path, EEGPATH=eeg_data_path)

    test_dataset = PreTrainedDataset(train=False, DATAPATH=ecg_data_path, EEGPATH=eeg_data_path)
    if VERBOSE:
        LOGGER.info(f"Train dataset:\n{train_dataset}")
        LOGGER.info(f"Test dataset:\n{test_dataset}")

    train_dataloader = get_dataloader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=gpc.config.BATCH_SIZE,
        num_workers=6,
        pin_memory=True,
        drop_last=True,
    )

    test_dataloader = get_dataloader(
        dataset=test_dataset,
        shuffle=True,
        batch_size=gpc.config.BATCH_SIZE,
        num_workers=6,
        pin_memory=True,
        drop_last=False,
    )

    return train_dataloader, test_dataloader, len(train_dataset)


def init_global_states(config: Config):
    global VERBOSE, DEBUG
    VERBOSE = config.VERBOSE
    DEBUG = config.DEBUG


def init_engine(config: Config):
    _model = model(config.NORM_PIX_LOSS)
    _optimizer = optimizer(_model, config.LEARNING_RATE, config.WEIGHT_DECAY)
    _criterion = criterion()
    train_dataloader, test_dataloader, length = pretrain_dataloaders(
        config.DATAPATH, config.EEGPATH
    )
    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(
        _model,
        _optimizer,
        _criterion,
        train_dataloader,
        test_dataloader,
    )
    return engine, train_dataloader, test_dataloader, _model, length


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
    engine, train_dataloader, _, m, length = init_engine(config)
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

    times = np.linspace(0, 30, 7680)  # 注意修改
    freqs = m.morlet.freqs
    writer = SummaryWriter()
    LOGGER.info(f"Start pre-training for {config.NUM_EPOCHS} epochs")
    start_time = time.time()
    for epoch in range(start_epoch, config.NUM_EPOCHS):

        engine.train()
        batch_length = int(length / config.BATCH_SIZE)
        pbar = tqdm(train_dataloader,
                    total=batch_length,
                    leave=True,
                    desc=f"epoch {epoch}",
                    unit_scale=False,
                    colour="red")

        for idx, sig in enumerate(pbar):
            # we use a per iteration (instead of per epoch) lr scheduler
            sig = sig.cuda()
            engine.zero_grad()
            loss, time_freq, pred, mask = engine.model(sig, mask_ratio=config.MASK_RATIO)

            # add loss into tensorboard
            writer.add_scalar(tag='loss/train', scalar_value=loss, global_step=epoch * batch_length + idx)

            engine.backward(loss)
            engine.step()
            lr_scheduler.step()

            pbar.set_postfix({"loss": loss.detach().cpu().numpy()})

            if idx % 100 == 0:
                pred = m.unpatchify(pred)
                # visualize the mask
                mask = mask.detach()
                mask = mask.unsqueeze(-1).repeat(1, 1,
                                                 m.patch_embed.patch_size[0] * m.patch_embed.patch_size[
                                                     1])  # (N, H*W, p*p)
                mask = m.unpatchify(mask)

                # masked image
                im_masked = time_freq * (1 - mask)

                # MAE reconstruction pasted with visible patches
                im_paste = time_freq * (1 - mask) + pred * mask

                fig, axes = plt.subplots(4, 1, figsize=(20, 24))
                time_freq = time_freq[0, 0, ...].detach().cpu().numpy()
                im_masked = im_masked[0, 0, ...].detach().cpu().numpy()
                pred = pred[0, 0, ...].detach().cpu().numpy()
                im_paste = im_paste[0, 0, ...].detach().cpu().numpy()
                axes[0].pcolormesh(times, freqs, time_freq, cmap='PRGn', vmax=time_freq.max(), vmin=-time_freq.max())
                axes[1].pcolormesh(times, freqs, im_masked, cmap='PRGn', vmax=im_masked.max(), vmin=-im_masked.max())
                axes[2].pcolormesh(times, freqs, pred, cmap='PRGn', vmax=pred.max(), vmin=-pred.max())
                axes[3].pcolormesh(times, freqs, im_paste, cmap='PRGn', vmax=im_paste.max(), vmin=-im_paste.max())
                writer.add_figure(tag=f'epoch{epoch} | step {idx} | raw | masked | reconstruct | reconstruct + visible',
                                  figure=fig, global_step=epoch * idx + idx)

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
    parser.add_argument('--config', type=str, default=Path(__file__).parent / 'config.py',
                        help='path to the config file')
    parser.add_argument('--host', default='localhost', type=str, help='the master address for distributed training')
    parser.add_argument('--port', type=int, default='8910', help='the master port for distributed training')
    parser.add_argument('--world_size', type=int, default=int(os.environ['WORLD_SIZE']),
                        help='world size for distributed training')
    parser.add_argument('--rank', type=int, default=int(os.environ['RANK']), help='rank for the default process group')
    parser.add_argument('--local_rank', type=int, default=int(os.environ['LOCAL_RANK']), help='local rank on the node')
    parser.add_argument('--backend', type=str, default='nccl', help='backend for distributed communication')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
