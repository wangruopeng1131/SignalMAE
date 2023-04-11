import sys
from pathlib import Path
from colossalai.logging import get_dist_logger
import colossalai
import torch
import os
from colossalai.core import global_context as gpc
from colossalai.utils import get_dataloader, MultiTimer
from colossalai.trainer import Trainer, hooks
from colossalai.nn.metric import Accuracy
from torchvision import transforms
from colossalai.nn.lr_scheduler import CosineAnnealingLR
from .baseline.ResidualNet import load_model
from tqdm import tqdm
from titans.utils import barrier_context
from dataIO.load_dataset import NoiseDataset

sys.path.extend(['/home/wangruopeng/noise_detect_2023_vision/'])

DATA_ROOT = Path(os.environ.get('DATA', './data'))


def main():
    parser = colossalai.get_default_parser()
    parser.add_argument('--use_trainer', action='store_true', help='whether to use trainer')
    args = parser.parse_args()

    colossalai.launch_from_torch(config='./config.py')

    logger = get_dist_logger()

    # build resnet
    model = load_model('resnet18')
    model = torch.compile(model, fullgraph=True)
    with barrier_context():
        # build dataloaders
        train_dataset = NoiseDataset(train=True)

    test_dataset = NoiseDataset(train=False)

    train_dataloader = get_dataloader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=gpc.config.BATCH_SIZE,
        pin_memory=True,
    )

    test_dataloader = get_dataloader(
        dataset=test_dataset,
        add_sampler=False,
        batch_size=gpc.config.BATCH_SIZE,
        pin_memory=True,
    )

    # build criterion
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, betas=(0.9, 0.95), weight_decay=0.05)

    # lr_scheduler
    lr_scheduler = CosineAnnealingLR(optimizer, total_steps=gpc.config.NUM_EPOCHS)

    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(
        model,
        optimizer,
        criterion,
        train_dataloader,
        test_dataloader,
    )

    if not args.use_trainer:
        for epoch in range(gpc.config.NUM_EPOCHS):
            engine.train()
            if gpc.get_global_rank() == 0:
                train_dl = tqdm(train_dataloader)
            else:
                train_dl = train_dataloader
            for img, label in train_dl:
                img = img.cuda()
                label = label.cuda()

                engine.zero_grad()
                output = engine(img)
                train_loss = engine.criterion(output, label)
                engine.backward(train_loss)
                engine.step()
            lr_scheduler.step()

            engine.eval()
            correct = 0
            total = 0
            for img, label in test_dataloader:
                img = img.cuda()
                label = label.cuda()

                with torch.no_grad():
                    output = engine(img)
                    test_loss = engine.criterion(output, label)
                pred = torch.argmax(output, dim=-1)
                correct += torch.sum(pred == label)
                total += img.size(0)

            logger.info(
                f"Epoch {epoch} - train loss: {train_loss:.5}, test loss: {test_loss:.5}, acc: {correct / total:.5}, lr: {lr_scheduler.get_last_lr()[0]:.5g}",
                ranks=[0])
    else:
        # build a timer to measure time
        timer = MultiTimer()

        # create a trainer object
        trainer = Trainer(engine=engine, timer=timer, logger=logger)

        # define the hooks to attach to the trainer
        hook_list = [
            hooks.LossHook(),
            hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=True),
            hooks.AccuracyHook(accuracy_func=Accuracy()),
            hooks.LogMetricByEpochHook(logger),
            hooks.LogMemoryByEpochHook(logger),
            hooks.LogTimingByEpochHook(timer, logger),

            # you can uncomment these lines if you wish to use them
            # hooks.TensorboardHook(log_dir='./tb_logs', ranks=[0]),
            # hooks.SaveCheckpointHook(checkpoint_dir='./ckpt')
        ]

        # start training
        trainer.fit(train_dataloader=train_dataloader,
                    epochs=gpc.config.NUM_EPOCHS,
                    test_dataloader=test_dataloader,
                    test_interval=1,
                    hooks=hook_list,
                    display_progress=True)


if __name__ == '__main__':
    main()
