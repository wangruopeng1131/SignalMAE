import torch
from SignalMAE.mae import mae_vit_base_patch25
from dataIO.load_dataset import PreTrainedDataset
from colossalai.utils import get_dataloader
import matplotlib.pyplot as plt

if __name__ == '__main__':
    train_dataset = PreTrainedDataset(train=True, DATAPATH=r'/mnt/d/database/icentia11k-single-lead-continuous-raw-electrocardiogram-dataset-1.0')
    train_dataloader = get_dataloader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=2,
        pin_memory=True,
    )

    for i, d in enumerate(train_dataloader):
        print(i, d)
        break

    model = mae_vit_base_patch25()
    loss, time_freq, pred, mask = model(d)
    y = model.unpatchify(pred)
