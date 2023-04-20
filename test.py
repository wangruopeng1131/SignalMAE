import torch
from SignalMAE.mae import mae_vit_base_patch25
from dataIO.load_dataset import PreTrainedDataset
from torch.utils.data.dataloader import DataLoader
from SignalMAE.conv import ConvHeader

if __name__ == '__main__':
    dataset = PreTrainedDataset(DATAPATH=r'/home/wang/ecg_data', EEGPATH=r'/home/wang/chb-mit-scalp-eeg')
    train = DataLoader(dataset, 10)
    for d in train:
        break
    model = mae_vit_base_patch25()
    loss, time_freq, pred, mask = model(d)
    loss.backward()
    y = model.unpatchify(pred)
