import torch
from SignalMAE.mae import mae_vit_base_patch25
from dataIO.utlis import get_eeg_path
from mne.io import read_raw_edf
from SignalMAE.conv import ConvHeader

if __name__ == '__main__':
    d = torch.rand(10, 1, 2560)
    conv = ConvHeader()
    a = conv(d)
    model = mae_vit_base_patch25()
    loss, time_freq, pred, mask = model(d)
    loss.backward()
    y = model.unpatchify(pred)
