import torch
from torch.nn import functional as F
# from torchaudio.transforms import TimeMasking, FrequencyMasking


def multi_crop():
    pass


def random_crop(data, sfreq, win=4):
    data = F.layer_norm(data, (data.size()[-1],))
    time = int(data.shape[-1] / sfreq)
    if time < win:
        return data[[0, -1]], data[1]
    start = torch.randint(0, time - win, size=(1,))
    times = torch.arange(int(start)*sfreq, int(start+win)*sfreq)
    return data[[0, -1]][:, times], data[1][times]