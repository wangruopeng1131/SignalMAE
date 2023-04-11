import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from torchvision import transforms

from transform import random_crop

np.random.seed(8848)


class Dataset(Dataset):
    def __init__(self, files, sfreq):
        self.files = files
        self.sfreq = sfreq

    def _read(self, file):
        data = np.load(file)
        return torch.from_numpy(data).float()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        data = self._read(self.files[item])
        x, y = random_crop(data, self.sfreq)
        return x.permute(1, 0), y.unsqueeze(-1)


def get_dataset(root=r'/home/wangruopeng/database/blood pressure', ratio=0.8, sfreq=125):
    files = np.array([os.path.join(root, file) for file in os.listdir(root)])
    idx = np.arange(len(files))
    np.random.shuffle(idx)
    length = int(ratio * len(files))
    train = Dataset(files[list(idx[:length])], sfreq)
    test = Dataset(files[list(idx[length:])], sfreq)
    return train, test


if __name__ == '__main__':
    train, test = get_dataset()
    t = DataLoader(train, batch_size=10, shuffle=True)
    for x, y in t:
        print(x.shape, y.shape)
