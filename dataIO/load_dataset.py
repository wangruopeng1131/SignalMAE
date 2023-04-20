import os
import numpy as np
from wfdb import rdrecord
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchaudio.transforms import Resample
from colossalai.registry import DATASETS

import mne
from mne.io import read_raw_edf

mne.utils.use_log_level('error')
from .utlis import get_path, get_eeg_path

ecg_resample = Resample(250, 256)
eeg_transform = transforms.RandomCrop(size=(1, 256 * 10))  # 256采样率 * 30S
ecg_transform = transforms.RandomCrop(size=(1, 250 * 10))  # 250采样率 * 30S

np.random.seed(1024)


@DATASETS.register_module
class NoiseDataset(Dataset):
    def __init__(self, file_path=None):
        super(NoiseDataset, self).__init__()
        self.file_path = file_path
        self.file_length = len(file_path)
        self.label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

    def __getitem__(self, index):
        data_dict = np.load(self.file_path[index], allow_pickle=True).item()
        data = data_dict['data'][np.newaxis]
        data = torch.from_numpy(data).float()
        labels = torch.from_numpy(data_dict['labels'])
        return data, labels.long()

    def __len__(self):
        return self.file_length


@DATASETS.register_module
class PreTrainedDataset(Dataset):
    def __init__(self, train=True, DATAPATH=None, EEGPATH=None):
        super(PreTrainedDataset, self).__init__()
        self.train = train
        self.ecg_file_path = get_path(DATAPATH)
        np.random.shuffle(self.ecg_file_path)
        self.ecg_file_path = self.ecg_file_path[: int(len(self.ecg_file_path) / 2)]
        self.eeg_file_path = get_eeg_path(EEGPATH) * 10

        self.file_path = self.ecg_file_path + self.eeg_file_path
        np.random.shuffle(self.file_path)

        self.file_length = len(self.file_path)
        self.train = self.file_path
        self.test = self.file_path[int(self.file_length * 0.95):]

    def __getitem__(self, index):
        if self.train:
            file = self.train[index]
            if 'icentia11k' in file:
                data = rdrecord(file).p_signal.reshape(1, -1)
                data = torch.from_numpy(data).float()
                data = ecg_transform(data)
                data = ecg_resample(data)

            elif '.edf' in file:
                raw = read_raw_edf(file, verbose='error').load_data(verbose=False)
                random_ch = np.random.randint(0, len(raw.ch_names))
                data = raw.get_data(raw.ch_names[random_ch])
                data = torch.from_numpy(data)
                data = eeg_transform(data)
        else:
            file = self.test[index]
            if 'icentia11k' in file:
                data = rdrecord(file).p_signal.reshape(1, -1)
                data = torch.from_numpy(data).float()
                data = ecg_transform(data)
                data = ecg_resample(data)
            elif '.edf' in file:
                raw = read_raw_edf(file, verbose='error').load_data(verbose=False)
                random_ch = np.random.randint(0, len(raw.ch_names))
                data = raw.get_data(raw.ch_names[random_ch])
                data = torch.from_numpy(data)
                data = eeg_transform(data)
        return data.float()

    def __len__(self):
        if self.train:
            return self.file_length
        else:
            return len(self.test)


def load_noise_dataset(NOISE_DATA_ROOT=None):
    file_path = [os.path.join(NOISE_DATA_ROOT, i) for i in os.listdir(NOISE_DATA_ROOT)]
    np.random.shuffle(file_path)
    train_files = file_path[:int(0.8 * len(file_path))]
    valid_files = file_path[int(0.8 * len(file_path)):int(0.9 * len(file_path))]
    test_files = file_path[int(0.9 * len(file_path)):]
    return NoiseDataset(train_files), NoiseDataset(valid_files), NoiseDataset(test_files)
