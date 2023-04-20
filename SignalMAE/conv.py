import torch
import torch.nn as nn
import torch.nn.functional as F

from torchaudio.functional import filtfilt

from scipy.signal import butter


class IIRFilter(nn.Module):
    def __init__(self, fs: float = 256, l_freq: float = 0.1, h_freq: float = 100):
        super(IIRFilter, self).__init__()
        b, a = butter(4, [l_freq, h_freq], fs=fs, btype='bandpass')
        self.b = torch.from_numpy(b).float()
        self.a = torch.from_numpy(a).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return filtfilt(x, self.a, self.b, False)


class ConvHeader(nn.Module):
    def __init__(self, channels: int = 64, fs: int = 256):
        super(ConvHeader, self).__init__()

        self.conv1 = nn.Conv1d(1, channels, kernel_size=int(fs / 2) - 1, padding='same')
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=int(fs / 2) - 1, padding='same')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        return x