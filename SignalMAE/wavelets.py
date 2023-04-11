import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal
from scipy.signal import ricker, morlet2
from scipy import fft
import pywt

from .patch_embedding import PatchEmbed


def meyer(N, LB=-8, UB=8):
    lint = (UB - LB) / (2 * np.pi)
    x = np.arange(-N, N, 2) / (2 * lint)
    xa = np.abs(x)

    int1 = np.where((xa >= 2 * np.pi / 3) & (xa < 4 * np.pi / 3))
    int2 = np.where((xa >= 4 * np.pi / 3) & (xa < 8 * np.pi / 3))

    psihat = np.zeros(N, dtype=np.complex128)
    if int1[0].size != 0:
        psihat[int1] = np.exp(x[int1] * 0.5j) * np.sin(np.pi / 2 * meyeraux(3 / 2 / np.pi * xa[int1] - 1))
    if int2[0].size != 0:
        psihat[int2] = np.exp(x[int2] * 0.5j) * np.cos(np.pi / 2 * meyeraux(3 / 4 / np.pi * xa[int2] - 1))

    psihat = instdfft(psihat, LB, UB)
    return psihat


def meyeraux(x):
    p = np.array([-20, 70, -84, 35, 0, 0, 0, 0])
    y = np.zeros_like(x)
    y = np.polyval(p, x) * np.bitwise_and(x > 0, x <= 1)
    y[x > 1] = 1
    return y


def instdfft(xhat, lowb, uppb):
    n = len(xhat)

    delta = (uppb - lowb) / n

    omega = np.arange(-n, n, 2) / (2 * n * delta)

    xhat = fft.fftshift(xhat * np.exp(2j * np.pi * omega * lowb) / delta)

    x = fft.ifft(xhat)

    w = np.where((x.imag < np.sqrt(1e-6)))
    x[w] = x[w].real

    return x.real


class ContinuousWaveletsTransform(nn.Module):
    def __init__(self, freqs: np.ndarray = np.linspace(1, 51, 50), wavelet: str = 'morl', precision: int = 10):
        super(ContinuousWaveletsTransform, self).__init__()
        self.wavList = pywt.wavelist(kind='continuous')
        # 判断是不是复小波
        self.dtype = np.float64
        if 'c' in wavelet:
            self.dtype = np.complex128

        int_psi, x = pywt.integrate_wavelet(wavelet, precision=precision)
        int_psi = np.conj(int_psi) if self.dtype == np.float64 else int_psi

        step = x[1] - x[0]
        length = len(np.arange(freqs[-1] * (x[-1] - x[0]) + 1) / (freqs[-1] * step))

        wavelets = np.zeros((len(widths), length), dtype=self.dtype)
        for i, freq in enumerate(freqs):
            j = np.arange(freq * (x[-1] - x[0]) + 1) / (freq * step)
            j = j.astype(int)  # floor
            if j[-1] >= int_psi.size:
                j = np.extract(j < int_psi.size, j)
            int_psi_scale = int_psi[j][::-1]
            y = int((length - 1) / 2 - (int_psi_scale.shape[-1] - 1) / 2)
            wavelets[i, y:y + int_psi_scale.shape[-1]] = int_psi_scale

        self.wavelet = wavelets
        self.freqs = freqs
        real = torch.from_numpy(wavelets.real).unsqueeze(1)
        self.real = real.float()
        if self.dtype == np.complex128:
            imag = torch.from_numpy(wavelets.imag).unsqueeze(1)
            self.imag = imag.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.real.device != x.device:
            self.real = self.real.to(x.device)
            if self.dtype == np.complex128:
                self.imag = self.imag.to(x.device)
        x_real = F.conv1d(x, self.real, padding='same')
        if self.dtype == np.complex128:
            x_imag = F.conv1d(x, self.imag, padding='same')
            x_real = (x_real ** 2 + x_imag ** 2) ** 0.5
        return x_real


class Morlet(nn.Module):
    """
    Morlet 复小波
    """

    def __init__(self, w: int = 5, freqs: np.ndarray = np.linspace(1, 51, 50), fs: int = 128):
        super(Morlet, self).__init__()
        widths = w * fs / (2 * freqs * np.pi)
        length = int(10 * widths[0])  # 小波长度
        length = length if length % 2 != 0 else length + 1

        # 复小波需要分别对实部虚部卷积
        wavelets = np.empty((len(widths), length), dtype=np.complex128)
        for ind, width in enumerate(widths):
            cwavelet = np.conj(morlet2(length, width, w)[::-1])
            wavelets[ind] = cwavelet
        self.widths = widths
        self.freqs = freqs
        self.wavelet = wavelets
        real = torch.from_numpy(wavelets.real).unsqueeze(1)
        self.real = real.float()
        imag = torch.from_numpy(wavelets.imag).unsqueeze(1)
        self.imag = imag.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.real.device != x.device:
            self.real = self.real.to(x.device)
            self.imag = self.imag.to(x.device)
        x_real = F.conv1d(x, self.real, padding='same')
        x_imag = F.conv1d(x, self.imag, padding='same')
        return (x_real ** 2 + x_imag ** 2) ** 0.5


class Ricker(nn.Module):
    def __init__(self, widths=np.arange(1, 51)):
        super(Ricker, self).__init__()
        length = int(10 * widths[-1])  # 小波长度
        length = length if length % 2 != 0 else length + 1

        self.conv = nn.Conv1d(1, len(widths), kernel_size=length, bias=False, padding="same")
        wavelets = np.empty((len(widths), length))
        for ind, width in enumerate(widths):
            cwavelet = np.conj(ricker(length, width)[::-1])
            wavelets[ind] = cwavelet
        self.wavelet = wavelets
        wavelets = torch.from_numpy(wavelets).unsqueeze(1)
        self.conv.weight = nn.Parameter(wavelets)
        self.conv = self.conv.float()
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv.device != x.device:
            self.conv = self.conv.to(x.device)
        return self.conv(x)


if __name__ == "__main__":
    ricker = Ricker()
    widths = np.arange(1, 51, 0.5)
    cwt = ContinuousWaveletsTransform(widths, wavelet='cmor0.5-5.0')
    morlet = Morlet(widths=widths)
    t = np.linspace(-1, 1, 7500, endpoint=False)
    sig = np.cos(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)
    Sig = torch.from_numpy(sig).float()
    y = cwt(Sig.unsqueeze(0).unsqueeze(0))