import torch
from SignalMAE.mae import mae_vit_base_patch25

if __name__ == '__main__':
    d = torch.rand(10, 1, 2560)

    model = mae_vit_base_patch25()
    loss, time_freq, pred, mask = model(d)
    loss.backward()
    y = model.unpatchify(pred)
