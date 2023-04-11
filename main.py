import torch
from SignalMAE.mae import mae_vit_base_patch25


if __name__ == "__main__":
    x = torch.randn(10, 1, 7500)
    model = mae_vit_base_patch25()
    loss, pred, mask = model(x)
    a = 1