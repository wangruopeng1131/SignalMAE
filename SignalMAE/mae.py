# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Resample

from timm.models.vision_transformer import Block

from .tools.pos_embed import get_2d_sincos_pos_embed
from .patch_embedding import PatchEmbed
from .wavelets import Ricker, Morlet
from .conv import IIRFilter, ConvHeader


class MaskedAutoencoderSignal(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, wavelets_width, img_size: tuple, patch_size=tuple, in_chans: int = 1,
                 embed_dim: int = 1024, depth: int = 24, num_heads: int = 16,
                 decoder_embed_dim: int = 512, decoder_depth: int = 8, decoder_num_heads: int = 16,
                 mlp_ratio: float = 4., norm_layer=nn.LayerNorm, norm_pix_loss: bool = False):
        super().__init__()

        # Signal specifics
        # --------------------------------------------------------------------------
        signal_input = len(wavelets_width)
        self.morlet = Morlet(freqs=wavelets_width)
        self.iir = IIRFilter(fs=256, l_freq=0.1, h_freq=100)
        self.conv = ConvHeader(channels=signal_input, fs=256)
        # self.conv = nn.Conv1d(in_channels=signal_input, out_channels=signal_input, kernel_size=3, stride=2, padding=1)
        # self.resample = {'ecg': Resample(250, 256), 'eeg': Resample(256, 256)}
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop=0.1, drop_path=0.1,
                  attn_drop=0.1) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop=0.1,
                  drop_path=0.1, attn_drop=0.1) for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[0] * patch_size[1] * in_chans,
                                      bias=True)  # decoder to patch

        # self.decoder_classification = nn.Linear(decoder_embed_dim, patch_size[0] * patch_size[1] * in_chans,
        #                                         bias=True)  # For infoNCE
        # --------------------------------------------------------------------------
        self.reconstruction_weight = 1
        self.classification_weight = 0
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def reset_wavelets(self, wavelets_width):
        self.morlet = Morlet(freqs=wavelets_width)

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size,
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    self.patch_embed.grid_size, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size**2 *1)
        """
        p_h = self.patch_embed.patch_size[0]
        p_w = self.patch_embed.patch_size[1]
        assert imgs.shape[2] % p_h == 0

        h = imgs.shape[2] // p_h
        w = imgs.shape[3] // p_w
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p_h, w, p_w))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p_h * p_w))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *1)
        imgs: (N, 1, H, W)
        """
        p_h = self.patch_embed.patch_size[0]
        p_w = self.patch_embed.patch_size[1]
        h = self.patch_embed.grid_size[0]
        w = self.patch_embed.grid_size[1]
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p_h, p_w, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p_h, w * p_w))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def preprocess(self, x):
        # x = self.resample[signal_type](x)
        x = self.iir(x)
        x = self.conv(x)
        # x = self.morlet(x)  # batch, 1, 15 * 250 ->  batch, wavelet channels, 15 * 250
        # x = self.conv(x)  # batch, wavelet channels, 15 * 250 -> batch, wavelet channels, 15 * 125
        # B, t, f = x.shape
        # x = F.layer_norm(x, [t, f])  # normalize wavelet channels, 15 * 250
        return x.unsqueeze(1)  # batch, wavelet channels, 15 * 125 -> batch, 1, wavelet channels, 15 * 125

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)  # batch, 1, wavelet channels, 15 * 125 -> batch, patch, patch embedding

        B, P, D = x.shape
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:P + 1, :]  # suit for different signal length

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x_mse = self.decoder_pred(x)
        # x_infoNCE = self.decoder_classification(x)

        # remove cls token
        x_mse = x_mse[:, 1:, :]
        # x_infoNCE = x_infoNCE[:, 1:, :]

        return x_mse

    def forward_loss(self, imgs, mse, mask, tau=0.1):
        """
        imgs: [N, 1, H, W]
        pred: [N, L, p*p*1]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)

        # mse loss
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss_mse = (mse - target) ** 2
        loss_mse = loss_mse.mean(dim=-1)  # [N, L], mean loss per patch

        loss_mse = (loss_mse * mask).sum() / mask.sum()  # mean loss on removed patches

        # infoNCE loss
        # target_m_list = target[mask.to(torch.bool)]
        # logits_masked = logits[mask.to(torch.bool)]
        # all_dots = torch.matmul(logits_masked / tau, target_m_list.transpose(-1, -2))
        # log_softmax = torch.log_softmax(all_dots, dim=-1)
        # loss_info_nce = -torch.mean(torch.diagonal(log_softmax, dim1=-2, dim2=-1))

        # loss = self.reconstruction_weight * loss_mse + self.classification_weight * loss_info_nce
        return loss_mse

    def forward(self, signal, mask_ratio=0.75):
        time_freq = self.preprocess(signal)
        latent, mask, ids_restore = self.forward_encoder(time_freq, mask_ratio)
        mse = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*1]
        loss = self.forward_loss(time_freq, mse, mask)
        return loss, time_freq, mse, mask


def mae_vit_base_patch16_dec512d8b(**kwargs) -> nn.Module:
    model = MaskedAutoencoderSignal(wavelets_width=np.linspace(1, 32, 64), img_size=(64, 2560),
                                    patch_size=(16, 32), embed_dim=768, depth=12, num_heads=12,
                                    decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                                    mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs) -> nn.Module:
    model = MaskedAutoencoderSignal(wavelets_width=np.linspace(1, 32, 64), img_size=(64, 2560),
                                    patch_size=(16, 32), embed_dim=1024, depth=24, num_heads=16,
                                    decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                                    mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs) -> nn.Module:
    model = MaskedAutoencoderSignal(wavelets_width=np.linspace(1, 26, 64), img_size=(64, 3750),
                                    patch_size=(16, 16), embed_dim=1280, depth=32, num_heads=16,
                                    decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                                    mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch25 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch25 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch25 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
