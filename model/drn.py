# Closed-loop Matters: Dual Regression Networks for Single Image Super-Resolution
# https://arxiv.org/pdf/2003.07018.pdf
# TODO: fix bugs in this code

import numpy as np
import torch
import torch.nn as nn

from .blocks import PixelUpSampler
from .rcan import RCAB


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class DownBlock(nn.Module):
    def __init__(self, arg, scale, nFeat=None, in_channels=None, out_channels=None):
        super().__init__()
        negval = arg.negval

        if nFeat is None:
            nFeat = arg.n_feats

        if in_channels is None:
            in_channels = arg.n_colors

        if out_channels is None:
            out_channels = arg.n_colors

        dual_block = [
            nn.Sequential(
                nn.Conv2d(in_channels, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=negval, inplace=True)
            )
        ]

        for _ in range(1, int(np.log2(scale))):
            dual_block.append(
                nn.Sequential(
                    nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=negval, inplace=True)
                )
            )

        dual_block.append(nn.Conv2d(nFeat, out_channels, kernel_size=3, stride=1, padding=1, bias=False))

        self.dual_module = nn.Sequential(*dual_block)

    def forward(self, x):
        x = self.dual_module(x)
        return x


class DRN(nn.Module):
    def __init__(self, arg):
        super().__init__()
        self.arg = arg
        self.scale = arg.scale
        self.phase = len(arg.scale)
        n_blocks = arg.n_blocks
        n_feats = arg.n_feats
        kernel_size = 3

        act = nn.ReLU()

        self.upsample = nn.Upsample(scale_factor=max(arg.scale), mode='bicubic', align_corners=False)

        self.head = default_conv(arg.n_colors, n_feats, kernel_size)

        self.down = [
            DownBlock(arg, 2, n_feats * pow(2, p), n_feats * pow(2, p), n_feats * pow(2, p + 1)
                      ) for p in range(self.phase)
        ]

        self.down = nn.ModuleList(self.down)

        up_body_blocks = [[RCAB(default_conv, n_feats * pow(2, p), kernel_size, act=act) for _ in range(n_blocks)]
                          for p in range(self.phase, 1, -1)]

        up_body_blocks.insert(0, [RCAB(default_conv, n_feats * pow(2, self.phase), kernel_size, act=act)
                                  for _ in range(n_blocks)])

        # The fisrt upsample block
        up = [[
            PixelUpSampler(default_conv, 2, n_feats * pow(2, self.phase), act=False),
            default_conv(n_feats * pow(2, self.phase), n_feats * pow(2, self.phase - 1), kernel_size=1)
        ]]

        # The rest upsample blocks
        for p in range(self.phase - 1, 0, -1):
            up.append([
                PixelUpSampler(default_conv, 2, 2 * n_feats * pow(2, p), act=False),
                default_conv(2 * n_feats * pow(2, p), n_feats * pow(2, p - 1), kernel_size=1)
            ])

        self.up_blocks = nn.ModuleList()
        for idx in range(self.phase):
            self.up_blocks.append(
                nn.Sequential(*up_body_blocks[idx], *up[idx])
            )

        # tail conv that output sr imgs
        tail = [default_conv(n_feats * pow(2, self.phase), arg.n_colors, kernel_size)]
        for p in range(self.phase, 0, -1):
            tail.append(
                default_conv(n_feats * pow(2, p), arg.n_colors, kernel_size)
            )
        self.tail = nn.ModuleList(tail)

    def forward(self, x):
        # upsample x to target sr size
        x = self.upsample(x)

        # preprocess
        x = self.head(x)

        # down phases,
        copies = []
        for idx in range(self.phase):
            copies.append(x)
            x = self.down[idx](x)

        # up phases
        sr = self.tail[0](x)
        results = [sr]
        for idx in range(self.phase):
            # upsample to SR features
            x = self.up_blocks[idx](x)
            # concat down features and upsample features
            x = torch.cat((x, copies[self.phase - idx - 1]), 1)
            # output sr imgs
            sr = self.tail[idx + 1](x)

            results.append(sr)

        return results
