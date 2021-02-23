# Pyramid Attention Networks for Image Restoration
# https://arxiv.org/pdf/2004.13824.pdf

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from model.blocks import BasicBlock, PixelUpSampler


def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = t.nn.ZeroPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError(
            'Unsupported padding type: {}.; Only "same" or "valid" are supported.'.format(padding))

    unfold = t.nn.Unfold(kernel_size=ksizes, dilation=rates, padding=0, stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = t.sum(x, dim=i, keepdim=keepdim)
    return x


class PyramidAttention(nn.Module):
    def __init__(self, level=5, res_scale=1, channel=64, reduction=2, stride=1, softmax_scale=10, average=True):
        super().__init__()
        self.ksize = 3
        self.stride = stride
        self.res_scale = res_scale
        self.softmax_scale = softmax_scale
        self.scale = [1 - i / 10 for i in range(level)]
        self.average = average
        self.escape_nan = nn.Parameter(t.tensor(1e-4, dtype=t.float32), True)
        self.conv_match_L_base = BasicBlock(channel, channel // reduction, 1, bn=False, act=nn.PReLU())
        self.conv_match = BasicBlock(channel, channel // reduction, 1, bn=False, act=nn.PReLU())
        self.conv_assembly = BasicBlock(channel, channel, 1, bn=False, act=nn.PReLU())

    def forward(self, x):
        res = x
        # theta
        match_base = self.conv_match_L_base(x)
        shape_base = list(res.size())
        input_groups = t.split(match_base, 1, dim=0)
        # patch size for matching
        kernel = self.ksize
        # raw_w is for reconstruction
        raw_w = []
        # w is for matching
        w = []
        # build feature pyramid
        for i in range(len(self.scale)):
            ref = x
            if self.scale[i] != 1:
                ref = F.interpolate(x, scale_factor=self.scale[i], mode='bicubic', align_corners=False)
            # feature transformation function f
            base = self.conv_assembly(ref)
            shape_input = base.shape
            # sampling
            raw_w_i = extract_image_patches(base, ksizes=[kernel, kernel],
                                            strides=[self.stride, self.stride],
                                            rates=[1, 1],
                                            padding='same')  # [N, C*k*k, L]
            raw_w_i = raw_w_i.view(shape_input[0], shape_input[1], kernel, kernel, -1)
            raw_w_i = raw_w_i.permute(0, 4, 1, 2, 3)  # raw_shape: [N, L, C, k, k]
            # raw_w_i_groups = t.split(raw_w_i, 1, dim=0)
            # raw_w.append(raw_w_i_groups)
            raw_w.append(raw_w_i.unsqueeze(1))

            # feature transformation function g
            ref_i = self.conv_match(ref)
            shape_ref = ref_i.shape
            # sampling
            w_i = extract_image_patches(ref_i, ksizes=[self.ksize, self.ksize],
                                        strides=[self.stride, self.stride],
                                        rates=[1, 1],
                                        padding='same')
            w_i = w_i.view(shape_ref[0], shape_ref[1], self.ksize, self.ksize, -1)
            w_i = w_i.permute(0, 4, 1, 2, 3)  # w shape: [N, L, C, k, k]
            # w_i_groups = t.split(w_i, 1, dim=0)
            # w.append(w_i_groups)
            w.append(w_i.unsqueeze(1))

        y = []
        for idx, xi in enumerate(input_groups):
            # group in a filter
            wi = t.cat([w[i][idx][0] for i in range(len(self.scale))], dim=0)  # [L, C, k, k]
            # normalize
            max_wi = t.max(t.sqrt(reduce_sum(t.pow(wi, 2), axis=[1, 2, 3], keepdim=True)),
                           self.escape_nan)
            wi_normed = wi / max_wi
            # matching
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)  # [1, L, H, W] L = shape_ref[2]*shape_ref[3]
            yi = yi.view(1, wi.shape[0], shape_base[2], shape_base[3])  # (B=1, C=32*32, H=32, W=32)
            # softmax matching score
            yi = F.softmax(yi * self.softmax_scale, dim=1)

            if not self.average:
                yi = (yi == yi.max(dim=1, keepdim=True)[0]).float()

            # deconv for patch pasting
            raw_wi = t.cat([raw_w[i][idx][0] for i in range(len(self.scale))], dim=0)
            yi = F.conv_transpose2d(yi, raw_wi, stride=self.stride, padding=1) / 4.
            y.append(yi)

        y = t.cat(y, dim=0) + res * self.res_scale  # back to the mini-batch
        return y


class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, bias=True, bn=False, act=nn.PReLU(), res_scale=1):

        super().__init__()
        m = []
        for i in range(2):
            m.append(BasicBlock(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x

        return res


class PANet(nn.Module):
    def __init__(self, args):
        super().__init__()

        n_resblock = args.panet.num_resblocks
        n_feats = args.panet.num_features
        kernel_size = 3
        scale = args.upscale_factor
        res_scale = args.panet.res_scale

        act = nn.ReLU()

        self.msa = PyramidAttention(channel=n_feats, reduction=8, res_scale=res_scale)
        # define head module
        m_head = [BasicBlock(args.model.in_channels, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(n_feats, kernel_size, act=act, res_scale=res_scale
                     ) for _ in range(n_resblock // 2)
        ]
        m_body.append(self.msa)
        for _ in range(n_resblock // 2):
            m_body.append(ResBlock(
                n_feats, kernel_size, act=act, res_scale=res_scale
            ))
        m_body.append(BasicBlock(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            PixelUpSampler(BasicBlock, scale, n_feats, act=False),
            BasicBlock(n_feats, args.model.out_channels, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x
