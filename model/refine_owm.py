import torch as t
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=True, act=nn.ReLU(), res_scale=1, collect_data=False):
        super().__init__()
        if collect_data:
            bias = False
            bn = False
        if bn:
            self.conv1 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size, bias=bias),
                nn.GroupNorm(32, n_feats))
            self.conv2 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size, bias=bias),
                nn.GroupNorm(32, n_feats))
        else:
            self.conv1 = conv(n_feats, n_feats, kernel_size, bias=bias)
            self.conv2 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.act = act
        self.res_scale = res_scale

    def forward(self, x):
        if isinstance(x, tuple):
            x, input_list = x
        else:
            input_list = None
        identity = x
        if input_list is not None:
            input_list.append(x.detach().mean(0, True))
        x = self.conv1(x)
        x = self.act(x)
        if input_list:
            input_list.append(x.detach().mean(0, True))
        x = self.conv2(x)
        x = x * self.res_scale + identity
        if input_list is not None:
            return x, input_list
        return x


class RefineOWM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.scale = cfg.upscale_factor
        in_channels = 3
        out_channels = 3

        self.num_features = num_features = cfg.edsr.num_features
        num_blocks = cfg.edsr.num_blocks
        res_scale = cfg.edsr.res_scale
        conv = default_conv

        kernel_size = 3
        act = nn.ReLU()

        # define head module
        m_head = [conv(in_channels, num_features, kernel_size)]

        # define body module
        m_body = [ResBlock(conv, num_features, kernel_size, act=act, res_scale=res_scale) for _ in range(num_blocks)]
        m_body.append(conv(num_features, num_features, kernel_size))

        # define tail module
        m_tail = [ResBlock(conv, num_features, kernel_size, act=act, res_scale=res_scale, collect_data=True) for _ in
                  range(3)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.upsample = nn.functional.interpolate
        self.tail = nn.Sequential(*m_tail)
        self.last = conv(num_features, out_channels, kernel_size, bias=False)

    def forward(self, x):
        input_list = []
        size = t.tensor(x.shape[2:]) * self.scale
        x = self.head(x)

        res = self.body(x)
        res += x

        res = self.upsample(res, size=tuple(size), mode='bicubic', align_corners=True)

        if self.training:
            res, _ = self.tail((res, input_list))
            input_list.append(res.detach().mean(0, True))
            res = self.last(res)
            return res, input_list
        else:
            return self.last(self.tail(res))
