import torch.nn as nn
import model
from model.blocks import BasicBlock


class Translation(nn.Module):
    def __init__(self, num_layers=3):
        super().__init__()
        layers = []
        in_channel = 3
        out_channel = 64
        kernel = 3
        for i in range(num_layers):
            if i:
                in_channel = out_channel
                out_channel *= 2
            layers.append(BasicBlock(in_channel, out_channel, kernel, stride=2, act='leaky', bn=True))
        self.encode = nn.Sequential(*layers)
        layers = []
        for i in range(num_layers):
            layers.append(BasicBlock(out_channel, in_channel, kernel, conv='deconv', stride=2,
                                     act='leaky', bn=True, channel=out_channel))
            out_channel = in_channel
            if i != 1:
                in_channel //= 2
            else:
                in_channel = 3
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, patch_size):
        super().__init__()

        in_channels = 3
        out_channels = 64
        depth = 7
        bn = True
        act = nn.LeakyReLU(negative_slope=0.2)

        m_features = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            act,
            nn.BatchNorm2d(out_channels)
        ]
        for i in range(depth):
            in_channels = out_channels
            if i % 2 == 1:
                stride = 1
                out_channels *= 2
            else:
                stride = 2
            m_features.append(BasicBlock(
                in_channels, out_channels, 3, stride=stride, bn=bn, act=act
            ))

        self.features = nn.Sequential(*m_features)

        assert not patch_size % (
                2 ** ((depth + 1) // 2)), f'input size: {patch_size} % {2 ** ((depth + 1) // 2)} must be 0!'
        patch_size = patch_size // (2 ** ((depth + 1) // 2))
        m_classifier = [
            nn.Linear(out_channels * patch_size ** 2, 1024),
            act,
            nn.Linear(1024, 1)
        ]
        self.classifier = nn.Sequential(*m_classifier)

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features.view(features.size(0), -1))

        return output


class CycleSR(nn.Module):

    def __init__(self, arg):
        super().__init__()
        self.backbone_real: nn.Module = getattr(model, arg.cyclesr.srnet)(arg)
        self.backbone_synthetic: nn.Module = getattr(model, arg.cyclesr.srnet)(arg)
        self.translation_s2r = Translation()
        self.translation_r2s = Translation()
        self.discriminator_synthetic = Discriminator(arg.datasets.input_size)
        self.discriminator_real = Discriminator(arg.datasets.input_size)
        self.scale = arg.upscale_factor

    def forward(self, x, synthetic=True, is_train=False):
        if not is_train:
            return self.backbone_synthetic(x)
        if synthetic:
            dis_synthetic = self.discriminator_synthetic(x)
            trans = self.translation_s2r(x)
            sr_real = self.backbone_real(trans)
            dis_real = self.discriminator_real(trans)
            sr_synthetic = self.backbone_synthetic(x)
            return sr_synthetic, sr_real, dis_synthetic, dis_real
        else:
            dis_real = self.discriminator_real(x)
            trans = self.translation_r2s(x)
            # sr_synthetic = self.backbone_synthetic(trans)
            dis_synthetic = self.discriminator_synthetic(trans)
            # sr_real = self.backbone_real(x)
            return dis_synthetic, dis_real


if __name__ == '__main__':
    from config import get_cfg
    import torch as t

    net = CycleSR(get_cfg()).to('cuda')

    dummy_input = t.randn((1, 3, 32, 32)).to('cuda')
    result = net(dummy_input)
    print(type(result), len(result), result[0].shape, result[1].shape, result[2].shape, result[3].shape)
