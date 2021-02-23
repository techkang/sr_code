import torch.nn.functional as F


def mse_loss(lr_image, hr_image, sr_image, reduction='mean'):
    return F.mse_loss(hr_image, sr_image, reduction=reduction)


def l1_loss(lr_image, hr_image, sr_image, reduction='mean'):
    return F.l1_loss(hr_image, sr_image, reduction=reduction)
