import numpy as np

from tools.comm import rgb_to_ycbcr


def augment(*images, h_flip=True, rot=True):
    h_flip = h_flip and np.random.random() < 0.5
    v_flip = rot and np.random.random() < 0.5
    rot90 = rot and np.random.random() < 0.5

    def _augment(img):
        if h_flip:
            img = img[:, ::-1, :]
        if v_flip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)

        return img

    return [_augment(i) for i in images]


def sub_mean(image, rgb_range=255):
    """image shape: [batch size] * height * width * channel"""
    assert image.shape[-1] == 3, f'channel must be last dimension, get: {image.shape}'
    rgb_mean = np.array((0.4488, 0.4371, 0.4040), np.float32)
    image = image.astype(np.float32) / rgb_range - rgb_mean
    return image


def add_mean(image, rgb_range=255):
    """image shape: [batch size] * height * width * channel"""
    assert image.shape[-1] == 3, 'channel must be last dimension'
    rgb_mean = np.array((0.4488, 0.4371, 0.4040), np.float32)
    image = (image + rgb_mean) * rgb_range
    image = np.clip(image, 0, rgb_range)
    if rgb_range == 255:
        image = image.astype(np.uint8)
    return image


def set_channel(image, n_channels=3):
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    c = image.shape[-1]
    if n_channels == 1 and c == 3:
        image = np.expand_dims(rgb_to_ycbcr(image), -1)
    elif n_channels == 3 and c == 1:
        image = np.concatenate([image] * n_channels, -1)

    return image
