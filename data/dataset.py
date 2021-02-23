import logging
from pathlib import Path

import numpy as np
import torch as t
import torch.utils.data as data
from skimage import io

from data.common import sub_mean, set_channel
from tools.matlabimresize import imresize


class PreLoadDataset(data.Dataset):
    def __init__(self, cfg, mode):
        self.scale = cfg.upscale_factor
        self.mode = mode
        self.cubic_a = cfg.dataset.cubic_a
        self.input_size = cfg.dataset.input_size
        self.method = cfg.dataset.interpolation
        dataset = cfg.dataset.train if mode == 'train' else cfg.dataset.test
        self.labels = [i for i in Path(cfg.dataset.path, dataset).iterdir()]
        assert len(self)
        logging.info(f'{self.__class__.__name__} created, total image:{len(self)}, dataset: {dataset}, mode: {mode}')

    def _get_images(self, index):
        label = self.labels[index]
        hr_image = set_channel(io.imread(str(label)))
        hr_image = sub_mean(hr_image).astype(np.float32)
        return hr_image

    def augment(self, item, mode):
        hr_image = self._get_images(item)
        if mode == 'train':
            patch_size = self.input_size * self.scale
            x = np.random.randint(0, hr_image.shape[0] - patch_size)
            y = np.random.randint(0, hr_image.shape[1] - patch_size)
            hr_image = hr_image[x:x + patch_size, y:y + patch_size]
            lr_image = self.interpolation(hr_image, 1 / self.scale)
            if t.rand(1) < 0.5:
                hr_image = hr_image[:, ::-1].copy()
                lr_image = lr_image[:, ::-1].copy()
            if t.rand(1) < 0.5:
                hr_image = hr_image[:, :, ::-1].copy()
                lr_image = lr_image[:, :, ::-1].copy()
        else:
            crop = self.scale ** 2
            height, width = np.array(hr_image.shape[:2]) // crop * crop
            hr_image = hr_image[:height, :width]
            lr_image = self.interpolation(hr_image, 1 / self.scale)
        return lr_image.transpose(2, 0, 1), hr_image.transpose(2, 0, 1)

    def interpolation(self, image, scale):
        result_image = imresize(image, scale, a=self.cubic_a)
        # result_image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return result_image

    def __getitem__(self, item):
        return self.augment(item, self.mode)

    def __len__(self):
        return len(self.labels)
