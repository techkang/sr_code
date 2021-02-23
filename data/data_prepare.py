import logging
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from skimage import io
from tqdm import tqdm

from config import default_argument_parser, get_cfg, default_setup
from data.common import sub_mean
from tools.matlabimresize import imresize


class DataGenerator(object):
    def __init__(self, cfg):
        self.method = cfg.dataset.interpolation
        self.folders = cfg.dataset.train
        self.dataset_path = Path(cfg.dataset.path)
        self.train_folder = Path(cfg.dataset.train_folder)
        self.scale = cfg.upscale_factor
        self.input_size = cfg.dataset.input_size
        self.channel = 1 if cfg.dataset.gray else 3

        if self.channel == 3:
            self.rgb_mean = np.array((0.4488, 0.4371, 0.4040)).reshape((1, 3, 1, 1))
        else:
            self.rgb_mean = np.array(0.5)
        self.thread_num = cfg.dataloader.num_workers
        self.reshape_size = [1., 0.8, 0.7, 0.6, 0.5]
        self.hr_folder = None
        self.lr_folder = None

    def get_train_files(self, folder):
        """return training files according to file structure."""
        return self.dataset_path / folder / self.train_folder

    def pre_process(self):
        logging.info(f'start to process dataset:{self.folders}')
        hr_folders = [self.dataset_path / d / 'Augment' / 'HR' / f'x{self.scale}' for d in self.folders]
        lr_folders = [self.dataset_path / d / 'Augment' / 'LR' / f'x{self.scale}' for d in self.folders]
        logging.info(f'Using multi processing with {self.thread_num} threads.')
        for i, (hr_folder, lr_folder) in enumerate(zip(hr_folders, lr_folders)):
            self.hr_folder = hr_folder
            self.lr_folder = lr_folder
            shutil.rmtree(hr_folder, ignore_errors=True)
            shutil.rmtree(lr_folder, ignore_errors=True)
            hr_folder.mkdir(parents=True)
            lr_folder.mkdir(parents=True)
            files = self.get_train_files(self.folders[i])
            files = list(files.iterdir())
            start = time.perf_counter()
            pool = Pool(self.thread_num)
            for _ in tqdm(pool.imap(self.process, zip(files)), total=len(files)):
                pass
            # if you need debug, uncomment this line and comment two lines up.
            # for file in files:
            #     self.process([file])
            pool.close()
            pool.join()
            logging.info(f'Finish generate dataset {self.folders[i]}, using {time.perf_counter() - start:.2f}s.')

    def process(self, files):
        file = files[0]
        source = io.imread(str(file))
        for i, size in enumerate(self.reshape_size):
            height, width, channel = source.shape
            scale = self.input_size * self.scale / size
            if height < scale or width < scale:
                continue
            image = imresize(source, scale=size)
            split = self.split_image(image)
            label = self.image_aug(split)
            lr_image = self.interpolation(label, 1 / self.scale)
            self.save(sub_mean(label), self.hr_folder / (file.stem + f'_{i}'))
            self.save(sub_mean(lr_image), self.lr_folder / (file.stem + f'_{i}'))

    def split_image(self, image: np.array):
        height, width, channel = image.shape
        scale = self.input_size * self.scale
        image = image[:height // scale * scale, :width // scale * scale]
        split = image.reshape(height // scale, scale, width // scale, scale, channel)
        split = split.transpose(0, 2, 1, 3, 4).reshape(-1, scale, scale, channel)
        return split

    def image_aug(self, images):
        def aug(image):
            h_flip = np.random.rand() < 0.5
            v_flip = np.random.rand() < 0.5
            rot90 = np.random.rand() < 0.5
            if h_flip:
                image = image[:, ::-1, :]
            if v_flip:
                image = image[::-1, :, :]
            if rot90:
                image = image.transpose(1, 0, 2)
            return image

        return np.array([aug(i) for i in images])

    def save(self, images, name):
        for i, image in enumerate(images):
            np.save(f'{name}_{str(i).zfill(3)}', image)

    def interpolation(self, images, scale):
        result_images = np.array([imresize(i, scale=scale) for i in images])
        return result_images


class RealDataGenerator(DataGenerator):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.folders = cfg.dataset.train_real

    def get_train_files(self, folder):
        """return training files according to file structure."""
        return self.dataset_path / folder / self.train_folder / f'{self.scale}'

    def split_image_pair(self, hr_image, lr_image):
        stride_lr = self.input_size
        stride_hr = stride_lr * self.scale
        lr_images = []
        hr_images = []
        for i in range(lr_image.shape[0] // stride_lr):
            for j in range(lr_image.shape[1] // stride_lr):
                lr_images.append(lr_image[i * stride_lr:(i + 1) * stride_lr, j * stride_lr:(j + 1) * stride_lr])
                hr_images.append(hr_image[i * stride_hr:(i + 1) * stride_hr, j * stride_hr:(j + 1) * stride_hr])
        return list(zip(hr_images, lr_images))

    def image_aug_pair(self, image_pairs):
        def aug(image, h_flip, v_flip, rot90):
            if h_flip:
                image = image[:, ::-1, :]
            if v_flip:
                image = image[::-1, :, :]
            if rot90:
                image = image.transpose(1, 0, 2)
            return image

        result = []
        for h in (True, False):
            for v in (True, False):
                for r in (True, False):
                    for images in image_pairs:
                        result.append([aug(img, h, v, r) for img in images])
        return result

    def process(self, files):
        file = files[0]
        if 'LR' in str(file):
            return
        hr_image = io.imread(str(file))
        lr_image = io.imread(str(file)[::-1].replace('RH', f'{self.scale}RL', 1)[::-1])
        image_pairs = self.split_image_pair(hr_image, lr_image)
        image_pairs = self.image_aug_pair(image_pairs)
        for i, (label, lr_image) in enumerate(image_pairs):
            self.save(sub_mean(label), self.hr_folder / (file.stem + f'_{i}'))
            self.save(sub_mean(lr_image), self.lr_folder / (file.stem + f'_{i}'))


def setup(args):
    cfg = get_cfg()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


if __name__ == '__main__':
    arg = default_argument_parser().parse_args()
    print(f'Command Line Args: {arg}')
    config = setup(arg)
    # generator = DataGenerator(config)
    generator = RealDataGenerator(config).pre_process()
