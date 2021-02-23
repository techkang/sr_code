import logging
import math
import shutil
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.nn.functional as F
from torch.utils import tensorboard
from tqdm import tqdm

import tools.dist as du
from data import build_dataloader
from data.common import add_mean
from loss.build_loss import build_loss
from lr_scheduler.build_lr_scheduler import build_lr_scheduler
from model import build_model
from tools.checkpointer import Checkpointer
from tools.comm import cal_psnr, cal_ssim
from tools.matlabimresize import imresize


class Trainer:
    def __init__(self, cfg, resume):
        self.loss_thresh = 1000
        self.cfg = cfg
        self.device = t.device(cfg.device)
        self.model = build_model(cfg)
        self.output_dir = Path(cfg.output_dir)
        self.optimizer = self.build_optimizer(cfg, self.model)
        self.train_loader = build_dataloader(cfg, 'train')
        self.test_loader = build_dataloader(cfg, 'test')
        self.scheduler = build_lr_scheduler(cfg, self.optimizer)
        self.loss_function = build_loss(cfg)
        self.log_name = cfg.log_name if cfg.log_name else cfg.model.name
        self.checkpointer = Checkpointer(self.model, self.output_dir / 'checkpoint' / self.log_name,
                                         optimizer=self.optimizer,
                                         scheduler=self.scheduler)
        self.writer = None
        self.iter = self.start_iter = self.resume_or_load(resume)
        self.max_iter = cfg.solver.max_iter

        self._data_loader_iter = None
        self.all_metrics_list = []
        self.best_score = {'psnr': None}
        self.tqdm = None

        baseline = self.get_baseline()
        if cfg.num_gpus > 1 and du.is_master_proc():
            logging.warning(f'baseline is inaccurate due to num gpus > 1')
        logging.info(f'dataset: {self.cfg.dataset.test}, upscale:{self.cfg.upscale_factor}')
        logging.info(f'baseline psnr: {baseline["psnr"]:.3f}, ssim: {baseline["ssim"]:.3f}')

    def resume_or_load(self, resume=True):
        """
        If `resume==True`, and last checkpoint exists, resume from it.

        Otherwise, load a model specified by the config.

        Args:
            resume (bool): whether to do resume or not
        """
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration (or iter zero if there's no checkpoint).
        return (
                self.checkpointer.resume_or_load(self.cfg.model.weights, resume=resume).get(
                    "iteration", self.cfg.start_iter - 1
                )
                + 1
        )

    def build_optimizer(self, cfg, net):
        optimizer = getattr(t.optim, cfg.optimizer.name)
        return optimizer(net.parameters(), cfg.lr_scheduler.base_lr)

    def to(self, obj, device):
        if hasattr(obj, 'to'):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: self.to(v, device) for k, v in obj.items()}
        elif isinstance(obj, Iterable):
            return type(obj)([self.to(item, device) for item in obj])
        else:
            logging.warning(f'object {obj} can not be moved to device {device}!')
            return obj

    def _detect_anomaly(self, losses):
        if not t.isfinite(losses).all():
            raise FloatingPointError(f"Loss became infinite or NaN at iteration"
                                     f"={self.iter}!\nlosses = {losses}")

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu() if isinstance(v, t.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        self.all_metrics_list.append(metrics_dict)

    def prepare_for_tbx(self, images, squeeze=False):
        flag = False
        if isinstance(images, t.Tensor):
            flag = True
            images = [images]
        images = [image.detach().cpu().numpy() for image in images]
        images = [add_mean(image.transpose(0, 2, 3, 1)) for image in images]
        if squeeze:
            images = [np.squeeze(image) for image in images]
        if flag:
            return images[0]
        else:
            return images

    def train(self):
        logging.info("Starting training from iteration {}".format(self.start_iter))

        self.before_train()
        for self.iter in range(self.start_iter, self.max_iter):
            self.before_step()
            self.run_step()
            self.after_step()
        self.after_train()

    @t.no_grad()
    def get_baseline(self):
        test_gray_psnr = []
        test_gray_ssim = []
        baseline = {}
        for i, (lr_image, hr_image) in enumerate(self.test_loader):
            lr_image, hr_image = self.prepare_for_tbx([lr_image, hr_image], squeeze=True)
            sr_image = imresize(lr_image, shape=hr_image.shape[:2])
            test_gray_psnr.append(cal_psnr(sr_image, hr_image, bolder=self.cfg.upscale_factor, gray=True))
            test_gray_ssim.append(cal_ssim(sr_image, hr_image, bolder=self.cfg.upscale_factor, gray=True))
        baseline['psnr'] = np.mean(test_gray_psnr)
        baseline['ssim'] = np.mean(test_gray_ssim)
        return baseline

    def before_train(self):
        # prepare for tensorboard
        if du.is_master_proc():
            if self.cfg.tensorboard.name:
                folder = self.output_dir / 'tbfile' / self.cfg.tensorboard.name
            else:
                folder = self.output_dir / 'tbfile' / self.log_name
            if self.cfg.tensorboard.clear_before:
                shutil.rmtree(folder, ignore_errors=True)
            folder.mkdir(parents=True, exist_ok=True)
            self.writer = tensorboard.SummaryWriter(folder)

            plt.switch_backend('agg')

            total_num = sum(p.numel() for p in self.model.parameters())
            trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logging.info(f'total parameters:{total_num:.4e}, trainable parameters: {trainable_num:.4e}')

            self.tqdm = tqdm(total=self.cfg.solver.max_iter)
            self.tqdm.update(self.start_iter)
            self.tqdm.display('')
        else:
            self.tqdm = tqdm(disable=True)
        if self._data_loader_iter is None:
            logging.info('init dataloader')
            self._data_loader_iter = iter(self.train_loader)
            logging.info('train dataloader init finished')
        logging.info('begin to train model.')

    def before_step(self):
        if not self.iter % self.cfg.solver.test_interval:
            self.test()

    def run_step(self):
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        batch = next(self._data_loader_iter)
        batch = self.to(batch, self.device)
        data_time = time.perf_counter() - start

        lr_image, hr_image = batch
        sr_image = self.model(lr_image)
        losses = self.loss_function(lr_image, hr_image, sr_image)
        if isinstance(losses, dict):
            loss_backward = sum(losses.values())
        else:
            loss_backward = losses
        self.optimizer.zero_grad()
        loss_backward.backward()
        self.optimizer.step()

        if isinstance(losses, dict):
            du.all_reduce(list(losses.values()))
            loss_dict = losses
            losses = sum(losses.values())
        else:
            loss_dict = {}
        loss_dict.update({'loss': losses})
        self._detect_anomaly(losses)
        postfix_dict = {k: f'{v.detach().cpu():.2e}' for k, v in loss_dict.items()}
        self.tqdm.set_postfix(postfix_dict)

        metrics_dict = {**loss_dict, "data_time": data_time, "lr": self.scheduler.get_last_lr()[0]}
        self._write_metrics(metrics_dict)

        self.scheduler.step(None)
        self.tqdm.update()

    def after_step(self):
        if not du.is_master_proc():
            return
        if not self.iter % self.cfg.tensorboard.save_freq:
            if "data_time" in self.all_metrics_list[0]:
                data_time = np.max([x.pop("data_time") for x in self.all_metrics_list])
                self.writer.add_scalar("data_time", data_time, self.iter)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in self.all_metrics_list]) for k in self.all_metrics_list[0].keys()
            }
            self.all_metrics_list.clear()

            for k, v in metrics_dict.items():
                self.writer.add_scalar(k, v, self.iter)
        if not (self.iter + 1) % self.cfg.solver.save_interval:
            self.checkpointer.save(f'{self.cfg.model.name}_SR{self.cfg.upscale_factor}_{self.iter + 1}')

    def save_best(self, score):
        if du.is_master_proc() and not self.cfg.eval_only:
            for k, v in self.best_score.items():
                if v is None:
                    self.best_score[k] = score[k]
                elif v < score[k]:
                    self.best_score[k] = score[k]
                    logging.info(f'Find max {k} with best score: {v:.3f} at iter: {self.iter}')
                    self.checkpointer.save(f'{self.cfg.model.name}_max_{k}', iteration=self.iter)

    def collect_info(self, lr_image, hr_image, sr_image):
        psnr = cal_psnr(hr_image, sr_image)
        ssim = cal_ssim(hr_image, sr_image)
        return {'psnr': psnr, 'ssim': ssim}

    def analyse_info(self, info_list):
        info = defaultdict(float)
        for d in info_list:
            for k, v in d.items():
                info[k] += v
        for k, v in info.items():
            info[k] /= len(info_list)
        return info

    def chop_pred(self, lr_image, hr_image):
        _, _, h_lr, w_lr = lr_image.shape
        stride = self.cfg.solver.chop_size
        pad_h_lr = math.ceil(h_lr / stride) * stride - h_lr
        pad_w_lr = math.ceil(w_lr / stride) * stride - w_lr
        lr_test = F.pad(lr_image, [0, pad_w_lr, 0, pad_h_lr])

        batch_size, channel, h_hr, w_hr = hr_image.shape
        stride_hr = stride * self.cfg.upscale_factor
        pred_h = math.ceil(h_hr / stride_hr) * stride_hr
        pred_w = math.ceil(w_hr / stride_hr) * stride_hr

        pred = t.zeros((batch_size, channel, pred_h, pred_w), device=hr_image.device)
        for j in range(math.ceil(h_lr / stride)):
            for k in range(math.ceil(w_lr / stride)):
                pred[:, :, j * stride_hr:(j + 1) * stride_hr, k * stride_hr:(k + 1) * stride_hr] = \
                    self.model(lr_test[:, :, j * stride:(j + 1) * stride, k * stride:(k + 1) * stride])
        pred = pred[:, :, :h_hr, :w_hr]
        return pred

    @t.no_grad()
    def test(self):
        logging.info(f'iter: {self.iter} testing...')
        self.model.eval()
        all_info = []
        for i, (lr_image, hr_image) in enumerate(self.test_loader):
            lr_image, hr_image = self.to([lr_image, hr_image], self.device)
            if not self.cfg.solver.chop_size:
                pred = self.model(lr_image)
            else:
                pred = self.chop_pred(lr_image, hr_image)

            lr_image, hr_image, pred = self.prepare_for_tbx([lr_image, hr_image, pred], squeeze=True)
            all_info.append(self.collect_info(lr_image, hr_image, pred))

            if self.writer and i < self.cfg.tensorboard.image_num:
                image_to_show = np.concatenate([pred, hr_image], axis=1)
                self.writer.add_image(f'Test/pred_label_{i}', image_to_show, self.iter, dataformats='HWC')

        analysed_info = self.analyse_info(all_info)
        for k, v in analysed_info.items():
            analysed_info[k] = t.tensor(v)
        for k, v in analysed_info.items():
            analysed_info[k] = du.all_reduce(v)
        self.save_best(analysed_info)
        info_str = ''
        for j, (k, v) in enumerate(analysed_info.items()):
            if isinstance(v, dict):
                format_v = {key: f'{v[key]:.3f}' for key in sorted(list(v.keys()))}
            else:
                format_v = f'{v:.3f}'
            info_str += f'\t{k}:\t {format_v}\n'
        logging.info(info_str)

        if self.writer:
            for k, v in analysed_info.items():
                if isinstance(v, dict):
                    if self.cfg.tensorboard.add_scalars:
                        writer_dict = {str(key): v[key] for key in sorted(list(v.keys()))}
                        self.writer.add_scalars(f'Test/{k}', writer_dict, self.iter)
                else:
                    self.writer.add_scalar(f'Test/{k}', v, self.iter)
        self.model.train()

    def after_train(self):
        self.iter += 1
        if self.tqdm is not None:
            self.tqdm.disable = True
        self.after_step()
        if not self.iter % self.cfg.solver.test_interval:
            self.test()
        if self.writer is not None:
            self.writer.close()
        logging.info('train finished.')
