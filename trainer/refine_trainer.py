import time
from functools import reduce

import torch as t

import tools.dist as du
from tools.checkpointer import Checkpointer
from .trainer import Trainer


def multiply(tensor):
    return reduce(lambda x, y: x * y, tensor)


class RefineTrainer(Trainer):
    def __init__(self, cfg, resume):
        super().__init__(cfg, resume)
        flag = False
        self.param_dict = t.nn.Module()
        for name, tensor in list(self.model.named_parameters()):
            if 'tail' in name:
                flag = True
            if flag:
                tensor = t.eye(multiply(tensor.shape[1:]), dtype=t.float32, device=self.device)
                setattr(self.param_dict, name.replace('.', '-'), t.nn.Parameter(tensor))
        self.checkpointer = Checkpointer(self.model, self.output_dir / 'checkpoint' / self.log_name,
                                         optimizer=self.optimizer,
                                         scheduler=self.scheduler,
                                         param_dict=self.param_dict)

    def pro_weight(self, proj, input_data, weight):
        alpha = self.scheduler.get_last_lr()[0]
        if len(weight.shape) > 2:
            _, _, input_h, input_w = input_data.shape
            out_batch_size, _, weight_h, weight_w = weight.shape
            Ho = int(1 + (input_h - weight_h) / 2)
            Wo = int(1 + (input_w - weight_w) / 2)
            for i in range(Ho):
                for j in range(Wo):
                    # N*C*weight_h*weight_w, C*weight_h*weight_w = N*C*weight_h*weight_w, sum -> N*1
                    r = input_data[:, :, i * 2: i * 2 + weight_h, j * 2: j * 2 + weight_w].reshape(
                        1, -1)
                    k = t.mm(proj, t.t(r))
                    proj.sub_(t.mm(k, t.t(k)) / (alpha + t.mm(r, k)))
            weight.grad.data = t.mm(weight.grad.data.reshape(out_batch_size, -1), t.t(proj.data)).view_as(
                weight)
        else:
            r = input_data
            k = t.mm(proj, t.t(r))
            proj.sub_(t.mm(k, t.t(k)) / (alpha + t.mm(r, k)))
            weight.grad.data = t.mm(weight.grad.data, t.t(proj.data))

    def run_step(self):
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        batch = next(self._data_loader_iter)
        batch = self.to(batch, self.device)
        data_time = time.perf_counter() - start

        lr_image, hr_image = batch
        sr_image, input_list = self.model(lr_image)
        losses = self.loss_function(lr_image, hr_image, sr_image)
        if isinstance(losses, dict):
            loss_backward = sum(losses.values())
        else:
            loss_backward = losses
        self.optimizer.zero_grad()
        loss_backward.backward()

        # start owm
        with t.no_grad():
            flag = False
            count = 0
            for n, w in self.model.named_parameters():
                if 'tail' in n:
                    flag = True
                if flag:
                    p = getattr(self.param_dict, n.replace('.', '-'))
                    self.pro_weight(p, input_list[count], w)
                    count += 1
        # end owm

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
