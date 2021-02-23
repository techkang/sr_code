# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

import data
from tools.dist import get_rank, get_world_size


class TrainingSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.

    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(self, data_source, shuffle: bool = True):
        """
        Args:
            data_source (Dataset): the dataset to be sampled
            shuffle (bool): whether to shuffle the indices or not
        """
        self._size = len(data_source)
        assert self._size > 0
        self._shuffle = shuffle

        self._rank = get_rank()
        self._world_size = get_world_size()

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)


class InferenceSampler(Sampler):
    """
    Produce indices for inference.
    Inference needs to run on the __exact__ set of samples,
    therefore when the total number of samples is not divisible by the number of workers,
    this sampler produces different number of samples on different workers.
    """

    def __init__(self, data_source):
        """
        Args:
            data_source (Dataset): target dataset
        """
        self._size = len(data_source)
        assert self._size > 0
        self._rank = get_rank()
        self._world_size = get_world_size()

        shard_size = (self._size - 1) // self._world_size + 1
        begin = shard_size * self._rank
        end = min(shard_size * (self._rank + 1), self._size)
        self._local_indices = range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def build_dataloader(cfg, mode):
    dataset = getattr(data, cfg.dataset.name)(cfg, mode=mode)
    if mode == 'train':
        pin_memory = drop_last = True
        batch_size = cfg.dataloader.batch_size
        sampler = TrainingSampler(dataset, shuffle=True)
    else:
        pin_memory = drop_last = False
        batch_size = 1
        sampler = InferenceSampler(dataset)

    dataloader = DataLoader(dataset, batch_size,
                            sampler=sampler,
                            num_workers=cfg.dataloader.num_workers,
                            pin_memory=pin_memory,
                            drop_last=drop_last)
    return dataloader
