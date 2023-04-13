#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# adpated from torch.utils.data.DistributedSampler

import random
import numpy as np
from typing import TypeVar

import torch
from torch.utils.data import DataLoader

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils import DataParallelSampler
from prefetch_generator import BackgroundGenerator

T_co = TypeVar('T_co', covariant=True)


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def get_dataloader(dataset,
                   shuffle=False,
                   seed=1024,
                   add_sampler=True,
                   drop_last=False,
                   pin_memory=False,
                   num_workers=0,
                   **kwargs):
    r"""Set up a deterministic dataloader (also configure seed workers, samplers and whether shuffle or not)

    Note:
        When pipeline parallel is enabled, shuffle cannot be True as it will result in mismatch between input data
        on the 1st stage and label on the last stage.

    Args:
        dataset (:class:`torch.utils.data.Dataset`): The dataset to be loaded.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        seed (int, optional): Random worker seed for sampling, defaults to 1024.
        add_sampler: Whether to add ``DistributedDataParallelSampler`` to the dataset. Defaults to True.
        drop_last (bool, optional): Set to True to drop the last incomplete batch, if the dataset size
            is not divisible by the batch size. If False and the size of dataset is not divisible by
            the batch size, then the last batch will be smaller, defaults to False.
        pin_memory (bool, optional): Whether to pin memory address in CPU memory. Defaults to False.
        num_workers (int, optional): Number of worker threads for this dataloader. Defaults to 0.
        kwargs (dict): optional parameters for ``torch.utils.data.DataLoader``, more details could be found in
                `DataLoader <https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader>`_.

    Returns:
        :class:`torch.utils.data.DataLoader`: A DataLoader used for training or testing.
    """
    _kwargs = kwargs.copy()

    if add_sampler and gpc.is_initialized(ParallelMode.DATA) and gpc.get_world_size(ParallelMode.DATA) > 1:
        sampler = DataParallelSampler(dataset, shuffle=shuffle)
    else:
        sampler = None

    # Deterministic dataloader
    def seed_worker(worker_id):
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    if sampler is None:
        return DataLoaderX(dataset,
                          worker_init_fn=seed_worker,
                          shuffle=shuffle,
                          drop_last=drop_last,
                          pin_memory=pin_memory,
                          num_workers=num_workers,
                          **_kwargs)
    else:
        return DataLoaderX(dataset,
                          sampler=sampler,
                          worker_init_fn=seed_worker,
                          drop_last=drop_last,
                          pin_memory=pin_memory,
                          num_workers=num_workers,
                          **_kwargs)
