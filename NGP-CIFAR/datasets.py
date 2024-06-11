# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Dataset getter utility."""

import json
import logging
from typing import Any, List, Tuple, Union
import warnings

import tensorflow as tf
import tensorflow_datasets as tfds
from base import BaseDataset
from cifar import Cifar100Dataset
from cifar import Cifar10CorruptedDataset
from cifar import Cifar10Dataset
from cifar100_corrupted import Cifar100CorruptedDataset

DATASETS = {
    'cifar100': Cifar100Dataset,
    'cifar10': Cifar10Dataset,
    'cifar10_corrupted': Cifar10CorruptedDataset,
    'cifar100_corrupted': Cifar100CorruptedDataset,
}


def get_dataset_names() -> List[str]:
  return list(DATASETS.keys())


def get(
    dataset_name: str,
    split: Union[Tuple[str, float], str, tfds.Split],
    **hyperparameters: Any) -> BaseDataset:
  """Gets a dataset builder by name.

  Note that the user still needs to call
  `distribution_strategy.experimental_distribute_dataset(dataset)` on the loaded
  dataset if they are running in a distributed environment.

  Args:
    dataset_name: Name of the dataset builder class.
    split: a dataset split, either a custom tfds.Split or one of the
      tfds.Split enums [TRAIN, VALIDAITON, TEST] or their lowercase string
      names.
    **hyperparameters: dict of possible kwargs to be passed to the dataset
      constructor.

  Returns:
    A dataset builder class with a method .build(split) which can be called to
    get the tf.data.Dataset, which has elements that are a dict described by
    dataset_builder.info.

  Raises:
    ValueError: If dataset_name is unrecognized.
  """
  hyperparameters_py = {
      k: (v.numpy().tolist() if isinstance(v, tf.Tensor) else v)
      for k, v in hyperparameters.items()
  }
  logging.info(
      'Building dataset %s with additional kwargs:\n%s',
      dataset_name,
      json.dumps(hyperparameters_py, indent=2, sort_keys=True))
  if dataset_name not in DATASETS:
    raise ValueError('Unrecognized dataset name: {!r}'.format(dataset_name))

  dataset_class = DATASETS[dataset_name]
  return dataset_class(
      split=split,
      **hyperparameters)
