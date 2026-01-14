# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils for training recipes."""
# pylint: disable=import-error
# pylint: disable=protected-access
import glob
import math
import os
import pathlib
import signal
from typing import Optional, Union
import warnings

from megatron.core.datasets.indexed_dataset import _IndexReader
from nemo.utils import logging
import numpy as np


# These are defined in the run.py file. Copy them here to be shared by recipes.
DATA_DIR = "/data"
LOGS_DIR = "/logs"
CKPT_DIR = "/checkpoints"
CACHE_DIR = pathlib.Path.home() / ".cache"


def filter_warnings():
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  warnings.filterwarnings("ignore", category=FutureWarning)
  warnings.filterwarnings("ignore", module="typing_extensions")
  warnings.filterwarnings(
      "ignore", module="megatron.core.distributed.param_and_grad_buffer"
  )
  warnings.filterwarnings("ignore", message=r".*deprecated.*")


def filter_grad_bucket_logs():
  """Filter the noisy `Number of buckets...` log dumped by megatron."""

  def _filter(record):
    del record
    return False

  for handler in logging._logger.handlers:
    handler.addFilter(_filter)


def ignore_sigprof():
  logging.warning("Ignore SIGPROF due to b/405683768")
  signal.signal(signal.SIGPROF, signal.SIG_IGN)


def expand_folder_to_data_prefix(
    folder: str, split: Optional[str] = None
) -> Union[list[str], dict[str, list[str]]]:
  """Expand a folder to data prefixes which can be used with NeMo's data module.

  For example:
  If the folder contains files like:
  - folder/data_1.bin
  - folder/data_1.idx
  - folder/data_2.bin
  - folder/data_2.idx

  then the function will return:
  ["folder/data_1", "folder/data_2"].

  If the split "80,10,10" is specified, the function will return a dictionary of
  data prefixes with split:
  {
      "train": ["folder/data_1"],
      "validation": ["folder/data_2"],
      "test": ["folder/data_2"],
  }

  NOTE: The split is shard wise, not sample wise. Each split will contain
  at least 1 shard. The train split and other splits will be mutually exclusive.

  Args:
    folder: The data folder to expand.
    split: The train/validation/test split, in the format of like "80,10,10".

  Returns:
    A list of data prefixes, or a dictionary of data prefixes if split presents.
  """
  if not os.path.isdir(folder):
    raise ValueError(f"Folder {folder} does not exist.")

  bin_files = glob.glob(os.path.join(folder, "*.bin"))
  bin_files = sorted(bin_files)

  if not bin_files:
    raise ValueError(f"No bin files found in folder {folder}.")

  data_prefix = [f.split(".bin")[-2] for f in bin_files]

  if split is None:
    return data_prefix

  if len(data_prefix) < 2:
    raise ValueError(
        "At least 2 shards are required in the folder to split, "
        f"but got {len(data_prefix)}."
    )

  # "70,20,10" -> [70.0, 20.0, 10.0]
  split_val = [float(v) for v in split.split(",")]
  # [70.0, 20.0, 10.0] -> [0.7, 0.2, 0.1]
  normalized_split_val = [v / sum(split_val) for v in split_val]
  # 10, [0.7, 0.2, 0.1] -> [7, 2, 1]
  split_length = [
      max(1, math.floor(v * len(data_prefix))) for v in normalized_split_val
  ]
  # 10, [7, 2, 1] -> [[0, 7], [7, 9], [9, 10]]
  split_slice = []
  for length in split_length:
    if not split_slice:
      split_slice.append([0, length])
    else:
      last_start = min(len(data_prefix) - 1, split_slice[-1][-1])
      split_slice.append([last_start, last_start + length])

  result = {
      "train": data_prefix[split_slice[0][0] : split_slice[0][1]],
      "validation": data_prefix[split_slice[1][0] : split_slice[1][1]],
      "test": data_prefix[split_slice[2][0] : split_slice[2][1]],
  }

  log_str = " | ".join([f"{k}: {len(v)} shards" for k, v in result.items()])
  logging.info(f"Expanded data folder to data prefixes, {log_str}")
  return result


def get_train_val_batches_one_epoch(
    paths: dict[str, list[str]],
    seq_length: int = 8192,
    global_batch_size: int = 128,
) -> tuple[int, int]:
  """Calculate the train and val batches for one epoch of the dataset.

  The NeMo's llm.PreTrainingDataModule doesn't support training with epochs
  when set trainer.max_epochs. It only supports trainer.max_steps.

  So this function is used to calculate the max steps with epochs ahead of
  constructing the trainer.

  A more elegant way is to enhance the llm.PreTrainingDataModule to support
  trainer.max_epochs. But that requires to reset the trainer.max_steps which
  is not allowed according to trainer's design.

  Args:
    paths: A dictionary of data prefixes with train/validation/test split.
    seq_length: The sequence length of each sample.
    global_batch_size: The global number of samples in one batch.

  Returns:
    A tuple of (train_batches, val_batches)
  """
  train_batches = _get_batches(paths, "train", seq_length, global_batch_size)
  val_batches = _get_batches(paths, "validation", seq_length, global_batch_size)
  return (train_batches, val_batches)


def _get_batches(
    paths: dict[str, list[str]],
    split: str,
    seq_length: int = 8192,
    global_batch_size: int = 128,
) -> int:
  """Calculate the number of batches for a split of the dataset.

  Args:
    paths: A dictionary of data prefixes with train/validation/test split.
    split: A string of the split name, like "train", "validation", "test".
    seq_length: The sequence length of each sample.
    global_batch_size: The global number of samples in one batch.

  Returns:
    The number of batches for the dataset.
  """
  data_idx = [t + ".idx" for t in paths[split]]
  num_samples = 0
  for idx in data_idx:
    dataset = _IndexReader(idx, multimodal=False)
    num_tokens = int(np.sum(dataset.sequence_lengths))
    # Use the same logic as:
    # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/gpt_dataset.py#L381-L383
    # Apply `-1` because we assume `add_extra_token_to_sequence` is True.
    num_samples += (num_tokens - 1) // seq_length
  batches = num_samples // global_batch_size
  logging.info(
      f"Calculated {split} dataset, samples: {num_samples} | batches: {batches}"
  )
  return batches
