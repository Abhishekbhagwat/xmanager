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
"""Llama 3.1 1.71B training recipe for fineweb dataset ablation.

https://arxiv.org/pdf/2406.17557
Minimal setup: 1 H100 GPUs.
Real experiment: 16x8 = 128 H100 GPUs.
"""
# pylint: disable=g-import-not-at-top
# pylint: disable=g-importing-member
# pylint: disable=g-bad-import-order
# pylint: disable=import-error
import utils

# Need to run these filters before importing nemo.
utils.filter_warnings()
utils.filter_grad_bucket_logs()

import argparse
import os

import torch
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers import WandbLogger
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.gpt.model.llama import Llama31Config2B
from nemo.collections.llm.gpt.model.llama import LlamaModel
from nemo.lightning.pytorch.callbacks.flops_callback import FLOPsMeasurementCallback
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback


def main(args: argparse.Namespace):
  num_nodes = int(os.getenv("SLURM_NNODES", "1"))
  if num_nodes == 1:
    utils.ignore_sigprof()

  seq_length = 4096
  global_batch_size = 128 * num_nodes

  # Config the dataset
  if os.path.isdir(utils.DATA_DIR):
    # Data mix: 100% of roots_vi dataset.
    paths = [
        100,
        "/data/roots_vi_document_tokens",
    ]
    data = llm.PreTrainingDataModule(
        paths=paths,
        split="99,1,0",
        # The GPTDataset writes cache to this dir
        index_mapping_dir=os.path.join(utils.CACHE_DIR, "dataset"),
        seq_length=seq_length,
        global_batch_size=global_batch_size,
        micro_batch_size=4,
    )
    train_steps = 6000
    val_steps = 64
    val_check_interval = 1000
  else:
    data = llm.MockDataModule(
        num_train_samples=1_000_000,
        seq_length=seq_length,
        global_batch_size=global_batch_size,
        micro_batch_size=4,
    )
    train_steps = 10
    val_steps = 8
    val_check_interval = 5

  # Config the model
  model_config = Llama31Config2B(seq_length=seq_length)
  model = LlamaModel(model_config)

  # Config the parallelism strategy
  strategy = nl.MegatronStrategy(
      tensor_model_parallel_size=1,
      pipeline_model_parallel_size=1,
      pipeline_dtype=torch.bfloat16,
      virtual_pipeline_model_parallel_size=None,
      context_parallel_size=1,
      expert_model_parallel_size=1,
      sequence_parallel=False,
      account_for_embedding_in_pipeline_split=True,
      account_for_loss_in_pipeline_split=True,
      gradient_as_bucket_view=True,
      # Configuring checkpoint save/load optimizations
      # https://github.com/NVIDIA/NeMo/blob/main/docs/source/checkpoints/dist_ckpt.rst
      ckpt_async_save=True,
      ckpt_parallel_save=True,
      ckpt_parallel_load=True,
      ckpt_parallel_save_optim=True,
      ckpt_load_strictness="log_all",
      ddp=DistributedDataParallelConfig(
          check_for_nan_in_grad=True,
          grad_reduce_in_fp32=True,
          overlap_grad_reduce=True,
          overlap_param_gather=True,
          average_in_collective=True,
      ),
  )

  # Combine to the trainer
  trainer = nl.Trainer(
      accelerator="gpu",
      devices=8,
      num_nodes=num_nodes,
      max_steps=train_steps,
      limit_val_batches=val_steps,
      val_check_interval=val_check_interval,
      # Set this to 0 to make trainer validation log correct
      num_sanity_val_steps=0,
      log_every_n_steps=1,
      strategy=strategy,
      # Will let nemo tune automatically
      accumulate_grad_batches=1,
      # Will use nemo's sampler
      use_distributed_sampler=False,
      plugins=nl.MegatronMixedPrecision(
          precision="bf16-mixed", params_dtype=torch.bfloat16
      ),
      # Will let NeMoLogger to setup checkpoint
      enable_checkpointing=False,
      callbacks=[
          TimingCallback(),
          FLOPsMeasurementCallback(model_config, data, "llama3.1-2b"),
      ],
  )

  # Config the optimizer
  opt_config = OptimizerConfig(
      optimizer="adam",
      lr=3e-4,
      weight_decay=0.1,
      bf16=True,
      fp16=False,
      adam_beta1=0.9,
      adam_beta2=0.95,
      adam_eps=1e-8,
      use_distributed_optimizer=True,
      clip_grad=1.0,
  )
  lr_scheduler = CosineAnnealingScheduler(
      warmup_steps=500,
      constant_steps=0,
      min_lr=2.5e-4,
  )
  opt = MegatronOptimizerModule(config=opt_config, lr_scheduler=lr_scheduler)

  # Setup checkpoint and tensorboard for logger
  ckpt = nl.ModelCheckpoint(
      save_top_k=2,
      # Generate a *-last ckpt copy (link) whenever a ckpt is saved.
      # This is required when using auto resume.
      save_last=True,
      # Set to True if the final ckpt will be used by auto resume
      save_optim_on_train_end=True,
      filename="{val_loss:.2f}-{step}-{consumed_samples}",
  )
  tb = TensorBoardLogger(
      save_dir="tensorboard",  # The name of tfevents folder
      name="",  # No need further subfolder
  )
  wandb = (
      WandbLogger(project=args.wandb_project, name=args.exp_name)
      if args.wandb_project
      else None
  )
  logger = nl.NeMoLogger(
      # The centralized dir for loggings, tensorboard, checkpoints
      explicit_log_dir="/logs",
      log_global_rank_0_only=True,
      update_logger_directory=True,
      # Remove this argument to disable checkpointing
      ckpt=ckpt,
      tensorboard=tb,
      wandb=wandb,
  )

  # Config auto resume
  resume = nl.AutoResume(
      # Disable resuming from the last ckpt in log_dir
      resume_if_exists=False,
      # Do not raise error if ckpt does not exist
      resume_ignore_no_checkpoint=False,
      restore_config=nl.RestoreConfig(
          path=args.resume_from_ckpt,
          load_optim_state=False,
      ),
  )

  # Call nl.trainer.fit
  llm.pretrain(
      model=model,
      data=data,
      trainer=trainer,
      log=logger,
      resume=resume,
      optim=opt,
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Llama 3.1 2B Continuous Pre-Training."
  )
  parser.add_argument(
      "--exp_name", type=str, required=False, default="llama3p1_2b_cpt"
  )
  parser.add_argument("--wandb_project", type=str, required=False, default=None)
  parser.add_argument(
      "--resume_from_ckpt",
      type=str,
      required=True,
      default=None,
  )
  parsed_args = parser.parse_args()
  main(parsed_args)
