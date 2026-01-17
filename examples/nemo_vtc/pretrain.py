# Copyright 2025 Google LLC
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
#

# Llama31-70B pretraining recipe for NeMo 2.0

from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from dataclasses import dataclass
from typing import List
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.recipes import llama31_70b as llama31
from nemo.lightning.pytorch.callbacks.flops_callback import FLOPsMeasurementCallback

from typing import Optional, Union

import nemo_run as run


@run.cli.factory
def configure_recipe(
    max_steps: int = 50,
    val_check_interval: int = 10,
    val_batches: int = 5,
    explicit_log_dir: str = "/mnt/logs",
    tensorboard_log_dir: Optional[str] = None,
    resume_if_exists: bool = True,
    resume_ignore_no_checkpoint: bool = True,
    tensor_model_parallel_size: int = 4,
    pipeline_model_parallel_size: int = 4,
    virtual_pipeline_model_parallel_size: int = 5,
    context_parallel_size: int = 2,
) -> run.Partial[llm.pretrain]:
    """Factory function to configure Llama31-70b pretraining recipe.

    Args:
        explicit_log_dir: Directory for checkpoints (on Lustre).
        tensorboard_log_dir: Directory for TensorBoard logs only (on GCS).
            Set to $AIP_TENSORBOARD_LOG_DIR for VMDS TensorBoard integration.
    """
    pretrain = llama31.pretrain_recipe(
        performance_mode=False)

    # Configure TensorBoard logger separately if tensorboard_log_dir is specified
    # This allows checkpoints on Lustre while TensorBoard logs go to GCS
    tb_logger = None
    if tensorboard_log_dir:
        tb_logger = run.Config(
            TensorBoardLogger,
            save_dir=tensorboard_log_dir,
            name="",
            version="",
        )

    pretrain.log = run.Config(
        nl.NeMoLogger,
        explicit_log_dir=explicit_log_dir,
        log_global_rank_0_only=True,
        update_logger_directory=False if tensorboard_log_dir else True,
        tensorboard=tb_logger,
        ckpt=run.Config(
            nl.ModelCheckpoint,
            save_top_k=3,
            save_last=True,
            save_optim_on_train_end=True,
            filename="{val_loss:.2f}-{step}-{consumed_samples}",
        )
    )

    pretrain.resume = run.Config(
        nl.AutoResume,
        resume_if_exists=resume_if_exists,
        resume_ignore_no_checkpoint=resume_ignore_no_checkpoint,
    )

    pretrain.trainer.callbacks.append(
        run.Config(
            FLOPsMeasurementCallback,
            model_name="llama3",
            model_config=pretrain.model.config,
            data_config=pretrain.data,
        )
    )

    pretrain.trainer.strategy.tensor_model_parallel_size = tensor_model_parallel_size
    pretrain.trainer.strategy.pipeline_model_parallel_size = pipeline_model_parallel_size
    pretrain.trainer.strategy.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
    pretrain.trainer.strategy.context_parallel_size = context_parallel_size

    pretrain.trainer.strategy.ckpt_async_save = True
    pretrain.trainer.limit_val_batches = val_batches
    pretrain.trainer.log_every_n_steps = 1
    pretrain.trainer.max_steps = max_steps
    pretrain.trainer.check_val_every_n_epoch = 1
    pretrain.trainer.val_check_interval = val_check_interval

    return pretrain


if __name__ == "__main__":
    run.cli.main(
        llm.pretrain,
        default_factory=configure_recipe,
        cmd_defaults={"skip_confirmation": True})