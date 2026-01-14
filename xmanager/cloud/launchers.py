# Copyright 2024 DeepMind Technologies Limited
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

"""Launcher abstraction for distributed training jobs.

This module provides an abstraction for different distributed training launchers
that can be used to launch training jobs on various compute backends.
"""

import abc
from typing import Any, Dict, List, Optional

import attr


@attr.s(auto_attribs=True)
class LauncherContext:
  """Context information for launching distributed training jobs.

  Attributes:
    num_nodes: Number of nodes to use for training.
    gpus_per_node: Number of GPUs per node.
    cluster_config: Optional cluster-specific configuration.
    work_dir: Working directory for the job.
    partition: Slurm partition name.
    master_addr: Address of the master node (can use environment variable).
    master_port: Port for the master node.
    nccl_env_vars: Environment variables for NCCL configuration.
  """
  num_nodes: int
  gpus_per_node: int = 8
  cluster_config: Optional[Any] = None
  work_dir: Optional[str] = None
  partition: Optional[str] = None
  master_addr: str = "${MASTER_ADDR}"
  master_port: int = 29500
  nccl_env_vars: Dict[str, str] = attr.Factory(dict)

  @property
  def world_size(self) -> int:
    """Total number of processes across all nodes."""
    return self.num_nodes * self.gpus_per_node

  @property
  def local_world_size(self) -> int:
    """Number of processes per node."""
    return self.gpus_per_node


class Launcher(abc.ABC):
  """Abstract base class for distributed training launchers."""

  @abc.abstractmethod
  def get_launch_command(
      self,
      script: str,
      script_args: List[str],
      ctx: LauncherContext,
  ) -> str:
    """Generate the launch command for the given script and context.

    Args:
      script: Path to the training script.
      script_args: Arguments to pass to the training script.
      ctx: Launcher context with configuration.

    Returns:
      The complete launch command as a string.
    """
    raise NotImplementedError

  def get_env_vars(self, ctx: LauncherContext) -> Dict[str, str]:
    """Get environment variables to set for the launch.

    Args:
      ctx: Launcher context with configuration.

    Returns:
      Dictionary of environment variables.
    """
    env_vars = {}

    # Add NCCL environment variables
    if ctx.nccl_env_vars:
      env_vars.update(ctx.nccl_env_vars)

    return env_vars

  def validate(self, ctx: LauncherContext) -> None:
    """Validate the launcher context.

    Args:
      ctx: Launcher context to validate.

    Raises:
      ValueError: If the context is invalid.
    """
    if ctx.num_nodes <= 0:
      raise ValueError(f"num_nodes must be positive, got {ctx.num_nodes}")
    if ctx.gpus_per_node <= 0:
      raise ValueError(f"gpus_per_node must be positive, got {ctx.gpus_per_node}")


@attr.s(auto_attribs=True)
class NemoRunLauncher(Launcher):
  """Launcher for NeMo Run based training.

  This launcher generates commands for running NeMo training jobs using
  the nemo-run framework with SLURM backend.

  Attributes:
    nemorun_script: Path to the nemo-run entry point script.
    recipe: Path to the training recipe script.
    container_image: Path to the container image (.sqsh file).
    experiment_name: Optional name for the experiment.
    extra_args: Additional arguments to pass to nemo-run.
  """
  nemorun_script: str = "run.py"
  recipe: str = ""
  container_image: str = ""
  experiment_name: Optional[str] = None
  extra_args: List[str] = attr.Factory(list)

  def get_launch_command(
      self,
      script: str,
      script_args: List[str],
      ctx: LauncherContext,
  ) -> str:
    """Generate the NeMo Run launch command.

    Args:
      script: Path to the training script (unused for NeMo Run, uses recipe).
      script_args: Arguments to pass to the script (unused for NeMo Run).
      ctx: Launcher context with configuration.

    Returns:
      The complete NeMo Run launch command.
    """
    self.validate(ctx)

    if not self.recipe:
      raise ValueError("recipe must be specified for NemoRunLauncher")
    if not self.container_image:
      raise ValueError("container_image must be specified for NemoRunLauncher")

    work_dir = ctx.work_dir or "."
    exp_name = self.experiment_name or "nemo_experiment"

    # Extract cluster type from cluster_config, partition from context
    cluster_type = "xmanager"
    if ctx.cluster_config:
      cluster_type = getattr(ctx.cluster_config, "cluster_type", cluster_type)
    partition = ctx.partition or "default"

    # Build the nemo-run command
    cmd_parts = [
        f"cd {work_dir}",
        f"export NEMORUN_HOME={work_dir}",
        f"python3 {self.nemorun_script}",
        "-e slurm",
        f"--slurm-type {cluster_type}",
        f"--partition {partition}",
        f"-d {work_dir}",
        f"-i {self.container_image}",
        f"-s {self.recipe}",
        f"-n {ctx.num_nodes}",
        f"--experiment-name {exp_name}",
    ]

    # Add any extra arguments
    if self.extra_args:
      cmd_parts.extend(self.extra_args)

    return " && ".join(cmd_parts[:2]) + " && " + " ".join(cmd_parts[2:])

  def validate(self, ctx: LauncherContext) -> None:
    """Validate the NeMo Run launcher context.

    Args:
      ctx: Launcher context to validate.

    Raises:
      ValueError: If the context is invalid.
    """
    super().validate(ctx)

    if not self.recipe:
      raise ValueError("recipe must be specified for NemoRunLauncher")
    if not self.container_image:
      raise ValueError("container_image must be specified for NemoRunLauncher")


@attr.s(auto_attribs=True)
class CustomLauncher(Launcher):
  """Launcher for custom user-defined commands.

  This launcher allows users to specify a custom command template with
  placeholders that will be filled in at runtime.

  Available placeholders:
    - {num_nodes}: Number of nodes
    - {gpus_per_node}: Number of GPUs per node
    - {world_size}: Total number of processes
    - {local_world_size}: Number of processes per node
    - {script}: Path to the training script
    - {args}: Arguments to pass to the script (space-separated)
    - {work_dir}: Working directory
    - {master_addr}: Master node address
    - {master_port}: Master node port

  Attributes:
    template: Command template with placeholders.
    env_vars: Additional environment variables to set.
  """
  template: str = ""
  env_vars: Dict[str, str] = attr.Factory(dict)

  def get_launch_command(
      self,
      script: str,
      script_args: List[str],
      ctx: LauncherContext,
  ) -> str:
    """Generate the custom launch command.

    Args:
      script: Path to the training script.
      script_args: Arguments to pass to the training script.
      ctx: Launcher context with configuration.

    Returns:
      The complete launch command with placeholders filled in.
    """
    self.validate(ctx)

    if not self.template:
      raise ValueError("template must be specified for CustomLauncher")

    # Prepare placeholder values
    args_str = " ".join(script_args)
    work_dir = ctx.work_dir or "."

    # Fill in the template
    command = self.template.format(
        num_nodes=ctx.num_nodes,
        gpus_per_node=ctx.gpus_per_node,
        world_size=ctx.world_size,
        local_world_size=ctx.local_world_size,
        script=script,
        args=args_str,
        work_dir=work_dir,
        master_addr=ctx.master_addr,
        master_port=ctx.master_port,
    )

    return command

  def get_env_vars(self, ctx: LauncherContext) -> Dict[str, str]:
    """Get environment variables including custom ones.

    Args:
      ctx: Launcher context with configuration.

    Returns:
      Dictionary of environment variables.
    """
    env_vars = super().get_env_vars(ctx)
    env_vars.update(self.env_vars)
    return env_vars

  def validate(self, ctx: LauncherContext) -> None:
    """Validate the custom launcher context.

    Args:
      ctx: Launcher context to validate.

    Raises:
      ValueError: If the context is invalid.
    """
    super().validate(ctx)

    if not self.template:
      raise ValueError("template must be specified for CustomLauncher")
