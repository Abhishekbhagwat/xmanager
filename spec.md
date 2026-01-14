# Consolidated Implementation Plan: Vertex Training Cluster Backend for XManager

## Overview

Add a deep, XManager-native `VertexTrainingCluster` executor for running distributed ML experiments on Slurm-based Vertex Training Clusters. This backend generates sbatch scripts, submits jobs via `sbatch`, tracks them via `squeue`/`sacct`, and integrates fully with XManager's experiment tracking.

**Framework-agnostic**: Works with any ML framework (NeMo, PyTorch, JAX, TensorFlow, DeepSpeed, HuggingFace Accelerate) through a Launcher abstraction.

**Key Design Principles**:
1. Deep XManager integration (not just a "Slurm wrapper")
2. Framework-agnostic sbatch generation
3. User flexibility (provide own scripts OR use templates OR auto-generate)
4. Proper job lifecycle management (submit, poll, cancel, stream logs)

---

## Critical Design Decisions

### 1. Use `replicas` for Node Count
- **Decision**: Use `requirements.replicas` for node count, not a separate `num_nodes`
- **Rationale**: Maintains XManager's standard resource abstraction, enabling easy executor swapping

### 2. Three Submission Modes
- **Raw Script**: User provides complete sbatch script
- **Template Mode**: User provides Jinja2 template, we fill in variables
- **Auto-Generate**: We generate sbatch from Launcher + cluster config

### 3. Launcher Abstraction
- **Decision**: Abstract "how to run" from "where to run"
- **Launchers**: TorchrunLauncher, PythonLauncher, MPIRunLauncher, AccelerateLauncher, DeepSpeedLauncher, CustomLauncher

### 4. Jinja2 Templates for sbatch
- **Decision**: Use Jinja2 with shell escaping for safe sbatch generation
- **Rationale**: Flexible, user-customizable, proper escaping

### 5. Database Storage Pattern
- **Decision**: Job data stored as serialized proto in `job_data` column
- **Rationale**: Follows existing pattern (Vertex, Kubernetes); no migration needed

### 6. Handle Type
- **Decision**: Extend `ExecutionHandle` (not `LocalExecutionHandle`)
- **Rationale**: Matches Kubernetes/Vertex pattern for remote executors

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Code                                 │
│  xm_local.VertexTrainingCluster(cluster_type='hcc-a4', ...)     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     VertexTrainingCluster                        │
│  ExecutorSpec + Executor + JobRequirements                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Launcher                                  │
│  TorchrunLauncher / AccelerateLauncher / CustomLauncher / ...   │
│  Generates: torchrun --nnodes=N ... train.py                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   SbatchTemplateRenderer                         │
│  Jinja2 template → sbatch script with NCCL config               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Client                                   │
│  SSH/local → sbatch script.sh → job_id                          │
│  squeue/sacct → status, scancel → cancel                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                VertexTrainingClusterHandle                       │
│  wait() → poll squeue/sacct until done                          │
│  stop() → scancel job_id                                        │
│  get_status() → PENDING/RUNNING/COMPLETED/FAILED                │
│  monitor() → tail -f slurm-{job_id}.out                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Cluster-Specific Configurations

Each cluster type has unique NCCL/networking requirements:

| Cluster | GPU | NCCL Dir | Network | Setup Method |
|---------|-----|----------|---------|--------------|
| hcc-a3m | H100 | /var/lib/tcpxo | TCPXO | Source nccl-env-profile.sh |
| hcc-a3u | H200 | /usr/local/gib | gIB | Source set_nccl_env.sh |
| hcc-a4 | B200 | /usr/local/gib | gIB | Explicit env vars |
| hcc-a3h | H100 | /usr/local/gib | gIB | Source set_nccl_env.sh |

---

## Files to Create/Modify

### New Files

1. **`xmanager/cloud/launchers.py`** - Launcher abstraction
2. **`xmanager/cloud/sbatch_templates.py`** - Jinja2 template rendering
3. **`xmanager/cloud/launchers_test.py`** - Launcher tests
4. **`xmanager/cloud/sbatch_templates_test.py`** - Template tests

### Files to Modify

1. **`xmanager/xm_local/packaging/cloud.py`** - FIX BUG: Add VertexTrainingClusterSpec case
2. **`xmanager/xm_local/executors.py`** - Add new fields to executor classes
3. **`xmanager/cloud/vertex_training_cluster.py`** - Complete rewrite with sbatch submission
4. **`xmanager/xm_local/__init__.py`** - Export Launcher classes

---

## P0 Bug Fix: cloud.py

**File**: `xmanager/xm_local/packaging/cloud.py`
**Line**: ~29-39

The `_get_push_image_tag()` function is missing the VertexTrainingClusterSpec case:

```python
def _get_push_image_tag(executor_spec: xm.ExecutorSpec) -> Optional[str]:
  """Get the push_image_tag from executor or None."""
  match executor_spec:
    case local_executors.CaipSpec() as caip_spec:
      return caip_spec.push_image_tag
    case local_executors.KubernetesSpec() as kubernetes_spec:
      return kubernetes_spec.push_image_tag
    case local_executors.VertexTrainingClusterSpec() as vtc_spec:  # ADD THIS
      return vtc_spec.push_image_tag                               # ADD THIS
    case _:
      raise TypeError(
          f'Unsupported executor specification: {executor_spec!r}. '
      )
```

---

## Implementation: launchers.py

```python
# Copyright 2021 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0
"""Launcher abstraction for Vertex Training Cluster distributed training.

This module provides a flexible launcher system that supports multiple
distributed training frameworks (PyTorch, MPI, HuggingFace, DeepSpeed, etc.)
while abstracting away the complexity of multi-node execution.

Architecture:
  - Launcher: Base class defining the interface
  - Concrete Launchers: Framework-specific implementations
  - LauncherContext: Runtime information (nodes, GPUs, cluster config)
"""

import abc
import os
from typing import Any, Dict, List, Optional, Sequence
import attr


@attr.s(auto_attribs=True)
class LauncherContext:
  """Runtime context for launcher command generation.

  Provides the launcher with information about the cluster environment,
  allowing it to generate appropriate distributed training commands.
  """

  # Cluster configuration
  num_nodes: int = 1
  gpus_per_node: int = 8
  cluster_config: Optional[Any] = None  # ClusterConfig

  # Job information
  work_dir: Optional[str] = None

  # Network configuration
  master_addr: str = "${MASTER_ADDR}"  # Populated by sbatch script
  master_port: int = 29500

  # NCCL environment variables (from cluster_config)
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
  """Base class for distributed training launchers.

  A launcher generates the command prefix needed to execute a training script
  in a distributed environment. For example:

    - Torchrun: torchrun --nnodes=2 --nproc_per_node=8 train.py
    - MPI: mpirun -np 16 train.py
    - Python: python train.py (single-process)

  The launcher abstracts framework-specific details while providing a
  consistent interface for XManager to execute jobs.
  """

  @abc.abstractmethod
  def get_launch_command(
      self,
      script: str,
      script_args: Sequence[str],
      ctx: LauncherContext,
  ) -> str:
    """Generate the full launch command.

    Args:
      script: Path to the training script (e.g., "train.py")
      script_args: Arguments to pass to the script
      ctx: Runtime context with cluster information

    Returns:
      Complete shell command to execute the training job
    """
    pass

  def get_env_vars(self, ctx: LauncherContext) -> Dict[str, str]:
    """Get environment variables required by this launcher."""
    return {}

  def validate(self, ctx: LauncherContext) -> None:
    """Validate launcher compatibility with the context."""
    pass


@attr.s(auto_attribs=True)
class TorchrunLauncher(Launcher):
  """PyTorch distributed launcher using torchrun.

  Example command:
    torchrun --nnodes=2 --nproc_per_node=8 \
             --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
             train.py --batch_size=32
  """

  rdzv_backend: str = "c10d"
  rdzv_endpoint: Optional[str] = None  # Auto-detect from Slurm if None
  standalone: bool = False
  extra_args: List[str] = attr.Factory(list)

  def get_launch_command(
      self,
      script: str,
      script_args: Sequence[str],
      ctx: LauncherContext,
  ) -> str:
    cmd_parts = ["torchrun"]

    if self.standalone or ctx.num_nodes == 1:
      cmd_parts.extend([
          "--standalone",
          f"--nproc_per_node={ctx.gpus_per_node}",
      ])
    else:
      endpoint = self.rdzv_endpoint or f"{ctx.master_addr}:{ctx.master_port}"
      cmd_parts.extend([
          f"--nnodes={ctx.num_nodes}",
          f"--nproc_per_node={ctx.gpus_per_node}",
          f"--rdzv_backend={self.rdzv_backend}",
          f"--rdzv_endpoint={endpoint}",
          "--rdzv_id=$SLURM_JOB_ID",
      ])

    cmd_parts.extend(self.extra_args)
    cmd_parts.append(script)
    cmd_parts.extend(script_args)

    return " ".join(cmd_parts)

  def get_env_vars(self, ctx: LauncherContext) -> Dict[str, str]:
    env_vars = ctx.nccl_env_vars.copy()
    env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return env_vars


@attr.s(auto_attribs=True)
class PythonLauncher(Launcher):
  """Simple Python launcher for single-process jobs."""

  python_path: str = "python"

  def get_launch_command(
      self,
      script: str,
      script_args: Sequence[str],
      ctx: LauncherContext,
  ) -> str:
    cmd_parts = [self.python_path, script]
    cmd_parts.extend(script_args)
    return " ".join(cmd_parts)

  def validate(self, ctx: LauncherContext) -> None:
    if ctx.world_size > 1:
      import warnings
      warnings.warn(
          f"PythonLauncher is single-process but context has {ctx.world_size} "
          f"GPUs. Consider using TorchrunLauncher."
      )


@attr.s(auto_attribs=True)
class MPIRunLauncher(Launcher):
  """MPI launcher for MPI-based frameworks (Horovod, etc.)."""

  mpi_impl: str = "openmpi"
  bind_to: str = "none"
  extra_args: List[str] = attr.Factory(list)

  def get_launch_command(
      self,
      script: str,
      script_args: Sequence[str],
      ctx: LauncherContext,
  ) -> str:
    cmd_parts = ["mpirun", "-np", str(ctx.world_size)]

    if self.bind_to != "none":
      cmd_parts.extend(["--bind-to", self.bind_to])

    cmd_parts.extend(self.extra_args)
    cmd_parts.extend(["python", script])
    cmd_parts.extend(script_args)

    return " ".join(cmd_parts)

  def get_env_vars(self, ctx: LauncherContext) -> Dict[str, str]:
    env_vars = ctx.nccl_env_vars.copy()
    env_vars["HOROVOD_GPU_OPERATIONS"] = "NCCL"
    return env_vars


@attr.s(auto_attribs=True)
class AccelerateLauncher(Launcher):
  """HuggingFace Accelerate launcher."""

  config_file: Optional[str] = None
  mixed_precision: str = "bf16"
  extra_args: List[str] = attr.Factory(list)

  def get_launch_command(
      self,
      script: str,
      script_args: Sequence[str],
      ctx: LauncherContext,
  ) -> str:
    cmd_parts = ["accelerate", "launch"]

    if self.config_file:
      cmd_parts.extend(["--config_file", self.config_file])
    else:
      cmd_parts.extend([
          f"--num_machines={ctx.num_nodes}",
          f"--num_processes={ctx.world_size}",
          f"--mixed_precision={self.mixed_precision}",
      ])

    cmd_parts.extend(self.extra_args)
    cmd_parts.append(script)
    cmd_parts.extend(script_args)

    return " ".join([p for p in cmd_parts if p])


@attr.s(auto_attribs=True)
class DeepSpeedLauncher(Launcher):
  """DeepSpeed launcher for DeepSpeed-enabled training."""

  hostfile: Optional[str] = None
  launcher: str = "slurm"
  extra_args: List[str] = attr.Factory(list)

  def get_launch_command(
      self,
      script: str,
      script_args: Sequence[str],
      ctx: LauncherContext,
  ) -> str:
    cmd_parts = ["deepspeed"]

    if self.launcher == "slurm":
      cmd_parts.extend([
          "--launcher=slurm",
          "--launcher_args='--overlap --export=ALL'",
      ])
    else:
      cmd_parts.extend([
          f"--num_nodes={ctx.num_nodes}",
          f"--num_gpus={ctx.gpus_per_node}",
      ])
      if self.hostfile:
        cmd_parts.extend(["--hostfile", self.hostfile])

    cmd_parts.extend(self.extra_args)
    cmd_parts.append(script)
    cmd_parts.extend(script_args)

    return " ".join(cmd_parts)


@attr.s(auto_attribs=True)
class SrunLauncher(Launcher):
  """Direct srun launcher for container workloads.

  This is the most common launcher for VTC as it handles container
  execution properly with NCCL mounts.
  """

  container_image: Optional[str] = None
  container_mounts: List[str] = attr.Factory(list)
  extra_srun_args: List[str] = attr.Factory(list)

  def get_launch_command(
      self,
      script: str,
      script_args: Sequence[str],
      ctx: LauncherContext,
  ) -> str:
    cmd_parts = ["srun"]

    # Standard srun args for containers
    cmd_parts.extend([
        "--ntasks-per-node=1",
        "--gpus-per-node=8",
        "--container-writable",
        "--no-container-mount-home",
        "--mpi=pmix",
    ])

    # Container image
    if self.container_image:
      cmd_parts.append(f"--container-image={self.container_image}")

    # Container mounts (including NCCL)
    if ctx.cluster_config:
      nccl_dir = ctx.cluster_config.nccl_dir
      cmd_parts.append(f"--container-mounts={nccl_dir}:{nccl_dir}")

    for mount in self.container_mounts:
      cmd_parts.append(f"--container-mounts={mount}")

    cmd_parts.extend(self.extra_srun_args)

    # The actual command to run inside container
    cmd_parts.append(script)
    cmd_parts.extend(script_args)

    return " ".join(cmd_parts)


@attr.s(auto_attribs=True)
class CustomLauncher(Launcher):
  """Custom user-defined launcher with template string.

  Allows users to provide a template command with placeholders.

  Example:
    template = "srun -N {num_nodes} -n {world_size} python {script} {args}"

  Available placeholders:
    - {num_nodes}: Number of nodes
    - {gpus_per_node}: GPUs per node
    - {world_size}: Total processes
    - {master_addr}: Master node address
    - {master_port}: Master node port
    - {work_dir}: Working directory
    - {script}: Training script path
    - {args}: Script arguments (pre-joined)
  """

  template: str
  env_vars: Dict[str, str] = attr.Factory(dict)

  def get_launch_command(
      self,
      script: str,
      script_args: Sequence[str],
      ctx: LauncherContext,
  ) -> str:
    substitutions = {
        "num_nodes": str(ctx.num_nodes),
        "gpus_per_node": str(ctx.gpus_per_node),
        "world_size": str(ctx.world_size),
        "local_world_size": str(ctx.local_world_size),
        "master_addr": ctx.master_addr,
        "master_port": str(ctx.master_port),
        "work_dir": ctx.work_dir or "",
        "script": script,
        "args": " ".join(script_args),
    }
    return self.template.format(**substitutions)

  def get_env_vars(self, ctx: LauncherContext) -> Dict[str, str]:
    env_vars = ctx.nccl_env_vars.copy()
    env_vars.update(self.env_vars)
    return env_vars


@attr.s(auto_attribs=True)
class FaultTolerantLauncher(Launcher):
  """Fault-tolerant wrapper around another launcher.

  Adds automatic retry logic and checkpointing support.
  """

  base_launcher: Launcher
  max_restarts: int = 3
  restart_interval: int = 60
  enable_elastic: bool = False

  def get_launch_command(
      self,
      script: str,
      script_args: Sequence[str],
      ctx: LauncherContext,
  ) -> str:
    base_cmd = self.base_launcher.get_launch_command(script, script_args, ctx)

    # Wrap in retry loop
    return f"""
for attempt in $(seq 1 {self.max_restarts}); do
  echo "Starting training attempt $attempt/{self.max_restarts}..."
  {base_cmd}
  exit_code=$?
  if [ $exit_code -eq 0 ]; then
    echo "Training completed successfully"
    exit 0
  else
    echo "Training failed with exit code $exit_code"
    if [ $attempt -lt {self.max_restarts} ]; then
      echo "Restarting in {self.restart_interval} seconds..."
      sleep {self.restart_interval}
    fi
  fi
done
echo "Max restarts reached. Exiting."
exit 1
"""

  def get_env_vars(self, ctx: LauncherContext) -> Dict[str, str]:
    env_vars = self.base_launcher.get_env_vars(ctx)
    if self.enable_elastic:
      env_vars["TORCH_ELASTIC_ENABLED"] = "1"
      env_vars["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    return env_vars

  def validate(self, ctx: LauncherContext) -> None:
    self.base_launcher.validate(ctx)
```

---

## Implementation: sbatch_templates.py

```python
# Copyright 2021 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0
"""Sbatch script generation using Jinja2 templates.

This module provides safe, flexible sbatch script generation with:
  - Jinja2 templating with proper shell escaping
  - Default template for common use cases
  - Support for custom user templates
  - Cluster-specific NCCL configuration
"""

import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, StrictUndefined


# Default sbatch template for Vertex Training Clusters
DEFAULT_SBATCH_TEMPLATE = r'''#!/bin/bash
{%- if job_name %}
#SBATCH --job-name={{ job_name }}
{%- endif %}
{%- if partition %}
#SBATCH --partition={{ partition }}
{%- endif %}
{%- if account %}
#SBATCH --account={{ account }}
{%- endif %}
#SBATCH --nodes={{ num_nodes }}
#SBATCH --ntasks-per-node={{ ntasks_per_node|default(gpus_per_node, true) }}
#SBATCH --gpus-per-node={{ gpus_per_node }}
{%- if cpus_per_task %}
#SBATCH --cpus-per-task={{ cpus_per_task }}
{%- endif %}
#SBATCH --time={{ time_limit|default("0", true) }}
{%- if exclusive %}
#SBATCH --exclusive
{%- endif %}
{%- if output_file %}
#SBATCH --output={{ output_file }}
{%- endif %}
{%- if error_file %}
#SBATCH --error={{ error_file }}
{%- endif %}
{%- for key, value in sbatch_flags.items() %}
#SBATCH --{{ key }}={{ value }}
{%- endfor %}

set -euo pipefail

# Get master node address for distributed training
MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_ADDR
export MASTER_PORT={{ master_port|default("29500", true) }}
export WORLD_SIZE=$SLURM_NNODES
export NODE_RANK=$SLURM_NODEID

{%- if nccl_setup_script %}

# NCCL environment setup (cluster-specific)
source {{ nccl_setup_script }}
export LD_LIBRARY_PATH={{ nccl_lib_path }}:$LD_LIBRARY_PATH
{%- endif %}

{%- if nccl_env_vars %}

# NCCL environment variables
{%- for key, value in nccl_env_vars.items() %}
export {{ key }}={{ value|shell_quote }}
{%- endfor %}
{%- endif %}

{%- if env_vars %}

# User environment variables
{%- for key, value in env_vars.items() %}
export {{ key }}={{ value|shell_quote }}
{%- endfor %}
{%- endif %}

{%- if prologue_commands %}

# Prologue commands
{%- for cmd in prologue_commands %}
{{ cmd }}
{%- endfor %}
{%- endif %}

{%- if working_dir %}

# Change to working directory
cd {{ working_dir|shell_quote }}
{%- endif %}

# Main command
{{ command }}

{%- if epilogue_commands %}

# Epilogue commands
{%- for cmd in epilogue_commands %}
{{ cmd }}
{%- endfor %}
{%- endif %}
'''


class SbatchTemplateRenderer:
  """Renders sbatch scripts from Jinja2 templates with proper shell escaping."""

  def __init__(
      self,
      custom_template: Optional[str] = None,
      template_path: Optional[Path] = None,
  ):
    """Initialize template renderer.

    Args:
      custom_template: String containing custom Jinja2 template
      template_path: Path to custom template file
    """
    self.env = Environment(
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )

    # Add shell escaping filter
    self.env.filters['shell_quote'] = shlex.quote

    if custom_template:
      self.template = self.env.from_string(custom_template)
    elif template_path:
      with open(template_path) as f:
        self.template = self.env.from_string(f.read())
    else:
      self.template = self.env.from_string(DEFAULT_SBATCH_TEMPLATE)

  def render(self, **variables) -> str:
    """Render the sbatch script with given variables.

    Standard variables:
      job_name: str - Job name
      partition: str - Slurm partition
      account: str - Slurm account
      num_nodes: int - Number of nodes
      ntasks_per_node: int - Tasks per node (default: gpus_per_node)
      gpus_per_node: int - GPUs per node
      cpus_per_task: int - CPUs per task
      time_limit: str - Time limit (e.g., "01:00:00", "0" for unlimited)
      exclusive: bool - Exclusive node access
      output_file: str - Stdout file path
      error_file: str - Stderr file path
      sbatch_flags: Dict[str, str] - Additional SBATCH parameters

      nccl_setup_script: str - Path to NCCL setup script to source
      nccl_lib_path: str - Path to NCCL library directory
      nccl_env_vars: Dict[str, str] - NCCL environment variables

      env_vars: Dict[str, str] - User environment variables
      prologue_commands: List[str] - Commands to run before main command
      epilogue_commands: List[str] - Commands to run after main command
      working_dir: str - Working directory
      command: str - Main command to execute
    """
    defaults = {
        'sbatch_flags': {},
        'nccl_env_vars': {},
        'env_vars': {},
        'prologue_commands': [],
        'epilogue_commands': [],
    }

    render_vars = {**defaults, **variables}

    # Validate required variables
    if 'command' not in render_vars or not render_vars['command']:
      raise ValueError("'command' is required")
    if 'num_nodes' not in render_vars:
      raise ValueError("'num_nodes' is required")
    if 'gpus_per_node' not in render_vars:
      raise ValueError("'gpus_per_node' is required")

    return self.template.render(**render_vars)


def render_sbatch_for_cluster(
    cluster_config: Any,  # ClusterConfig from vertex_training_cluster.py
    job_name: str,
    num_nodes: int,
    command: str,
    partition: Optional[str] = None,
    account: Optional[str] = None,
    time_limit: str = "0",
    working_dir: Optional[str] = None,
    env_vars: Optional[Dict[str, str]] = None,
    sbatch_flags: Optional[Dict[str, str]] = None,
    prologue_commands: Optional[List[str]] = None,
    epilogue_commands: Optional[List[str]] = None,
    custom_template: Optional[str] = None,
) -> str:
  """Convenience function to render sbatch for a specific cluster type.

  Args:
    cluster_config: ClusterConfig object with NCCL settings
    job_name: Name for the Slurm job
    num_nodes: Number of nodes
    command: Main command to execute
    partition: Slurm partition (optional)
    account: Slurm account (optional)
    time_limit: Time limit string (default: "0" unlimited)
    working_dir: Working directory on cluster
    env_vars: Additional environment variables
    sbatch_flags: Additional SBATCH flags
    prologue_commands: Commands before main command
    epilogue_commands: Commands after main command
    custom_template: Custom Jinja2 template string

  Returns:
    Complete sbatch script as string
  """
  renderer = SbatchTemplateRenderer(custom_template=custom_template)

  # Build NCCL configuration from cluster_config
  nccl_setup_script = cluster_config.setup_script if cluster_config else None
  nccl_lib_path = cluster_config.get_nccl_lib_path() if cluster_config else None
  nccl_env_vars = cluster_config.nccl_env_vars if cluster_config else {}
  gpus_per_node = cluster_config.gpus_per_node if cluster_config else 8

  return renderer.render(
      job_name=job_name,
      partition=partition,
      account=account,
      num_nodes=num_nodes,
      gpus_per_node=gpus_per_node,
      time_limit=time_limit,
      exclusive=True,
      nccl_setup_script=nccl_setup_script,
      nccl_lib_path=nccl_lib_path,
      nccl_env_vars=nccl_env_vars,
      env_vars=env_vars or {},
      sbatch_flags=sbatch_flags or {},
      prologue_commands=prologue_commands or [],
      epilogue_commands=epilogue_commands or [],
      working_dir=working_dir,
      command=command,
  )
```

---

## Updated Executor Classes: executors.py

Add/update these classes in `xmanager/xm_local/executors.py`:

```python
from typing import Callable, Dict, List, Optional, Union
from xmanager.cloud import launchers  # Import after creating launchers.py


@attr.s(auto_attribs=True)
class VertexTrainingClusterSpec(xm.ExecutorSpec):
  """Vertex Training Cluster spec for packaging.

  Attributes:
    push_image_tag: Image registry path to push (for Docker packaging)
    squashfs_path: Path to pre-built .squashfs on cluster storage
    auto_convert_to_squashfs: Convert Docker image to squashfs on cluster
  """

  push_image_tag: Optional[str] = None
  squashfs_path: Optional[str] = None
  auto_convert_to_squashfs: bool = False


@attr.s(auto_attribs=True)
class VertexTrainingCluster(xm.Executor):
  """Executor for Slurm-based Vertex Training Clusters.

  This executor provides deep XManager integration:
    - Generates sbatch scripts (or uses user-provided)
    - Submits jobs via sbatch
    - Tracks jobs via squeue/sacct
    - Streams logs via SSH
    - Integrates with experiment tracking

  Submission Modes:
    1. Raw Script: Provide complete sbatch script via `sbatch_script`
    2. Template: Provide Jinja2 template via `sbatch_template`
    3. Auto-Generate: Use `launcher` + cluster config to generate script

  Supported cluster types:
    - hcc-a3m: H100 GPUs with TCPXO networking
    - hcc-a3u: H200 GPUs with gIB networking
    - hcc-a4: B200 GPUs with gIB networking
    - hcc-a3h: H100 GPUs with gIB networking

  Example (Auto-Generate):
    executor = xm_local.VertexTrainingCluster(
        cluster_type='hcc-a4',
        partition='a4',
        launcher=launchers.TorchrunLauncher(),
    )

  Example (Template):
    executor = xm_local.VertexTrainingCluster(
        cluster_type='hcc-a4',
        sbatch_template=Path('my_template.j2'),
    )

  Example (Raw Script):
    executor = xm_local.VertexTrainingCluster(
        cluster_type='hcc-a4',
        sbatch_script='''#!/bin/bash
        #SBATCH --partition=a4
        #SBATCH --nodes=2
        srun torchrun train.py
        ''',
    )
  """

  requirements: xm.JobRequirements = attr.Factory(xm.JobRequirements)
  # NOTE: Use requirements.replicas for node count (maps to --nodes)

  # Cluster configuration
  cluster_type: str = 'hcc-a3m'  # hcc-a3m, hcc-a3u, hcc-a4, hcc-a3h
  partition: Optional[str] = None
  account: Optional[str] = None
  time_limit: str = "0"  # "0" = unlimited
  exclusive: bool = True

  # Submission mode 1: Raw sbatch script (complete script, no templating)
  sbatch_script: Optional[str] = None

  # Submission mode 2: Custom template (Jinja2)
  sbatch_template: Optional[Union[str, 'Path']] = None

  # Submission mode 3: Auto-generate (use launcher)
  # Type hint is string to avoid circular import; actual type is Launcher
  launcher: Optional[Any] = None  # Launcher from launchers.py

  # Additional sbatch flags (arbitrary --key=value pairs)
  # From Gemini's plan - allows users to add any sbatch flags
  sbatch_flags: Dict[str, str] = attr.Factory(dict)

  # Connection settings
  use_gcloud_ssh: bool = False
  login_node: Optional[str] = None
  ssh_hostname: Optional[str] = None

  # Working directories on cluster
  work_dir: Optional[str] = None
  log_dir: Optional[str] = None  # Directory for slurm-*.out files

  # Container configuration
  container_image: Optional[str] = None  # .squashfs or docker image path
  container_mounts: List[str] = attr.Factory(list)

  # Environment configuration
  env_vars: Dict[str, str] = attr.Factory(dict)

  # Prologue/Epilogue commands (from Gemini's plan)
  prologue_commands: List[str] = attr.Factory(list)
  epilogue_commands: List[str] = attr.Factory(list)

  # Log streaming
  stream_output: bool = True

  Spec = VertexTrainingClusterSpec

  def __attrs_post_init__(self):
    vtc_execution = importlib.import_module(
        'xmanager.cloud.vertex_training_cluster'
    )
    vtc_execution.register()

  def get_cluster_config(self):
    """Get NCCL and networking configuration for this cluster type."""
    vtc = importlib.import_module('xmanager.cloud.vertex_training_cluster')
    return vtc.get_cluster_config(self.cluster_type)

  @override
  @classmethod
  async def launch(
      cls, local_experiment_unit: Any, job_group: xm.JobGroup
  ) -> Sequence[handles.ExecutionHandle]:
    return await registry.get_launch_method(cls)(
        local_experiment_unit, job_group
    )
```

---

## Implementation: vertex_training_cluster.py (Complete Rewrite)

```python
# Copyright 2021 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0
"""Vertex Training Cluster executor with full Slurm integration.

This module provides:
  1. Cluster configurations (NCCL, networking) per GPU type
  2. Client for SSH/local command execution
  3. Sbatch script generation via templates
  4. Job submission via sbatch
  5. Job lifecycle management (squeue, sacct, scancel)
  6. Handle for experiment tracking
"""

import asyncio
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import attr
from xmanager import xm
from xmanager.xm import utils
from xmanager.xm_local import executors as local_executors
from xmanager.xm_local import handles
from xmanager.xm_local import registry
from xmanager.xm_local import status as local_status
from xmanager.xm_local.storage import database
from xmanager.cloud import launchers
from xmanager.cloud import sbatch_templates


# =============================================================================
# Cluster Configurations
# =============================================================================

@attr.s(auto_attribs=True)
class ClusterConfig:
  """Configuration for a specific Vertex Training Cluster type."""

  cluster_type: str
  gpu_type: str  # H100, H200, B200
  gpus_per_node: int = 8

  nccl_dir: str = ""
  nccl_lib_subdir: str = "lib64"
  setup_script: Optional[str] = None
  nccl_env_vars: Dict[str, str] = attr.Factory(dict)

  container_srun_args: List[str] = attr.Factory(lambda: [
      "--container-writable",
      "--no-container-mount-home",
      "--mpi=pmix",
  ])

  def get_nccl_lib_path(self) -> str:
    return os.path.join(self.nccl_dir, self.nccl_lib_subdir)

  def get_setup_commands(self) -> Optional[str]:
    if self.setup_script:
      return f"source {self.setup_script}"
    return None


CLUSTER_CONFIGS: Dict[str, ClusterConfig] = {
    "hcc-a3m": ClusterConfig(
        cluster_type="hcc-a3m",
        gpu_type="H100",
        nccl_dir="/var/lib/tcpxo",
        setup_script="/var/lib/tcpxo/lib64/nccl-env-profile.sh",
    ),
    "hcc-a3u": ClusterConfig(
        cluster_type="hcc-a3u",
        gpu_type="H200",
        nccl_dir="/usr/local/gib",
        setup_script="/usr/local/gib/scripts/set_nccl_env.sh",
    ),
    "hcc-a4": ClusterConfig(
        cluster_type="hcc-a4",
        gpu_type="B200",
        nccl_dir="/usr/local/gib",
        nccl_env_vars={
            "NCCL_IB_TC": "52",
            "NCCL_IB_FIFO_TC": "84",
            "NCCL_NVLS_CHUNKSIZE": "524288",
            "NCCL_SOCKET_IFNAME": "enp0s19,enp192s20",
            "NCCL_NET_GDR_LEVEL": "PIX",
            "NCCL_NET": "gIB",
            "NCCL_IB_GID_INDEX": "3",
            "NCCL_P2P_NET_CHUNKSIZE": "131072",
            "NCCL_IB_QPS_PER_CONNECTION": "4",
            "NCCL_P2P_PCI_CHUNKSIZE": "131072",
            "NCCL_P2P_NVL_CHUNKSIZE": "524288",
            "NCCL_TUNER_CONFIG_PATH": "/usr/local/gib/configs/tuner_config_a4.txtpb",
            "NCCL_IB_ADAPTIVE_ROUTING": "1",
            "NCCL_CROSS_NIC": "0",
        },
    ),
    "hcc-a3h": ClusterConfig(
        cluster_type="hcc-a3h",
        gpu_type="H100",
        nccl_dir="/usr/local/gib",
        setup_script="/usr/local/gib/scripts/set_nccl_env.sh",
    ),
}


def get_cluster_config(cluster_type: str) -> ClusterConfig:
  if cluster_type not in CLUSTER_CONFIGS:
    raise ValueError(
        f"Unknown cluster_type: {cluster_type}. "
        f"Supported: {list(CLUSTER_CONFIGS.keys())}"
    )
  return CLUSTER_CONFIGS[cluster_type]


# =============================================================================
# SSH Client
# =============================================================================

SSH_OPTIONS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
]


class Client:
  """Client for interacting with Vertex Training Cluster via Slurm."""

  def __init__(self, executor: local_executors.VertexTrainingCluster):
    self.executor = executor
    self.cluster_config = get_cluster_config(executor.cluster_type)

  def _build_ssh_prefix(self) -> List[str]:
    if self.executor.use_gcloud_ssh and self.executor.login_node:
      prefix = ['gcloud', 'compute', 'ssh', self.executor.login_node, '--']
      if self.executor.ssh_hostname:
        prefix.extend(['-o', f'Hostname={self.executor.ssh_hostname}'])
      return prefix
    elif self.executor.login_node:
      return ['ssh'] + SSH_OPTIONS + [self.executor.login_node]
    return []

  def run_command(self, cmd: str) -> subprocess.CompletedProcess:
    """Run command on cluster (locally or via SSH)."""
    ssh_prefix = self._build_ssh_prefix()
    if ssh_prefix:
      full_cmd = ssh_prefix + [cmd]
      return subprocess.run(full_cmd, capture_output=True, text=True)
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

  async def run_command_async(
      self, cmd: str
  ) -> asyncio.subprocess.Process:
    """Run command asynchronously for streaming output."""
    ssh_prefix = self._build_ssh_prefix()
    if ssh_prefix:
      return await asyncio.create_subprocess_exec(
          *ssh_prefix, cmd,
          stdout=asyncio.subprocess.PIPE,
          stderr=asyncio.subprocess.STDOUT,
      )
    return await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

  def submit_sbatch(self, script_content: str, work_dir: Optional[str] = None) -> str:
    """Submit sbatch script and return job ID.

    Args:
      script_content: Complete sbatch script content
      work_dir: Working directory on cluster (optional)

    Returns:
      Slurm job ID as string

    Raises:
      RuntimeError: If sbatch submission fails
    """
    # Write script to temp file on cluster
    script_name = f"xm_job_{os.getpid()}.sh"
    if work_dir:
      script_path = f"{work_dir}/{script_name}"
    else:
      script_path = f"/tmp/{script_name}"

    # Escape script content for shell
    import shlex
    escaped_content = script_content.replace("'", "'\\''")
    write_cmd = f"cat > {script_path} << 'XMANAGER_SBATCH_EOF'\n{script_content}\nXMANAGER_SBATCH_EOF"

    result = self.run_command(write_cmd)
    if result.returncode != 0:
      raise RuntimeError(f"Failed to write sbatch script: {result.stderr}")

    # Submit via sbatch
    submit_cmd = f"sbatch {script_path}"
    result = self.run_command(submit_cmd)

    if result.returncode != 0:
      raise RuntimeError(f"sbatch failed: {result.stderr}")

    # Parse job ID from "Submitted batch job 12345"
    match = re.search(r'Submitted batch job (\d+)', result.stdout)
    if not match:
      raise RuntimeError(f"Could not parse job ID from: {result.stdout}")

    return match.group(1)

  def get_job_status(self, job_id: str) -> str:
    """Get job status via squeue/sacct.

    Returns: One of PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, UNKNOWN
    """
    # First try squeue (for running/pending jobs)
    result = self.run_command(f"squeue -j {job_id} -h -o '%T'")
    if result.returncode == 0 and result.stdout.strip():
      status = result.stdout.strip().upper()
      if status in ('PENDING', 'RUNNING', 'CONFIGURING'):
        return status

    # Fall back to sacct (for completed jobs)
    result = self.run_command(
        f"sacct -j {job_id} -n -o State -X"
    )
    if result.returncode == 0 and result.stdout.strip():
      status = result.stdout.strip().split()[0].upper()
      if 'COMPLETED' in status:
        return 'COMPLETED'
      if 'FAILED' in status or 'TIMEOUT' in status or 'NODE_FAIL' in status:
        return 'FAILED'
      if 'CANCELLED' in status:
        return 'CANCELLED'
      return status

    return 'UNKNOWN'

  def cancel_job(self, job_id: str) -> None:
    """Cancel job via scancel."""
    self.run_command(f"scancel {job_id}")

  def get_job_log_path(self, job_id: str, work_dir: Optional[str] = None) -> str:
    """Get path to job's stdout log file."""
    base_dir = work_dir or "."
    return f"{base_dir}/slurm-{job_id}.out"


# =============================================================================
# Sbatch Generation
# =============================================================================

def _generate_sbatch_script(
    executor: local_executors.VertexTrainingCluster,
    job: xm.Job,
    job_name: str,
    cluster_config: ClusterConfig,
) -> str:
  """Generate sbatch script for job submission.

  Handles all three submission modes:
    1. Raw script (sbatch_script)
    2. Custom template (sbatch_template)
    3. Auto-generate (launcher)
  """
  num_nodes = executor.requirements.replicas or 1

  # Mode 1: Raw script - return as-is
  if executor.sbatch_script:
    return executor.sbatch_script

  # Build the main command
  args = xm.merge_args(job.executable.args, job.args).to_list(utils.ARG_ESCAPER)
  entrypoint = getattr(job.executable, 'entrypoint', None)

  if executor.launcher:
    # Mode 3: Auto-generate using launcher
    ctx = launchers.LauncherContext(
        num_nodes=num_nodes,
        gpus_per_node=cluster_config.gpus_per_node,
        cluster_config=cluster_config,
        work_dir=executor.work_dir,
        nccl_env_vars=cluster_config.nccl_env_vars,
    )

    if entrypoint:
      script = entrypoint
      script_args = args
    else:
      script = args[0] if args else ""
      script_args = args[1:] if len(args) > 1 else []

    command = executor.launcher.get_launch_command(script, script_args, ctx)
    launcher_env_vars = executor.launcher.get_env_vars(ctx)
  else:
    # No launcher - just run the command directly
    if entrypoint:
      command = f"{entrypoint} {' '.join(args)}".strip()
    else:
      command = ' '.join(args)
    launcher_env_vars = {}

  # Combine environment variables
  all_env_vars = {**launcher_env_vars, **executor.env_vars}

  # Mode 2: Custom template
  if executor.sbatch_template:
    if isinstance(executor.sbatch_template, Path):
      renderer = sbatch_templates.SbatchTemplateRenderer(
          template_path=executor.sbatch_template
      )
    else:
      renderer = sbatch_templates.SbatchTemplateRenderer(
          custom_template=executor.sbatch_template
      )
  else:
    # Default template
    renderer = sbatch_templates.SbatchTemplateRenderer()

  # Determine output file path
  if executor.log_dir:
    output_file = f"{executor.log_dir}/slurm-%j.out"
    error_file = f"{executor.log_dir}/slurm-%j.err"
  else:
    output_file = None
    error_file = None

  return renderer.render(
      job_name=job_name,
      partition=executor.partition,
      account=executor.account,
      num_nodes=num_nodes,
      gpus_per_node=cluster_config.gpus_per_node,
      time_limit=executor.time_limit,
      exclusive=executor.exclusive,
      output_file=output_file,
      error_file=error_file,
      sbatch_flags=executor.sbatch_flags,
      nccl_setup_script=cluster_config.setup_script,
      nccl_lib_path=cluster_config.get_nccl_lib_path(),
      nccl_env_vars=cluster_config.nccl_env_vars,
      env_vars=all_env_vars,
      prologue_commands=executor.prologue_commands,
      epilogue_commands=executor.epilogue_commands,
      working_dir=executor.work_dir,
      command=command,
  )


# =============================================================================
# Execution Handle
# =============================================================================

@attr.s(auto_attribs=True)
class VertexTrainingClusterHandle(handles.ExecutionHandle):
  """Handle for tracking Slurm jobs on Vertex Training Cluster."""

  job_name: str
  slurm_job_id: str
  client: Client
  executor: local_executors.VertexTrainingCluster
  _cached_status: Optional[str] = None

  async def wait(self) -> None:
    """Wait for the job to complete by polling squeue/sacct."""
    while True:
      status = self.client.get_job_status(self.slurm_job_id)
      if status in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT'):
        break
      await asyncio.sleep(30)  # Poll every 30 seconds

  def stop(self) -> None:
    """Cancel the job via scancel."""
    self.client.cancel_job(self.slurm_job_id)

  def get_status(self) -> local_status.LocalWorkUnitStatus:
    """Get current job status."""
    status = self.client.get_job_status(self.slurm_job_id)

    status_mapping = {
        'PENDING': local_status.LocalWorkUnitStatusEnum.NOT_STARTED,
        'CONFIGURING': local_status.LocalWorkUnitStatusEnum.NOT_STARTED,
        'RUNNING': local_status.LocalWorkUnitStatusEnum.RUNNING,
        'COMPLETED': local_status.LocalWorkUnitStatusEnum.COMPLETED,
        'FAILED': local_status.LocalWorkUnitStatusEnum.FAILED,
        'CANCELLED': local_status.LocalWorkUnitStatusEnum.FAILED,
        'TIMEOUT': local_status.LocalWorkUnitStatusEnum.FAILED,
    }

    enum_status = status_mapping.get(
        status, local_status.LocalWorkUnitStatusEnum.UNKNOWN
    )

    return local_status.LocalWorkUnitStatus(
        enum_status,
        message=f"Slurm job {self.slurm_job_id}: {status}"
    )

  def save_to_storage(self, experiment_id: int, work_unit_id: int) -> None:
    """Save job info to database."""
    database.database().insert_vertex_training_cluster_job(
        experiment_id=experiment_id,
        work_unit_id=work_unit_id,
        job_name=self.job_name,
        slurm_job_id=self.slurm_job_id,
        login_node=self.executor.login_node or "",
        cluster_type=self.executor.cluster_type,
        partition=self.executor.partition or "",
    )

  async def monitor(self) -> None:
    """Stream job output via SSH tail -f."""
    if not self.executor.stream_output:
      return

    log_path = self.client.get_job_log_path(
        self.slurm_job_id, self.executor.work_dir
    )

    # Wait for log file to appear
    await asyncio.sleep(5)

    tail_cmd = f"tail -f {log_path} 2>/dev/null || echo 'Log not available'"
    process = await self.client.run_command_async(tail_cmd)

    try:
      while True:
        line = await process.stdout.readline()
        if not line:
          # Check if job is still running
          status = self.client.get_job_status(self.slurm_job_id)
          if status not in ('PENDING', 'RUNNING', 'CONFIGURING'):
            break
          await asyncio.sleep(1)
          continue
        print(f"[{self.job_name}] {line.decode().strip()}")
    finally:
      process.terminate()


# =============================================================================
# Launch Logic
# =============================================================================

def _vtc_job_predicate(job: xm.Job) -> bool:
  return isinstance(job.executor, local_executors.VertexTrainingCluster)


async def launch(
    local_experiment_unit: Any,
    job_group: xm.JobGroup,
) -> List[VertexTrainingClusterHandle]:
  """Launch jobs on Vertex Training Cluster via sbatch."""
  jobs = xm.job_operators.collect_jobs_by_filter(
      job_group, _vtc_job_predicate
  )

  if not jobs:
    return []

  handles_list = []
  experiment_title = local_experiment_unit.experiment_title
  work_unit_name = local_experiment_unit.work_unit_name

  for idx, job in enumerate(jobs):
    executor = job.executor
    client = Client(executor)
    cluster_config = get_cluster_config(executor.cluster_type)

    job_name = f"{experiment_title}_{work_unit_name}_{idx}"

    # Generate sbatch script
    sbatch_script = _generate_sbatch_script(
        executor, job, job_name, cluster_config
    )

    print(f"Submitting job {job_name} to {executor.cluster_type}...")

    # Submit and get job ID
    slurm_job_id = client.submit_sbatch(sbatch_script, executor.work_dir)

    print(f"  Submitted: Slurm job ID {slurm_job_id}")

    handle = VertexTrainingClusterHandle(
        job_name=job_name,
        slurm_job_id=slurm_job_id,
        client=client,
        executor=executor,
    )

    handles_list.append(handle)

    # Start log streaming in background if enabled
    if executor.stream_output:
      asyncio.create_task(handle.monitor())

  return handles_list


def _create_handle(
    *args,
    data,
    vertex_training_cluster_jobs,
) -> VertexTrainingClusterHandle:
  """Restore handle from database."""
  del args

  executor = local_executors.VertexTrainingCluster(
      cluster_type=data.vertex_training_cluster.cluster_type,
      login_node=data.vertex_training_cluster.login_node or None,
      partition=data.vertex_training_cluster.partition or None,
  )

  handle = VertexTrainingClusterHandle(
      job_name="restored_job",
      slurm_job_id=data.vertex_training_cluster.slurm_job_id,
      client=Client(executor),
      executor=executor,
  )

  vertex_training_cluster_jobs.append(handle)
  return handle


def register():
  """Registers Vertex Training Cluster execution logic."""
  registry.register(
      local_executors.VertexTrainingCluster,
      launch=launch,
      create_handle=_create_handle,
  )
```

---

## Usage Examples

### Example 1: Auto-Generate with TorchrunLauncher

```python
from xmanager import xm, xm_local
from xmanager.cloud import launchers

with xm_local.create_experiment(experiment_title='llama_training') as exp:

  executor = xm_local.VertexTrainingCluster(
      cluster_type='hcc-a4',
      partition='a4',
      requirements=xm.JobRequirements(replicas=4),  # 4 nodes
      login_node='vmdsa405-login-001',
      use_gcloud_ssh=True,
      work_dir='/workspace/training',
      launcher=launchers.TorchrunLauncher(
          rdzv_backend='c10d',
          extra_args=['--max_restarts=3'],
      ),
  )

  job = xm.Job(
      executable=xm.PythonContainer(
          path='.',
          entrypoint='train.py',
      ),
      executor=executor,
      args={'--batch_size': 32, '--epochs': 100},
  )

  exp.add(xm.JobGroup(job))
```

### Example 2: Custom Template

```python
my_template = '''#!/bin/bash
#SBATCH --job-name={{ job_name }}
#SBATCH --partition={{ partition }}
#SBATCH --nodes={{ num_nodes }}
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00

echo "Starting training on $(hostname)"
{{ command }}
echo "Training complete"
'''

executor = xm_local.VertexTrainingCluster(
    cluster_type='hcc-a3m',
    sbatch_template=my_template,
    work_dir='/workspace',
)
```

### Example 3: Raw Sbatch Script

```python
executor = xm_local.VertexTrainingCluster(
    cluster_type='hcc-a4',
    sbatch_script='''#!/bin/bash
#SBATCH --job-name=custom-job
#SBATCH --partition=a4
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --time=48:00:00
#SBATCH --exclusive

source /usr/local/gib/scripts/set_nccl_env.sh
srun --container-image=my-container.sqsh torchrun train.py
''',
)
```

### Example 4: With sbatch_flags and prologue/epilogue

```python
executor = xm_local.VertexTrainingCluster(
    cluster_type='hcc-a4',
    partition='a4',
    requirements=xm.JobRequirements(replicas=2),
    launcher=launchers.TorchrunLauncher(),

    # Additional SBATCH flags (Gemini's suggestion)
    sbatch_flags={
        'mem': '500G',
        'cpus-per-task': '48',
        'mail-type': 'END,FAIL',
        'mail-user': 'user@example.com',
    },

    # Prologue/epilogue commands (Gemini's suggestion)
    prologue_commands=[
        'echo "Job starting at $(date)"',
        'nvidia-smi',
    ],
    epilogue_commands=[
        'echo "Job finished at $(date)"',
    ],

    env_vars={
        'WANDB_PROJECT': 'my-project',
        'HF_HOME': '/workspace/.cache/huggingface',
    },
)
```

### Example 5: DeepSpeed Launcher

```python
executor = xm_local.VertexTrainingCluster(
    cluster_type='hcc-a4',
    requirements=xm.JobRequirements(replicas=4),
    launcher=launchers.DeepSpeedLauncher(
        launcher='slurm',
        extra_args=['--no_ssh_check'],
    ),
)

job = xm.Job(
    executable=xm.PythonContainer(
        path='.',
        entrypoint='train.py',
    ),
    executor=executor,
    args={'--deepspeed_config': 'ds_config.json'},
)
```

### Example 6: HuggingFace Accelerate Launcher

```python
executor = xm_local.VertexTrainingCluster(
    cluster_type='hcc-a3u',
    requirements=xm.JobRequirements(replicas=2),
    launcher=launchers.AccelerateLauncher(
        config_file='configs/accelerate_fsdp.yaml',
        mixed_precision='bf16',
    ),
)
```

---

## Template Variables Reference

The default sbatch template supports these variables:

| Variable | Type | Description |
|----------|------|-------------|
| `job_name` | str | Job name |
| `partition` | str | Slurm partition |
| `account` | str | Slurm account |
| `num_nodes` | int | Number of nodes |
| `gpus_per_node` | int | GPUs per node |
| `ntasks_per_node` | int | Tasks per node (default: gpus_per_node) |
| `cpus_per_task` | int | CPUs per task |
| `time_limit` | str | Time limit (e.g., "01:00:00", "0" for unlimited) |
| `exclusive` | bool | Exclusive node access |
| `output_file` | str | Stdout file path |
| `error_file` | str | Stderr file path |
| `sbatch_flags` | Dict | Additional SBATCH flags |
| `nccl_setup_script` | str | NCCL setup script path |
| `nccl_lib_path` | str | NCCL library path |
| `nccl_env_vars` | Dict | NCCL environment variables |
| `env_vars` | Dict | User environment variables |
| `prologue_commands` | List | Commands before main command |
| `epilogue_commands` | List | Commands after main command |
| `working_dir` | str | Working directory |
| `command` | str | Main command |

---

## Testing Plan

### Unit Tests (launchers_test.py)

```python
def test_torchrun_launcher_single_node():
    launcher = TorchrunLauncher()
    ctx = LauncherContext(num_nodes=1, gpus_per_node=8)
    cmd = launcher.get_launch_command("train.py", ["--lr=0.001"], ctx)
    assert "--standalone" in cmd
    assert "--nproc_per_node=8" in cmd

def test_torchrun_launcher_multi_node():
    launcher = TorchrunLauncher()
    ctx = LauncherContext(num_nodes=4, gpus_per_node=8)
    cmd = launcher.get_launch_command("train.py", [], ctx)
    assert "--nnodes=4" in cmd
    assert "--rdzv_backend=c10d" in cmd

def test_custom_launcher_template():
    launcher = CustomLauncher(
        template="srun -N {num_nodes} python {script} {args}"
    )
    ctx = LauncherContext(num_nodes=2, gpus_per_node=8)
    cmd = launcher.get_launch_command("train.py", ["--batch=32"], ctx)
    assert cmd == "srun -N 2 python train.py --batch=32"
```

### Unit Tests (sbatch_templates_test.py)

```python
def test_shell_escaping():
    renderer = SbatchTemplateRenderer()
    script = renderer.render(
        num_nodes=1,
        gpus_per_node=8,
        command="echo 'hello'",
        env_vars={"PATH": "/path/with spaces:/other"},
    )
    assert "'/path/with spaces:/other'" in script

def test_conditional_sections():
    renderer = SbatchTemplateRenderer()
    script = renderer.render(
        num_nodes=1,
        gpus_per_node=8,
        command="test",
        partition="gpu",
    )
    assert "#SBATCH --partition=gpu" in script

    script = renderer.render(
        num_nodes=1,
        gpus_per_node=8,
        command="test",
    )
    assert "--partition" not in script
```

---

## Implementation Priority Order

### P0 - Blocking Bug Fix
1. Fix `_get_push_image_tag()` in `xmanager/xm_local/packaging/cloud.py`

### P1 - Core Infrastructure
2. Create `xmanager/cloud/launchers.py`
3. Create `xmanager/cloud/sbatch_templates.py`

### P2 - Executor Updates
4. Update `xmanager/xm_local/executors.py` with new fields

### P3 - Main Implementation
5. Rewrite `xmanager/cloud/vertex_training_cluster.py`

### P4 - Integration
6. Update `xmanager/xm_local/__init__.py` to export Launcher classes
7. Update example in `examples/vertex_training_cluster/launcher.py`

### P5 - Testing
8. Create `xmanager/cloud/launchers_test.py`
9. Create `xmanager/cloud/sbatch_templates_test.py`

---

## Summary

This consolidated plan provides:

1. **Three Submission Modes**: Raw script, Template, Auto-generate
2. **Launcher Abstraction**: Framework-agnostic distributed training
3. **Jinja2 Templates**: Safe, flexible sbatch generation
4. **Full Job Lifecycle**: Submit, poll, cancel, stream logs
5. **Deep XManager Integration**: Handles, database storage, experiment tracking
6. **User Flexibility**: Custom templates, sbatch_flags, prologue/epilogue commands
