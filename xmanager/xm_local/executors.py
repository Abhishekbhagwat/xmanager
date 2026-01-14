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
"""Local backend executors."""

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import attr
from typing_extensions import override
from xmanager import xm
from xmanager.docker import docker_adapter
from xmanager.xm_local import handles
from xmanager.xm_local import registry


GOOGLE_KUBERNETES_ENGINE_CLOUD_PROVIDER = 'GOOGLE_KUBERNETES_ENGINE'


@attr.s(auto_attribs=True)
class LocalSpec(xm.ExecutorSpec):
  """Current machine executor's specification."""


@attr.s(auto_attribs=True)
class DockerOptions:
  """Options of the container to be run.

  Attributes:
      ports: In the simplest form -- a dictionary from `int` to `int`, where the
        keys represent the ports inside the container and the values represent
        the ports of the host to bind. See the specification at
        https://docker-py.readthedocs.io/en/stable/containers.html.
      volumes: A dictionary from `str` to `str`, where the keys represent paths
        inside the host to mount and the values represent paths in the
        container.
      mount_gcs_path: If True, checks for the `~/gcs` directory on the host and
        mounts it (if found) at `/gcs` in the container. Defaults to True.
      interactive: If True, requests a run with interactive shell.
  """

  ports: Optional[docker_adapter.Ports] = None
  volumes: Optional[Dict[str, str]] = None
  mount_gcs_path: bool = True
  interactive: bool = False


@attr.s(auto_attribs=True)
class Local(xm.Executor):
  """Current machine executor.

  Attributes:
    requirements: Resources to be requested from the host.
      Note: Currently, only the `local_gpu` resource is supported (and only with
        a container-based executable). Any other resource requirement will be
        ignored.
    docker_options: Options applied if the job is a container-based executable.
    experimental_stream_output: Whether to pipe the job's stdout and stderr to
      the terminal. Might be removed once we decide on the logging design.
  """

  requirements: xm.JobRequirements = attr.Factory(xm.JobRequirements)
  docker_options: Optional[DockerOptions] = None
  experimental_stream_output: bool = True

  Spec = LocalSpec  # pylint: disable=invalid-name

  def __attrs_post_init__(self):
    local_execution = importlib.import_module('xmanager.xm_local.execution')
    local_execution.register()

  @override
  @classmethod
  async def launch(
      cls, local_experiment_unit: Any, job_group: xm.JobGroup
  ) -> Sequence[handles.ExecutionHandle]:
    return await registry.get_launch_method(cls)(
        local_experiment_unit, job_group
    )


@attr.s(auto_attribs=True)
class TpuCapability:
  """TPU capability configures the TPU software requested by an executor."""

  # Read about TPU versions:
  # https://cloud.google.com/tpu/docs/version-switching
  tpu_runtime_version: str


@attr.s(auto_attribs=True)
class TensorboardCapability:
  """Tensorboard capability integrates a Vertex AI Job with Tensorboard."""

  # The name of the tensorboard to use.
  name: str
  # The "gs://$GCS_BUCKET/dir_name" to save output.
  # Tensorboard will read the logs from $BASE_OUTPUT_DIRECTORY/logs/
  # If None, then the root of the default bucket will be used.
  base_output_directory: Optional[str] = None


@attr.s(auto_attribs=True)
class VertexSpec(xm.ExecutorSpec):
  """Vertex AI spec describes the Google Cloud Platform (GCP) location."""

  # An image registry name tag to push.
  # The image tag should be in the form 'myregistryhost/name:tag'
  push_image_tag: Optional[str] = None


@attr.s(auto_attribs=True)
class Vertex(xm.Executor):
  """Vertex AI Executor describes the runtime environment of GCP."""

  requirements: xm.JobRequirements = attr.Factory(xm.JobRequirements)
  tensorboard: Optional[TensorboardCapability] = None

  Spec = VertexSpec  # pylint: disable=invalid-name

  def __attrs_post_init__(self):
    vertex_execution = importlib.import_module('xmanager.cloud.vertex')
    vertex_execution.register()

  @override
  @classmethod
  async def launch(
      cls, local_experiment_unit: Any, job_group: xm.JobGroup
  ) -> Sequence[handles.ExecutionHandle]:
    return await registry.get_launch_method(cls)(
        local_experiment_unit, job_group
    )


# Declaring variable aliases for legacy compatability.
# New code should not use these aliases.
Caip = Vertex
CaipSpec = VertexSpec


@attr.s(auto_attribs=True)
class KubernetesSpec(xm.ExecutorSpec):
  """K8s spec describes the K8s location."""

  # An image registry name tag to push.
  # The image tag should be in the form 'myregistryhost/name:tag'
  push_image_tag: Optional[str] = None


@attr.s(auto_attribs=True)
class Kubernetes(xm.Executor):
  """K8s Executor describes the runtime environment of Kubernetes."""

  requirements: xm.JobRequirements = attr.Factory(xm.JobRequirements)
  cloud_provider: str = GOOGLE_KUBERNETES_ENGINE_CLOUD_PROVIDER
  tpu_capability: TpuCapability | None = None

  Spec = KubernetesSpec  # pylint: disable=invalid-name

  def __attrs_post_init__(self):
    k8s_execution = importlib.import_module('xmanager.cloud.kubernetes')
    k8s_execution.register()

  @override
  @classmethod
  async def launch(
      cls, local_experiment_unit: Any, job_group: xm.JobGroup
  ) -> Sequence[handles.ExecutionHandle]:
    return await registry.get_launch_method(cls)(
        local_experiment_unit, job_group
    )


@attr.s(auto_attribs=True)
class VertexTrainingClusterSpec(xm.ExecutorSpec):
  """Vertex Training Cluster spec for packaging.

  Attributes:
    push_image_tag: Image registry path to push (for Docker packaging).
    squashfs_path: Path to pre-built .squashfs on cluster storage.
    auto_convert_to_squashfs: Convert Docker image to squashfs on cluster.
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

  Example (Auto-Generate with NemoRunLauncher):
    from xmanager.cloud import launchers

    executor = xm_local.VertexTrainingCluster(
        cluster_type='hcc-a4',
        partition='a4',
        requirements=xm.JobRequirements(replicas=4),
        launcher=launchers.NemoRunLauncher(
            recipe='pretrain/llama3p1_2b_pt.py',
            container_image='nemo.sqsh',
        ),
    )

  Example (Template):
    executor = xm_local.VertexTrainingCluster(
        cluster_type='hcc-a4',
        sbatch_template=Path('my_template.j2'),
    )

  Example (Raw Script):
    executor = xm_local.VertexTrainingCluster(
        cluster_type='hcc-a4',
        sbatch_script='#!/bin/bash\\n#SBATCH --nodes=2\\nsrun train.py',
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

  # Submission mode 2: Custom template (Jinja2 string or path)
  sbatch_template: Optional[Union[str, Path]] = None

  # Submission mode 3: Auto-generate (use launcher)
  # Type is Any to avoid circular import; actual type is Launcher
  launcher: Optional[Any] = None

  # Additional sbatch flags (arbitrary --key=value pairs)
  sbatch_flags: Dict[str, str] = attr.Factory(dict)

  # Connection settings
  use_gcloud_ssh: bool = False  # Use gcloud compute ssh
  login_node: Optional[str] = None  # SSH target (omit if on login node)
  ssh_hostname: Optional[str] = None  # Hostname for gcloud ssh -o Hostname=...

  # Working directories on cluster
  work_dir: Optional[str] = None
  log_dir: Optional[str] = None  # Directory for slurm-*.out files

  # Container configuration
  container_image: Optional[str] = None  # .squashfs or docker image path
  container_mounts: List[str] = attr.Factory(list)

  # Environment configuration
  env_vars: Dict[str, str] = attr.Factory(dict)

  # Prologue/Epilogue commands
  prologue_commands: List[str] = attr.Factory(list)
  epilogue_commands: List[str] = attr.Factory(list)

  # Log streaming
  stream_output: bool = True

  Spec = VertexTrainingClusterSpec  # pylint: disable=invalid-name

  def __attrs_post_init__(self):
    vertex_training_cluster_execution = importlib.import_module(
        'xmanager.cloud.vertex_training_cluster'
    )
    vertex_training_cluster_execution.register()

  def get_cluster_config(self):
    """Get NCCL and networking configuration for this cluster type.

    Returns:
      ClusterConfig with nccl_dir, nccl_env_vars, setup_script, etc.
    """
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
