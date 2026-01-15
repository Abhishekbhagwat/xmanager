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

r"""XManager launcher for MaxText on Vertex Training Cluster.

This example demonstrates how to run MaxText training on Slurm-based
Vertex Training Clusters using the XManager integration with JAX distributed
training support.

Features:
  - JAX distributed training with automatic coordinator setup
  - Rsync of local config files to the cluster
  - Container-based execution with proper GCS and data mounting
  - Automatic NCCL configuration per cluster type
  - Job status monitoring and log streaming

The executor provides optimized NCCL configurations per cluster type:
  - hcc-a3m: H100 with TCPXO
  - hcc-a3u: H200 with gIB
  - hcc-a4: B200 with gIB
  - hcc-a3h: H100 with gIB

Usage:
   xmanager launch examples/maxtext_vtc/launcher.py -- \
    --cluster_type=hcc-a4 \
    --partition=a4 \
    --login_node=vmdsa405-login-001 \
    --use_gcloud_ssh \
    --ssh_hostname=nic0.vmdsa405-login-001.asia-southeast1-b.c.ai-infra-recipe-validation.internal.gcpnode.com \
    --work_dir=/home/abhishekbhgwt_google_com/vertexai-mds/maxtext \
    --maxtext_dir=/workspace/MaxText \
    --config_file=./config.yaml \
    --account=aaie \
    --nodes=2

Prerequisites:
  1. A squashfs container image with JAX/MaxText installed
  2. A config.yaml file with MaxText hyperparameters
  3. Access to a Vertex Training Cluster
  4. SSH access to the cluster login node

Container Setup:
  Your container should include:
  - JAX with GPU support
  - MaxText and dependencies
  - Any custom model code or datasets
"""

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_local

# Cluster configuration
_CLUSTER_TYPE = flags.DEFINE_string(
    'cluster_type', 'hcc-a4',
    'Cluster type: hcc-a3m (H100), hcc-a3u (H200), hcc-a4 (B200), hcc-a3h (H100)')
_PARTITION = flags.DEFINE_string('partition', 'a4', 'Slurm partition')
_ACCOUNT = flags.DEFINE_string('account', None, 'Slurm account (optional)')
_NODES = flags.DEFINE_integer('nodes', 4, 'Number of nodes for training')
_TIME_LIMIT = flags.DEFINE_string('time_limit', '0', 'Time limit (0=unlimited)')

# Paths on cluster
_WORK_DIR = flags.DEFINE_string(
    'work_dir', None, 'Working directory on cluster (required)')
_MAXTEXT_DIR = flags.DEFINE_string(
    'maxtext_dir', '/workspace/MaxText',
    'MaxText installation directory (in container or on cluster)')
_IMAGE = flags.DEFINE_string(
    'image', 'maxtext-jax.sqsh',
    'Container image (squashfs) relative to work_dir or absolute path')

# Local paths to rsync
_CONFIG_FILE = flags.DEFINE_string(
    'config_file', './config.yaml',
    'Local path to MaxText config.yaml to rsync to cluster')
_EXTRA_FILES = flags.DEFINE_multi_string(
    'extra_files', [],
    'Additional local files/directories to rsync (e.g., custom datasets, scripts)')

# MaxText training configuration
_TRAIN_SCRIPT = flags.DEFINE_string(
    'train_script', 'train.py',
    'Training script name (relative to maxtext_dir)')
_RUN_NAME = flags.DEFINE_string(
    'run_name', 'maxtext-run', 'Run name for experiment tracking')

# GCS and data paths
_GCS_BUCKET = flags.DEFINE_string(
    'gcs_bucket', None,
    'GCS bucket for checkpoints and logs (e.g., gs://my-bucket)')
_DATA_DIR = flags.DEFINE_string(
    'data_dir', '/mnt/data',
    'Data directory path (on cluster or mounted storage)')

# Connection
_LOGIN_NODE = flags.DEFINE_string('login_node', None, 'SSH login node')
_USE_GCLOUD_SSH = flags.DEFINE_bool('use_gcloud_ssh', False, 'Use gcloud SSH')
_SSH_HOSTNAME = flags.DEFINE_string(
    'ssh_hostname', None,
    'SSH hostname override (for gcloud ssh -o Hostname=...)')

# Flags
_DRY_RUN = flags.DEFINE_bool(
    'dry_run', False, 'Print configuration but do not submit job')
_STREAM_OUTPUT = flags.DEFINE_bool(
    'stream_output', True, 'Stream job output via SSH tail -f')


def main(_):
  if not _WORK_DIR.value:
    raise app.UsageError('--work_dir is required')

  print("=" * 60)
  print("MaxText on Vertex Training Cluster - XManager")
  print("=" * 60)

  with xm_local.create_experiment(
      experiment_title=f'maxtext_{_RUN_NAME.value}'
  ) as experiment:

    # Build container image path
    if _IMAGE.value.startswith('/'):
      container_image = _IMAGE.value
    else:
      container_image = f"{_WORK_DIR.value}/{_IMAGE.value}"

    # Prepare local files to rsync
    local_files = [_CONFIG_FILE.value] + list(_EXTRA_FILES.value)

    # Build container mounts
    # Format: host_path:container_path[:options]
    container_mounts = []

    # Mount working directory for configs and outputs
    container_mounts.append(f"{_WORK_DIR.value}:/workspace/work")

    # Mount data directory if specified
    if _DATA_DIR.value:
      container_mounts.append(f"{_DATA_DIR.value}:/data:ro")

    # Mount GCS fuse if available (common on Vertex clusters)
    container_mounts.append("/gcs:/gcs")

    # Join mounts with comma for srun
    container_mounts_str = ",".join(container_mounts)

    # Build training command
    # MaxText train.py reads config.yaml and uses JAX_COORDINATOR_ADDRESS
    config_basename = _CONFIG_FILE.value.split('/')[-1]
    train_command = (
        f"cd {_MAXTEXT_DIR.value} && "
        f"python {_TRAIN_SCRIPT.value} "
        f"/workspace/work/{config_basename} "
        f"run_name={_RUN_NAME.value}"
    )

    # Add GCS bucket if specified
    if _GCS_BUCKET.value:
      train_command += f" base_output_directory={_GCS_BUCKET.value}"

    # Environment variables for JAX
    env_vars = {
        # JAX configuration
        'JAX_PLATFORMS': 'cuda',
        'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.95',
        'CUDA_DEVICE_MAX_CONNECTIONS': '1',

        # Logging
        'TF_CPP_MIN_LOG_LEVEL': '0',
        'JAX_TRACEBACK_FILTERING': 'off',
    }

    # Create executor with full configuration
    executor = xm_local.VertexTrainingCluster(
        # Cluster settings
        cluster_type=_CLUSTER_TYPE.value,
        partition=_PARTITION.value,
        account=_ACCOUNT.value,
        time_limit=_TIME_LIMIT.value,

        # Resource requirements (replicas = number of nodes)
        requirements=xm.JobRequirements(replicas=_NODES.value),

        # Connection settings
        login_node=_LOGIN_NODE.value,
        use_gcloud_ssh=_USE_GCLOUD_SSH.value,
        ssh_hostname=_SSH_HOSTNAME.value,

        # Working directory
        work_dir=_WORK_DIR.value,

        # Local files to rsync before job submission
        local_files=local_files,

        # Container configuration
        container_image=container_image,
        container_mounts=container_mounts_str,

        # JAX distributed training setup
        setup_jax_coordinator=True,  # Exports JAX_COORDINATOR_ADDRESS
        master_port=29500,  # Port for JAX coordinator

        # Environment variables
        env_vars=env_vars,

        # Log streaming
        stream_output=_STREAM_OUTPUT.value,
    )

    # Get cluster configuration for display
    cluster_config = executor.get_cluster_config()

    print(f"\nConfiguration:")
    print(f"  Cluster: {cluster_config.cluster_type}")
    print(f"  GPU: {cluster_config.gpu_type}")
    print(f"  GPUs per node: {cluster_config.gpus_per_node}")
    print(f"  Nodes: {_NODES.value}")
    print(f"  Total GPUs: {_NODES.value * cluster_config.gpus_per_node}")
    print(f"  Work dir: {_WORK_DIR.value}")
    print(f"  MaxText dir: {_MAXTEXT_DIR.value}")
    print(f"  Image: {container_image}")
    print(f"  Config file: {_CONFIG_FILE.value}")
    print(f"  Run name: {_RUN_NAME.value}")
    if _GCS_BUCKET.value:
      print(f"  GCS bucket: {_GCS_BUCKET.value}")
    print(f"\nContainer mounts:")
    for mount in container_mounts:
      print(f"    {mount}")
    print(f"\nLocal files to rsync:")
    for file in local_files:
      print(f"    {file}")
    print(f"\nTraining command:")
    print(f"  {train_command}")
    print(f"\nEnvironment variables:")
    for key, value in env_vars.items():
      print(f"  {key}={value}")

    if _DRY_RUN.value:
      print("\n[DRY RUN] Configuration validated, not submitting job")
      return

    if not _LOGIN_NODE.value:
      print("\n[SKIP] No login_node specified, cannot submit job")
      print("Add --login_node (and --use_gcloud_ssh if needed) to submit")
      return

    # Create job with the executor
    # The Binary represents the training script that will be executed
    job = xm.Job(
        executable=xm.Binary(
            path=f"{_MAXTEXT_DIR.value}/{_TRAIN_SCRIPT.value}",
        ),
        args=[
            f"/workspace/work/{config_basename}",
            f"run_name={_RUN_NAME.value}",
        ] + ([f"base_output_directory={_GCS_BUCKET.value}"]
             if _GCS_BUCKET.value else []),
        executor=executor,
    )

    # Add job to experiment - this triggers:
    # 1. Rsync of local_files to work_dir
    # 2. Sbatch script generation with JAX coordinator setup
    # 3. Job submission via sbatch
    print("\nSubmitting job to Slurm via sbatch...")
    experiment.add(xm.JobGroup(job=job))

    print(f"\nExperiment ID: {experiment.experiment_id}")
    print("\nJob submitted! Use the following to monitor:")
    print(f"  squeue --me")
    print(f"  xmanager list")
    print(f"\nCheckpoints and logs will be saved to: {_GCS_BUCKET.value or 'work_dir'}")


if __name__ == '__main__':
  app.run(main)
