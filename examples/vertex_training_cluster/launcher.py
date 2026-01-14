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
r"""XManager launcher for Vertex Training Cluster with NeMo Run.

This example demonstrates how to run training jobs on Slurm-based
Vertex Training Clusters using the deep XManager integration.

Features:
  - Automatic sbatch script generation via NemoRunLauncher
  - Job submission via sbatch with proper tracking
  - Job status monitoring via squeue/sacct
  - Log streaming via SSH
  - Full experiment tracking integration

The executor provides NCCL configurations per cluster type:
  - hcc-a3m: H100 with TCPXO
  - hcc-a3u: H200 with gIB
  - hcc-a4: B200 with gIB
  - hcc-a3h: H100 with gIB

Usage:
   xmanager launch examples/vertex_training_cluster/launcher.py -- \
    --cluster_type=hcc-a4 \
    --partition=a4 \
    --login_node=vmdsa405-login-001 \
    --use_gcloud_ssh \
    --ssh_hostname=nic0.vmdsa405-login-001.asia-southeast1-b.c.ai-infra-recipe-validation.internal.gcpnode.com \
    --work_dir=/home/abhishekbhgwt_google_com/vertexai-mds/nemo \
    --recipe=pretrain/llama3p1_2b_pt.py \
    --account=aaie \
    --nodes=2                        
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
_NODES = flags.DEFINE_integer('nodes', 1, 'Number of nodes')
_TIME_LIMIT = flags.DEFINE_string('time_limit', '0', 'Time limit (0=unlimited)')

# Paths
_WORK_DIR = flags.DEFINE_string(
    'work_dir', None, 'Working directory on cluster (required)')
_IMAGE = flags.DEFINE_string(
    'image', 'nemo-demo.sqsh',
    'Container image (squashfs) relative to work_dir')

# Recipe
_RECIPE = flags.DEFINE_string(
    'recipe', 'pretrain/llama3p1_2b_pt.py', 'Training recipe script')
_EXP_NAME = flags.DEFINE_string(
    'exp_name', 'xmanager-nemo-run', 'Experiment name')

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
  print("Vertex Training Cluster - XManager Deep Integration")
  print("=" * 60)

  with xm_local.create_experiment(
      experiment_title=f'vtc_{_EXP_NAME.value}'
  ) as experiment:

    # Build container image path
    container_image = f"{_WORK_DIR.value}/{_IMAGE.value}"

    # Create NemoRunLauncher for automatic sbatch generation
    # Pass account via extra_args if specified
    extra_args = []
    if _ACCOUNT.value:
      extra_args.extend(['--account', _ACCOUNT.value])

    launcher = xm_local.launchers.NemoRunLauncher(
        recipe=_RECIPE.value,
        container_image=container_image,
        experiment_name=_EXP_NAME.value,
        extra_args=extra_args,
    )

    # Create executor with full configuration
    executor = xm_local.VertexTrainingCluster(
        # Cluster settings
        cluster_type=_CLUSTER_TYPE.value,
        partition=_PARTITION.value,
        account=_ACCOUNT.value,
        time_limit=_TIME_LIMIT.value,

        # Use NemoRunLauncher for sbatch generation
        launcher=launcher,

        # Resource requirements (replicas = number of nodes)
        requirements=xm.JobRequirements(replicas=_NODES.value),

        # Connection settings
        login_node=_LOGIN_NODE.value,
        use_gcloud_ssh=_USE_GCLOUD_SSH.value,
        ssh_hostname=_SSH_HOSTNAME.value,

        # Working directory
        work_dir=_WORK_DIR.value,

        # Log streaming
        stream_output=_STREAM_OUTPUT.value,
    )

    # Get NCCL configuration for this cluster type
    cluster_config = executor.get_cluster_config()

    print(f"\nConfiguration:")
    print(f"  Cluster: {cluster_config.cluster_type}")
    print(f"  GPU: {cluster_config.gpu_type}")
    print(f"  GPUs per node: {cluster_config.gpus_per_node}")
    print(f"  Nodes: {_NODES.value}")
    print(f"  Total GPUs: {_NODES.value * cluster_config.gpus_per_node}")
    print(f"  Work dir: {_WORK_DIR.value}")
    print(f"  Image: {_IMAGE.value}")
    print(f"  Recipe: {_RECIPE.value}")
    print(f"  Partition: {_PARTITION.value}")
    print(f"  Time limit: {_TIME_LIMIT.value}")

    if _DRY_RUN.value:
      print("\n[DRY RUN] Configuration validated, not submitting job")
      return

    if not _LOGIN_NODE.value:
      print("\n[SKIP] No login_node specified, cannot submit job")
      print("Add --login_node (and --use_gcloud_ssh if needed) to submit")
      return

    # Create a job with the executor
    # The executable is a placeholder since NemoRunLauncher handles the command
    job = xm.Job(
        executable=xm.Binary(path=_RECIPE.value),
        executor=executor,
    )

    # Add job to experiment - this triggers sbatch submission
    print("\nSubmitting job to Slurm via sbatch...")
    experiment.add(xm.JobGroup(job=job))

    print(f"\nExperiment ID: {experiment.experiment_id}")
    print("\nJob submitted! Use the following to monitor:")
    print(f"  squeue --me")
    print(f"  xmanager list")


if __name__ == '__main__':
  app.run(main)
