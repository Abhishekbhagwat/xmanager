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
r"""XManager launcher for NeMo training on Vertex Training Cluster.

This example demonstrates how to run distributed NeMo training jobs on
Slurm-based Vertex Training Clusters using XManager.

Features:
  - PyTorch distributed training with MPI (--mpi=pmix)
  - Factory args and trainer args passed directly (no config file needed)
  - Container-based execution (work_dir mounted at same path in container)
  - torchrun with rendezvous endpoint for multi-node training
  - NCCL configuration per cluster type
  - Native Vertex AI TensorBoard integration via sbatch --extra flag

Directory structure on cluster:
  work_dir/
    pretrain.py      # Training script (already on cluster or rsynced via --local_files)
    logs/            # Output logs (optional subdirectory)
    slurm-*.out      # Slurm output files

To rsync local files to the cluster, use --local_files:
  --local_files=./pretrain.py --local_files=./my_config.yaml

Usage:
   # Set environment variables for your cluster
   export LUSTRE_INSTANCE_NAME=ls1-europe-west4-a
   export LOGS_PATH=/mnt/lustre/${LUSTRE_INSTANCE_NAME}/jobs

   xmanager launch examples/nemo_vtc/launcher.py -- \
    --cluster_type=hcc-a3u \
    --partition=a3u \
    --login_node=vmdsa3u04-login-001 \
    --use_gcloud_ssh \
    --ssh_hostname=nic0.vmdsa3u04-login-001.europe-west4-a.c.ai-infra-recipe-validation.internal.gcpnode.com \
    --work_dir=/home/abhishekbhgwt_google_com/vertexai-mds/nemo \
    --logs_path=${LOGS_PATH} \
    --container_image=/home/common/images/nemo.25.07.sqsh \
    --training_script=pretrain.py \
    --extra_args="--factory='configure_recipe(explicit_log_dir=${LOGS_PATH}/\${JOB_IDENTIFIER}/)'" \
    --extra_args="trainer.num_nodes=\${SLURM_NNODES}" \
    --extra_args="trainer.max_steps=15" \
    --account=aaie \
    --nodes=8

   # With TensorBoard integration (logs to GCS, checkpoints stay on Lustre):
   # VMDS sets AIP_TENSORBOARD_LOG_DIR - pass it to tensorboard_log_dir
   xmanager launch examples/nemo_vtc/launcher.py -- \
    --cluster_type=hcc-a3u \
    --partition=a3u \
    --login_node=vmdsa3u04-login-001 \
    --use_gcloud_ssh \
    --ssh_hostname=nic0.vmdsa3u04-login-001.europe-west4-a.c.ai-infra-recipe-validation.internal.gcpnode.com \
    --work_dir=/home/abhishekbhgwt_google_com/vertexai-mds/nemo \
    --logs_path=${LOGS_PATH} \
    --container_image=/home/common/images/nemo.25.07.sqsh \
    --tensorboard=ai-infra-europe-west4-tb \
    --tensorboard_region=europe-west4 \
    --tensorboard_project=ai-infra-recipe-validation \
    --tensorboard_gcs_path=ai-infra-gcs-europe-west4 \
    --local_files=/usr/local/google/home/abhishekbhgwt/xmanager/examples/nemo_vtc/pretrain.py \
    --extra_args="--factory='configure_recipe(explicit_log_dir=${LOGS_PATH}/\${JOB_IDENTIFIER}/, tensorboard_log_dir=\${AIP_TENSORBOARD_LOG_DIR})'" \
    --extra_args="trainer.num_nodes=\${SLURM_NNODES}" \
    --extra_args="trainer.max_steps=5" \
    --nodes=4
"""

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_local

# Cluster configuration
_CLUSTER_TYPE = flags.DEFINE_string(
    'cluster_type', 'hcc-a4',
    'Cluster type: hcc-a3m (H100+TCPXO), hcc-a3u (H200+gIB), hcc-a4 (B200+gIB), hcc-a3h (H100+gIB)')
_PARTITION = flags.DEFINE_string('partition', 'a4', 'Slurm partition')
_ACCOUNT = flags.DEFINE_string('account', None, 'Slurm account (optional)')
_NODES = flags.DEFINE_integer('nodes', 2, 'Number of nodes')
_TIME_LIMIT = flags.DEFINE_string('time_limit', None, 'Time limit (e.g., "1:00:00"). None=no limit directive')

# Paths on cluster
_WORK_DIR = flags.DEFINE_string(
    'work_dir', None,
    'Working directory on cluster (required). Contains scripts, etc. '
    'Mounted at same path inside container.')
_LOGS_PATH = flags.DEFINE_string(
    'logs_path', None,
    'Path for job logs and checkpoints on Managed Lustre (optional). '
    'If not set, logs are written to work_dir. Example: /mnt/lustre/<instance>/jobs')
_CONTAINER_IMAGE = flags.DEFINE_string(
    'container_image', None, 'Container image path (.sqsh) on cluster (required)')

# Training configuration
_TRAINING_SCRIPT = flags.DEFINE_string(
    'training_script', 'pretrain.py',
    'Training script name (relative to work_dir)')
_EXTRA_ARGS = flags.DEFINE_multi_string(
    'extra_args', [],
    'Extra args to pass to training script (factory, trainer, etc.)')
_EXP_NAME = flags.DEFINE_string(
    'exp_name', 'nemo-training', 'Experiment/job name prefix')

# Local files to rsync (optional)
_LOCAL_FILES = flags.DEFINE_multi_string(
    'local_files', [],
    'Local files/directories to rsync to work_dir before submission')

# Connection
_LOGIN_NODE = flags.DEFINE_string('login_node', None, 'SSH login node')
_USE_GCLOUD_SSH = flags.DEFINE_bool('use_gcloud_ssh', False, 'Use gcloud SSH')
_SSH_HOSTNAME = flags.DEFINE_string(
    'ssh_hostname', None,
    'SSH hostname override (for gcloud ssh -o Hostname=...)')

# Options
_USE_HOST_PLUGIN = flags.DEFINE_bool(
    'use_host_plugin', True, 'Use gIB from host (mount /usr/local/gib)')
_DRY_RUN = flags.DEFINE_bool(
    'dry_run', False, 'Print configuration but do not submit job')
_STREAM_OUTPUT = flags.DEFINE_bool(
    'stream_output', True, 'Stream job output via SSH tail -f')

# TensorBoard integration (native VTC support via sbatch --extra)
_TENSORBOARD = flags.DEFINE_string(
    'tensorboard', None,
    'Vertex AI TensorBoard display name (e.g., "ai-infra-europe-west4-tb"). '
    'If set, enables native VTC TensorBoard integration. '
    'Instance will be looked up or created automatically.')
_TENSORBOARD_REGION = flags.DEFINE_string(
    'tensorboard_region', 'us-central1',
    'Vertex AI region for TensorBoard instance (e.g., "europe-west4").')
_TENSORBOARD_PROJECT = flags.DEFINE_string(
    'tensorboard_project', None,
    'GCP project ID for TensorBoard. If not set, uses default ADC project.')
_TENSORBOARD_GCS_PATH = flags.DEFINE_string(
    'tensorboard_gcs_path', None,
    'GCS bucket path for TensorBoard logs (e.g., "my-bucket").')


def main(_):
  if not _WORK_DIR.value:
    raise app.UsageError('--work_dir is required')
  if not _CONTAINER_IMAGE.value:
    raise app.UsageError('--container_image is required')

  print("=" * 60)
  print("NeMo Training on Vertex Training Cluster")
  print("=" * 60)

  work_dir = _WORK_DIR.value
  logs_path = _LOGS_PATH.value if _LOGS_PATH.value else work_dir
  training_script = _TRAINING_SCRIPT.value

  with xm_local.create_experiment(
      experiment_title=f'vtc_nemo_{_EXP_NAME.value}'
  ) as experiment:

    # Build container mounts - mount work_dir and logs_path at same path inside container
    # Note: gIB mount (/usr/local/gib) is automatically added by vertex_training_cluster.py
    # based on cluster_config.nccl_dir, so we don't add it here
    container_mounts = [f"{work_dir}:{work_dir}"]
    if _LOGS_PATH.value and _LOGS_PATH.value != work_dir:
      container_mounts.append(f"{logs_path}:{logs_path}")

    # Environment variables
    env_vars = {
        'NEMORUN_HOME': work_dir,
        'GIB_PATH': '/usr/local/gib',
        'NCCL_SOCKET_IFNAME': 'enp0s19,enp192s20',
        'NCCL_DEBUG': 'VERSION',
        'CUDA_DEVICE_MAX_CONNECTIONS': '1',
        'OMP_NUM_THREADS': '12',
    }

    # Container env passthrough for srun --container-env
    container_env_passthrough = ['NCCL_SOCKET_IFNAME', 'NCCL_DEBUG', 'OMP_NUM_THREADS']

    # TensorBoard integration (optional)
    tensorboard = None
    if _TENSORBOARD.value or _TENSORBOARD_GCS_PATH.value:
      tensorboard = xm_local.TensorboardCapability(
          name=_TENSORBOARD.value or '',
          base_output_directory=_TENSORBOARD_GCS_PATH.value,
      )

    # Create executor
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
        work_dir=work_dir,

        # Container configuration
        container_image=_CONTAINER_IMAGE.value,
        container_mounts=container_mounts,
        container_env_passthrough=container_env_passthrough,

        # Use MPI for PyTorch distributed training
        use_mpi=True,

        # Rsync local files if specified
        local_files=list(_LOCAL_FILES.value) if _LOCAL_FILES.value else [],

        # Environment variables
        env_vars=env_vars,

        # Log streaming
        stream_output=_STREAM_OUTPUT.value,

        # TensorBoard integration
        tensorboard=tensorboard,
        tensorboard_region=_TENSORBOARD_REGION.value,
        tensorboard_project=_TENSORBOARD_PROJECT.value,
    )

    # Get cluster configuration for display
    cluster_config = executor.get_cluster_config()

    print(f"\nConfiguration:")
    print(f"  Cluster: {cluster_config.cluster_type}")
    print(f"  GPU: {cluster_config.gpu_type}")
    print(f"  GPUs per node: {cluster_config.gpus_per_node}")
    print(f"  Nodes: {_NODES.value}")
    print(f"  Total GPUs: {_NODES.value * cluster_config.gpus_per_node}")
    print(f"  Work dir: {work_dir}")
    print(f"  Logs path: {logs_path}")
    print(f"  Container: {_CONTAINER_IMAGE.value}")
    print(f"  Training script: {work_dir}/{training_script}")
    print(f"  Partition: {_PARTITION.value}")
    print(f"  Use host gIB: {_USE_HOST_PLUGIN.value}")
    if tensorboard:
      print(f"  TensorBoard: {_TENSORBOARD.value}")
      print(f"  TensorBoard Region: {_TENSORBOARD_REGION.value}")
      print(f"  TensorBoard Project: {_TENSORBOARD_PROJECT.value or '(default ADC project)'}")
      print(f"  TensorBoard GCS Path: {_TENSORBOARD_GCS_PATH.value}")

    if _DRY_RUN.value:
      print("\n[DRY RUN] Configuration validated, not submitting job")
      return

    if not _LOGIN_NODE.value:
      print("\n[SKIP] No login_node specified, cannot submit job")
      print("Add --login_node (and --use_gcloud_ssh if needed) to submit")
      return

    # Build torchrun command
    # The template exports: MASTER_ADDR, MASTER_PORT, GPUS_PER_NODE, JOB_IDENTIFIER
    # Use ShellSafeArg for args containing shell variables (prevents escaping)
    #
    # IMPORTANT: Variable expansion inside bash -c "..."
    #   ${VAR}  - expands in OUTER shell (sbatch script) before srun
    #   \${VAR} - expands INSIDE bash -c (after srun sets per-task vars)
    #
    # SLURM_PROCID is set BY srun for each task, so must use \${SLURM_PROCID}
    # Other vars (GPUS_PER_NODE, SLURM_NNODES, MASTER_ADDR) are set before srun
    torchrun_args = [
        xm.ShellSafeArg('--nproc-per-node=${GPUS_PER_NODE}'),
        xm.ShellSafeArg('--nnodes=${SLURM_NNODES}'),
        xm.ShellSafeArg('--node_rank=\\${SLURM_PROCID}'),  # Escaped: expands inside srun
        xm.ShellSafeArg('--rdzv_id=${JOB_IDENTIFIER}'),
        xm.ShellSafeArg('--rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT}'),
        '--rdzv-backend=static',
        f'{work_dir}/{training_script}',
    ]

    # Add extra args - always wrap in ShellSafeArg to preserve user's quoting
    # (shlex.quote would mangle quotes inside bash -c context)
    for extra_arg in _EXTRA_ARGS.value:
      torchrun_args.append(xm.ShellSafeArg(extra_arg))

    print(f"\nTorchrun args:")
    for arg in torchrun_args:
      if isinstance(arg, xm.ShellSafeArg):
        print(f"  {arg.arg}")
      else:
        print(f"  {arg}")
    print()

    # Create job - torchrun with distributed training args
    # The sbatch template wraps in bash -c, so shell variables will expand
    job = xm.Job(
        executable=xm.Binary(path='torchrun'),
        args=torchrun_args,
        executor=executor,
    )

    # Submit job
    print("Submitting job to Slurm via sbatch...")
    experiment.add(xm.JobGroup(job=job))

    print(f"\nExperiment ID: {experiment.experiment_id}")
    print("\nJob submitted! Use the following to monitor:")
    print(f"  squeue --me")
    print(f"  xmanager list")
    print(f"\nSlurm logs: {work_dir}/slurm-<job_id>.out")
    print(f"NeMo logs/checkpoints: {logs_path}/<job_id>/")


if __name__ == '__main__':
  app.run(main)
