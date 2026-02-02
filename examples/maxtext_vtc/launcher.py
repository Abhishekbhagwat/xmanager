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

r"""XManager launcher for MaxText Vizier hyperparameter optimization on VTC.

This launcher runs Vertex AI Vizier hyperparameter optimization for MaxText
training jobs on Slurm-based Vertex Training Clusters.

Features:
  - JAX distributed training with automatic coordinator setup
  - Vertex AI Vizier for Bayesian hyperparameter optimization
  - GCS-based metric reading from TensorBoard logs
  - Uses ADC (Application Default Credentials) for project/region
  - Local file sync via --local_files (rsync to work_dir before job submission)

Usage:
  xmanager launch examples/maxtext_vtc/launcher.py -- \
    --cluster_type=hcc-a3u \
    --partition=a3u \
    --login_node=vmdsa3u04-login-001 \
    --use_gcloud_ssh \
    --ssh_hostname=nic0.vmdsa3u04-login-001.europe-west4-a.c.ai-infra-recipe-validation.internal.gcpnode.com \
    --work_dir=/home/abhishekbhgwt_google_com/vertexai-mds/nemo \
    --container_image=/mnt/lustre/ls1-europe-west4-a/images/jax-maxtext-2025-10-01.sqsh \
    --config_file=/mnt/jobs/config.yaml \
    --gcs_bucket=ai-infra-gcs-europe-west4 \
    --cluster_id=vmdsa3u04-7070932889948389376 \
    --nodes=2 \
    --num_trials=5

To copy local config files to the cluster:
  --local_files=./gemma3-27b.yaml --local_files=./another-config.yaml

To find your cluster_id:
  gsutil ls gs://<your-bucket>/
"""

from google.cloud import aiplatform_v1beta1 as aip

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_local
from xmanager.vizier import vizier_cloud

# =============================================================================
# Cluster Configuration
# =============================================================================
_CLUSTER_TYPE = flags.DEFINE_string(
    'cluster_type', 'hcc-a3u',
    'Cluster type: hcc-a3m, hcc-a3u, hcc-a4, hcc-a3h')
_PARTITION = flags.DEFINE_string('partition', 'a3u', 'Slurm partition')
_ACCOUNT = flags.DEFINE_string('account', None, 'Slurm account (optional)')
_NODES = flags.DEFINE_integer('nodes', 2, 'Number of nodes')
_TIME_LIMIT = flags.DEFINE_string('time_limit', None, 'Time limit (e.g., "1:00:00")')

# =============================================================================
# Paths and Container
# =============================================================================
_WORK_DIR = flags.DEFINE_string(
    'work_dir', '/home/abhishekbhgwt_google_com/vertexai-mds/maxtext',
    'Working directory on cluster (required)')
_MAXTEXT_DIR = flags.DEFINE_string(
    'maxtext_dir', '/workspace/MaxText',
    'MaxText installation directory in container')
_CONTAINER_IMAGE = flags.DEFINE_string(
    'container_image', '/mnt/lustre/ls1-europe-west4-a/images/jax-maxtext-2025-10-01.sqsh',
    'Container image path (.sqsh) on cluster')
_CONFIG_FILE = flags.DEFINE_string(
    'config_file', '/mnt/jobs/gemma3-27b.yaml',
    'Path to MaxText config file on cluster')

# =============================================================================
# GCS Configuration
# =============================================================================
_GCS_BUCKET = flags.DEFINE_string(
    'gcs_bucket', None, 'GCS bucket name (required, e.g., my-bucket)')
_CLUSTER_ID = flags.DEFINE_string(
    'cluster_id', None,
    'Cluster ID for GCS paths (required). Find with: gsutil ls gs://<bucket>/')

# =============================================================================
# Training Configuration
# =============================================================================
_STEPS = flags.DEFINE_integer('steps', 10, 'Training steps per trial')

# =============================================================================
# Connection
# =============================================================================
_LOGIN_NODE = flags.DEFINE_string('login_node', None, 'SSH login node (required)')
_USE_GCLOUD_SSH = flags.DEFINE_bool('use_gcloud_ssh', False, 'Use gcloud compute ssh')
_SSH_HOSTNAME = flags.DEFINE_string('ssh_hostname', None, 'SSH hostname override')

# =============================================================================
# Vizier Configuration
# =============================================================================
_NUM_TRIALS = flags.DEFINE_integer('num_trials', 5, 'Number of Vizier trials')
_NUM_PARALLEL = flags.DEFINE_integer('num_parallel', 1, 'Parallel trials')
_VIZIER_METRIC = flags.DEFINE_string(
    'metric', 'learning/loss', 'Metric to optimize from TensorBoard')
_VIZIER_PROJECT = flags.DEFINE_string(
    'vizier_project', 'ai-infra-recipe-validation',
    'GCP project for Vizier study (overrides ADC)')
_VIZIER_REGION = flags.DEFINE_string(
    'vizier_region', 'europe-west4',
    'GCP region for Vizier study (overrides ADC)')

# =============================================================================
# Local Files (rsync to work_dir before submission)
# =============================================================================
_LOCAL_FILES = flags.DEFINE_multi_string(
    'local_files', [],
    'Local files/directories to rsync to work_dir before submission '
    '(e.g., config files)')

# =============================================================================
# Other
# =============================================================================
_DRY_RUN = flags.DEFINE_bool('dry_run', False, 'Validate config without submitting')


def get_study_spec() -> aip.StudySpec:
  """Define the Vizier study specification.

  Modify this function to change the hyperparameters being optimized.
  """
  return aip.StudySpec(
      # Let Vizier choose the algorithm (typically Bayesian optimization)
      algorithm=aip.StudySpec.Algorithm.ALGORITHM_UNSPECIFIED,

      parameters=[
          # Learning rate: log-scale search from 1e-5 to 1e-3
          aip.StudySpec.ParameterSpec(
              parameter_id='learning_rate',
              double_value_spec=aip.StudySpec.ParameterSpec.DoubleValueSpec(
                  min_value=1e-3,
                  max_value=1e-1,
              ),
              scale_type=aip.StudySpec.ParameterSpec.ScaleType.UNIT_LOG_SCALE,
          ),
          # Add more parameters here as needed:
          # aip.StudySpec.ParameterSpec(
          #     parameter_id='per_device_batch_size',
          #     integer_value_spec=aip.StudySpec.ParameterSpec.IntegerValueSpec(
          #         min_value=1,
          #         max_value=8,
          #     ),
          # ),
      ],

      metrics=[
          aip.StudySpec.MetricSpec(
              metric_id=_VIZIER_METRIC.value,
              goal=aip.StudySpec.MetricSpec.GoalType.MINIMIZE,
          )
      ],
  )


def create_executor() -> xm_local.VertexTrainingCluster:
  """Create the VTC executor with MaxText-specific configuration."""
  container_mounts = [
      f"{_WORK_DIR.value}:/mnt/jobs",
  ]

  env_vars = {
      'NCCL_SOCKET_IFNAME': 'enp0s19,enp192s20',
      'NCCL_DEBUG': 'VERSION',
      'CUDA_DEVICE_MAX_CONNECTIONS': '1',
      'TF_CPP_MIN_LOG_LEVEL': '0',
      'NVTE_FUSED_ATTN': '1',
      'JAX_REMOVE_CUSTOM_PARTITIONING_PTR_FROM_CACHE_KEY': 'true',
      'JAX_ENABLE_PGLE': 'false',
      'JAX_PLATFORMS': 'cuda',  # Force GPU backend, skip TPU
      'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.98',
      'SLURM_NTASKS_PER_NODE': '8',  # Required for JAX multiprocess
      'XLA_FLAGS': (
          '--xla_gpu_enable_latency_hiding_scheduler=true '
          '--xla_gpu_enable_triton_gemm=false '
          '--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL '
          '--xla_gpu_all_reduce_combine_threshold_bytes=2147483648 '
          '--xla_gpu_all_gather_combine_threshold_bytes=2147483648 '
          '--xla_gpu_reduce_scatter_combine_threshold_bytes=16777216 '
          '--xla_gpu_enable_pipelined_all_gather=true '
          '--xla_gpu_enable_pipelined_reduce_scatter=true '
          '--xla_gpu_enable_pipelined_all_reduce=true '
          '--xla_gpu_enable_while_loop_double_buffering=true '
          '--xla_gpu_enable_all_gather_combine_by_dim=false '
          '--xla_gpu_enable_reduce_scatter_combine_by_dim=false '
          '--xla_disable_hlo_passes=rematerialization'
      ),
  }

  # Env vars to pass into container via --container-env
  # JAX_COORDINATOR_ADDRESS is set by template, others from env_vars
  container_env_passthrough = [
      'JAX_COORDINATOR_ADDRESS',
      'JAX_PLATFORMS',
      'NCCL_DEBUG',
      'XLA_FLAGS',
      'SLURM_NTASKS_PER_NODE',
  ]

  tensorboard = xm_local.TensorboardCapability(
      name='',
      base_output_directory=_GCS_BUCKET.value,
  )

  return xm_local.VertexTrainingCluster(
      cluster_type=_CLUSTER_TYPE.value,
      partition=_PARTITION.value,
      account=_ACCOUNT.value,
      time_limit=_TIME_LIMIT.value,
      requirements=xm.JobRequirements(replicas=_NODES.value),
      login_node=_LOGIN_NODE.value,
      use_gcloud_ssh=_USE_GCLOUD_SSH.value,
      ssh_hostname=_SSH_HOSTNAME.value,
      work_dir=_WORK_DIR.value,
      container_image=_CONTAINER_IMAGE.value,
      container_mounts=container_mounts,
      container_env_passthrough=container_env_passthrough,
      setup_jax_coordinator=True,
      master_port=6002,
      env_vars=env_vars,
      stream_output=True,
      tensorboard=tensorboard,
      local_files=list(_LOCAL_FILES.value) if _LOCAL_FILES.value else [],
  )


def build_training_args():
  """Build MaxText training arguments."""
  return [
      f'src/MaxText/train.py',
      _CONFIG_FILE.value,
      xm.ShellSafeArg(f'steps={_STEPS.value}'),
      xm.ShellSafeArg(f'base_output_directory=gs://{_GCS_BUCKET.value}'),
      xm.ShellSafeArg(f'tensorboard_dir=gs://{_GCS_BUCKET.value}/{_CLUSTER_ID.value}/tensorboard/job-${{SLURM_JOB_ID}}'),
      xm.ShellSafeArg('run_name=job-${SLURM_JOB_ID}'),
      xm.ShellSafeArg("dataset_path=gs://davidsotomora-asia-southeast1"),
      xm.ShellSafeArg('packing=False'),
  ]

def main(_):
  # Validate required flags
  if not _GCS_BUCKET.value:
    raise app.UsageError('--gcs_bucket is required')
  if not _CLUSTER_ID.value:
    raise app.UsageError('--cluster_id is required. Find with: gsutil ls gs://<bucket>/')
  if not _LOGIN_NODE.value:
    raise app.UsageError('--login_node is required')

  import time
  timestamp = time.strftime("%Y%m%d-%H%M%S")

  print("=" * 60)
  print("MaxText Vizier Hyperparameter Optimization")
  print("=" * 60)

  executor = create_executor()
  cluster_config = executor.get_cluster_config()

  print(f"\nCluster:")
  print(f"  Type: {cluster_config.cluster_type} ({cluster_config.gpu_type})")
  print(f"  Nodes: {_NODES.value} ({_NODES.value * cluster_config.gpus_per_node} GPUs)")
  print(f"  Partition: {_PARTITION.value}")

  print(f"\nPaths:")
  print(f"  Work dir: {_WORK_DIR.value}")
  print(f"  Config: {_CONFIG_FILE.value}")
  print(f"  GCS bucket: gs://{_GCS_BUCKET.value}")
  print(f"  Cluster ID: {_CLUSTER_ID.value}")
  if _LOCAL_FILES.value:
    print(f"  Local files to sync: {_LOCAL_FILES.value}")

  print(f"\nVizier:")
  print(f"  Project: {_VIZIER_PROJECT.value}")
  print(f"  Region: {_VIZIER_REGION.value}")
  print(f"  Trials: {_NUM_TRIALS.value}")
  print(f"  Parallel: {_NUM_PARALLEL.value}")
  print(f"  Metric: {_VIZIER_METRIC.value}")
  print(f"  Steps per trial: {_STEPS.value}")

  if _DRY_RUN.value:
    print("\n[DRY RUN] Config validated, not submitting")
    return

  with xm_local.create_experiment(
      experiment_title=f'maxtext_vizier_{timestamp}'
  ) as experiment:

    job = xm.Job(
        executable=xm.Binary(path='python'),
        args=build_training_args(),
        executor=executor,
    )

    # Create Vizier exploration
    exploration = vizier_cloud.VizierExploration(
        experiment=experiment,
        job=job,
        study_factory=vizier_cloud.NewStudy(
            study_config=get_study_spec(),
            project=_VIZIER_PROJECT.value,
            location=_VIZIER_REGION.value,
        ),
        num_trials_total=_NUM_TRIALS.value,
        num_parallel_trial_runs=_NUM_PARALLEL.value,
        metric_name=_VIZIER_METRIC.value,
        gcs_log_base=f'gs://{_GCS_BUCKET.value}',
        # Don't pass cluster_id - MaxText uses different path pattern
    )

    print(f"\nStarting Vizier study...")
    print(f"Monitor: squeue --me")
    print(f"Vizier: https://console.cloud.google.com/vertex-ai/experiments")

    exploration.launch(poll_frequency_in_sec=60)

    print(f"\nCompleted! Experiment ID: {experiment.experiment_id}")


if __name__ == '__main__':
  app.run(main)
