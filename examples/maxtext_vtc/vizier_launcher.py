# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Vizier hyperparameter tuning with MaxText on Vertex Training Cluster.

Demonstrates Vizier-driven HPO with MaxText on VTC:
  - Vizier study creation with search space and objectives
  - Multi-objective optimization (minimize loss + maximize throughput)
  - Automatic metric collection from GCS TensorBoard logs
  - Parallel trial execution across work units
  - Parameter injection into MaxText CLI args

How it works:
  1. Creates a Vizier study with hyperparameter search space
  2. Requests trial suggestions from Vizier
  3. Launches work units with suggested parameters as MaxText args
  4. Monitors GCS for TensorBoard metrics
  5. Reports metrics back to Vizier
  6. Repeats until all trials complete

Usage:
  xmanager launch examples/maxtext_vtc/vizier_launcher.py -- \
    --login_node=vmdsa3u04-login-001 \
    --gcs_bucket=ai-infra-gcs-europe-west4 \
    --use_gcloud_ssh \
    --ssh_hostname=nic0.vmdsa3u04-login-001.europe-west4-a.c.ai-infra-recipe-validation.internal.gcpnode.com \
    --tensorboard=ai-infra-europe-west4-tb \
    --tensorboard_region=europe-west4 \
    --tensorboard_project=ai-infra-recipe-validation \
    --tensorboard_gcs_path=ai-infra-gcs-europe-west4 \
    --num_trials=10 \
    --num_parallel=2 \
    --steps=100
"""

import time

from absl import app
from absl import flags
from google.cloud import aiplatform_v1beta1 as aip

from xmanager import xm
from xmanager import xm_local
from xmanager.vizier import vizier_cloud

# Required
_LOGIN_NODE = flags.DEFINE_string('login_node', None, 'SSH login node (required)')
_GCS_BUCKET = flags.DEFINE_string('gcs_bucket', None, 'GCS bucket name (required)')

# Cluster config
_CLUSTER_TYPE = flags.DEFINE_string('cluster_type', 'hcc-a3u', 'Cluster type')
_PARTITION = flags.DEFINE_string('partition', 'a3u', 'Slurm partition')
_NODES = flags.DEFINE_integer('nodes', 2, 'Number of nodes')
_ACCOUNT = flags.DEFINE_string('account', None, 'Slurm account (optional)')
_TIME_LIMIT = flags.DEFINE_string('time_limit', None, 'Time limit (e.g., "1:00:00")')

# Paths
_WORK_DIR = flags.DEFINE_string(
    'work_dir', '/home/abhishekbhgwt_google_com/vertexai-mds/nemo', 'Work dir on cluster')
_CONTAINER_IMAGE = flags.DEFINE_string(
    'container_image', '/mnt/lustre/ls1-europe-west4-a/images/jax-maxtext-2025-10-01.sqsh',
    'Container image path')
_CONFIG_FILE = flags.DEFINE_string(
    'config_file', '/mnt/jobs/config.yaml', 'MaxText config file')

# SSH
_USE_GCLOUD_SSH = flags.DEFINE_bool('use_gcloud_ssh', True, 'Use gcloud compute ssh')
_SSH_HOSTNAME = flags.DEFINE_string('ssh_hostname', None, 'SSH hostname override')

# Training
_STEPS = flags.DEFINE_integer('steps', 10, 'Training steps per trial')

# Vizier
_NUM_TRIALS = flags.DEFINE_integer('num_trials', 2, 'Number of Vizier trials')
_NUM_PARALLEL = flags.DEFINE_integer('num_parallel', 1, 'Parallel trials')
_METRIC = flags.DEFINE_string('metric', 'learning/loss', 'Primary metric to optimize')
_PROJECT = flags.DEFINE_string('project', 'ai-infra-recipe-validation', 'GCP project')
_REGION = flags.DEFINE_string('region', 'europe-west4', 'GCP region')

# TensorBoard
_TENSORBOARD = flags.DEFINE_string('tensorboard', None, 'Vertex AI TensorBoard display name')
_TENSORBOARD_REGION = flags.DEFINE_string('tensorboard_region', 'us-central1', 'TensorBoard region')
_TENSORBOARD_PROJECT = flags.DEFINE_string('tensorboard_project', None, 'TensorBoard GCP project')
_TENSORBOARD_GCS_PATH = flags.DEFINE_string('tensorboard_gcs_path', None, 'TensorBoard GCS bucket')

# Debug
_DRY_RUN = flags.DEFINE_bool('dry_run', False, 'Validate config without submitting')
_STREAM_OUTPUT = flags.DEFINE_bool('stream_output', True, 'Stream slurm output via SSH')


def main(_):
  missing = []
  if not _LOGIN_NODE.value:
    missing.append('--login_node')
  if not _GCS_BUCKET.value:
    missing.append('--gcs_bucket')
  if missing:
    raise app.UsageError(f'Missing required flags: {", ".join(missing)}')

  timestamp = time.strftime('%Y%m%d-%H%M%S')

  env_vars = {
      'NCCL_SOCKET_IFNAME': 'enp0s19,enp192s20',
      'NCCL_DEBUG': 'VERSION',
      'CUDA_DEVICE_MAX_CONNECTIONS': '1',
      'JAX_PLATFORMS': 'cuda',
      'JAX_REMOVE_CUSTOM_PARTITIONING_PTR_FROM_CACHE_KEY': 'true',
      'JAX_ENABLE_PGLE': 'false',
      'SLURM_NTASKS_PER_NODE': '8',
      'TF_CPP_MIN_LOG_LEVEL': '0',
      'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.98',
      'NVTE_FUSED_ATTN': '1',
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

  container_env_passthrough = [
      'JAX_COORDINATOR_ADDRESS',
      'JAX_PLATFORMS',
      'NCCL_DEBUG',
      'XLA_FLAGS',
      'SLURM_NTASKS_PER_NODE',
  ]

  tensorboard = None
  if _TENSORBOARD.value or _TENSORBOARD_GCS_PATH.value:
    tensorboard = xm_local.TensorboardCapability(
        name=_TENSORBOARD.value or '',
        base_output_directory=_TENSORBOARD_GCS_PATH.value,
    )

  executor = xm_local.VertexTrainingCluster(
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
      container_mounts=[f'{_WORK_DIR.value}:/mnt/jobs'],
      container_env_passthrough=container_env_passthrough,
      setup_jax_coordinator=True,
      env_vars=env_vars,
      stream_output=_STREAM_OUTPUT.value,
      tensorboard=tensorboard,
      tensorboard_region=_TENSORBOARD_REGION.value,
      tensorboard_project=_TENSORBOARD_PROJECT.value,
  )

  config = executor.get_cluster_config()
  print(f'\nCluster: {config.cluster_type} ({config.gpu_type})')
  print(f'Nodes: {_NODES.value} ({_NODES.value * config.gpus_per_node} GPUs)')
  print(f'GCS: gs://{_GCS_BUCKET.value}/')
  print(f'Vizier: {_NUM_TRIALS.value} trials, {_NUM_PARALLEL.value} parallel, metric={_METRIC.value}')
  if tensorboard:
    print(f'TensorBoard: {_TENSORBOARD.value} ({_TENSORBOARD_REGION.value})')

  if _DRY_RUN.value:
    print('[DRY RUN] Config validated, not submitting')
    return

  maxtext_args = [
      f'steps={_STEPS.value}',
      f'base_output_directory=gs://{_GCS_BUCKET.value}',
      'tensorboard_dir=${AIP_TENSORBOARD_LOG_DIR}',
      'run_name=job-${SLURM_JOB_ID}',
      'packing=False',
  ]
  training_args = [
      'src/MaxText/train.py',
      _CONFIG_FILE.value,
  ] + [xm.ShellSafeArg(a) for a in maxtext_args]

  study_spec = aip.StudySpec(
      algorithm=aip.StudySpec.Algorithm.ALGORITHM_UNSPECIFIED,
      parameters=[
          aip.StudySpec.ParameterSpec(
              parameter_id='learning_rate',
              double_value_spec=aip.StudySpec.ParameterSpec.DoubleValueSpec(
                  min_value=1e-5, max_value=1e-3,
              ),
              scale_type=aip.StudySpec.ParameterSpec.ScaleType.UNIT_LOG_SCALE,
          ),
          aip.StudySpec.ParameterSpec(
              parameter_id='per_device_batch_size',
              discrete_value_spec=aip.StudySpec.ParameterSpec.DiscreteValueSpec(
                  values=[1, 2, 4],
              ),
          ),
      ],
      metrics=[
          aip.StudySpec.MetricSpec(
              metric_id=_METRIC.value,
              goal=aip.StudySpec.MetricSpec.GoalType.MINIMIZE,
          ),
          aip.StudySpec.MetricSpec(
              metric_id='perf/per_device_tflops_per_sec',
              goal=aip.StudySpec.MetricSpec.GoalType.MAXIMIZE,
          ),
      ],
  )

  with xm_local.create_experiment(
      experiment_title=f'maxtext_vizier_{timestamp}'
  ) as experiment:

    job = xm.Job(
        executable=xm.Binary(path='python'),
        args=training_args,
        executor=executor,
    )

    exploration = vizier_cloud.VizierExploration(
        experiment=experiment,
        job=job,
        study_factory=vizier_cloud.NewStudy(
            study_config=study_spec,
            project=_PROJECT.value,
            location=_REGION.value,
        ),
        num_trials_total=_NUM_TRIALS.value,
        num_parallel_trial_runs=_NUM_PARALLEL.value,
        metric_names=[_METRIC.value, 'perf/per_device_tflops_per_sec'],
        gcs_log_base=f'gs://{_GCS_BUCKET.value}',
    )

    print('Starting Vizier study...')
    print('Monitor jobs: squeue --me')

    exploration.launch(poll_frequency_in_sec=60)

    print(f'\nCompleted! Experiment ID: {experiment.experiment_id}')


if __name__ == '__main__':
  app.run(main)
