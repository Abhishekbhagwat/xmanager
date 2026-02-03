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

r"""Simplified XManager launcher for MaxText + Vizier on VTC.

Tests VTC + Vizier integration with minimal configuration.

Usage:
  xmanager launch examples/maxtext_vtc/launcher.py -- \
    --login_node=vmdsa3u04-login-001 \
    --use_gcloud_ssh \
    --ssh_hostname=nic0.vmdsa3u04-login-001.europe-west4-a.c.ai-infra-recipe-validation.internal.gcpnode.com \
    --gcs_bucket=ai-infra-gcs-europe-west4 \
    --cluster_type=hcc-a3u \
    --partition=a3u \
    --nodes=2 \
    --num_trials=2 \
    --steps=10

TensorBoard logs are written to:
  gs://{bucket}/job-{slurm_job_id}/tensorboard/job-{slurm_job_id}/
"""

import time

from absl import app
from absl import flags
from google.cloud import aiplatform_v1beta1 as aip

from xmanager import xm
from xmanager import xm_local
from xmanager.vizier import vizier_cloud

# Required flags
_LOGIN_NODE = flags.DEFINE_string('login_node', None, 'SSH login node (required)')
_GCS_BUCKET = flags.DEFINE_string('gcs_bucket', None, 'GCS bucket name (required)')

# Cluster config (with sensible defaults)
_CLUSTER_TYPE = flags.DEFINE_string('cluster_type', 'hcc-a3u', 'Cluster type')
_PARTITION = flags.DEFINE_string('partition', 'a3u', 'Slurm partition')
_NODES = flags.DEFINE_integer('nodes', 2, 'Number of nodes')

# Paths (with defaults)
_WORK_DIR = flags.DEFINE_string(
    'work_dir', '/home/abhishekbhgwt_google_com/vertexai-mds/nemo', 'Work dir on cluster')
_CONTAINER_IMAGE = flags.DEFINE_string(
    'container_image', '/mnt/lustre/ls1-europe-west4-a/images/jax-maxtext-2025-10-01.sqsh',
    'Container image path')
_CONFIG_FILE = flags.DEFINE_string(
    'config_file', '/mnt/jobs/config.yaml', 'MaxText config file')

# SSH options
_USE_GCLOUD_SSH = flags.DEFINE_bool('use_gcloud_ssh', True, 'Use gcloud compute ssh')
_SSH_HOSTNAME = flags.DEFINE_string('ssh_hostname', None, 'SSH hostname override')

# Training/Vizier config
_STEPS = flags.DEFINE_integer('steps', 10, 'Training steps per trial')
_NUM_TRIALS = flags.DEFINE_integer('num_trials', 2, 'Number of Vizier trials')
_NUM_PARALLEL = flags.DEFINE_integer('num_parallel', 1, 'Parallel trials')
_METRIC = flags.DEFINE_string('metric', 'learning/loss', 'Metric to optimize')

# Vizier project/region
_PROJECT = flags.DEFINE_string('project', 'ai-infra-recipe-validation', 'GCP project')
_REGION = flags.DEFINE_string('region', 'europe-west4', 'GCP region')

# Debug
_DRY_RUN = flags.DEFINE_bool('dry_run', False, 'Validate config without submitting')


def main(_):
  # Validate required flags
  missing = []
  if not _LOGIN_NODE.value:
    missing.append('--login_node')
  if not _GCS_BUCKET.value:
    missing.append('--gcs_bucket')
  if missing:
    raise app.UsageError(f'Missing required flags: {", ".join(missing)}')

  timestamp = time.strftime('%Y%m%d-%H%M%S')

  # JAX/XLA environment variables for distributed training
  env_vars = {
      # NCCL configuration
      'NCCL_SOCKET_IFNAME': 'enp0s19,enp192s20',
      'NCCL_DEBUG': 'VERSION',
      'CUDA_DEVICE_MAX_CONNECTIONS': '1',
      # JAX configuration
      'JAX_PLATFORMS': 'cuda',
      'JAX_REMOVE_CUSTOM_PARTITIONING_PTR_FROM_CACHE_KEY': 'true',
      'JAX_ENABLE_PGLE': 'false',
      'SLURM_NTASKS_PER_NODE': '8',  # Required for JAX multiprocess (8 GPUs/node)
      # XLA/TensorFlow configuration
      'TF_CPP_MIN_LOG_LEVEL': '0',
      'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.98',
      'NVTE_FUSED_ATTN': '1',
      # XLA optimization flags
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

  # Env vars to pass into container
  container_env_passthrough = [
      'JAX_COORDINATOR_ADDRESS',
      'JAX_PLATFORMS',
      'NCCL_DEBUG',
      'XLA_FLAGS',
      'SLURM_NTASKS_PER_NODE',
  ]

  # Create executor
  executor = xm_local.VertexTrainingCluster(
      cluster_type=_CLUSTER_TYPE.value,
      partition=_PARTITION.value,
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
      stream_output=True,
      tensorboard=xm_local.TensorboardCapability(
          name='',
          base_output_directory=_GCS_BUCKET.value,
      ),
  )

  # Print config
  config = executor.get_cluster_config()
  print(f'\n{"="*60}')
  print(f'MaxText + Vizier Integration Test')
  print(f'{"="*60}')
  print(f'Cluster: {config.cluster_type} ({config.gpu_type})')
  print(f'Nodes: {_NODES.value} ({_NODES.value * config.gpus_per_node} GPUs)')
  print(f'GCS: gs://{_GCS_BUCKET.value}/')
  print(f'Vizier: {_NUM_TRIALS.value} trials, metric={_METRIC.value}')
  print(f'{"="*60}\n')

  if _DRY_RUN.value:
    print('[DRY RUN] Config validated, not submitting')
    return

  # Build training args
  # TensorBoard path pattern: gs://{bucket}/job-{slurm_job_id}/tensorboard/job-{slurm_job_id}/
  # This matches the default MaxText pattern in VizierExploration
  training_args = [
      'src/MaxText/train.py',
      _CONFIG_FILE.value,
      xm.ShellSafeArg(f'steps={_STEPS.value}'),
      xm.ShellSafeArg(f'base_output_directory=gs://{_GCS_BUCKET.value}'),
      xm.ShellSafeArg(f'tensorboard_dir=gs://{_GCS_BUCKET.value}/job-${{SLURM_JOB_ID}}/tensorboard/job-${{SLURM_JOB_ID}}'),
      xm.ShellSafeArg('run_name=job-${SLURM_JOB_ID}'),
      xm.ShellSafeArg('packing=False'),
  ]

  # Vizier study spec
  study_spec = aip.StudySpec(
      algorithm=aip.StudySpec.Algorithm.ALGORITHM_UNSPECIFIED,
      parameters=[
          aip.StudySpec.ParameterSpec(
              parameter_id='learning_rate',
              double_value_spec=aip.StudySpec.ParameterSpec.DoubleValueSpec(
                  min_value=1e-3, max_value=1e-1,
              ),
              scale_type=aip.StudySpec.ParameterSpec.ScaleType.UNIT_LOG_SCALE,
          ),
      ],
      metrics=[
          aip.StudySpec.MetricSpec(
              metric_id=_METRIC.value,
              goal=aip.StudySpec.MetricSpec.GoalType.MINIMIZE,
          )
      ],
  )

  # Run experiment
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
        metric_name=_METRIC.value,
        gcs_log_base=f'gs://{_GCS_BUCKET.value}',
        # Uses default MaxText pattern: {gcs_log_base}/job-{slurm_job_id}/tensorboard/job-{slurm_job_id}/
    )

    print('Starting Vizier study...')
    print(f'Monitor jobs: squeue --me')
    print(f'Vizier console: https://console.cloud.google.com/vertex-ai/experiments')

    exploration.launch(poll_frequency_in_sec=60)

    print(f'\nCompleted! Experiment ID: {experiment.experiment_id}')


if __name__ == '__main__':
  app.run(main)
