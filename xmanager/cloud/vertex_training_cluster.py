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
"""Vertex Training Cluster executor with full Slurm integration.

This module provides:
  1. Cluster configurations (NCCL, networking) for different GPU types
  2. Slurm job submission, monitoring, and cancellation
  3. Three modes of job execution:
     - sbatch_script: Use a raw sbatch script provided by the user
     - sbatch_template: Use a Jinja2 template with variable substitution
     - launcher: Auto-generate sbatch script from launcher configurations

Supported cluster types:
  - hcc-a3m: H100 GPUs with TCPXO networking
  - hcc-a3u: H200 GPUs with gIB networking
  - hcc-a4: B200 GPUs with gIB networking
  - hcc-a3h: H100 GPUs with gIB networking
"""

import asyncio
import logging
import os
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional

import attr
from xmanager import xm
from xmanager.cloud import vertex
from xmanager.xm import utils
from xmanager.xm_local import executors as local_executors
from xmanager.xm_local import handles
from xmanager.xm_local import registry
from xmanager.xm_local import status as local_status
from xmanager.xm_local.storage import database

# Mapping from Slurm states to XManager status
_SLURM_STATE_TO_STATUS = {
    'PENDING': local_status.LocalWorkUnitStatusEnum.PENDING,
    'CONFIGURING': local_status.LocalWorkUnitStatusEnum.PENDING,
    'RUNNING': local_status.LocalWorkUnitStatusEnum.RUNNING,
    'COMPLETING': local_status.LocalWorkUnitStatusEnum.RUNNING,
    'COMPLETED': local_status.LocalWorkUnitStatusEnum.COMPLETED,
    'FAILED': local_status.LocalWorkUnitStatusEnum.FAILED,
    'TIMEOUT': local_status.LocalWorkUnitStatusEnum.FAILED,
    'OUT_OF_MEMORY': local_status.LocalWorkUnitStatusEnum.FAILED,
    'NODE_FAIL': local_status.LocalWorkUnitStatusEnum.FAILED,
    'CANCELLED': local_status.LocalWorkUnitStatusEnum.CANCELLED,
    'PREEMPTED': local_status.LocalWorkUnitStatusEnum.CANCELLED,
    'SUSPENDED': local_status.LocalWorkUnitStatusEnum.CANCELLED,
}


# =============================================================================
# Cluster Configurations
# =============================================================================

@attr.s(auto_attribs=True)
class ClusterConfig:
  """Configuration for a specific Vertex Training Cluster type."""

  cluster_type: str
  gpu_type: str  # H100, H200, B200
  gpus_per_node: int = 8

  # NCCL directory on the host (mounted into container)
  nccl_dir: str = ""
  nccl_lib_subdir: str = "lib64"

  # Setup script to source for NCCL environment (if applicable)
  # Some clusters use a script, others use explicit env vars
  setup_script: Optional[str] = None

  # Explicit NCCL environment variables (used when no setup_script)
  nccl_env_vars: Dict[str, str] = attr.Factory(dict)

  # Default srun arguments for container execution
  container_srun_args: List[str] = attr.Factory(lambda: [
      "--container-writable",
      "--no-container-mount-home",
      "--mpi=pmix",
  ])

  def get_nccl_lib_path(self) -> str:
    """Returns the full path to NCCL library directory."""
    return os.path.join(self.nccl_dir, self.nccl_lib_subdir)

  def get_setup_commands(self) -> Optional[str]:
    """Returns shell commands to setup NCCL environment."""
    if self.setup_script:
      nccl_lib = self.get_nccl_lib_path()
      return (
          f"source {self.setup_script}; "
          f"export LD_LIBRARY_PATH={nccl_lib}:$LD_LIBRARY_PATH"
      )
    return None


# Predefined cluster configurations
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
        # hcc-a4 uses explicit env vars instead of setup script
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
  """Get cluster configuration by type.

  Args:
    cluster_type: One of 'hcc-a3m', 'hcc-a3u', 'hcc-a4', 'hcc-a3h'

  Returns:
    ClusterConfig for the specified cluster type.

  Raises:
    ValueError: If cluster_type is not recognized.
  """
  if cluster_type not in CLUSTER_CONFIGS:
    raise ValueError(
        f"Unknown cluster_type: {cluster_type}. "
        f"Supported types: {list(CLUSTER_CONFIGS.keys())}"
    )
  return CLUSTER_CONFIGS[cluster_type]


# =============================================================================
# SSH Connection and Slurm Client
# =============================================================================

# SSH flags to handle ephemeral cluster nodes
SSH_OPTIONS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
]


class Client:
  """Client for running commands and Slurm operations on Vertex Training Cluster."""

  def __init__(self, executor: local_executors.VertexTrainingCluster):
    self.executor = executor
    self.cluster_config = get_cluster_config(executor.cluster_type)

  def _build_ssh_prefix(self) -> List[str]:
    """Build SSH command prefix for remote execution."""
    if self.executor.use_gcloud_ssh and self.executor.login_node:
      # gcloud compute ssh login-node -- -T -o Hostname=...
      # -T disables pseudo-terminal allocation for non-interactive commands
      prefix = ['gcloud', 'compute', 'ssh', self.executor.login_node, '--', '-T']
      if self.executor.ssh_hostname:
        prefix.extend(['-o', f'Hostname={self.executor.ssh_hostname}'])
      return prefix
    elif self.executor.login_node:
      return ['ssh', '-T'] + SSH_OPTIONS + [self.executor.login_node]
    else:
      return []

  def run_command(self, cmd: str) -> subprocess.CompletedProcess:
    """Run command on the cluster (locally or via SSH).

    Args:
      cmd: Shell command to execute

    Returns:
      CompletedProcess with stdout, stderr, returncode
    """
    ssh_prefix = self._build_ssh_prefix()
    if ssh_prefix:
      full_cmd = ssh_prefix + [cmd]
      return subprocess.run(full_cmd, capture_output=True, text=True)
    else:
      return subprocess.run(cmd, shell=True, capture_output=True, text=True)

  async def run_command_async(self, cmd: str) -> asyncio.subprocess.Process:
    """Run command asynchronously for streaming output.

    Args:
      cmd: Shell command to execute

    Returns:
      Async process with stdout/stderr streams
    """
    ssh_prefix = self._build_ssh_prefix()
    if ssh_prefix:
      full_cmd = ssh_prefix + [cmd]
      return await asyncio.create_subprocess_exec(
          *full_cmd,
          stdout=asyncio.subprocess.PIPE,
          stderr=asyncio.subprocess.STDOUT,
      )
    else:
      return await asyncio.create_subprocess_shell(
          cmd,
          stdout=asyncio.subprocess.PIPE,
          stderr=asyncio.subprocess.STDOUT,
      )

  def submit_sbatch(self, script_content: str, work_dir: str) -> str:
    """Submit a batch job via sbatch and return the job ID.

    Args:
      script_content: Complete sbatch script content
      work_dir: Working directory on cluster to write script

    Returns:
      Slurm job ID (e.g., "12345")

    Raises:
      RuntimeError: If sbatch submission fails
    """
    import base64

    # Create a temporary script name
    script_name = f"xmanager_job_{os.getpid()}.sh"
    script_path = os.path.join(work_dir, script_name)

    # Encode script content as base64 for safe transfer over SSH
    encoded_content = base64.b64encode(script_content.encode()).decode()

    # Write script to cluster via base64 decode
    write_cmd = f"echo '{encoded_content}' | base64 -d > {script_path} && chmod +x {script_path}"

    result = self.run_command(write_cmd)
    if result.returncode != 0:
      raise RuntimeError(
          f"Failed to write sbatch script: {result.stderr}"
      )

    # Submit the job
    submit_cmd = f"cd {work_dir} && sbatch {script_path}"
    result = self.run_command(submit_cmd)

    if result.returncode != 0:
      raise RuntimeError(
          f"sbatch submission failed: {result.stderr}"
      )

    # Parse job ID from "Submitted batch job 12345"
    match = re.search(r'Submitted batch job (\d+)', result.stdout)
    if not match:
      raise RuntimeError(
          f"Could not parse job ID from sbatch output: {result.stdout}"
      )

    return match.group(1)

  def get_job_status(self, job_id: str) -> str:
    """Get the status of a Slurm job.

    Args:
      job_id: Slurm job ID

    Returns:
      One of: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, UNKNOWN
    """
    # First try squeue (for active jobs)
    result = self.run_command(f"squeue -j {job_id} -h -o '%T'")
    if result.returncode == 0 and result.stdout.strip():
      status = result.stdout.strip()
      # Normalize Slurm states to our simplified states
      if status in ['PENDING', 'CONFIGURING']:
        return 'PENDING'
      elif status in ['RUNNING', 'COMPLETING']:
        return 'RUNNING'
      elif status in ['COMPLETED']:
        return 'COMPLETED'
      elif status in ['FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL']:
        return 'FAILED'
      elif status in ['CANCELLED', 'PREEMPTED', 'SUSPENDED']:
        return 'CANCELLED'
      else:
        return status

    # Fall back to sacct (for completed jobs)
    result = self.run_command(f"sacct -j {job_id} -n -o State -X")
    if result.returncode == 0 and result.stdout.strip():
      status = result.stdout.strip().split()[0]  # First word
      # Normalize sacct states
      if status in ['PENDING', 'CONFIGURING']:
        return 'PENDING'
      elif status in ['RUNNING', 'COMPLETING']:
        return 'RUNNING'
      elif status in ['COMPLETED']:
        return 'COMPLETED'
      elif status in ['FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL']:
        return 'FAILED'
      elif status in ['CANCELLED', 'PREEMPTED', 'SUSPENDED']:
        return 'CANCELLED'
      else:
        return 'FAILED'  # Treat unknown sacct states as failed

    return 'UNKNOWN'

  def cancel_job(self, job_id: str) -> None:
    """Cancel a Slurm job.

    Args:
      job_id: Slurm job ID to cancel
    """
    result = self.run_command(f"scancel {job_id}")
    if result.returncode != 0:
      # Log but don't raise - job might already be done
      logging.warning("scancel %s failed: %s", job_id, result.stderr)

  def get_job_log_path(self, job_id: str, work_dir: str) -> str:
    """Get the path to the job's log file.

    Args:
      job_id: Slurm job ID
      work_dir: Working directory where logs are stored

    Returns:
      Absolute path to the log file
    """
    # Default log path pattern used by Slurm
    return os.path.join(work_dir, f"slurm-{job_id}.out")

  def rsync_files(self, local_paths: List[str], remote_dir: str) -> None:
    """Rsync local files/directories to the cluster.

    Args:
      local_paths: List of local file or directory paths to sync
      remote_dir: Remote directory to sync files into

    Raises:
      RuntimeError: If rsync/scp fails
    """
    if not local_paths:
      return

    # Ensure remote directory exists
    mkdir_result = self.run_command(f"mkdir -p {remote_dir}")
    if mkdir_result.returncode != 0:
      raise RuntimeError(f"Failed to create remote directory {remote_dir}: {mkdir_result.stderr}")

    for local_path in local_paths:
      # Expand user home directory
      local_path = os.path.expanduser(local_path)

      if not os.path.exists(local_path):
        logging.warning("Local path %s does not exist, skipping", local_path)
        continue

      logging.info("Syncing %s to %s...", local_path, remote_dir)

      if self.executor.use_gcloud_ssh and self.executor.login_node:
        # Use gcloud compute ssh with cat to transfer files
        # This is needed because gcloud compute scp doesn't support hostname override
        if self.executor.ssh_hostname:
          # Build ssh command with hostname override
          remote_path = os.path.join(remote_dir, os.path.basename(local_path))
          ssh_cmd = ['gcloud', 'compute', 'ssh', self.executor.login_node,
                     '--', '-T',
                     '-o', f'Hostname={self.executor.ssh_hostname}',
                     '-o', 'StrictHostKeyChecking=no',
                     '-o', 'UserKnownHostsFile=/dev/null',
                     f'cat > {remote_path}']
          with open(local_path, 'rb') as f:
            result = subprocess.run(ssh_cmd, stdin=f, capture_output=True, text=True)
        else:
          # No hostname override needed, use standard gcloud scp
          cmd = ['gcloud', 'compute', 'scp', '--recurse',
                 local_path, f'{self.executor.login_node}:{remote_dir}/']
          result = subprocess.run(cmd, capture_output=True, text=True)
      elif self.executor.login_node:
        # Use rsync over SSH for regular SSH connections
        ssh_cmd = 'ssh ' + ' '.join(SSH_OPTIONS)
        cmd = ['rsync', '-avz', '--progress', '-e', ssh_cmd,
               local_path, f'{self.executor.login_node}:{remote_dir}/']
        result = subprocess.run(cmd, capture_output=True, text=True)
      else:
        # Local execution - just copy
        try:
          dest = os.path.join(remote_dir, os.path.basename(local_path))
          if os.path.isdir(local_path):
            shutil.copytree(local_path, dest, dirs_exist_ok=True)
          else:
            shutil.copy2(local_path, dest)
          continue  # Success, no subprocess result to check
        except Exception as e:
          raise RuntimeError(f"Failed to copy {local_path}: {e}")

      if result.returncode != 0:
        raise RuntimeError(
            f"Failed to sync {local_path} to {remote_dir}: {result.stderr}"
        )

    logging.info("Successfully synced %d path(s) to %s", len(local_paths), remote_dir)


# =============================================================================
# Sbatch Script Generation
# =============================================================================


def _build_command_from_job(job: xm.Job) -> str:
  """Build command string from job executable and args.

  Args:
    job: XManager job with executable and args

  Returns:
    Command string to execute
  """
  executable_args = getattr(job.executable, 'args', {})
  args = xm.merge_args(executable_args, job.args).to_list(utils.ARG_ESCAPER)
  entrypoint = getattr(job.executable, 'entrypoint', None)
  path = getattr(job.executable, 'path', None)

  if entrypoint:
    return f"{entrypoint} {' '.join(args)}".strip()
  elif path:
    return f"{path} {' '.join(args)}".strip()
  return ' '.join(args)


def _build_container_mounts(
    executor: local_executors.VertexTrainingCluster,
    cluster_config: ClusterConfig,
) -> str:
  """Build deduplicated container mounts string.

  Args:
    executor: VertexTrainingCluster executor
    cluster_config: Cluster configuration

  Returns:
    Comma-separated mount paths string
  """
  container_mounts_list = [cluster_config.nccl_dir] + list(executor.container_mounts)
  seen = set()
  unique_mounts = []
  for m in container_mounts_list:
    if m and m not in seen:
      seen.add(m)
      unique_mounts.append(m)
  return ','.join(unique_mounts)


def _build_sbatch_flags(
    executor: local_executors.VertexTrainingCluster,
) -> Dict[str, str]:
  """Build sbatch flags with TensorBoard integration if configured.

  Args:
    executor: VertexTrainingCluster executor

  Returns:
    Dictionary of sbatch flags
  """
  sbatch_flags = dict(executor.sbatch_flags)

  if executor.tensorboard:
    extra_parts = []
    # Add tensorboard_base_output_dir (GCS bucket path, e.g., bucket-name/path)
    # VMDS expects bucket/path format, not /gcs/... or gs://...
    if executor.tensorboard.base_output_directory:
      gcs_path = executor.tensorboard.base_output_directory
      # Strip /gcs/ prefix if present (mounted GCS path)
      if gcs_path.startswith('/gcs/'):
        gcs_path = gcs_path[5:]
      # Strip gs:// prefix if present
      elif gcs_path.startswith('gs://'):
        gcs_path = gcs_path[5:]
      extra_parts.append(f'tensorboard_base_output_dir={gcs_path}')
    # Add tensorboard_url (the Vertex AI TensorBoard instance URL)
    if executor.tensorboard.name:
      extra_parts.append(f'tensorboard_url={executor.tensorboard.name}')
    if extra_parts:
      sbatch_flags['extra'] = ','.join(extra_parts)

  return sbatch_flags


def _prepare_template_vars(
    executor: local_executors.VertexTrainingCluster,
    job: xm.Job,
    job_name: str,
    cluster_config: ClusterConfig,
) -> Dict[str, Any]:
  """Prepare template variables for sbatch script generation.

  Args:
    executor: VertexTrainingCluster executor with configuration
    job: XManager job with executable and args
    job_name: Name for the job
    cluster_config: Cluster-specific configuration

  Returns:
    Dictionary of template variables
  """
  num_nodes = executor.requirements.replicas or 1
  command = _build_command_from_job(job)
  container_mounts_str = _build_container_mounts(executor, cluster_config)
  sbatch_flags = _build_sbatch_flags(executor)

  template_vars = {
      'job_name': job_name,
      'num_nodes': num_nodes,
      'gpus_per_node': cluster_config.gpus_per_node,
      'partition': executor.partition,
      'account': executor.account,
      'time_limit': executor.time_limit,
      'exclusive': executor.exclusive,
      'command': command,
      'working_dir': executor.work_dir,
      'env_vars': executor.env_vars,
      'sbatch_flags': sbatch_flags,
      'prologue_commands': executor.prologue_commands,
      'epilogue_commands': executor.epilogue_commands,
      # NCCL configuration
      'nccl_setup_script': cluster_config.setup_script,
      'nccl_lib_path': cluster_config.get_nccl_lib_path(),
      'nccl_env_vars': cluster_config.nccl_env_vars,
      # Container configuration
      'container_image': executor.container_image,
      'container_mounts': container_mounts_str,
      'use_mpi': executor.use_mpi,
      'container_env_passthrough': executor.container_env_passthrough,
      # Distributed training
      'master_port': executor.master_port,
      'setup_jax_coordinator': executor.setup_jax_coordinator,
  }

  # Add output/error file paths if log_dir is set
  if executor.log_dir:
    template_vars['output_file'] = f"{executor.log_dir}/slurm-%j.out"
    template_vars['error_file'] = f"{executor.log_dir}/slurm-%j.err"

  return template_vars


def _generate_sbatch_script(
    executor: local_executors.VertexTrainingCluster,
    job: xm.Job,
    job_name: str,
    cluster_config: ClusterConfig,
) -> str:
  """Generate sbatch script content from executor configuration.

  Supports three modes:
    1. sbatch_script: Use raw script provided by user
    2. sbatch_template: Render Jinja2 template with variables
    3. launcher: Auto-generate script from launcher configuration

  Args:
    executor: VertexTrainingCluster executor with configuration
    job: XManager job with executable and args
    job_name: Name for the job
    cluster_config: Cluster-specific configuration

  Returns:
    Complete sbatch script content as string

  Raises:
    ValueError: If invalid configuration or missing required fields
  """
  # Mode 1: Raw sbatch script
  if executor.sbatch_script:
    return executor.sbatch_script

  # Import sbatch_templates (needed for both template and default modes)
  try:
    from xmanager.cloud import sbatch_templates
  except ImportError:
    raise ImportError(
        "sbatch script generation requires xmanager.cloud.sbatch_templates module"
    )

  # Prepare template variables (common for both modes)
  template_vars = _prepare_template_vars(executor, job, job_name, cluster_config)

  # Mode 2: Custom Jinja2 template
  if executor.sbatch_template:
    renderer = sbatch_templates.SbatchTemplateRenderer()
    return renderer.render(**template_vars)

  # Mode 3: Default template
  renderer = sbatch_templates.SbatchTemplateRenderer()
  return renderer.render(**template_vars)


# =============================================================================
# Execution Handle
# =============================================================================

@attr.s(auto_attribs=True)
class VertexTrainingClusterHandle(handles.ExecutionHandle):
  """Handle for tracking jobs on Vertex Training Cluster via Slurm."""

  job_name: str
  slurm_job_id: str
  client: Client
  executor: local_executors.VertexTrainingCluster
  _monitor_task: Optional[asyncio.Task] = None

  async def wait(self) -> None:
    """Wait for the job to complete by polling Slurm status."""
    poll_interval = 5  # seconds

    while True:
      status = self.client.get_job_status(self.slurm_job_id)

      if status in ['COMPLETED', 'FAILED', 'CANCELLED', 'UNKNOWN']:
        break

      await asyncio.sleep(poll_interval)

  def stop(self) -> None:
    """Stop the job by cancelling it in Slurm."""
    self.client.cancel_job(self.slurm_job_id)

    # Cancel monitor task if running
    if self._monitor_task and not self._monitor_task.done():
      self._monitor_task.cancel()

  def get_status(self) -> local_status.LocalWorkUnitStatus:
    """Get job status from Slurm.

    Returns:
      LocalWorkUnitStatus with current state
    """
    slurm_status = self.client.get_job_status(self.slurm_job_id)
    status = _SLURM_STATE_TO_STATUS.get(
        slurm_status, local_status.LocalWorkUnitStatusEnum.UNKNOWN
    )
    return local_status.LocalWorkUnitStatus(
        status, message=f"Slurm job {self.slurm_job_id}: {slurm_status}"
    )

  def save_to_storage(self, experiment_id: int, work_unit_id: int) -> None:
    """Save job info to database.

    Args:
      experiment_id: ID of the experiment
      work_unit_id: ID of the work unit
    """
    database.database().insert_vertex_training_cluster_job(
        experiment_id=experiment_id,
        work_unit_id=work_unit_id,
        job_name=self.job_name,
        slurm_job_id=self.slurm_job_id,
        login_node=self.executor.login_node or "",
        cluster_type=self.executor.cluster_type,
        partition=self.executor.partition or "",
        use_gcloud_ssh=self.executor.use_gcloud_ssh,
        ssh_hostname=self.executor.ssh_hostname or "",
        work_dir=self.executor.work_dir or "",
    )

  async def monitor(self) -> None:
    """Stream job output to console by tailing the log file.

    This monitors the Slurm output file and prints new lines as they appear.
    """
    if not self.executor.stream_output:
      return

    work_dir = self.executor.work_dir or os.getcwd()
    log_path = self.client.get_job_log_path(self.slurm_job_id, work_dir)

    # Wait a bit for the log file to be created
    await asyncio.sleep(2)

    # Tail the log file
    tail_cmd = f"tail -f {log_path}"

    try:
      process = await self.client.run_command_async(tail_cmd)

      while True:
        line = await process.stdout.readline()
        if not line:
          break
        logging.info("[%s] %s", self.job_name, line.decode().strip())

        # Check if job has completed
        status = self.client.get_job_status(self.slurm_job_id)
        if status in ['COMPLETED', 'FAILED', 'CANCELLED', 'UNKNOWN']:
          # Read any remaining output
          remaining = await process.stdout.read()
          if remaining:
            for line in remaining.decode().split('\n'):
              if line.strip():
                logging.info("[%s] %s", self.job_name, line.strip())
          break

    except asyncio.CancelledError:
      # Monitor was cancelled
      pass
    except Exception as e:
      logging.warning("Error monitoring job %s: %s", self.job_name, e)


# =============================================================================
# Launch Logic
# =============================================================================


def _vertex_training_cluster_job_predicate(job: xm.Job) -> bool:
  """Filter for Vertex Training Cluster jobs."""
  return isinstance(job.executor, local_executors.VertexTrainingCluster)


async def _resolve_tensorboard(
    executor: local_executors.VertexTrainingCluster,
) -> None:
  """Resolve TensorBoard instance if configured with display name.

  Updates executor.tensorboard.name to full resource URL if needed.

  Args:
    executor: VertexTrainingCluster executor with tensorboard config
  """
  if not executor.tensorboard:
    return

  tb_name = executor.tensorboard.name
  # If name doesn't look like a full resource URL, resolve it
  if tb_name and not tb_name.startswith('projects/'):
    try:
      vertex_client = vertex.Client(
          project=executor.tensorboard_project,
          location=executor.tensorboard_region,
      )
      full_tb_name = await vertex_client.get_or_create_tensorboard(tb_name)
      # Update the tensorboard name to full resource URL
      executor.tensorboard = local_executors.TensorboardCapability(
          name=full_tb_name,
          base_output_directory=executor.tensorboard.base_output_directory,
      )
      logging.info("Using TensorBoard: %s", full_tb_name)
    except Exception as e:
      logging.warning("Failed to resolve TensorBoard '%s': %s", tb_name, e)


def launch(
    experiment_id: int,
    experiment_unit_name: str,
    job_group: xm.JobGroup,
) -> List[VertexTrainingClusterHandle]:
  """Launch jobs on Vertex Training Cluster via Slurm.

  This function:
    1. Generates sbatch scripts from executor configuration
    2. Submits jobs to Slurm via sbatch
    3. Creates handles to track job status

  Args:
    experiment_id: ID of the experiment
    experiment_unit_name: Name of the experiment unit
    job_group: Job group containing jobs to launch

  Returns:
    List of execution handles for submitted jobs
  """
  jobs = xm.job_operators.collect_jobs_by_filter(
      job_group, _vertex_training_cluster_job_predicate
  )

  if not jobs:
    return []

  handles_list = []

  for job in jobs:
    executor = job.executor
    client = Client(executor)
    cluster_config = client.cluster_config

    # Generate job name
    job_name = f"exp{experiment_id}_{experiment_unit_name}"

    # Ensure work directory exists
    work_dir = executor.work_dir or os.getcwd()

    # Rsync local files to cluster if specified
    if executor.local_files:
      try:
        client.rsync_files(executor.local_files, work_dir)
      except Exception as e:
        raise RuntimeError(
            f"Failed to sync local files for job {job_name}: {e}"
        )

    # Generate sbatch script
    try:
      script_content = _generate_sbatch_script(
          executor, job, job_name, cluster_config
      )
    except Exception as e:
      raise RuntimeError(
          f"Failed to generate sbatch script for job {job_name}: {e}"
      )

    # Submit to Slurm
    try:
      slurm_job_id = client.submit_sbatch(script_content, work_dir)
      logging.info("Submitted Slurm job %s for %s", slurm_job_id, job_name)
    except Exception as e:
      raise RuntimeError(
          f"Failed to submit Slurm job for {job_name}: {e}"
      )

    # Create handle
    handle = VertexTrainingClusterHandle(
        job_name=job_name,
        slurm_job_id=slurm_job_id,
        client=client,
        executor=executor,
    )

    handles_list.append(handle)

  return handles_list


async def _async_launch(
    local_experiment_unit: Any, job_group: xm.JobGroup
) -> List[VertexTrainingClusterHandle]:
  """Async wrapper for launch that handles TensorBoard resolution and monitoring.

  Args:
    local_experiment_unit: Experiment unit with metadata
    job_group: Job group containing jobs to launch

  Returns:
    List of execution handles for submitted jobs
  """
  # Resolve TensorBoard for all VTC executors before launch
  jobs = xm.job_operators.collect_jobs_by_filter(
      job_group, _vertex_training_cluster_job_predicate
  )
  for job in jobs:
    await _resolve_tensorboard(job.executor)

  experiment_unit_name = (
      local_experiment_unit.experiment_unit_name
      or f"wu_{local_experiment_unit.work_unit_id}"
  )

  handles_list = launch(
      local_experiment_unit.experiment_id,
      experiment_unit_name,
      job_group,
  )

  # Start monitoring for handles that have stream_output enabled
  for handle in handles_list:
    if handle.executor.stream_output:
      monitor_task = asyncio.create_task(handle.monitor())
      handle._monitor_task = monitor_task

  return handles_list


def _create_vtc_handle(data) -> VertexTrainingClusterHandle:
  """Create a VTC handle from database record.

  Args:
    data: Database record with job information

  Returns:
    Restored execution handle
  """
  executor = local_executors.VertexTrainingCluster(
      cluster_type=data.vertex_training_cluster.cluster_type,
      login_node=data.vertex_training_cluster.login_node or None,
      partition=data.vertex_training_cluster.partition or None,
      use_gcloud_ssh=data.vertex_training_cluster.use_gcloud_ssh,
      ssh_hostname=data.vertex_training_cluster.ssh_hostname or None,
      work_dir=data.vertex_training_cluster.work_dir or None,
  )

  slurm_job_id = data.vertex_training_cluster.slurm_job_id or ""
  job_name = data.vertex_training_cluster.job_name or f"vtc_job_{slurm_job_id}"

  return VertexTrainingClusterHandle(
      job_name=job_name,
      slurm_job_id=slurm_job_id,
      client=Client(executor),
      executor=executor,
  )


def register():
  """Registers Vertex Training Cluster execution logic with XManager."""
  registry.register(
      local_executors.VertexTrainingCluster,
      launch=_async_launch,
      create_handle=lambda *args, data, **kwargs: _create_vtc_handle(data),
  )
