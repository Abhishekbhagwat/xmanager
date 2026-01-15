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
import os
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional

import attr
from xmanager import xm
from xmanager.xm import utils
from xmanager.xm_local import executors as local_executors
from xmanager.xm_local import handles
from xmanager.xm_local import registry
from xmanager.xm_local import status as local_status
from xmanager.xm_local.storage import database


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
      print(f"Warning: scancel {job_id} failed: {result.stderr}")

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
        print(f"Warning: Local path {local_path} does not exist, skipping")
        continue

      print(f"Syncing {local_path} to {remote_dir}...")

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

    print(f"Successfully synced {len(local_paths)} path(s) to {remote_dir}")


# =============================================================================
# Sbatch Script Generation
# =============================================================================

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
  if hasattr(executor, 'sbatch_script') and executor.sbatch_script:
    return executor.sbatch_script

  # Mode 2: Jinja2 template
  if hasattr(executor, 'sbatch_template') and executor.sbatch_template:
    try:
      from xmanager.cloud import sbatch_templates
    except ImportError:
      raise ImportError(
          "sbatch_template mode requires xmanager.cloud.sbatch_templates module"
      )

    # Prepare template variables
    num_nodes = getattr(executor.requirements, 'replicas', 1)
    gpus_per_node = cluster_config.gpus_per_node

    # Build command from job executable
    executable_args = getattr(job.executable, 'args', {})
    args = xm.merge_args(executable_args, job.args).to_list(utils.ARG_ESCAPER)
    entrypoint = getattr(job.executable, 'entrypoint', None)
    if entrypoint:
      command = f"{entrypoint} {' '.join(args)}".strip()
    else:
      command = ' '.join(args)

    # Build container mounts string (deduplicate to avoid redundant mounts)
    container_mounts_list = [cluster_config.nccl_dir] + list(getattr(executor, 'container_mounts', []))
    # Deduplicate while preserving order
    seen = set()
    unique_mounts = []
    for m in container_mounts_list:
      if m and m not in seen:
        seen.add(m)
        unique_mounts.append(m)
    container_mounts_str = ','.join(unique_mounts)

    template_vars = {
        'job_name': job_name,
        'num_nodes': num_nodes,
        'gpus_per_node': gpus_per_node,
        'partition': getattr(executor, 'partition', None),
        'account': getattr(executor, 'account', None),
        'time_limit': getattr(executor, 'time_limit', None),
        'exclusive': getattr(executor, 'exclusive', True),
        'command': command,
        'working_dir': executor.work_dir,
        'env_vars': getattr(executor, 'env_vars', {}),
        'sbatch_flags': getattr(executor, 'sbatch_flags', {}),
        'prologue_commands': getattr(executor, 'prologue_commands', []),
        'epilogue_commands': getattr(executor, 'epilogue_commands', []),
        # NCCL configuration
        'nccl_setup_script': cluster_config.setup_script,
        'nccl_lib_path': cluster_config.get_nccl_lib_path(),
        'nccl_env_vars': cluster_config.nccl_env_vars,
        # Container configuration
        'container_image': getattr(executor, 'container_image', None),
        'container_mounts': container_mounts_str,
        'use_mpi': getattr(executor, 'use_mpi', False),
        'container_env_passthrough': getattr(executor, 'container_env_passthrough', []),
        # Distributed training
        'master_port': getattr(executor, 'master_port', 29500),
        'setup_jax_coordinator': getattr(executor, 'setup_jax_coordinator', False),
    }

    # Add output/error file paths if log_dir is set
    log_dir = getattr(executor, 'log_dir', None)
    if log_dir:
        template_vars['output_file'] = f"{log_dir}/slurm-%j.out"
        template_vars['error_file'] = f"{log_dir}/slurm-%j.err"

    renderer = sbatch_templates.SbatchTemplateRenderer()
    return renderer.render(**template_vars)

  # No valid mode specified - use default template with job executable
  # This is the common case: user provides container_image and command via job args
  try:
    from xmanager.cloud import sbatch_templates
  except ImportError:
    raise ImportError(
        "Default template mode requires xmanager.cloud.sbatch_templates module"
    )

  num_nodes = getattr(executor.requirements, 'replicas', 1) or 1
  gpus_per_node = cluster_config.gpus_per_node

  # Build command from job executable
  executable_args = getattr(job.executable, 'args', {})
  args = xm.merge_args(executable_args, job.args).to_list(utils.ARG_ESCAPER)
  entrypoint = getattr(job.executable, 'entrypoint', None)
  path = getattr(job.executable, 'path', None)
  if entrypoint:
    command = f"{entrypoint} {' '.join(args)}".strip()
  elif path:
    command = f"{path} {' '.join(args)}".strip()
  else:
    command = ' '.join(args)

  # Build container mounts string (deduplicate to avoid redundant mounts)
  container_mounts_list = [cluster_config.nccl_dir] + list(getattr(executor, 'container_mounts', []))
  # Deduplicate while preserving order
  seen = set()
  unique_mounts = []
  for m in container_mounts_list:
    if m and m not in seen:
      seen.add(m)
      unique_mounts.append(m)
  container_mounts_str = ','.join(unique_mounts)

  renderer = sbatch_templates.SbatchTemplateRenderer()
  return renderer.render(
      job_name=job_name,
      num_nodes=num_nodes,
      gpus_per_node=gpus_per_node,
      partition=getattr(executor, 'partition', None),
      account=getattr(executor, 'account', None),
      time_limit=getattr(executor, 'time_limit', None),
      exclusive=getattr(executor, 'exclusive', True),
      command=command,
      working_dir=executor.work_dir,
      env_vars=getattr(executor, 'env_vars', {}),
      sbatch_flags=getattr(executor, 'sbatch_flags', {}),
      prologue_commands=getattr(executor, 'prologue_commands', []),
      epilogue_commands=getattr(executor, 'epilogue_commands', []),
      nccl_setup_script=cluster_config.setup_script,
      nccl_lib_path=cluster_config.get_nccl_lib_path(),
      nccl_env_vars=cluster_config.nccl_env_vars,
      # Container configuration
      container_image=getattr(executor, 'container_image', None),
      container_mounts=container_mounts_str,
      use_mpi=getattr(executor, 'use_mpi', False),
      container_env_passthrough=getattr(executor, 'container_env_passthrough', []),
      # Distributed training
      master_port=getattr(executor, 'master_port', 29500),
      setup_jax_coordinator=getattr(executor, 'setup_jax_coordinator', False),
  )


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

    # Map Slurm states to XManager states
    if slurm_status == 'PENDING':
      return local_status.LocalWorkUnitStatus(
          local_status.LocalWorkUnitStatusEnum.PENDING,
          message=f"Slurm job {self.slurm_job_id} pending"
      )
    elif slurm_status == 'RUNNING':
      return local_status.LocalWorkUnitStatus(
          local_status.LocalWorkUnitStatusEnum.RUNNING,
          message=f"Slurm job {self.slurm_job_id} running"
      )
    elif slurm_status == 'COMPLETED':
      return local_status.LocalWorkUnitStatus(
          local_status.LocalWorkUnitStatusEnum.COMPLETED,
          message=f"Slurm job {self.slurm_job_id} completed successfully"
      )
    elif slurm_status == 'FAILED':
      return local_status.LocalWorkUnitStatus(
          local_status.LocalWorkUnitStatusEnum.FAILED,
          message=f"Slurm job {self.slurm_job_id} failed"
      )
    elif slurm_status == 'CANCELLED':
      return local_status.LocalWorkUnitStatus(
          local_status.LocalWorkUnitStatusEnum.CANCELLED,
          message=f"Slurm job {self.slurm_job_id} cancelled"
      )
    else:
      return local_status.LocalWorkUnitStatus(
          local_status.LocalWorkUnitStatusEnum.UNKNOWN,
          message=f"Slurm job {self.slurm_job_id} status: {slurm_status}"
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
        print(f"[{self.job_name}] {line.decode().strip()}")

        # Check if job has completed
        status = self.client.get_job_status(self.slurm_job_id)
        if status in ['COMPLETED', 'FAILED', 'CANCELLED', 'UNKNOWN']:
          # Read any remaining output
          remaining = await process.stdout.read()
          if remaining:
            for line in remaining.decode().split('\n'):
              if line.strip():
                print(f"[{self.job_name}] {line.strip()}")
          break

    except asyncio.CancelledError:
      # Monitor was cancelled
      pass
    except Exception as e:
      print(f"Warning: Error monitoring job {self.job_name}: {e}")


# =============================================================================
# Launch Logic
# =============================================================================

def _vertex_training_cluster_job_predicate(job: xm.Job) -> bool:
  """Filter for Vertex Training Cluster jobs."""
  return isinstance(job.executor, local_executors.VertexTrainingCluster)


async def launch(
    local_experiment_unit: Any, job_group: xm.JobGroup
) -> List[VertexTrainingClusterHandle]:
  """Launch jobs on Vertex Training Cluster via Slurm.

  This function:
    1. Generates sbatch scripts from executor configuration
    2. Submits jobs to Slurm via sbatch
    3. Creates handles to track job status
    4. Optionally streams job output

  Args:
    local_experiment_unit: Experiment unit with metadata
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
  experiment_id = local_experiment_unit.experiment_id
  work_unit_id = local_experiment_unit.work_unit_id
  experiment_unit_name = local_experiment_unit.experiment_unit_name or f"wu_{work_unit_id}"

  for job in jobs:
    executor = job.executor
    client = Client(executor)
    cluster_config = client.cluster_config

    # Generate job name
    job_name = f"exp{experiment_id}_{experiment_unit_name}"

    # Ensure work directory exists
    work_dir = executor.work_dir or os.getcwd()

    # Rsync local files to cluster if specified
    local_files = getattr(executor, 'local_files', [])
    if local_files:
      try:
        client.rsync_files(local_files, work_dir)
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
      print(f"Submitted Slurm job {slurm_job_id} for {job_name}")
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

    # Note: save_to_storage is called by experiment.py's _save_handles_to_storage

    # Start monitoring if enabled
    if executor.stream_output:
      monitor_task = asyncio.create_task(handle.monitor())
      handle._monitor_task = monitor_task

    handles_list.append(handle)

  return handles_list


def _create_handle(*args, data, vertex_training_cluster_jobs) -> VertexTrainingClusterHandle:
  """Restore handle from database.

  Args:
    data: Database record with job information
    vertex_training_cluster_jobs: List to append restored handle

  Returns:
    Restored execution handle
  """
  del args

  executor = local_executors.VertexTrainingCluster(
      cluster_type=data.vertex_training_cluster.cluster_type,
      login_node=data.vertex_training_cluster.login_node or None,
      partition=data.vertex_training_cluster.partition or None,
      use_gcloud_ssh=data.vertex_training_cluster.use_gcloud_ssh,
      ssh_hostname=data.vertex_training_cluster.ssh_hostname or None,
      work_dir=data.vertex_training_cluster.work_dir or None,
  )

  slurm_job_id = data.vertex_training_cluster.slurm_job_id or ""
  job_name = f"vtc_job_{slurm_job_id}" if slurm_job_id else "restored_job"

  handle = VertexTrainingClusterHandle(
      job_name=job_name,
      slurm_job_id=slurm_job_id,
      client=Client(executor),
      executor=executor,
  )

  vertex_training_cluster_jobs.append(handle)
  return handle


def register():
  """Registers Vertex Training Cluster execution logic with XManager."""
  registry.register(
      local_executors.VertexTrainingCluster,
      launch=launch,
      create_handle=_create_handle,
  )
