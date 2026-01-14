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
r"""Test script to verify connection to Vertex Training Cluster.

This script tests:
  1. Cluster configuration (NCCL settings for GPU type)
  2. SSH connection to login node
  3. Basic Slurm commands on the cluster

Usage:
  xmanager launch examples/vertex_training_cluster/test_connection.py -- \
    --cluster_type=hcc-a4 \
    --login_node=vmdsa405-login-001 \
    --use_gcloud_ssh \
    --ssh_hostname=nic0.vmdsa405-login-001.asia-southeast1-b.c.ai-infra-recipe-validation.internal.gcpnode.com
"""

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_local

# Cluster configuration
_CLUSTER_TYPE = flags.DEFINE_string(
    'cluster_type', 'hcc-a4',
    'Cluster type: hcc-a3m (H100), hcc-a3u (H200), hcc-a4 (B200)')

# Connection
_LOGIN_NODE = flags.DEFINE_string('login_node', None, 'SSH login node')
_USE_GCLOUD_SSH = flags.DEFINE_bool('use_gcloud_ssh', False, 'Use gcloud SSH')
_SSH_HOSTNAME = flags.DEFINE_string(
    'ssh_hostname', None,
    'SSH hostname override (for gcloud ssh -o Hostname=...)')


def main(_):
  print("=" * 60)
  print("Vertex Training Cluster Connection Test")
  print("=" * 60)

  # Create executor to get cluster configuration
  executor = xm_local.VertexTrainingCluster(
      cluster_type=_CLUSTER_TYPE.value,
      login_node=_LOGIN_NODE.value,
      use_gcloud_ssh=_USE_GCLOUD_SSH.value,
      ssh_hostname=_SSH_HOSTNAME.value,
  )

  # Step 1: Get cluster configuration
  print("\n1. Cluster Configuration:")
  print("-" * 40)
  cluster_config = executor.get_cluster_config()
  print(f"   Cluster type: {cluster_config.cluster_type}")
  print(f"   GPU type: {cluster_config.gpu_type}")
  print(f"   GPUs per node: {cluster_config.gpus_per_node}")
  print(f"   NCCL dir: {cluster_config.nccl_dir}")

  if cluster_config.setup_script:
    print(f"   Setup script: {cluster_config.setup_script}")

  if cluster_config.nccl_env_vars:
    print("   NCCL environment variables:")
    for k, v in cluster_config.nccl_env_vars.items():
      print(f"      {k}={v}")

  print("   [OK] Cluster configuration loaded successfully")

  # Step 2: Test connection by running simple commands
  if not _LOGIN_NODE.value:
    print("\n2. SSH Connection Test:")
    print("-" * 40)
    print("   [SKIP] No login_node specified")
    print("   To test SSH, add: --login_node=<node> --use_gcloud_ssh")
    return

  print("\n2. SSH Connection Test:")
  print("-" * 40)
  print(f"   Login node: {_LOGIN_NODE.value}")
  print(f"   Use gcloud SSH: {_USE_GCLOUD_SSH.value}")
  if _SSH_HOSTNAME.value:
    print(f"   SSH hostname: {_SSH_HOSTNAME.value}")

  # Import the client directly to test connection
  from xmanager.cloud import vertex_training_cluster as vtc

  client = vtc.Client(executor)

  # Show the SSH command being used
  ssh_prefix = client._build_ssh_prefix()
  print(f"   SSH command: {' '.join(ssh_prefix)} '<command>'")

  # Test basic connection
  print("\n   Testing connection with 'hostname'...")
  result = client.run_command("hostname")
  if result.returncode == 0:
    print(f"   [OK] Connected to: {result.stdout.strip()}")
  else:
    print(f"   [FAIL] SSH failed: {result.stderr}")
    return

  # Step 3: Test Slurm commands
  print("\n3. Slurm Commands Test:")
  print("-" * 40)

  # Test sinfo
  print("   Running 'sinfo --summarize'...")
  result = client.run_command("sinfo --summarize")
  if result.returncode == 0:
    print("   [OK] Slurm cluster info:")
    for line in result.stdout.strip().split('\n')[:10]:
      print(f"      {line}")
  else:
    print(f"   [WARN] sinfo failed: {result.stderr}")

  # Test squeue
  print("\n   Running 'squeue'...")
  result = client.run_command("squeue 2>/dev/null | head -10")
  if result.returncode == 0:
    output = result.stdout.strip()
    if output:
      print("   [OK] Current jobs:")
      for line in output.split('\n')[:10]:
        print(f"      {line}")
    else:
      print("   [OK] No jobs currently running")
  else:
    print(f"   [WARN] squeue failed: {result.stderr}")

  # Check for NCCL directory
  print(f"\n   Checking NCCL directory '{cluster_config.nccl_dir}'...")
  result = client.run_command(f"ls -la {cluster_config.nccl_dir} 2>&1 | head -5")
  if result.returncode == 0:
    print(f"   [OK] NCCL directory exists:")
    for line in result.stdout.strip().split('\n'):
      print(f"      {line}")
  else:
    print(f"   [WARN] NCCL directory check: {result.stdout.strip()}")

  print("\n" + "=" * 60)
  print("Connection test complete!")
  print("=" * 60)


if __name__ == '__main__':
  app.run(main)
