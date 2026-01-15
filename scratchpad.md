# Vertex Training Cluster Integration: Research & Revised Spec

## Executive Summary

After researching the XManager codebase, lxm3, xm-slurm, and Vertex AI Training Clusters, I've identified several important considerations for the proposed integration. The original spec is mostly well-designed but has some areas that could be improved to better align with XManager principles and learn from lxm3's approach.

**Key Findings:**
1. The proposed spec is largely sound but over-engineers some aspects
2. lxm3's approach offers valuable lessons, especially for packaging
3. The "Launcher" abstraction is novel and useful, but may add unnecessary complexity
4. The spec correctly follows XManager's executor/handle patterns
5. Some naming and structural decisions should be reconsidered

---

## Part 1: Analysis of Existing Integrations

### 1.1 XManager Core Patterns

From analyzing the codebase, XManager follows these core patterns:

**Executor Pattern** (`xmanager/xm_local/executors.py`):
```python
@attr.s(auto_attribs=True)
class ExecutorSpec(xm.ExecutorSpec):
    """Minimal metadata for packaging (e.g., push_image_tag)"""
    push_image_tag: Optional[str] = None

@attr.s(auto_attribs=True)
class Executor(xm.Executor):
    """Runtime environment with requirements and platform-specific settings"""
    requirements: xm.JobRequirements = attr.Factory(xm.JobRequirements)
    # Platform-specific attributes...

    Spec = ExecutorSpec

    def __attrs_post_init__(self):
        # Lazy import and registration
        execution_module = importlib.import_module('...')
        execution_module.register()

    @classmethod
    async def launch(cls, local_experiment_unit, job_group):
        return await registry.get_launch_method(cls)(...)
```

**Handle Pattern** (`xmanager/cloud/vertex.py`, `kubernetes.py`):
```python
@attr.s(auto_attribs=True)
class Handle(handles.ExecutionHandle):
    job_name: str  # or job reference

    async def wait(self) -> None: ...
    def stop(self) -> None: ...
    def get_status(self) -> local_status.LocalWorkUnitStatus: ...
    def save_to_storage(self, experiment_id: int, work_unit_id: int) -> None: ...
```

**Key Observations:**
- ExecutorSpec is minimal - just packaging-related config (push_image_tag)
- Executor contains runtime requirements and platform-specific settings
- Lazy registration via `__attrs_post_init__` avoids heavy imports
- Handles store minimal state (job identifiers) and delegate to clients
- Database storage follows existing patterns (insert_xxx_job methods)

### 1.2 lxm3's Approach

**Key Design Decisions in lxm3:**

1. **Two-Stage Packaging**: Separates runtime dependencies (heavy, cached) from application code (light, changes frequently). This is particularly important for HPC where:
   - File count limits exist on shared filesystems
   - Singularity images are single files (no layer caching)
   - rsync deployment is efficient for small archives

2. **Array Jobs**: lxm3 introduces `ArrayJob` as a new `JobConfig` to handle the HPC pattern of submitting many similar jobs as a single array. XManager's core API doesn't natively support this.

3. **Framework Agnostic**: lxm3 is a "launcher" not a "distributed training framework". It doesn't provide torchrun/mpirun wrappers - that's the user's responsibility.

4. **Container Strategy**: Uses Singularity/Apptainer (not Docker) for HPC compatibility. No root privileges required.

5. **Vendored XManager API**: lxm3 vendors XManager's API to avoid dependency conflicts. Uses `from lxm3 import xm` instead of `from xmanager import xm`.

**lxm3 Executor Example:**
```python
from lxm3 import xm
from lxm3 import xm_cluster

with xm_cluster.create_experiment(experiment_title="example") as experiment:
    executor = xm_cluster.Slurm()  # Minimal executor
    spec = xm_cluster.PythonPackage(
        path=".",
        entrypoint=xm_cluster.ModuleName("my_package.main"),
    )
    [executable] = experiment.package(
        [xm.Packageable(spec, executor_spec=executor.Spec())]
    )
    experiment.add(xm.Job(executable=executable, executor=executor))
```

### 1.3 Comparison: Proposed VTC Spec vs lxm3 vs XManager Patterns

| Aspect | XManager (Vertex/K8s) | lxm3 | Proposed VTC Spec |
|--------|----------------------|------|-------------------|
| **ExecutorSpec** | Minimal (push_image_tag) | Similar | More complex (squashfs_path, auto_convert) |
| **Container Strategy** | Docker/GCR | Singularity | Docker + squashfs conversion |
| **Distributed Training** | Worker pools via replicas | User's responsibility | Launcher abstraction |
| **Script Generation** | Cloud API handles it | Auto-generated job scripts | Jinja2 templates |
| **SSH/Remote** | Cloud APIs | rsync + SSH | SSH/gcloud SSH |
| **Job Monitoring** | Cloud client polling | Slurm commands | squeue/sacct polling |

---

## Part 2: Critique of the Proposed Spec

### 2.1 What the Spec Gets Right

1. **Three Submission Modes**: Raw script, template, and auto-generate is a good flexibility model that serves different user needs.

2. **Using `requirements.replicas` for Node Count**: Correctly maps to XManager's existing abstraction.

3. **Cluster Config Registry**: The `CLUSTER_CONFIGS` dictionary with NCCL settings per cluster type is well-designed.

4. **Handle Pattern**: Correctly extends `ExecutionHandle` and follows the Vertex/K8s patterns.

5. **Database Storage**: Following the existing pattern of storing job metadata is correct.

6. **Bug Fix**: The P0 fix for `_get_push_image_tag()` is necessary.

### 2.2 Concerns and Improvements

#### Concern 1: The "Launcher" Abstraction May Be Overengineered

**Issue**: The spec introduces a comprehensive Launcher abstraction (TorchrunLauncher, AccelerateLauncher, DeepSpeedLauncher, etc.). While elegant, this:
- Adds significant complexity
- Duplicates functionality that users already have in their training scripts
- Violates lxm3's philosophy of being a "launcher not a framework"
- May quickly become outdated as ML frameworks evolve

**lxm3's Approach**: lxm3 doesn't wrap torchrun/mpirun. Users are expected to:
- Have their distributed training setup in their application code
- Or use a wrapper script that sets up the distributed environment

**Recommendation**: Consider a simpler approach where:
- Users provide the complete command (like lxm3)
- Or provide a script that handles distributed setup
- Keep launchers optional and minimal (maybe just TorchrunLauncher for convenience)

#### Concern 2: ExecutorSpec May Be Too Heavy

**Issue**: The proposed `VertexTrainingClusterSpec` has:
```python
push_image_tag: Optional[str] = None
squashfs_path: Optional[str] = None
auto_convert_to_squashfs: bool = False
```

**XManager Pattern**: ExecutorSpec should be minimal, containing only packaging-related metadata.

**Recommendation**:
- `push_image_tag` is appropriate for ExecutorSpec
- `squashfs_path` should be on the Executor, not the Spec (it's a runtime artifact)
- `auto_convert_to_squashfs` is a runtime behavior, belongs on Executor

#### Concern 3: Container Strategy Needs Clarification

**Issue**: The spec mentions both Docker images and squashfs but doesn't clearly define the packaging workflow:
- How does a Docker image get converted to squashfs?
- Where does this conversion happen (local machine? cluster node?)
- How are images transferred to the cluster?

**lxm3's Approach**:
- Builds Singularity images locally via Docker
- Uses rsync to deploy to cluster
- Separates runtime image from application code

**Recommendation**: Define a clear packaging strategy:
1. User provides Docker image → XManager converts to squashfs locally
2. User provides pre-built squashfs → XManager copies to cluster
3. User provides nothing → XManager builds from source (like PythonContainer)

#### Concern 4: No Array Job Support

**Issue**: The spec doesn't address array jobs, which are critical for HPC workloads. lxm3 specifically introduced `ArrayJob` to handle hyperparameter sweeps efficiently.

**XManager Core Issue**: The core XManager API doesn't have native array job support. lxm3 solved this by adding `xm_cluster.ArrayJob`.

**Recommendation**: Consider adding array job support:
```python
# Option 1: Follow lxm3's pattern
class ArrayJob(xm.JobConfig):
    executable: xm.Executable
    executor: xm.Executor
    args: Sequence[xm.Args]  # One per array element
```

#### Concern 5: SSH/Connection Model

**Issue**: The spec has `use_gcloud_ssh` and `login_node` on the Executor. This is reasonable but could be cleaner.

**lxm3's Approach**: Uses a configuration file (`~/.config/lxm3/config.toml`) for cluster connection settings, keeping the Executor clean.

**Recommendation**: Consider separating connection config:
```python
@attr.s(auto_attribs=True)
class ClusterConnection:
    login_node: str
    use_gcloud_ssh: bool = False
    ssh_hostname: Optional[str] = None
    # ...

@attr.s(auto_attribs=True)
class VertexTrainingCluster(xm.Executor):
    connection: ClusterConnection
    cluster_type: str
    # ...
```

#### Concern 6: File Location

**Issue**: The spec puts everything in `xmanager/cloud/`. However:
- Kubernetes and Vertex are in `xmanager/cloud/`
- But this is for cloud-specific code that talks to cloud APIs
- Slurm is accessed via SSH, not a cloud API

**Recommendation**: Consider `xmanager/slurm/` or keep in `xmanager/cloud/` since it's still a "cloud" cluster (Vertex Training Cluster on GCP).

---

## Part 3: Revised Spec Proposal

### 3.1 Design Philosophy

Following XManager principles:
1. **Python as config language**: All configuration in Python, no external config files
2. **Modular design**: Keep components separate and replaceable
3. **Eager execution**: Submit jobs immediately when `experiment.add()` is called
4. **Avoid late bindings**: Be explicit about configuration
5. **REPL speed**: Fast iteration for researchers

Following lxm3 lessons:
1. **Framework agnostic**: Don't build a distributed training framework
2. **Simple command execution**: Users provide the command, we run it
3. **Efficient packaging**: Separate runtime from application code where beneficial

### 3.2 Simplified Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Code                                 │
│  xm_local.VertexTrainingCluster(cluster_type='hcc-a4', ...)     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     VertexTrainingCluster                        │
│  Executor with requirements, cluster_type, partition, etc.      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   SbatchScriptBuilder                            │
│  Builds sbatch script from executor settings + job executable   │
│  Injects NCCL config, container mounts, env vars                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        SlurmClient                               │
│  SSH/local → sbatch script.sh → job_id                          │
│  squeue/sacct → status, scancel → cancel                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              VertexTrainingClusterHandle                         │
│  wait() → poll squeue/sacct until done                          │
│  stop() → scancel job_id                                        │
│  get_status() → PENDING/RUNNING/COMPLETED/FAILED                │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Revised File Structure

```
xmanager/
├── cloud/
│   ├── vertex_training_cluster.py    # Main implementation
│   └── vertex_training_cluster_test.py
├── xm_local/
│   ├── executors.py                   # Add VertexTrainingCluster, VertexTrainingClusterSpec
│   ├── packaging/
│   │   ├── router.py                  # Add VTC case
│   │   └── cloud.py                   # Fix push_image_tag bug
│   └── __init__.py                    # Export new executor
```

### 3.4 Revised Executor Classes

```python
# xmanager/xm_local/executors.py

@attr.s(auto_attribs=True)
class VertexTrainingClusterSpec(xm.ExecutorSpec):
    """Vertex Training Cluster spec for packaging.

    Attributes:
        push_image_tag: Image registry path to push (for Docker packaging)
    """
    push_image_tag: Optional[str] = None


@attr.s(auto_attribs=True)
class VertexTrainingCluster(xm.Executor):
    """Executor for Slurm-based Vertex Training Clusters.

    This executor submits jobs to Vertex AI Training Clusters via sbatch.
    It handles NCCL configuration, container execution, and job lifecycle.

    Supported cluster types:
        - hcc-a3m: H100 GPUs with TCPXO networking
        - hcc-a3u: H200 GPUs with gIB networking
        - hcc-a4: B200 GPUs with gIB networking
        - hcc-a3h: H100 GPUs with gIB networking

    Example:
        executor = xm_local.VertexTrainingCluster(
            cluster_type='hcc-a4',
            partition='a4',
            login_node='cluster-login-001',
            requirements=xm.JobRequirements(replicas=4),  # 4 nodes
        )

        job = xm.Job(
            executable=xm.Container(
                image_path='my-container.sqsh',
                entrypoint='torchrun --nnodes=$SLURM_NNODES train.py',
            ),
            executor=executor,
        )
    """

    # Standard XManager requirements
    requirements: xm.JobRequirements = attr.Factory(xm.JobRequirements)
    # NOTE: Use requirements.replicas for node count (maps to --nodes)

    # Cluster identification
    cluster_type: str = 'hcc-a3m'  # hcc-a3m, hcc-a3u, hcc-a4, hcc-a3h
    partition: Optional[str] = None
    account: Optional[str] = None

    # Time and resource settings
    time_limit: str = "0"  # "0" = unlimited, or "HH:MM:SS"
    exclusive: bool = True

    # Connection settings
    login_node: Optional[str] = None
    use_gcloud_ssh: bool = False
    ssh_hostname: Optional[str] = None  # For gcloud compute ssh

    # Container settings
    container_image: Optional[str] = None  # Path to .sqsh file on cluster
    container_mounts: List[str] = attr.Factory(list)

    # Working directory and logs
    work_dir: Optional[str] = None
    log_dir: Optional[str] = None

    # Custom sbatch script (overrides auto-generation)
    sbatch_script: Optional[str] = None

    # Additional sbatch flags (arbitrary --key=value pairs)
    sbatch_flags: Dict[str, str] = attr.Factory(dict)

    # Environment variables
    env_vars: Dict[str, str] = attr.Factory(dict)

    # Hook commands
    prologue_commands: List[str] = attr.Factory(list)
    epilogue_commands: List[str] = attr.Factory(list)

    # Output streaming
    stream_output: bool = True

    Spec = VertexTrainingClusterSpec

    def __attrs_post_init__(self):
        vtc_execution = importlib.import_module(
            'xmanager.cloud.vertex_training_cluster'
        )
        vtc_execution.register()

    @override
    @classmethod
    async def launch(
        cls, local_experiment_unit: Any, job_group: xm.JobGroup
    ) -> Sequence[handles.ExecutionHandle]:
        return await registry.get_launch_method(cls)(
            local_experiment_unit, job_group
        )
```

### 3.5 Simplified Main Implementation

```python
# xmanager/cloud/vertex_training_cluster.py
"""Vertex Training Cluster executor implementation.

This module provides Slurm-based job submission to Vertex AI Training Clusters.
"""

import asyncio
import re
import subprocess
from typing import Any, Dict, List, Optional

import attr
from xmanager import xm
from xmanager.xm import utils
from xmanager.xm_local import executables as local_executables
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
    """NCCL and networking configuration for a cluster type."""
    cluster_type: str
    gpu_type: str
    gpus_per_node: int = 8
    nccl_dir: str = ""
    setup_script: Optional[str] = None
    nccl_env_vars: Dict[str, str] = attr.Factory(dict)


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
        nccl_env_vars={
            "NCCL_NET": "gIB",
            "NCCL_IB_TC": "52",
            "NCCL_CROSS_NIC": "0",
            # ... (full list from original spec)
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
    if cluster_type not in CLUSTER_CONFIGS:
        raise ValueError(
            f"Unknown cluster_type: {cluster_type}. "
            f"Supported: {list(CLUSTER_CONFIGS.keys())}"
        )
    return CLUSTER_CONFIGS[cluster_type]


# =============================================================================
# SSH Client
# =============================================================================

class SlurmClient:
    """Client for Slurm operations via SSH."""

    def __init__(self, executor: local_executors.VertexTrainingCluster):
        self.executor = executor
        self.cluster_config = get_cluster_config(executor.cluster_type)

    def _ssh_command(self, cmd: str) -> subprocess.CompletedProcess:
        """Execute command on cluster via SSH."""
        if self.executor.login_node:
            if self.executor.use_gcloud_ssh:
                prefix = ['gcloud', 'compute', 'ssh', self.executor.login_node, '--']
                if self.executor.ssh_hostname:
                    prefix.extend(['-o', f'Hostname={self.executor.ssh_hostname}'])
            else:
                prefix = ['ssh', '-o', 'StrictHostKeyChecking=no',
                         self.executor.login_node]
            return subprocess.run(prefix + [cmd], capture_output=True, text=True)
        return subprocess.run(cmd, shell=True, capture_output=True, text=True)

    def submit(self, script: str) -> str:
        """Submit sbatch script and return job ID."""
        # Write script to cluster
        script_path = f"/tmp/xm_job_{id(script)}.sh"
        write_cmd = f"cat > {script_path} << 'EOF'\n{script}\nEOF"
        self._ssh_command(write_cmd)

        # Submit
        result = self._ssh_command(f"sbatch {script_path}")
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed: {result.stderr}")

        match = re.search(r'Submitted batch job (\d+)', result.stdout)
        if not match:
            raise RuntimeError(f"Could not parse job ID: {result.stdout}")
        return match.group(1)

    def get_status(self, job_id: str) -> str:
        """Get job status via squeue/sacct."""
        # Try squeue first (running/pending)
        result = self._ssh_command(f"squeue -j {job_id} -h -o '%T'")
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().upper()

        # Fall back to sacct (completed jobs)
        result = self._ssh_command(f"sacct -j {job_id} -n -o State -X")
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split()[0].upper()
        return 'UNKNOWN'

    def cancel(self, job_id: str) -> None:
        """Cancel job via scancel."""
        self._ssh_command(f"scancel {job_id}")


# =============================================================================
# Sbatch Script Builder
# =============================================================================

def build_sbatch_script(
    executor: local_executors.VertexTrainingCluster,
    job: xm.Job,
    job_name: str,
) -> str:
    """Build sbatch script from executor and job settings."""

    # If user provides custom script, use it
    if executor.sbatch_script:
        return executor.sbatch_script

    cluster_config = get_cluster_config(executor.cluster_type)
    num_nodes = executor.requirements.replicas or 1
    gpus_per_node = cluster_config.gpus_per_node

    # Build command from executable
    executable = job.executable
    args = xm.merge_args(executable.args, job.args).to_list(utils.ARG_ESCAPER)

    if hasattr(executable, 'entrypoint') and executable.entrypoint:
        command = f"{executable.entrypoint} {' '.join(args)}".strip()
    else:
        command = ' '.join(args)

    # Build script
    lines = ['#!/bin/bash']
    lines.append(f'#SBATCH --job-name={job_name}')

    if executor.partition:
        lines.append(f'#SBATCH --partition={executor.partition}')
    if executor.account:
        lines.append(f'#SBATCH --account={executor.account}')

    lines.append(f'#SBATCH --nodes={num_nodes}')
    lines.append(f'#SBATCH --gpus-per-node={gpus_per_node}')
    lines.append(f'#SBATCH --time={executor.time_limit}')

    if executor.exclusive:
        lines.append('#SBATCH --exclusive')

    if executor.log_dir:
        lines.append(f'#SBATCH --output={executor.log_dir}/slurm-%j.out')
        lines.append(f'#SBATCH --error={executor.log_dir}/slurm-%j.err')

    for key, value in executor.sbatch_flags.items():
        lines.append(f'#SBATCH --{key}={value}')

    lines.append('')
    lines.append('set -euo pipefail')
    lines.append('')

    # Master node setup
    lines.append('# Distributed training setup')
    lines.append('MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)')
    lines.append('export MASTER_ADDR')
    lines.append('export MASTER_PORT=29500')
    lines.append('')

    # NCCL setup
    if cluster_config.setup_script:
        lines.append('# NCCL setup')
        lines.append(f'source {cluster_config.setup_script}')
        lines.append(f'export LD_LIBRARY_PATH={cluster_config.nccl_dir}/lib64:$LD_LIBRARY_PATH')
        lines.append('')

    for key, value in cluster_config.nccl_env_vars.items():
        lines.append(f'export {key}="{value}"')
    if cluster_config.nccl_env_vars:
        lines.append('')

    # User environment variables
    for key, value in executor.env_vars.items():
        lines.append(f'export {key}="{value}"')
    if executor.env_vars:
        lines.append('')

    # Prologue
    for cmd in executor.prologue_commands:
        lines.append(cmd)
    if executor.prologue_commands:
        lines.append('')

    # Working directory
    if executor.work_dir:
        lines.append(f'cd "{executor.work_dir}"')
        lines.append('')

    # Main command
    lines.append('# Main command')
    lines.append(command)
    lines.append('')

    # Epilogue
    for cmd in executor.epilogue_commands:
        lines.append(cmd)

    return '\n'.join(lines)


# =============================================================================
# Handle
# =============================================================================

@attr.s(auto_attribs=True)
class VertexTrainingClusterHandle(handles.ExecutionHandle):
    """Handle for Slurm jobs on Vertex Training Cluster."""

    job_name: str
    slurm_job_id: str
    client: SlurmClient

    async def wait(self) -> None:
        """Wait for job completion by polling status."""
        terminal_states = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT'}
        while True:
            status = self.client.get_status(self.slurm_job_id)
            if status in terminal_states:
                break
            await asyncio.sleep(30)

    def stop(self) -> None:
        """Cancel the job."""
        self.client.cancel(self.slurm_job_id)

    def get_status(self) -> local_status.LocalWorkUnitStatus:
        status = self.client.get_status(self.slurm_job_id)
        status_map = {
            'PENDING': local_status.LocalWorkUnitStatusEnum.NOT_STARTED,
            'RUNNING': local_status.LocalWorkUnitStatusEnum.RUNNING,
            'COMPLETED': local_status.LocalWorkUnitStatusEnum.COMPLETED,
            'FAILED': local_status.LocalWorkUnitStatusEnum.FAILED,
            'CANCELLED': local_status.LocalWorkUnitStatusEnum.FAILED,
        }
        return local_status.LocalWorkUnitStatus(
            status_map.get(status, local_status.LocalWorkUnitStatusEnum.UNKNOWN)
        )

    def save_to_storage(self, experiment_id: int, work_unit_id: int) -> None:
        # TODO: Add insert_vertex_training_cluster_job to database.py
        pass


# =============================================================================
# Launch
# =============================================================================

def _vtc_predicate(job: xm.Job) -> bool:
    return isinstance(job.executor, local_executors.VertexTrainingCluster)


async def launch(
    local_experiment_unit: Any,
    job_group: xm.JobGroup,
) -> List[VertexTrainingClusterHandle]:
    """Launch jobs on Vertex Training Cluster."""
    jobs = xm.job_operators.collect_jobs_by_filter(job_group, _vtc_predicate)
    if not jobs:
        return []

    handles_list = []
    experiment_title = local_experiment_unit._experiment_title
    work_unit_name = local_experiment_unit.experiment_unit_name

    for idx, job in enumerate(jobs):
        executor = job.executor
        client = SlurmClient(executor)
        job_name = f"{experiment_title}_{work_unit_name}_{idx}"

        script = build_sbatch_script(executor, job, job_name)
        print(f"Submitting {job_name} to {executor.cluster_type}...")

        slurm_job_id = client.submit(script)
        print(f"  Slurm job ID: {slurm_job_id}")

        handles_list.append(VertexTrainingClusterHandle(
            job_name=job_name,
            slurm_job_id=slurm_job_id,
            client=client,
        ))

    return handles_list


def register():
    """Register Vertex Training Cluster executor."""
    registry.register(
        local_executors.VertexTrainingCluster,
        launch=launch,
        create_handle=None,  # TODO: Implement for restoration from DB
    )
```

### 3.6 Changes to Packaging Router

```python
# xmanager/xm_local/packaging/router.py - Add this case

case executors.VertexTrainingClusterSpec():
    return cloud_packaging.package_cloud_executable(
        built_targets,
        packageable,
        packageable.executable_spec,
    )
```

### 3.7 Bug Fix for cloud.py

```python
# xmanager/xm_local/packaging/cloud.py

def _get_push_image_tag(executor_spec: xm.ExecutorSpec) -> Optional[str]:
    match executor_spec:
        case local_executors.CaipSpec() as caip_spec:
            return caip_spec.push_image_tag
        case local_executors.KubernetesSpec() as kubernetes_spec:
            return kubernetes_spec.push_image_tag
        case local_executors.VertexTrainingClusterSpec() as vtc_spec:  # ADD
            return vtc_spec.push_image_tag
        case _:
            raise TypeError(...)
```

---

## Part 4: Recommendations

### 4.1 Immediate Actions (P0)

1. **Fix the packaging bug** - Add VertexTrainingClusterSpec case to cloud.py

2. **Implement minimal executor** - Start with the simplified version above, add complexity as needed

3. **Test with real clusters** - Validate NCCL configs and sbatch generation

### 4.2 Near-term Improvements (P1)

1. **Add database storage** - Implement `insert_vertex_training_cluster_job()`

2. **Implement handle restoration** - For `create_handle` callback

3. **Add log streaming** - Via SSH tail -f

### 4.3 Future Enhancements (P2)

1. **Array job support** - Consider adding `ArrayJob` like lxm3

2. **Optional launchers** - Add TorchrunLauncher as a convenience, not a requirement

3. **Container conversion** - Automated Docker→squashfs conversion

4. **Configuration file** - For cluster connection settings (like lxm3's config.toml)

### 4.4 Key Design Decisions

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| Launcher abstraction | Optional, minimal | Users know their frameworks better |
| ExecutorSpec contents | Just push_image_tag | Keep minimal like other executors |
| Container strategy | User provides squashfs | Conversion is complex, defer to user |
| Array jobs | Future enhancement | Core API doesn't support it yet |
| SSH config | On Executor | Keep Python-centric like XManager |
| File location | xmanager/cloud/ | It's still a GCP-based cluster |

---

## Part 5: Comparison with Original Spec

| Aspect | Original Spec | Revised Spec |
|--------|---------------|--------------|
| **Launcher abstraction** | Full framework (6+ launchers) | Remove or make optional |
| **ExecutorSpec** | 3 attributes | 1 attribute (push_image_tag) |
| **Jinja2 templates** | Yes | Simple Python string building |
| **File count** | 4 new files | 1 new file + minor edits |
| **Complexity** | ~1500 lines | ~300 lines |
| **Dependencies** | Jinja2 | None new |

---

## Sources

- [XManager Launch API Principles](https://github.com/google-deepmind/xmanager/blob/main/docs/xm_launch_api_principles.md)
- [lxm3 GitHub Repository](https://github.com/ethanluoyc/lxm3)
- [lxm3 Documentation](https://lxm3.readthedocs.io/en/latest/)
- [XManager Issue #33: How to adapt the XManager API for HPC](https://github.com/google-deepmind/xmanager/issues/33)
- [Vertex AI Training Clusters Overview](https://docs.cloud.google.com/vertex-ai/docs/training/training-clusters/overview)
- [NVIDIA Pyxis - Slurm container plugin](https://github.com/NVIDIA/pyxis)

---

# Part 6: Implementation Review (feat/add-vtc-support branch)

## Files Changed Summary

| File | Lines Added | Purpose |
|------|-------------|---------|
| `xmanager/xm_local/executors.py` | +139 | VTC executor and spec classes |
| `xmanager/cloud/vertex_training_cluster.py` | +775 | Main implementation |
| `xmanager/cloud/launchers.py` | +295 | Launcher abstraction |
| `xmanager/cloud/sbatch_templates.py` | +258 | Jinja2 template rendering |
| `xmanager/xm_local/storage/database.py` | +39 | DB storage for VTC jobs |
| `xmanager/xm_local/storage/data.proto` | +18 | Proto for VTC job |
| `xmanager/xm_local/packaging/router.py` | +4 | Routing for VTC |
| `xmanager/xm_local/packaging/cloud.py` | +2 | push_image_tag support |
| `xmanager/xm_local/__init__.py` | +15 | Exports |
| `examples/vertex_training_cluster/launcher.py` | +182 | Example usage |
| `spec.md` | +1782 | Specification document |

**Total: ~5,000+ lines added**

---

## Critical Issues Found

### Issue 1: NEMORUN_HOME Hardcoding (CRITICAL)

**Location**: `vertex_training_cluster.py` lines 202, 223

```python
if set_nemorun_home and self.executor.work_dir:
    cmd = f"export NEMORUN_HOME={self.executor.work_dir} && {cmd}"
```

**Problem**: This couples XManager to NeMo conventions. XManager should be framework-agnostic.

**Fix**: Remove hardcoded NEMORUN_HOME. If users need it, they can add it to `env_vars`.

### Issue 2: Database Save Not Called (CRITICAL BUG)

**Location**: `vertex_training_cluster.py` launch function

**Problem**: The `launch()` function creates handles but never calls `save_to_storage()`. Jobs won't be persisted to the database, meaning:
- Jobs won't survive process restarts
- `xmanager list` won't show VTC jobs

**Fix**: Add `handle.save_to_storage(experiment_id, work_unit_id)` call in launch.

### Issue 3: ExecutorSpec Has Unused Attributes

**Location**: `executors.py` lines 186-198

```python
class VertexTrainingClusterSpec(xm.ExecutorSpec):
    push_image_tag: Optional[str] = None
    squashfs_path: Optional[str] = None  # NOT USED ANYWHERE
    auto_convert_to_squashfs: bool = False  # NOT USED ANYWHERE
```

**Problem**: `squashfs_path` and `auto_convert_to_squashfs` are defined but never referenced in the codebase.

**Fix**: Remove unused attributes. ExecutorSpec should be minimal (just `push_image_tag`).

### Issue 4: Launcher Type Safety Issue

**Location**: `executors.py` line 267

```python
launcher: Optional[Any] = None  # Type is Any to avoid circular import
```

**Problem**: Using `Any` weakens type checking.

**Fix**: Use `TYPE_CHECKING` pattern:
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from xmanager.cloud import launchers

launcher: Optional["launchers.Launcher"] = None
```

### Issue 5: Script File Cleanup Missing

**Location**: `vertex_training_cluster.py` line 257

**Problem**: Temporary sbatch scripts are created in work_dir but never cleaned up.

**Fix**: Clean up scripts after successful submission or add a note that they're intentionally kept for debugging.

---

## Moderate Issues

### Issue 6: Over-Engineered Launcher Abstraction

**Files**: `launchers.py` (295 lines), integration in `vertex_training_cluster.py`

**Problem**: Only 2 launchers implemented (NemoRunLauncher, CustomLauncher) out of 8+ planned. The abstraction adds complexity without proportional value.

**Evidence from lxm3**: lxm3 intentionally doesn't wrap distributed training frameworks. Users handle torchrun/mpirun themselves.

**Recommendation**: Consider simplifying to just:
1. Raw script mode
2. Simple template mode (without Jinja2)

### Issue 7: NCCL Configuration Inconsistency

**Location**: `vertex_training_cluster.py` CLUSTER_CONFIGS

**Problem**: Some clusters use `setup_script`, others use explicit `nccl_env_vars`. This dual approach adds complexity.

**Fix**: Document why the approaches differ, or unify them.

### Issue 8: Monitoring Efficiency

**Location**: `vertex_training_cluster.py` monitor() method

**Problem**: Checks job status on EVERY log line read (extremely inefficient).

**Fix**: Check status less frequently (every N lines or every N seconds).

### Issue 9: Print Statements Instead of Logging

**Location**: Throughout `vertex_training_cluster.py`

**Problem**: Uses `print()` instead of proper logging.

**Fix**: Use Python's `logging` module.

### Issue 10: Status Mapping Duplication

**Location**: `Client.get_job_status()` and `VertexTrainingClusterHandle.get_status()`

**Problem**: Status mapping logic is duplicated.

**Fix**: Use a shared constant or function.

---

## Minor Issues

### Issue 11: No Validation for Mutually Exclusive Modes

**Location**: `executors.py` VertexTrainingCluster

**Problem**: User can set both `sbatch_script` AND `launcher`, but only one will be used.

**Fix**: Add validation in `__attrs_post_init__` to ensure only one mode is set.

### Issue 12: Hardcoded Default Cluster Type

**Location**: `executors.py` line 253

```python
cluster_type: str = 'hcc-a3m'
```

**Problem**: Default might not be appropriate for all users.

**Fix**: Consider making it required (no default) or document why hcc-a3m is the default.

### Issue 13: No Timeout Protection in wait()

**Location**: `vertex_training_cluster.py` VertexTrainingClusterHandle.wait()

**Problem**: Could wait forever if job never completes.

**Fix**: Add optional timeout parameter.

---

## Comparison: Implementation vs Scratchpad Recommendations

| Aspect | Scratchpad Recommendation | Current Implementation | Status |
|--------|---------------------------|------------------------|--------|
| Launcher abstraction | Remove or make optional | Full implementation with 2 launchers | **Exceeds** |
| ExecutorSpec | Just `push_image_tag` | 3 attributes (2 unused) | **Needs cleanup** |
| Jinja2 templates | Remove, use simple string building | Full Jinja2 implementation | **Exceeds** |
| Database storage | Follow existing patterns | Implemented correctly | **Good** |
| Handle pattern | Follow Vertex/K8s | Implemented correctly | **Good** |
| Packaging router | Add VTC case | Done | **Good** |
| Framework agnostic | Yes | No (NEMORUN_HOME hardcoded) | **Needs fix** |
| Error handling | Proper logging | Uses print() | **Needs fix** |

---

## Cleanup Priority List

### P0 - Critical Fixes (Must do before merge)

1. **Remove NEMORUN_HOME hardcoding** - XManager must be framework-agnostic
2. **Add `save_to_storage()` call in launch** - Jobs must be persisted
3. **Remove unused ExecutorSpec attributes** - `squashfs_path`, `auto_convert_to_squashfs`

### P1 - Important Improvements

4. **Fix launcher type hint** - Use TYPE_CHECKING pattern
5. **Add mode validation** - Ensure only one submission mode is set
6. **Replace print() with logging** - Throughout vertex_training_cluster.py
7. **Fix monitoring efficiency** - Don't check status on every log line

### P2 - Nice to Have

8. **Simplify launcher abstraction** - Consider removing or making optional
9. **Remove Jinja2 dependency** - Use simple string building
10. **Add timeout to wait()** - Prevent infinite waiting
11. **Document NCCL config differences** - Explain why clusters have different setup methods
12. **Clean up temp script files** - Or document why they're kept

### P3 - Future Consideration

13. **Add array job support** - Like lxm3
14. **Add connection config object** - Group SSH settings
15. **Add validation for work_dir existence** - Before submission

---

## Recommended Code Changes

### Change 1: Remove NEMORUN_HOME (vertex_training_cluster.py)

```python
# BEFORE (lines 201-204)
def run_command(self, cmd: str, set_nemorun_home: bool = True) -> ...:
    if set_nemorun_home and self.executor.work_dir:
        cmd = f"export NEMORUN_HOME={self.executor.work_dir} && {cmd}"

# AFTER
def run_command(self, cmd: str) -> ...:
    # Removed NEMORUN_HOME - users should add it to env_vars if needed
```

### Change 2: Add save_to_storage call (vertex_training_cluster.py)

```python
# In launch() function, after creating handle:
handle = VertexTrainingClusterHandle(...)

# ADD THIS:
handle.save_to_storage(experiment_id, work_unit_id)

handles_list.append(handle)
```

### Change 3: Simplify ExecutorSpec (executors.py)

```python
# BEFORE
@attr.s(auto_attribs=True)
class VertexTrainingClusterSpec(xm.ExecutorSpec):
    push_image_tag: Optional[str] = None
    squashfs_path: Optional[str] = None  # REMOVE
    auto_convert_to_squashfs: bool = False  # REMOVE

# AFTER
@attr.s(auto_attribs=True)
class VertexTrainingClusterSpec(xm.ExecutorSpec):
    """Vertex Training Cluster spec for packaging."""
    push_image_tag: Optional[str] = None
```

### Change 4: Fix launcher type hint (executors.py)

```python
# At top of file, add:
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from xmanager.cloud import launchers as launchers_module

# In VertexTrainingCluster class:
launcher: Optional["launchers_module.Launcher"] = None
```

### Change 5: Add mode validation (executors.py)

```python
def __attrs_post_init__(self):
    # Validate mutually exclusive submission modes
    modes_set = sum([
        self.sbatch_script is not None,
        self.sbatch_template is not None,
        self.launcher is not None,
    ])
    if modes_set > 1:
        raise ValueError(
            "Only one submission mode allowed: sbatch_script, sbatch_template, or launcher"
        )

    # Continue with registration...
    vtc_execution = importlib.import_module(...)
    vtc_execution.register()
```

---

## Summary

The implementation is **functionally complete** and follows XManager patterns well in most areas. However, there are:

- **2 critical bugs** that must be fixed (NEMORUN_HOME, database save)
- **3 unnecessary components** (unused ExecutorSpec attrs, over-engineered launchers)
- **Multiple minor issues** (logging, efficiency, validation)

The core architecture is sound. With the P0 fixes, this is mergeable. The P1/P2 improvements would make it production-ready.

---

# Part 7: Consolidated Template Design for Real Workloads

## Real Workload Analysis

### Workload 1: MaxText (JAX)

```bash
# Key patterns:
# 1. Head node discovery
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# 2. JAX coordinator setup
export JAX_COORDINATOR_ADDRESS="${MASTER_ADDR}:${MASTER_PORT}"

# 3. srun with pyxis
srun --container-image=... --container-mounts=... bash -c "python train.py ..."
```

### Workload 2: NeMo (PyTorch)

```bash
# Key patterns:
# 1. Head node discovery (identical)
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )

# 2. torchrun with rendezvous
torchrun --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} ...

# 3. srun with pyxis + MPI
srun --mpi=pmix --container-image=... bash -c "torchrun ..."
```

## Common Pattern Extraction

| Component | Handled By | Notes |
|-----------|------------|-------|
| SBATCH directives | Template | Standard flags |
| Head node discovery | Template | Always needed for distributed |
| MASTER_ADDR/PORT | Template | Exported automatically |
| NCCL setup | Template | Based on cluster_type |
| Container wrapper | Template | srun with pyxis |
| MPI support | Executor attr | Optional `--mpi=pmix` |
| User command | Job args | Framework-specific |
| Framework env vars | Executor env_vars | JAX_COORDINATOR_ADDRESS, etc. |

## Proposed Solution

### New Executor Attributes

```python
@attr.s(auto_attribs=True)
class VertexTrainingCluster(xm.Executor):
    # ... existing attrs ...

    # Container execution (ADDED)
    use_mpi: bool = False  # Add --mpi=pmix to srun
    container_env_passthrough: List[str] = attr.Factory(list)  # --container-env

    # Distributed training (ADDED)
    master_port: int = 29500
    setup_jax_coordinator: bool = False  # Export JAX_COORDINATOR_ADDRESS
```

### New Consolidated Template

```jinja2
#!/bin/bash
#SBATCH --job-name={{ job_name }}
{% if partition %}#SBATCH --partition={{ partition }}{% endif %}
{% if account %}#SBATCH --account={{ account }}{% endif %}
#SBATCH --nodes={{ num_nodes }}
#SBATCH --gpus-per-node={{ gpus_per_node }}
#SBATCH --ntasks-per-node=1
{% if exclusive %}#SBATCH --exclusive{% endif %}
#SBATCH --mem=0
#SBATCH --time={{ time_limit | default("0") }}
{% if output_file %}#SBATCH --output={{ output_file }}{% endif %}
{% if error_file %}#SBATCH --error={{ error_file }}{% endif %}
{% for key, value in sbatch_flags.items() %}
#SBATCH --{{ key }}={{ value }}
{% endfor %}

set -euo pipefail

echo "========================================================"
echo "XManager Vertex Training Cluster Job"
echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Started: $(date)"
echo "========================================================"

# =============================================================================
# HEAD NODE DISCOVERY (Required for distributed training)
# =============================================================================
echo "Discovering head node..."
nodes_array=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Head node: $head_node ($head_node_ip)"

export MASTER_ADDR="$head_node_ip"
export MASTER_PORT={{ master_port | default(29500) }}
export GPUS_PER_NODE={{ gpus_per_node }}

{% if setup_jax_coordinator %}
# JAX Distributed Setup
export JAX_COORDINATOR_ADDRESS="${MASTER_ADDR}:${MASTER_PORT}"
echo "JAX_COORDINATOR_ADDRESS: $JAX_COORDINATOR_ADDRESS"
{% endif %}

# =============================================================================
# NCCL CONFIGURATION
# =============================================================================
{% if nccl_setup_script %}
echo "Setting up NCCL..."
source {{ nccl_setup_script }}
{% endif %}
{% if nccl_lib_path %}
export LD_LIBRARY_PATH={{ nccl_lib_path }}:${LD_LIBRARY_PATH:-}
{% endif %}
{% for key, value in nccl_env_vars.items() %}
export {{ key }}={{ value | shell_quote }}
{% endfor %}

# =============================================================================
# USER ENVIRONMENT VARIABLES
# =============================================================================
{% for key, value in env_vars.items() %}
export {{ key }}={{ value | shell_quote }}
{% endfor %}

# =============================================================================
# PROLOGUE COMMANDS
# =============================================================================
{% for cmd in prologue_commands %}
{{ cmd }}
{% endfor %}

# =============================================================================
# JOB IDENTIFIER
# =============================================================================
TIME=$(TZ="America/Los_Angeles" date +"%Y%m%d_%H%M%S")
export JOB_IDENTIFIER="{{ job_name }}-${SLURM_NNODES}n-${TIME}"
echo "JOB_IDENTIFIER: $JOB_IDENTIFIER"

{% if working_dir %}
cd {{ working_dir | shell_quote }}
{% endif %}

# =============================================================================
# CONTAINER EXECUTION
# =============================================================================
{% if container_image %}
echo "Launching container..."
CONTAINER_MOUNTS="{{ container_mounts }}"

srun \
  --job-name=${JOB_IDENTIFIER} \
  --nodes=${SLURM_NNODES} \
  --container-image={{ container_image }} \
  --container-mounts=${CONTAINER_MOUNTS} \
  --no-container-mount-home \
  --container-writable \
{% if use_mpi %}  --mpi=pmix \{% endif %}
{% if container_env_passthrough %}  --container-env {{ container_env_passthrough | join(',') }} \{% endif %}
  bash -c "
set -e
{% if nccl_lib_path %}
export LD_LIBRARY_PATH={{ nccl_lib_path }}:\${LD_LIBRARY_PATH:-}
{% endif %}
echo 'Container started on node rank \${SLURM_PROCID} of \${SLURM_NNODES}'
{{ command }}
echo 'Container finished on \$(hostname)'
"
{% else %}
# Direct execution (no container)
{{ command }}
{% endif %}

# =============================================================================
# EPILOGUE
# =============================================================================
{% for cmd in epilogue_commands %}
{{ cmd }}
{% endfor %}

echo "========================================================"
echo "Job completed at $(date)"
echo "========================================================"
```

## Updated _generate_sbatch_script Function

```python
def _generate_sbatch_script(
    executor: local_executors.VertexTrainingCluster,
    job: xm.Job,
    job_name: str,
    cluster_config: ClusterConfig,
) -> str:
    """Generate sbatch script for job submission."""

    # Mode 1: Raw script - return as-is
    if executor.sbatch_script:
        return executor.sbatch_script

    num_nodes = executor.requirements.replicas or 1

    # Build the main command from executable + job args
    executable = job.executable
    args = xm.merge_args(executable.args, job.args).to_list(utils.ARG_ESCAPER)

    if hasattr(executable, 'entrypoint') and executable.entrypoint:
        command = f"{executable.entrypoint} {' '.join(args)}".strip()
    elif hasattr(executable, 'path') and executable.path:
        command = f"{executable.path} {' '.join(args)}".strip()
    else:
        command = ' '.join(args)

    # Build container mounts string
    container_mounts_list = [cluster_config.nccl_dir] + executor.container_mounts
    container_mounts_str = ','.join(m for m in container_mounts_list if m)

    # Combine environment variables
    all_env_vars = {**executor.env_vars}

    # Get or create renderer
    if executor.sbatch_template:
        if isinstance(executor.sbatch_template, Path):
            renderer = SbatchTemplateRenderer(template_path=executor.sbatch_template)
        else:
            renderer = SbatchTemplateRenderer(custom_template=executor.sbatch_template)
    else:
        renderer = SbatchTemplateRenderer()  # Uses new consolidated template

    # Render template with ALL necessary variables
    return renderer.render(
        # Job identification
        job_name=job_name,

        # SBATCH directives
        partition=executor.partition,
        account=executor.account,
        num_nodes=num_nodes,
        gpus_per_node=cluster_config.gpus_per_node,
        time_limit=executor.time_limit,
        exclusive=executor.exclusive,
        sbatch_flags=executor.sbatch_flags,
        output_file=f"{executor.log_dir}/slurm-%j.out" if executor.log_dir else None,
        error_file=f"{executor.log_dir}/slurm-%j.err" if executor.log_dir else None,

        # Distributed training
        master_port=getattr(executor, 'master_port', 29500),
        setup_jax_coordinator=getattr(executor, 'setup_jax_coordinator', False),

        # NCCL configuration
        nccl_setup_script=cluster_config.setup_script,
        nccl_lib_path=cluster_config.get_nccl_lib_path(),
        nccl_env_vars=cluster_config.nccl_env_vars,

        # Container configuration
        container_image=executor.container_image,
        container_mounts=container_mounts_str,
        use_mpi=getattr(executor, 'use_mpi', False),
        container_env_passthrough=getattr(executor, 'container_env_passthrough', []),

        # Environment and commands
        env_vars=all_env_vars,
        prologue_commands=executor.prologue_commands,
        epilogue_commands=executor.epilogue_commands,
        working_dir=executor.work_dir,

        # Main command
        command=command,
    )
```

## Example Usage: MaxText

```python
from xmanager import xm, xm_local

executor = xm_local.VertexTrainingCluster(
    cluster_type='hcc-a4',
    requirements=xm.JobRequirements(replicas=32),

    # Container
    container_image='/home/common/images/jax-maxtext-2025-10-01.sqsh',
    container_mounts=[
        f"{os.environ['HOME']}/.cache",
        f"{os.environ['RECIPE_ROOT']}:/mnt/jobs",
        f"/mnt/lustre/{os.environ['LUSTRE_INSTANCE']}:/data",
        "/gcs:/gcs",
    ],

    # JAX-specific
    setup_jax_coordinator=True,  # NEW: Exports JAX_COORDINATOR_ADDRESS

    # Environment
    env_vars={
        "MAXTEXT_CONFIG_PATH": "/mnt/jobs/llama3-1-405b.yaml",
        "XLA_FLAGS": "...",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.98",
    },

    # Connection
    login_node='vmdsa405-login-001',
    use_gcloud_ssh=True,
    work_dir='/mnt/jobs',
)

job = xm.Job(
    executable=xm.Binary(path='python'),
    args=[
        'src/MaxText/train.py',
        '${MAXTEXT_CONFIG_PATH}',
        'steps=100',
        'base_output_directory=gs://my-bucket',
        'run_name=${JOB_IDENTIFIER}',
    ],
    executor=executor,
)
```

**Generated script will include:**
- Head node discovery ✅
- `JAX_COORDINATOR_ADDRESS` export ✅
- srun with container ✅
- All env vars ✅

## Example Usage: NeMo

```python
executor = xm_local.VertexTrainingCluster(
    cluster_type='hcc-a4',
    requirements=xm.JobRequirements(replicas=4),

    # Container
    container_image='/path/to/nemo.sqsh',
    container_mounts=[
        f"{os.environ['HOME']}/.cache",
        f"{os.environ['LOGS_PATH']}:/mnt/logs",
        f"{os.environ['RECIPE_ROOT']}:/mnt/workspace",
    ],

    # NeMo/PyTorch-specific
    use_mpi=True,  # NEW: Adds --mpi=pmix
    container_env_passthrough=['NCCL_SOCKET_IFNAME', 'NCCL_DEBUG'],  # NEW

    # Environment
    env_vars={
        "NEMORUN_HOME": "/mnt/logs",
        "NCCL_DEBUG": "VERSION",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    },

    work_dir='/mnt/workspace',
)

# The torchrun command with all its flags
job = xm.Job(
    executable=xm.Binary(path='torchrun'),
    args=[
        '--nproc-per-node=${GPUS_PER_NODE}',
        '--nnodes=${SLURM_NNODES}',
        '--node_rank=${SLURM_PROCID}',
        '--rdzv_id=${JOB_IDENTIFIER}',
        '--rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT}',
        '--rdzv-backend=static',
        '/mnt/workspace/pretrain.py',
        '--factory=configure_recipe(explicit_log_dir=/mnt/logs/${JOB_IDENTIFIER})',
        'trainer.num_nodes=${SLURM_NNODES}',
        'trainer.max_steps=15',
    ],
    executor=executor,
)
```

**Generated script will include:**
- Head node discovery ✅
- MASTER_ADDR/PORT exports ✅
- srun with `--mpi=pmix` ✅
- Container env passthrough ✅
- torchrun command with all flags ✅

## Implementation Checklist

### Files to Modify

1. **`xmanager/xm_local/executors.py`**
   - Add `master_port: int = 29500`
   - Add `setup_jax_coordinator: bool = False`
   - Add `use_mpi: bool = False`
   - Add `container_env_passthrough: List[str] = attr.Factory(list)`

2. **`xmanager/cloud/sbatch_templates.py`**
   - Replace `DEFAULT_SBATCH_TEMPLATE` with consolidated version
   - Add new template variables to `render()` method

3. **`xmanager/cloud/vertex_training_cluster.py`**
   - Update `_generate_sbatch_script()` to pass all template variables
   - Remove NEMORUN_HOME hardcoding
   - Add `save_to_storage()` call in launch

### New Template Variables

| Variable | Type | Purpose |
|----------|------|---------|
| `container_image` | str | Path to squashfs |
| `container_mounts` | str | Comma-separated mount string |
| `use_mpi` | bool | Add --mpi=pmix to srun |
| `container_env_passthrough` | list | Vars for --container-env |
| `master_port` | int | Rendezvous port |
| `setup_jax_coordinator` | bool | Export JAX_COORDINATOR_ADDRESS |

## Summary

The consolidated template handles both MaxText and NeMo by:
1. **Always** doing head node discovery
2. **Always** exporting MASTER_ADDR/PORT
3. **Optionally** setting up JAX coordinator (`setup_jax_coordinator=True`)
4. **Optionally** adding MPI support (`use_mpi=True`)
5. **Always** wrapping command in srun with pyxis when `container_image` is set
6. **Passing** container env vars via `container_env_passthrough`

The user just provides:
- The command to run (python/torchrun with args)
- Framework-specific env vars
- Container image and mounts
