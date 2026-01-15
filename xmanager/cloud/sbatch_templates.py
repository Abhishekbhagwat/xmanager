# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Jinja2-based SBATCH template rendering for SLURM job scripts."""

import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, StrictUndefined, Template


DEFAULT_SBATCH_TEMPLATE = r"""#!/bin/bash
#SBATCH --job-name={{ job_name }}
{% if partition is defined and partition %}
#SBATCH --partition={{ partition }}
{% endif %}
{% if account is defined and account %}
#SBATCH --account={{ account }}
{% endif %}
#SBATCH --nodes={{ num_nodes }}
{% if gpus_per_node is defined and gpus_per_node %}
#SBATCH --gpus-per-node={{ gpus_per_node }}
{% endif %}
#SBATCH --ntasks-per-node=1
{% if cpus_per_task is defined and cpus_per_task %}
#SBATCH --cpus-per-task={{ cpus_per_task }}
{% endif %}
{% if time_limit is defined and time_limit %}
#SBATCH --time={{ time_limit }}
{% endif %}
{% if exclusive is defined and exclusive %}
#SBATCH --exclusive
{% endif %}
#SBATCH --mem=0
{% if output_file is defined and output_file %}
#SBATCH --output={{ output_file }}
{% endif %}
{% if error_file is defined and error_file %}
#SBATCH --error={{ error_file }}
{% endif %}
{% if sbatch_flags is defined and sbatch_flags %}
{% for key, value in sbatch_flags.items() %}
{% if value is none or value == true %}
#SBATCH --{{ key }}
{% else %}
#SBATCH --{{ key }}={{ value }}
{% endif %}
{% endfor %}
{% endif %}

set -x
set -e

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
export GPUS_PER_NODE={{ gpus_per_node | default(8) }}

{% if setup_jax_coordinator is defined and setup_jax_coordinator %}
# JAX Distributed Setup
export JAX_COORDINATOR_ADDRESS="${MASTER_ADDR}:${MASTER_PORT}"
echo "JAX_COORDINATOR_ADDRESS: $JAX_COORDINATOR_ADDRESS"
{% endif %}

# =============================================================================
# NCCL CONFIGURATION
# =============================================================================
{% if nccl_setup_script is defined and nccl_setup_script %}
echo "Setting up NCCL..."
source {{ nccl_setup_script | shell_quote }}
{% endif %}
{% if nccl_lib_path is defined and nccl_lib_path %}
export LD_LIBRARY_PATH={{ nccl_lib_path | shell_quote }}:${LD_LIBRARY_PATH:-}
{% endif %}
{% if nccl_env_vars is defined and nccl_env_vars %}
{% for key, value in nccl_env_vars.items() %}
export {{ key }}={{ value | shell_quote }}
{% endfor %}
{% endif %}

# =============================================================================
# USER ENVIRONMENT VARIABLES
# =============================================================================
{% if env_vars is defined and env_vars %}
{% for key, value in env_vars.items() %}
export {{ key }}={{ value | shell_quote }}
{% endfor %}
{% endif %}

# =============================================================================
# PROLOGUE COMMANDS
# =============================================================================
{% if prologue_commands is defined and prologue_commands %}
{% for cmd in prologue_commands %}
{{ cmd }}
{% endfor %}
{% endif %}

# =============================================================================
# JOB IDENTIFIER
# =============================================================================
TIME=$(TZ="America/Los_Angeles" date +"%Y%m%d_%H%M%S")
export JOB_IDENTIFIER="{{ job_name }}-${SLURM_NNODES}n-${TIME}"
echo "JOB_IDENTIFIER: $JOB_IDENTIFIER"

{% if working_dir is defined and working_dir %}
cd {{ working_dir | shell_quote }}
{% endif %}

# =============================================================================
# MAIN EXECUTION
# =============================================================================
{% if container_image is defined and container_image %}
echo "Launching container..."
CONTAINER_MOUNTS="{{ container_mounts | default('') }}"

srun \
  --job-name=${JOB_IDENTIFIER} \
  --nodes=${SLURM_NNODES} \
  --container-image={{ container_image | shell_quote }} \
  --container-mounts=${CONTAINER_MOUNTS} \
  --no-container-mount-home \
  --container-writable \
{% if use_mpi is defined and use_mpi %}
  --mpi=pmix \
{% endif %}
{% if container_env_passthrough is defined and container_env_passthrough %}
  --container-env={{ container_env_passthrough | join(',') }} \
{% endif %}
  bash -c "
set -x
set -e
{% if nccl_lib_path is defined and nccl_lib_path %}
export LD_LIBRARY_PATH={{ nccl_lib_path | shell_quote }}:\${LD_LIBRARY_PATH:-}
{% endif %}
echo 'Container started on node rank '\${SLURM_PROCID}' of '\${SLURM_NNODES}
{{ command }}
echo 'Container finished on '\$(hostname)
"
{% else %}
# Direct execution (no container)
{{ command }}
{% endif %}

# =============================================================================
# EPILOGUE
# =============================================================================
{% if epilogue_commands is defined and epilogue_commands %}
{% for cmd in epilogue_commands %}
{{ cmd }}
{% endfor %}
{% endif %}

echo "========================================================"
echo "Job completed at $(date)"
echo "========================================================"
"""


class SbatchTemplateRenderer:
    """Renders SBATCH job scripts using Jinja2 templates."""

    def __init__(
        self,
        custom_template: Optional[str] = None,
        template_path: Optional[Path] = None,
    ):
        """Initialize the SBATCH template renderer.

        Args:
            custom_template: Custom Jinja2 template string to use instead of default.
            template_path: Path to a Jinja2 template file to load.

        Raises:
            ValueError: If both custom_template and template_path are provided.
        """
        if custom_template and template_path:
            raise ValueError(
                "Cannot specify both custom_template and template_path"
            )

        # Create Jinja2 environment with strict undefined checking
        self.env = Environment(
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Add shell_quote filter for safe shell escaping
        self.env.filters["shell_quote"] = shlex.quote

        # Load template
        if template_path:
            with open(template_path, "r") as f:
                template_string = f.read()
        elif custom_template:
            template_string = custom_template
        else:
            template_string = DEFAULT_SBATCH_TEMPLATE

        self.template: Template = self.env.from_string(template_string)

    def render(self, **variables: Any) -> str:
        """Render the SBATCH script with the given variables.

        Args:
            **variables: Template variables to render.

        Returns:
            Rendered SBATCH script as a string.

        Raises:
            jinja2.UndefinedError: If a required template variable is missing.
        """
        return self.template.render(**variables)


def render_sbatch_for_cluster(
    cluster_config: Any,
    job_name: str,
    num_nodes: int,
    command: str,
    partition: Optional[str] = None,
    account: Optional[str] = None,
    time_limit: Optional[str] = None,
    working_dir: Optional[str] = None,
    env_vars: Optional[Dict[str, str]] = None,
    sbatch_flags: Optional[Dict[str, Any]] = None,
    prologue_commands: Optional[List[str]] = None,
    epilogue_commands: Optional[List[str]] = None,
    output_file: Optional[str] = None,
    error_file: Optional[str] = None,
    master_port: Optional[int] = None,
    container_image: Optional[str] = None,
    container_mounts: Optional[str] = None,
    use_mpi: bool = False,
    container_env_passthrough: Optional[List[str]] = None,
    setup_jax_coordinator: bool = False,
) -> str:
    """Convenience function to render an SBATCH script for a cluster.

    Args:
        cluster_config: ClusterConfig object with cluster settings.
        job_name: Name of the SLURM job.
        num_nodes: Number of nodes to allocate.
        command: Command to execute in the job.
        partition: SLURM partition to use.
        account: SLURM account to charge.
        time_limit: Time limit for the job (e.g., "1:00:00").
        working_dir: Working directory for the job.
        env_vars: Dictionary of environment variables to set.
        sbatch_flags: Additional SBATCH flags as key-value pairs.
        prologue_commands: Commands to run before the main command.
        epilogue_commands: Commands to run after the main command.
        output_file: Path for stdout output file.
        error_file: Path for stderr output file.
        master_port: Port for distributed training master.
        container_image: Path to container image (.sqsh or docker path).
        container_mounts: Comma-separated mount string for srun.
        use_mpi: Whether to add --mpi=pmix to srun.
        container_env_passthrough: List of env vars to pass via --container-env.
        setup_jax_coordinator: Whether to export JAX_COORDINATOR_ADDRESS.

    Returns:
        Rendered SBATCH script as a string.
    """
    renderer = SbatchTemplateRenderer()

    # Build template variables from cluster config
    variables = {
        "job_name": job_name,
        "num_nodes": num_nodes,
        "command": command,
    }

    # Add cluster config attributes if available
    if hasattr(cluster_config, "gpus_per_node"):
        variables["gpus_per_node"] = cluster_config.gpus_per_node
    if hasattr(cluster_config, "ntasks_per_node"):
        variables["ntasks_per_node"] = cluster_config.ntasks_per_node
    if hasattr(cluster_config, "cpus_per_task"):
        variables["cpus_per_task"] = cluster_config.cpus_per_task
    if hasattr(cluster_config, "exclusive"):
        variables["exclusive"] = cluster_config.exclusive
    if hasattr(cluster_config, "nccl_setup_script"):
        variables["nccl_setup_script"] = cluster_config.nccl_setup_script
    if hasattr(cluster_config, "nccl_lib_path"):
        variables["nccl_lib_path"] = cluster_config.nccl_lib_path
    if hasattr(cluster_config, "nccl_env_vars"):
        variables["nccl_env_vars"] = cluster_config.nccl_env_vars

    # Add optional parameters if provided
    if partition:
        variables["partition"] = partition
    if account:
        variables["account"] = account
    if time_limit:
        variables["time_limit"] = time_limit
    if working_dir:
        variables["working_dir"] = working_dir
    if env_vars:
        variables["env_vars"] = env_vars
    if sbatch_flags:
        variables["sbatch_flags"] = sbatch_flags
    if prologue_commands:
        variables["prologue_commands"] = prologue_commands
    if epilogue_commands:
        variables["epilogue_commands"] = epilogue_commands
    if output_file:
        variables["output_file"] = output_file
    if error_file:
        variables["error_file"] = error_file
    if master_port:
        variables["master_port"] = master_port

    # Container and distributed training options
    if container_image:
        variables["container_image"] = container_image
    if container_mounts:
        variables["container_mounts"] = container_mounts
    if use_mpi:
        variables["use_mpi"] = use_mpi
    if container_env_passthrough:
        variables["container_env_passthrough"] = container_env_passthrough
    if setup_jax_coordinator:
        variables["setup_jax_coordinator"] = setup_jax_coordinator

    return renderer.render(**variables)
