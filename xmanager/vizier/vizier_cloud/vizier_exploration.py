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
"""Interface for launching Vizier Explorations using Vertex Vizier."""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from xmanager import xm
from xmanager.vizier.vizier_cloud import study_factory as sf
from xmanager.vizier.vizier_cloud import vizier_controller

_DEFAULT_LOCATION = 'us-central1'


def _get_slurm_job_id(work_unit: xm.WorkUnit) -> Optional[str]:
  """Extract Slurm job ID from work unit's VTC handles.

  Args:
    work_unit: The work unit to extract the Slurm job ID from.

  Returns:
    The Slurm job ID as a string, or None if not found.
  """
  # Use public method if available, fallback to private attribute for compatibility
  if hasattr(work_unit, 'get_execution_handles'):
    handles = work_unit.get_execution_handles()
  else:
    handles = getattr(work_unit, '_non_local_execution_handles', [])

  for handle in handles:
    if hasattr(handle, 'slurm_job_id'):
      return handle.slurm_job_id
  return None


# TODO: Add vizier_controller as auxiliary Job generator.
class VizierExploration:
  """An API for launching experiment as a Vizier-based Exploration."""

  def __init__(
      self,
      experiment: xm.Experiment,
      job: xm.JobType,
      study_factory: sf.StudyFactory,
      num_trials_total: int,
      num_parallel_trial_runs: int,
      # GCS metric fetching parameters (for VTC jobs)
      metric_name: Optional[str] = None,
      gcs_log_base: Optional[str] = None,
      cluster_id: Optional[str] = None,
      tensorboard_path_fn: Optional[Callable[[str, str], str]] = None,
      param_to_arg_fn: Optional[Callable[[str, Any], str]] = None,
  ) -> None:
    """Create a VizierExploration.

    Args:
      experiment: the experiment who does the exploration.
      job: a job to run.
      study_factory: the VizierStudyFactory used to create or load the study.
      num_trials_total: total number of trials the experiment want to explore.
      num_parallel_trial_runs: number of parallel runs evaluating the trials.
      metric_name: Name of the metric to extract from TensorBoard logs (e.g.,
        'reduced_train_loss'). If provided along with gcs_log_base, enables
        GCS-based metric fetching for VTC jobs.
      gcs_log_base: Base GCS path for TensorBoard logs (e.g.,
        'gs://my-bucket'). Used with tensorboard_path_fn to construct full path.
      cluster_id: Cluster identifier for constructing GCS paths (NeMo pattern).
      tensorboard_path_fn: Optional function to construct the tensorboard path.
        Takes (gcs_log_base, slurm_job_id) and returns the full GCS path.
        Defaults to NeMo pattern if cluster_id provided, MaxText pattern otherwise.
        Example for MaxText: lambda base, job_id: f'{base}/job-{job_id}/tensorboard/job-{job_id}/'
        Example for NeMo: lambda base, job_id: f'{base}/{cluster_id}/tensorboard/job-{job_id}/'
      param_to_arg_fn: Optional function to convert Vizier parameter (name,
        value) pairs to command-line argument strings. Defaults to
        '{name}={value}' format. Use this for NeMo-style args like
        'data.micro_batch_size=4'.
    """
    self._metric_name = metric_name
    self._gcs_log_base = gcs_log_base
    self._cluster_id = cluster_id
    self._tensorboard_path_fn = tensorboard_path_fn
    self._param_to_arg_fn = param_to_arg_fn or (lambda n, v: f'{n}={v}')

    async def work_unit_generator(
        work_unit: xm.WorkUnit, vizier_params: Dict[str, Any]
    ):
      await work_unit.add(job, self._to_job_params(vizier_params))

    if not study_factory.display_name:
      study_factory.display_name = f'X{experiment.experiment_id}'

    # Create metric fetcher if GCS parameters are provided
    metric_fetcher = None
    metric_id = None
    if metric_name and gcs_log_base:
      metric_fetcher = self._create_metric_fetcher()
      metric_id = metric_name

    self._controller = vizier_controller.VizierController(
        experiment,
        work_unit_generator,
        study_factory.vz_client,
        study_factory.study(),
        num_trials_total,
        num_parallel_trial_runs,
        metric_fetcher=metric_fetcher,
        metric_id=metric_id,
    )

  def _create_metric_fetcher(
      self,
  ) -> Callable[[xm.WorkUnit], Optional[List[Tuple[int, float]]]]:
    """Create a metric fetcher function for GCS-based metric reading.

    Returns:
      A callable that takes a WorkUnit and returns a list of (step, value)
      tuples, or None if no metrics are found.
    """
    from xmanager.vizier.vizier_cloud import gcs_metric_reader

    def fetch_metrics(
        work_unit: xm.WorkUnit,
    ) -> Optional[List[Tuple[int, float]]]:
      # Get Slurm job ID from work unit handles
      slurm_job_id = _get_slurm_job_id(work_unit)
      if not slurm_job_id:
        logging.warning(
            'Could not find Slurm job ID for work unit %s. '
            'Cannot fetch metrics from GCS.',
            work_unit.work_unit_id,
        )
        return None

      # Construct GCS path using custom function or default patterns
      if self._tensorboard_path_fn:
        gcs_path = self._tensorboard_path_fn(self._gcs_log_base, slurm_job_id)
      elif self._cluster_id:
        # NeMo pattern: {gcs_log_base}/{cluster_id}/tensorboard/job-{slurm_job_id}/
        gcs_path = (
            f'{self._gcs_log_base}/{self._cluster_id}/'
            f'tensorboard/job-{slurm_job_id}/'
        )
      else:
        # MaxText pattern: {gcs_log_base}/job-{slurm_job_id}/tensorboard/job-{slurm_job_id}/
        gcs_path = (
            f'{self._gcs_log_base}/job-{slurm_job_id}/'
            f'tensorboard/job-{slurm_job_id}/'
        )

      logging.info('Fetching metrics from: %s', gcs_path)

      reader = gcs_metric_reader.GCSMetricReader(
          gcs_path=gcs_path,
          metric_name=self._metric_name,
      )

      # Get all metrics (for full measurement history)
      metrics = reader.get_all_metrics()
      return metrics if metrics else None

    return fetch_metrics

  def _to_job_params(self, vizier_params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Vizier parameters to job parameters.

    For VTC jobs with MaxText/NeMo, we need to convert parameters to
    command-line arguments in the format expected (e.g., 'learning_rate=0.001').
    These are wrapped in ShellSafeArg to prevent the '--' prefix from being
    added by merge_args.

    Args:
      vizier_params: Dictionary of parameter_id -> value from Vizier.

    Returns:
      Dictionary with 'args' key containing ShellSafeArg arguments.
    """
    # Convert parameters to ShellSafeArg with 'key=value' format
    param_args = []
    for name, value in vizier_params.items():
      if name == 'trial_name':
        continue
      # Use the param_to_arg_fn to format, then wrap in ShellSafeArg
      arg_str = self._param_to_arg_fn(name, value)
      param_args.append(xm.ShellSafeArg(arg_str))

    return {'args': param_args}

  def launch(self, **kwargs) -> None:
    self._controller.run(**kwargs)
