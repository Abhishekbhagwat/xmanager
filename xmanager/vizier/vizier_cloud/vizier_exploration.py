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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
      metric_names: Optional[Union[str, List[str]]] = None,
      gcs_log_base: Optional[str] = None,
      param_to_arg_fn: Optional[Callable[[str, Any], str]] = None,
  ) -> None:
    """Create a VizierExploration.

    Args:
      experiment: the experiment who does the exploration.
      job: a job to run.
      study_factory: the VizierStudyFactory used to create or load the study.
      num_trials_total: total number of trials the experiment want to explore.
      num_parallel_trial_runs: number of parallel runs evaluating the trials.
      metric_names: Name(s) of the metric(s) to extract from TensorBoard logs.
        Can be a single string for backward compatibility (e.g.,
        'reduced_train_loss') or a list for multi-objective optimization (e.g.,
        ['learning/loss', 'perf/per_device_tflops_per_sec']). If provided along
        with gcs_log_base, enables GCS-based metric fetching for VTC jobs.
      gcs_log_base: Base GCS path for TensorBoard logs (e.g.,
        'gs://my-bucket'). The actual tensorboard directory is discovered
        automatically via GCS blob listing, so this works with any framework
        (MaxText, NeMo, etc.) regardless of directory structure.
      param_to_arg_fn: Optional function to convert Vizier parameter (name,
        value) pairs to command-line argument strings. Defaults to
        '{name}={value}' format. Use this for NeMo-style args like
        'data.micro_batch_size=4'.
    """
    # Normalize metric_names to list for internal handling
    if metric_names is None:
      self._metric_names = None
    elif isinstance(metric_names, str):
      self._metric_names = [metric_names]
    else:
      self._metric_names = list(metric_names)
    self._gcs_log_base = gcs_log_base
    self._param_to_arg_fn = param_to_arg_fn or (lambda n, v: f'{n}={v}')

    async def work_unit_generator(
        work_unit: xm.WorkUnit, vizier_params: Dict[str, Any]
    ):
      await work_unit.add(job, self._to_job_params(vizier_params))

    if not study_factory.display_name:
      study_factory.display_name = f'X{experiment.experiment_id}'

    # Create metric fetcher if GCS parameters are provided
    metric_fetcher = None
    metric_ids = None
    if self._metric_names and gcs_log_base:
      metric_fetcher = self._create_metric_fetcher()
      metric_ids = self._metric_names

    self._controller = vizier_controller.VizierController(
        experiment,
        work_unit_generator,
        study_factory.vz_client,
        study_factory.study(),
        num_trials_total,
        num_parallel_trial_runs,
        metric_fetcher=metric_fetcher,
        metric_ids=metric_ids,
    )

  def _create_metric_fetcher(
      self,
  ) -> Callable[[xm.WorkUnit], Optional[List[Tuple[int, Dict[str, float]]]]]:
    """Create a metric fetcher function for GCS-based metric reading.

    Uses GCS discovery to find TensorBoard log directories automatically,
    regardless of the framework-specific directory structure. Caches
    GCSMetricReader instances per slurm_job_id so that _last_step_reported
    is preserved across polls (only new metrics are returned each cycle).

    Returns:
      A callable that takes a WorkUnit and returns a list of
      (step, metrics_dict) tuples, or None if no metrics are found.
    """
    from xmanager.vizier.vizier_cloud import gcs_metric_reader

    _readers = {}  # Cache: slurm_job_id -> GCSMetricReader

    def fetch_metrics(
        work_unit: xm.WorkUnit,
    ) -> Optional[List[Tuple[int, Dict[str, float]]]]:
      slurm_job_id = _get_slurm_job_id(work_unit)
      if not slurm_job_id:
        logging.warning(
            'Could not find Slurm job ID for work unit %s. '
            'Cannot fetch metrics from GCS.',
            work_unit.work_unit_id,
        )
        return None

      if slurm_job_id not in _readers:
        # Discover TB path via GCS search (framework-agnostic)
        gcs_path = gcs_metric_reader.GCSMetricReader.discover_tensorboard_path(
            self._gcs_log_base, slurm_job_id
        )
        if not gcs_path:
          logging.warning(
              'No TB logs found for job %s under %s',
              slurm_job_id,
              self._gcs_log_base,
          )
          return None

        logging.info('Discovered TB path: %s', gcs_path)
        _readers[slurm_job_id] = gcs_metric_reader.GCSMetricReader(
            gcs_path=gcs_path,
            metric_names=self._metric_names,
        )

      metrics = _readers[slurm_job_id].get_new_metrics()
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
