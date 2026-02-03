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
"""Main code that the Vertex Cloud Vizier Controller runs."""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from google.cloud import aiplatform_v1beta1 as aip

from xmanager import xm
from xmanager.vizier.vizier_cloud import vizier_worker

# Type alias for metric fetcher callback
# Takes a work unit and returns Optional[(step, value)] or list of (step, value)
MetricFetcherType = Callable[[xm.WorkUnit], Optional[List[Tuple[int, float]]]]


class VizierController:
  """A Controller that runs Vizier suggested hyperparameters in multiple work units."""

  def __init__(
      self,
      experiment: xm.Experiment,
      work_unit_generator: Callable[[xm.WorkUnit, Dict[str, Any]], Any],
      vz_client: aip.VizierServiceClient,
      study_name: str,
      num_work_units_total: int,
      num_parallel_work_units: int,
      metric_fetcher: Optional[MetricFetcherType] = None,
      metric_id: Optional[str] = None,
  ) -> None:
    """Create a VizierController.

    Args:
      experiment: XM experiment.
      work_unit_generator: the function that generates WorkUnit from
        hyperparameters.
      vz_client: the Vizier Client used for interacting with Vizier.
      study_name: the study name the controller works on.
      num_work_units_total: number of work units to create in total. (TODO:
        remove this and retrieve from study spec stopping criteria once it is
        settable there.)
      num_parallel_work_units: number of work units to run in parallel.
      metric_fetcher: Optional callback to fetch metrics from external sources
        (e.g., GCS TensorBoard logs). Takes a WorkUnit and returns a list of
        (step, value) tuples, or None if no metrics available.
      metric_id: The metric ID to use when reporting measurements to Vizier.
        Required if metric_fetcher is provided.
    """
    self._experiment = experiment
    self._work_unit_generator = work_unit_generator
    self._vz_client = vz_client
    self._study_name = study_name
    self._num_work_units_total = num_work_units_total
    self._num_parallel_work_units = num_parallel_work_units
    self._metric_fetcher = metric_fetcher
    self._metric_id = metric_id

    self._work_unit_updaters: List[WorkUnitVizierUpdater] = []

  def run(self, poll_frequency_in_sec: float = 60) -> None:
    """Periodically check and sync status between vizier and work units."""
    # Use the experiment's event loop to avoid loop mismatch.
    # The Experiment class runs its own event loop in a background thread,
    # so we schedule our async work there using run_coroutine_threadsafe.
    loop = self._experiment._event_loop
    future = asyncio.run_coroutine_threadsafe(
        self._run_async(poll_frequency_in_sec), loop
    )
    # Block until the async work completes
    future.result()

  async def _run_async(self, poll_frequency_in_sec: float) -> None:
    """Async implementation of the run loop."""
    while True:
      # 1. Complete trial for completed work unit; Early stop first if needed.
      for work_unit_updater in self._work_unit_updaters:
        if not work_unit_updater.completed:
          work_unit_updater.check_for_completion()

      # 2. Check if all work units are done
      num_existing_work_units = len(self._work_unit_updaters)
      num_completed_work_units = sum(
          [wuu.completed for wuu in self._work_unit_updaters]
      )
      if (
          num_existing_work_units == self._num_work_units_total
          and num_completed_work_units == self._num_work_units_total
      ):
        logging.info('All done! Exiting VizierController...')
        return

      # 3. Get new trials and assign to new work units.
      await self._launch_new_work_units()

      # Sleep to allow event loop processing
      await asyncio.sleep(poll_frequency_in_sec)

  async def _launch_new_work_units(self) -> None:
    """Get hyperparameter suggestions from Vizier and assign to new work units."""
    # Count existing and in-flight work units
    num_existing_work_units = len(self._work_unit_updaters)
    num_not_completed = len(
        [wuu for wuu in self._work_unit_updaters if not wuu.completed]
    )

    num_work_units_to_create_total = (
        self._num_work_units_total - num_existing_work_units
    )
    num_work_units_to_create_next = min(
        self._num_parallel_work_units - num_not_completed,
        num_work_units_to_create_total,
    )

    if num_work_units_to_create_next <= 0:
      return

    # Create work units sequentially, awaiting each one
    start_index = num_existing_work_units + 1
    for i in range(start_index, start_index + num_work_units_to_create_next):
      trial = (
          self._vz_client.suggest_trials(
              request=aip.SuggestTrialsRequest(
                  parent=self._study_name,
                  suggestion_count=1,
                  client_id=f'work unit {i}',
              )
          )
          .result()
          .trials[0]
      )
      logging.info('Trial for work unit (index: %d) is retrieved: %s', i, trial)
      logging.info('Creating work unit (index: %d)...', i)

      # Create work unit and await completion
      work_unit = await self._create_work_unit(i, trial)

      logging.info(
          'Work unit (index: %d, id: %s) created.',
          i,
          work_unit.work_unit_id,
      )

      self._work_unit_updaters.append(
          WorkUnitVizierUpdater(
              vz_client=self._vz_client,
              work_unit=work_unit,
              trial=trial,
              metric_fetcher=self._metric_fetcher,
              metric_id=self._metric_id,
          )
      )

  async def _create_work_unit(
      self, index: int, trial: aip.Trial
  ) -> xm.WorkUnit:
    """Create a single work unit for a trial.

    Args:
      index: The work unit index.
      trial: The Vizier trial with suggested parameters.

    Returns:
      The created WorkUnit.
    """
    args = {
        'trial_name': trial.name,
        **{p.parameter_id: p.value for p in trial.parameters},
    }

    async def gen_work_unit(work_unit: xm.WorkUnit, **kwargs):
      await self._work_unit_generator(work_unit, kwargs)

    # Await the experiment.add() to ensure work unit is fully created
    work_unit = await self._experiment.add(gen_work_unit, args)
    return work_unit


class WorkUnitVizierUpdater:
  """An updater for syncing completion state between work unit and vizier trial."""

  def __init__(
      self,
      vz_client: aip.VizierServiceClient,
      work_unit: xm.WorkUnit,
      trial: aip.Trial,
      metric_fetcher: Optional[MetricFetcherType] = None,
      metric_id: Optional[str] = None,
  ) -> None:
    self.completed = False
    self._vz_client = vz_client  # Still needed for early stopping check
    self._work_unit = work_unit
    self._trial = trial
    self._metric_fetcher = metric_fetcher
    self._metric_id = metric_id

    # Use VizierWorker for metric reporting and trial completion
    self._worker = vizier_worker.VizierWorker(trial.name)

  def work_unit_status(self) -> xm.ExperimentUnitStatus:
    return self._work_unit.get_status()

  def _is_pending(self, status: xm.ExperimentUnitStatus) -> bool:
    """Check if the status indicates a pending/queued state.

    Args:
      status: The experiment unit status.

    Returns:
      True if the job is pending (queued but not yet running).
    """
    # Use public is_pending property if available
    if hasattr(status, 'is_pending'):
      return status.is_pending
    # Fallback: if not active and not completed/failed, assume pending
    return not status.is_active

  def check_for_completion(self) -> None:
    """Sync the completion status between WorkUnit and Vizier Trial if needed."""
    if self.completed:
      return

    status = self.work_unit_status()

    # Check if job is running
    if status.is_active:
      logging.info('Work unit %s is running.', self._work_unit.work_unit_id)

      # Fetch intermediate metrics for early stopping / progress
      if self._metric_fetcher and self._metric_id:
        self._fetch_and_report_metrics()

      # Check for early stopping from Vizier
      if (
          self._vz_client.check_trial_early_stopping_state(
              request=aip.CheckTrialEarlyStoppingStateRequest(
                  trial_name=self._trial.name
              )
          )
          .result()
          .should_stop
      ):
        logging.info(
            'Early stopping work unit %s.', self._work_unit.work_unit_id
        )
        self._work_unit.stop()
      return

    # Check if job is pending (queued but not running)
    if self._is_pending(status):
      logging.info(
          'Work unit %s is pending/queuing.', self._work_unit.work_unit_id
      )
      return

    # Job finished (COMPLETED, FAILED, CANCELLED, UNKNOWN)
    logging.info('Work unit %s has finished.', self._work_unit.work_unit_id)

    # Fetch final metrics with retry
    metrics_reported = False
    if self._metric_fetcher and self._metric_id:
      metrics_reported = self._fetch_metrics_with_retry()

    # Complete the trial
    if metrics_reported:
      self._complete_trial(self._trial)
    else:
      self._complete_trial(
          self._trial, infeasible_reason='No metrics available from job output'
      )
    self.completed = True

  def _fetch_and_report_metrics(self) -> bool:
    """Fetch metrics from external source and report to Vizier.

    Returns:
        True if at least one metric was reported.
    """
    try:
      metrics = self._metric_fetcher(self._work_unit)
      if not metrics:
        logging.info(
            'No metrics found for work unit %s.', self._work_unit.work_unit_id
        )
        return False

      for step, value in metrics:
        self._report_measurement(step, value)

      logging.info(
          'Reported %d measurement(s) for work unit %s.',
          len(metrics),
          self._work_unit.work_unit_id,
      )
      return True
    except Exception as e:
      logging.warning(
          'Error fetching metrics for work unit %s: %s',
          self._work_unit.work_unit_id,
          e,
      )
      return False

  def _fetch_metrics_with_retry(
      self, max_retries: int = 3, delay: float = 30.0
  ) -> bool:
    """Fetch metrics with retry for GCS sync delays.

    Args:
        max_retries: Number of attempts to fetch metrics.
        delay: Seconds to wait between retries.

    Returns:
        True if at least one metric was reported.
    """
    for attempt in range(max_retries):
      if self._fetch_and_report_metrics():
        return True
      if attempt < max_retries - 1:
        logging.info(
            'Retrying metric fetch in %.0fs (attempt %d/%d)...',
            delay,
            attempt + 2,
            max_retries,
        )
        time.sleep(delay)
    return False

  def _report_measurement(self, step: int, value: float) -> None:
    """Report a single measurement to Vizier.

    Args:
      step: The training step for this measurement.
      value: The metric value at this step.
    """
    self._worker.add_trial_measurement(step, {self._metric_id: value})

  def _complete_trial(
      self, trial: aip.Trial, infeasible_reason: Optional[str] = None
  ) -> None:
    """Complete a trial."""
    del trial  # Not needed - worker already has trial name
    self._worker.complete_trial(infeasible_reason)
