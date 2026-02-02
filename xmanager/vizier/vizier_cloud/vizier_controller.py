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
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from google.cloud import aiplatform_v1beta1 as aip

from xmanager import xm

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

    self._work_unit_updaters = []
    # Counter incremented synchronously to avoid race condition with async
    # work unit creation. This tracks how many work units we've requested,
    # even if the async callbacks haven't completed yet.
    self._num_work_units_requested = 0

  def run(self, poll_frequency_in_sec: float = 60) -> None:
    """Peridically check and sync status between vizier and work units and create new work units when needed."""
    while True:
      # 1. Complete trial for completed work unit; Early stop first if needed.
      for work_unit_updater in self._work_unit_updaters:
        if not work_unit_updater.completed:
          work_unit_updater.check_for_completion()

      # 2. TODO: Return by Vizier's indication that study is done
      # when such API is ready on Vizier side.
      num_completed_work_units = sum(
          [wuu.completed for wuu in self._work_unit_updaters]
      )
      if (
          self._num_work_units_requested == self._num_work_units_total
          and num_completed_work_units == self._num_work_units_total
      ):
        print('All done! Exiting VizierController... \n')
        return

      # 3. Get new trials and assign to new work units.
      self._launch_new_work_units()

      # Use asyncio.sleep instead of time.sleep to allow the event loop
      # to process async callbacks (like experiment.add completion).
      # This is critical in notebook environments with nest_asyncio.
      try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(asyncio.sleep(poll_frequency_in_sec))
      except RuntimeError:
        # Fallback to time.sleep if no event loop available
        time.sleep(poll_frequency_in_sec)

  def _launch_new_work_units(self) -> None:
    """Get hyperparmeter suggestions from Vizier and assign to new work units to run."""
    # 1. Compute num of work units to create next.
    # Use _num_work_units_requested (sync counter) instead of len(_work_unit_updaters)
    # to avoid race condition with async work unit creation callbacks.
    num_existing_work_units = self._num_work_units_requested

    # Count "pending" work units (requested but async callback not yet completed)
    # These must be treated as "in flight" to avoid creating duplicates.
    num_pending_work_units = (
        self._num_work_units_requested - len(self._work_unit_updaters)
    )

    # Count confirmed work units that haven't completed yet
    # This includes both PENDING (queued in Slurm) and RUNNING jobs
    num_not_completed = len(
        [wuu for wuu in self._work_unit_updaters if not wuu.completed]
    )

    # Total "in flight" = pending (not yet confirmed) + not completed
    num_in_flight = num_pending_work_units + num_not_completed

    num_work_units_to_create_total = (
        self._num_work_units_total - num_existing_work_units
    )
    num_work_units_to_create_next = min(
        self._num_parallel_work_units - num_in_flight,
        num_work_units_to_create_total,
    )

    # 2. Create the work units.
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
      print(f'Trial for work unit (index: {i}) is retrieved：\n{trial}')

      print(f'Creating work unit (index: {i})... \n')

      def create_gen(index: int, trial: aip.Trial) -> xm.JobGeneratorType:
        async def gen_work_unit(work_unit: xm.WorkUnit, **kwargs):
          await self._work_unit_generator(work_unit, kwargs)

          # TODO: Add an utility to handle logging conditionally
          # (use print when run local otherwise logging.info.)
          print(
              f'Work unit (index: {index}, '
              f'id: {work_unit.work_unit_id}) created. \n'
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

        return gen_work_unit

      args = {
          'trial_name': trial.name,
          **{p.parameter_id: p.value for p in trial.parameters},
      }
      # Increment counter BEFORE experiment.add() to avoid race condition.
      # The async callback in create_gen populates _work_unit_updaters later,
      # but we need to track the request count synchronously.
      self._num_work_units_requested += 1
      self._experiment.add(create_gen(i, trial), args)


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
    self._vz_client = vz_client
    self._work_unit = work_unit
    self._trial = trial
    self._metric_fetcher = metric_fetcher
    self._metric_id = metric_id

  def work_unit_status(self) -> xm.ExperimentUnitStatus:
    return self._work_unit.get_status()

  def check_for_completion(self) -> None:
    """Sync the completion status between WorkUnit and Vizier Trial if needed."""
    if self.completed:
      return

    status = self.work_unit_status()

    # Check if job is running
    if status.is_active:
      print(f'Work unit {self._work_unit.work_unit_id} is running.\n')

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
        print(f'Early stopping work unit {self._work_unit.work_unit_id}.\n')
        self._work_unit.stop()
      return

    # Job is not active - check if PENDING or finished
    # Import here to avoid circular imports
    from xmanager.xm_local.status import LocalWorkUnitStatusEnum

    if (
        hasattr(status, '_status')
        and status._status == LocalWorkUnitStatusEnum.PENDING
    ):
      print(f'Work unit {self._work_unit.work_unit_id} is pending/queuing.\n')
      return

    # Job finished (COMPLETED, FAILED, CANCELLED, UNKNOWN)
    print(f'Work unit {self._work_unit.work_unit_id} has finished.\n')

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
        print(f'No metrics found for work unit {self._work_unit.work_unit_id}.\n')
        return False

      for step, value in metrics:
        self._report_measurement(step, value)

      print(
          f'Reported {len(metrics)} measurement(s) for work unit '
          f'{self._work_unit.work_unit_id}.\n'
      )
      return True
    except Exception as e:
      print(
          f'Error fetching metrics for work unit '
          f'{self._work_unit.work_unit_id}: {e}\n'
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
        print(
            f'Retrying metric fetch in {delay}s '
            f'(attempt {attempt + 2}/{max_retries})...\n'
        )
        time.sleep(delay)
    return False

  def _report_measurement(self, step: int, value: float) -> None:
    """Report a single measurement to Vizier.

    Args:
      step: The training step for this measurement.
      value: The metric value at this step.
    """
    self._vz_client.add_trial_measurement(
        request=aip.AddTrialMeasurementRequest(
            trial_name=self._trial.name,
            measurement=aip.Measurement(
                step_count=step,
                metrics=[
                    aip.Measurement.Metric(
                        metric_id=self._metric_id,
                        value=value,
                    )
                ],
            ),
        )
    )

  def _complete_trial(
      self, trial: aip.Trial, infeasible_reason: Optional[str] = None
  ) -> None:
    """Complete a trial."""
    self._vz_client.complete_trial(
        request=aip.CompleteTrialRequest(
            name=trial.name,
            trial_infeasible=infeasible_reason is not None,
            infeasible_reason=infeasible_reason,
        )
    )
    print(f'Trial {trial.name} is completed\n')
