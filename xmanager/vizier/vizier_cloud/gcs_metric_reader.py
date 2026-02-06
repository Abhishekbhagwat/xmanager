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
"""GCS-based metric reader for TensorBoard event files.

This module provides functionality to read TensorBoard event files from GCS
and extract scalar metrics for use with Vizier hyperparameter optimization.
"""

import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple, Union

from google.cloud import storage


class GCSMetricReader:
  """Reads metrics from TensorBoard event files stored on GCS.

  This class is designed to work with NeMo/PyTorch training jobs that write
  TensorBoard logs to GCS. It downloads event files and extracts scalar
  metrics for reporting to Vizier.

  Supports both single metric and multiple metric extraction for multi-objective
  optimization scenarios.

  Example usage:
    # Single metric
    reader = GCSMetricReader(
        gcs_path='gs://my-bucket/tensorboard/job-123/',
        metric_names='reduced_train_loss',
    )
    result = reader.get_latest_metric()
    if result:
        step, metrics = result
        print(f"Latest metric at step {step}: {metrics}")

    # Multiple metrics
    reader = GCSMetricReader(
        gcs_path='gs://my-bucket/tensorboard/job-123/',
        metric_names=['learning/loss', 'perf/per_device_tflops_per_sec'],
    )
    result = reader.get_latest_metric()
    if result:
        step, metrics = result
        print(f"Latest metrics at step {step}: {metrics}")
  """

  def __init__(
      self,
      gcs_path: str,
      metric_names: Union[str, List[str]] = 'reduced_train_loss',
  ) -> None:
    """Initialize the GCS metric reader.

    Args:
      gcs_path: GCS path to TensorBoard logs (e.g., 'gs://bucket/path/').
      metric_names: Name(s) of the scalar metric(s) to extract. Can be a single
        string for backward compatibility or a list of metric names for
        multi-objective optimization.
    """
    self._gcs_path = gcs_path.rstrip('/')
    # Normalize to list for internal handling
    if isinstance(metric_names, str):
      self._metric_names = [metric_names]
    else:
      self._metric_names = list(metric_names)
    self._storage_client = None  # Lazy initialization
    self._last_step_reported = -1

  def _get_storage_client(self) -> storage.Client:
    """Get or create the storage client (lazy initialization)."""
    if self._storage_client is None:
      self._storage_client = storage.Client()
    return self._storage_client

  def _parse_gcs_path(self) -> Tuple[str, str]:
    """Parse GCS path into bucket name and prefix.

    Returns:
      Tuple of (bucket_name, prefix).

    Raises:
      ValueError: If the GCS path is invalid.
    """
    if not self._gcs_path.startswith('gs://'):
      raise ValueError(f"Invalid GCS path: {self._gcs_path}. Must start with 'gs://'")

    path = self._gcs_path[5:]  # Remove 'gs://'
    parts = path.split('/', 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ''
    return bucket_name, prefix

  @staticmethod
  def _parse_gcs_path_static(gcs_path: str) -> Tuple[str, str]:
    """Parse a GCS path into bucket name and prefix.

    Args:
      gcs_path: GCS path (e.g., 'gs://my-bucket/some/prefix').

    Returns:
      Tuple of (bucket_name, prefix).

    Raises:
      ValueError: If the GCS path is invalid.
    """
    if not gcs_path.startswith('gs://'):
      raise ValueError(
          f"Invalid GCS path: {gcs_path}. Must start with 'gs://'"
      )
    path = gcs_path.rstrip('/')[5:]  # Remove 'gs://'
    parts = path.split('/', 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ''
    return bucket_name, prefix

  @classmethod
  def discover_tensorboard_path(
      cls, gcs_base: str, slurm_job_id: str
  ) -> Optional[str]:
    """Discover TensorBoard log directory by searching GCS.

    Searches for event files matching */tensorboard/job-{slurm_job_id}/*
    under the given GCS base path. Framework-agnostic -- works with any
    directory structure (MaxText, NeMo, or VMDS-provided paths).

    Args:
      gcs_base: Base GCS path (e.g., 'gs://my-bucket').
      slurm_job_id: Slurm job ID to search for.

    Returns:
      Full GCS path to the TB directory, or None if not found.
    """
    client = storage.Client()
    bucket_name, prefix = cls._parse_gcs_path_static(gcs_base)
    bucket = client.bucket(bucket_name)

    search_prefix = prefix
    target_pattern = f'tensorboard/job-{slurm_job_id}/'

    blobs = bucket.list_blobs(prefix=search_prefix, max_results=1000)
    for blob in blobs:
      if (
          target_pattern in blob.name
          and 'events.out.tfevents.' in blob.name
      ):
        # Extract the directory path (everything before the event filename)
        idx = blob.name.index(target_pattern) + len(target_pattern)
        dir_path = blob.name[:idx]
        return f'gs://{bucket_name}/{dir_path}'

    return None

  def _list_event_files(self) -> List[str]:
    """List TensorBoard event files in the GCS path.

    Returns:
      List of blob names for event files.
    """
    bucket_name, prefix = self._parse_gcs_path()
    client = self._get_storage_client()
    bucket = client.bucket(bucket_name)

    event_files = []
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
      # TensorBoard event files match pattern: events.out.tfevents.*
      if 'events.out.tfevents.' in blob.name:
        event_files.append(blob.name)

    return sorted(event_files)

  def _download_event_files(self, temp_dir: str) -> List[str]:
    """Download event files from GCS to a temporary directory.

    Args:
      temp_dir: Local directory to download files to.

    Returns:
      List of local file paths.
    """
    bucket_name, prefix = self._parse_gcs_path()
    client = self._get_storage_client()
    bucket = client.bucket(bucket_name)

    event_files = self._list_event_files()
    local_files = []

    for blob_name in event_files:
      blob = bucket.blob(blob_name)
      # Preserve directory structure relative to prefix
      relative_path = blob_name[len(prefix):].lstrip('/')
      local_path = os.path.join(temp_dir, relative_path)

      # Create parent directories
      os.makedirs(os.path.dirname(local_path), exist_ok=True)

      blob.download_to_filename(local_path)
      local_files.append(local_path)

    return local_files

  def _parse_events(self, log_dir: str) -> List[Tuple[int, Dict[str, float]]]:
    """Parse TensorBoard events and extract the specified metrics.

    Args:
      log_dir: Local directory containing event files.

    Returns:
      List of (step, metrics_dict) tuples where metrics_dict maps metric names
      to their values at that step.
    """
    try:
      from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
      raise ImportError(
          "tensorboard is required for parsing event files. "
          "Install it with: pip install tensorboard"
      )

    # Create event accumulator
    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={
            event_accumulator.SCALARS: 0,  # Load all scalars
        },
    )
    ea.Reload()

    # Get available scalar tags
    available_tags = ea.Tags().get('scalars', [])

    # Find matching tags for each requested metric
    metric_to_tag = {}
    for metric_name in self._metric_names:
      matching_tag = None
      for tag in available_tags:
        if tag == metric_name:
          matching_tag = tag
          break
        elif metric_name in tag:
          matching_tag = tag
          # Don't break - prefer exact match

      if matching_tag:
        metric_to_tag[metric_name] = matching_tag
      else:
        logging.warning(
            "Metric '%s' not found. Available metrics: %s",
            metric_name,
            available_tags,
        )

    if not metric_to_tag:
      return []

    # Extract scalar values for all metrics, grouped by step
    step_to_metrics: Dict[int, Dict[str, float]] = {}
    for metric_name, tag in metric_to_tag.items():
      scalars = ea.Scalars(tag)
      for s in scalars:
        if s.step not in step_to_metrics:
          step_to_metrics[s.step] = {}
        step_to_metrics[s.step][metric_name] = s.value

    # For multi-objective optimization, only return steps where ALL requested
    # metrics are present. Vizier rejects partial measurements.
    num_expected_metrics = len(self._metric_names)
    complete_steps = [
        (step, metrics)
        for step, metrics in sorted(step_to_metrics.items())
        if len(metrics) == num_expected_metrics
    ]

    if len(complete_steps) < len(step_to_metrics):
      logging.info(
          'Filtered %d incomplete steps (missing some metrics). '
          'Returning %d complete steps.',
          len(step_to_metrics) - len(complete_steps),
          len(complete_steps),
      )

    return complete_steps

  def get_latest_metric(self) -> Optional[Tuple[int, Dict[str, float]]]:
    """Get the latest metric values from TensorBoard events.

    Downloads event files from GCS, parses them, and returns the most
    recent values for the configured metrics.

    Returns:
      Tuple of (step, metrics_dict) for the latest step, or None if not found.
      metrics_dict maps metric names to their values.
    """
    try:
      with tempfile.TemporaryDirectory() as temp_dir:
        # Download event files
        local_files = self._download_event_files(temp_dir)
        if not local_files:
          logging.info("No event files found at %s", self._gcs_path)
          return None

        # Parse events
        metrics = self._parse_events(temp_dir)
        if not metrics:
          return None

        # Return the latest (highest step) metric
        return max(metrics, key=lambda x: x[0])

    except Exception as e:
      logging.warning("Error reading metrics from GCS: %s", e)
      return None

  def get_all_metrics(self) -> List[Tuple[int, Dict[str, float]]]:
    """Get all metric values from TensorBoard events.

    This is useful for reporting intermediate measurements to Vizier
    for early stopping decisions.

    Returns:
      List of (step, metrics_dict) tuples, sorted by step. Each metrics_dict
      maps metric names to their values at that step.
    """
    try:
      with tempfile.TemporaryDirectory() as temp_dir:
        local_files = self._download_event_files(temp_dir)
        if not local_files:
          return []

        metrics = self._parse_events(temp_dir)
        return sorted(metrics, key=lambda x: x[0])

    except Exception as e:
      logging.warning("Error reading metrics from GCS: %s", e)
      return []

  def get_new_metrics(self) -> List[Tuple[int, Dict[str, float]]]:
    """Get metrics newer than the last reported step.

    This is useful for incremental reporting during long training runs.

    Returns:
      List of (step, metrics_dict) tuples for steps > last_step_reported.
    """
    all_metrics = self.get_all_metrics()
    new_metrics = [(s, m) for s, m in all_metrics if s > self._last_step_reported]

    if new_metrics:
      self._last_step_reported = max(s for s, m in new_metrics)

    return new_metrics
