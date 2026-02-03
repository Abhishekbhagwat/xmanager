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
from typing import List, Optional, Tuple

from google.cloud import storage


class GCSMetricReader:
  """Reads metrics from TensorBoard event files stored on GCS.

  This class is designed to work with NeMo/PyTorch training jobs that write
  TensorBoard logs to GCS. It downloads event files and extracts scalar
  metrics for reporting to Vizier.

  Example usage:
    reader = GCSMetricReader(
        gcs_path='gs://my-bucket/tensorboard/job-123/',
        metric_name='reduced_train_loss',
    )
    result = reader.get_latest_metric()
    if result:
        step, value = result
        print(f"Latest metric at step {step}: {value}")
  """

  def __init__(
      self,
      gcs_path: str,
      metric_name: str = 'reduced_train_loss',
  ) -> None:
    """Initialize the GCS metric reader.

    Args:
      gcs_path: GCS path to TensorBoard logs (e.g., 'gs://bucket/path/').
      metric_name: Name of the scalar metric to extract (e.g., 'reduced_train_loss').
    """
    self._gcs_path = gcs_path.rstrip('/')
    self._metric_name = metric_name
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
      if 'events.out.tfevents' in blob.name or 'tfevents' in blob.name:
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

  def _parse_events(self, log_dir: str) -> List[Tuple[int, float]]:
    """Parse TensorBoard events and extract the specified metric.

    Args:
      log_dir: Local directory containing event files.

    Returns:
      List of (step, value) tuples for the metric.
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

    # Try to find the metric (exact match or partial match)
    matching_tag = None
    for tag in available_tags:
      if tag == self._metric_name:
        matching_tag = tag
        break
      elif self._metric_name in tag:
        matching_tag = tag
        # Don't break - prefer exact match

    if not matching_tag:
      logging.warning(
          "Metric '%s' not found. Available metrics: %s",
          self._metric_name,
          available_tags,
      )
      return []

    # Extract scalar values
    scalars = ea.Scalars(matching_tag)
    return [(s.step, s.value) for s in scalars]

  def get_latest_metric(self) -> Optional[Tuple[int, float]]:
    """Get the latest metric value from TensorBoard events.

    Downloads event files from GCS, parses them, and returns the most
    recent value for the configured metric.

    Returns:
      Tuple of (step, value) for the latest metric, or None if not found.
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

  def get_all_metrics(self) -> List[Tuple[int, float]]:
    """Get all metric values from TensorBoard events.

    This is useful for reporting intermediate measurements to Vizier
    for early stopping decisions.

    Returns:
      List of (step, value) tuples, sorted by step.
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

  def get_new_metrics(self) -> List[Tuple[int, float]]:
    """Get metrics newer than the last reported step.

    This is useful for incremental reporting during long training runs.

    Returns:
      List of (step, value) tuples for steps > last_step_reported.
    """
    all_metrics = self.get_all_metrics()
    new_metrics = [(s, v) for s, v in all_metrics if s > self._last_step_reported]

    if new_metrics:
      self._last_step_reported = max(s for s, v in new_metrics)

    return new_metrics
