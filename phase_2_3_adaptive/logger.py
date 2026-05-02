"""
logger.py — Extended CSV Logger (Phase 2+3)

Writes one row per processed frame with:
  - Inference metrics (fps, latency, detections, accuracy)
  - Full state vector  (gpu%, vram%, cpu%, ram%, temp, fps)
  - Controller action  (mode chosen, reason)
  - Frame-accounting   (source frames seen, processed, skipped)

The CSV doubles as the **dataset** for Phase 4 (RL training data)
since each row contains (state, action, metrics-for-reward).
"""

import csv
import os
from pathlib import Path

# Column order — designed to be directly usable for Phase 4 dataset
CSV_COLUMNS = [
    # --- Timestamp ---
    "timestamp",
    # --- Controller decision ---
    "mode",
    "mode_imgsz",
    "mode_skip",
    "controller_reason",
    "switch_count",
    # --- State vector (RL observation) ---
    "state_gpu_percent",
    "state_vram_percent",
    "state_cpu_percent",
    "state_ram_percent",
    "state_temperature",
    "state_fps",
    # --- Inference metrics ---
    "fps",
    "latency_ms",
    "num_detections",
    "confidence_score_avg",
    "accuracy",
    # --- Resource metrics (raw) ---
    "gpu_usage_percent",
    "vram_usage_percent",
    "cpu_usage_percent",
    "ram_usage_percent",
    "temperature",
    # --- Frame accounting ---
    "source_frames_seen",
    "processed_frames",
    "intentional_skips_total",
    "intentional_drop_rate_percent",
    "frame_drop_rate_percent",
]


class CSVLogger:
    """Append-mode CSV logger with guaranteed headers and periodic flush."""

    def __init__(self, output_path: str, flush_every: int = 10):
        """
        Args:
            output_path: Full path to the CSV file (e.g. output/logs.csv).
            flush_every: Flush the file buffer after this many rows.
        """
        self._path = Path(output_path)
        self._flush_every = flush_every
        self._row_count = 0
        self._file = None
        self._writer = None

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #
    def open(self) -> None:
        """Create / open the CSV and write headers if the file is new."""
        os.makedirs(self._path.parent, exist_ok=True)

        file_exists = self._path.exists() and self._path.stat().st_size > 0
        self._file = open(self._path, mode="a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_COLUMNS)

        if not file_exists:
            self._writer.writeheader()
            self._file.flush()

    def close(self) -> None:
        """Flush and close the file handle."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None

    # ------------------------------------------------------------------ #
    #  Writing
    # ------------------------------------------------------------------ #
    def log(self, row: dict) -> None:
        """Write a single row to the CSV.

        Args:
            row: Dictionary whose keys match CSV_COLUMNS.
                 Missing keys will be written as empty strings.
        """
        if self._writer is None:
            raise RuntimeError("CSVLogger.open() must be called before log().")

        self._writer.writerow(row)
        self._row_count += 1

        if self._row_count % self._flush_every == 0:
            self._file.flush()

    # ------------------------------------------------------------------ #
    #  Context-manager support
    # ------------------------------------------------------------------ #
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
