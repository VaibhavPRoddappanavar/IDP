"""
logger.py — CSV metrics logger.

Writes one row per frame with inference + system metrics.
Guarantees headers are written once and the file is flushed regularly
so data is not lost on unexpected termination.
"""

import csv
import os
from pathlib import Path

# Column order for the CSV (must match what main.py produces)
CSV_COLUMNS = [
    "timestamp",
    "mode",
    "mode_imgsz",
    "mode_skip",
    "fps",
    "latency_ms",
    "ram_usage_percent",
    "cpu_usage_percent",
    "gpu_usage_percent",
    "temperature",
    "num_detections",
    "confidence_score_avg",
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
        """
        Write a single row to the CSV.

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
