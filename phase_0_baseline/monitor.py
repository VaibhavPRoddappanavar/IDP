"""
monitor.py — Non-blocking hardware monitoring module.

Collects CPU usage, RAM usage, and CPU temperature in a background thread
so that the inference loop is never blocked by sensor polling.
"""

import threading
import time
import psutil


class SystemMonitor:
    """Background system resource monitor using psutil."""

    def __init__(self, poll_interval: float = 0.5):
        """
        Args:
            poll_interval: How often (seconds) to refresh hardware metrics.
        """
        self._poll_interval = poll_interval

        # Latest readings (thread-safe via GIL for simple types)
        self._cpu_percent: float = 0.0
        self._ram_percent: float = 0.0
        self._temperature: float = 0.0  # °C, 0 if unavailable

        self._running = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ #
    #  Background polling
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Start the background polling thread."""
        if self._running:
            return
        self._running = True
        # Prime the CPU counter so the first real reading isn't 0
        psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the polling thread to stop."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _poll_loop(self) -> None:
        """Continuously update cached metrics."""
        while self._running:
            self._cpu_percent = psutil.cpu_percent(interval=None)
            self._ram_percent = psutil.virtual_memory().percent
            self._temperature = self._read_temperature()
            time.sleep(self._poll_interval)

    # ------------------------------------------------------------------ #
    #  Temperature helper
    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_temperature() -> float:
        """
        Attempt to read CPU temperature via psutil.
        Returns 0.0 if sensors are unavailable (common on Windows).
        """
        try:
            temps = psutil.sensors_temperatures()
            if not temps:
                return 0.0
            # Pick the first available sensor group
            for _name, entries in temps.items():
                if entries:
                    return entries[0].current
        except (AttributeError, OSError):
            # sensors_temperatures() may not exist on all platforms
            pass
        return 0.0

    # ------------------------------------------------------------------ #
    #  Public accessors (called from the inference thread)
    # ------------------------------------------------------------------ #
    @property
    def cpu_percent(self) -> float:
        return self._cpu_percent

    @property
    def ram_percent(self) -> float:
        return self._ram_percent

    @property
    def temperature(self) -> float:
        return self._temperature

    def snapshot(self) -> dict:
        """Return a dict of all current readings."""
        return {
            "cpu_usage_percent": self._cpu_percent,
            "ram_usage_percent": self._ram_percent,
            "temperature": self._temperature,
        }
