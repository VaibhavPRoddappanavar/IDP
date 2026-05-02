"""
monitor.py — Enhanced Non-blocking Hardware Monitor (Phase 2)

Collects CPU, RAM, GPU utilization, VRAM, and temperature in a
background thread (~500 ms polling) so the inference loop is never
blocked.

GPU metrics are read via **pynvml** (fast, in-process) with a
subprocess fallback to nvidia-smi when pynvml is unavailable.
"""

from __future__ import annotations

import subprocess
import threading
import time

import psutil

# Try to import pynvml for native GPU monitoring
_HAS_PYNVML = False
try:
    import pynvml

    _HAS_PYNVML = True
except ImportError:
    pynvml = None  # type: ignore[assignment]


class SystemMonitor:
    """Background system resource monitor using psutil + pynvml."""

    def __init__(self, poll_interval: float = 0.5):
        """
        Args:
            poll_interval: How often (seconds) to refresh hardware metrics.
        """
        self._poll_interval = poll_interval

        # Latest readings (thread-safe via GIL for simple float writes)
        self._cpu_percent: float = 0.0
        self._ram_percent: float = 0.0
        self._gpu_percent: float = 0.0
        self._vram_percent: float = 0.0
        self._temperature: float = 0.0  # °C — GPU temp if available, else CPU

        self._running = False
        self._thread: threading.Thread | None = None

        # pynvml handle (initialised in start())
        self._gpu_handle = None
        self._has_gpu: bool = False

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Start the background polling thread."""
        if self._running:
            return
        self._running = True

        # Prime the CPU counter so the first real reading isn't 0
        psutil.cpu_percent(interval=None)

        # Initialise pynvml if available
        self._init_nvml()

        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the polling thread to stop and clean up."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._shutdown_nvml()

    # ------------------------------------------------------------------ #
    #  pynvml helpers
    # ------------------------------------------------------------------ #
    def _init_nvml(self) -> None:
        """Try to initialise NVIDIA Management Library."""
        if not _HAS_PYNVML:
            return
        try:
            pynvml.nvmlInit()
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._has_gpu = True
            name = pynvml.nvmlDeviceGetName(self._gpu_handle)
            if isinstance(name, bytes):
                name = name.decode()
            print(f"[MONITOR] pynvml initialised — GPU: {name}")
        except Exception as exc:
            print(f"[MONITOR] pynvml init failed ({exc}). Falling back to nvidia-smi.")
            self._has_gpu = False

    def _shutdown_nvml(self) -> None:
        if _HAS_PYNVML and self._has_gpu:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    #  Background polling loop
    # ------------------------------------------------------------------ #
    def _poll_loop(self) -> None:
        """Continuously update cached metrics."""
        while self._running:
            self._cpu_percent = psutil.cpu_percent(interval=None)
            self._ram_percent = psutil.virtual_memory().percent
            self._poll_gpu()
            self._temperature = self._read_temperature()
            time.sleep(self._poll_interval)

    def _poll_gpu(self) -> None:
        """Read GPU utilization + VRAM via pynvml or nvidia-smi fallback."""
        if self._has_gpu and _HAS_PYNVML:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                self._gpu_percent = float(util.gpu)

                mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                self._vram_percent = (mem.used / mem.total) * 100.0 if mem.total > 0 else 0.0
                return
            except Exception:
                pass  # fall through to subprocess

        # Subprocess fallback (slower, ~200 ms)
        self._gpu_percent = self._read_gpu_smi()
        # VRAM not available via simple smi query — leave at 0

    # ------------------------------------------------------------------ #
    #  Temperature helper
    # ------------------------------------------------------------------ #
    def _read_temperature(self) -> float:
        """Read temperature: prefer GPU temp (pynvml), fall back to CPU (psutil)."""
        # GPU temperature via pynvml
        if self._has_gpu and _HAS_PYNVML:
            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    self._gpu_handle, pynvml.NVML_TEMPERATURE_GPU
                )
                return float(temp)
            except Exception:
                pass

        # CPU temperature via psutil
        try:
            temps = psutil.sensors_temperatures()
            if not temps:
                return 0.0
            for _name, entries in temps.items():
                if entries:
                    return entries[0].current
        except (AttributeError, OSError):
            pass
        return 0.0

    # ------------------------------------------------------------------ #
    #  nvidia-smi subprocess fallback
    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_gpu_smi() -> float:
        """Read GPU utilization via nvidia-smi subprocess."""
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=0.3,
                check=False,
            )
            if proc.returncode != 0:
                return 0.0
            lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
            return float(lines[0]) if lines else 0.0
        except (FileNotFoundError, ValueError, OSError, subprocess.SubprocessError):
            return 0.0

    # ------------------------------------------------------------------ #
    #  Public accessors
    # ------------------------------------------------------------------ #
    @property
    def cpu_percent(self) -> float:
        return self._cpu_percent

    @property
    def ram_percent(self) -> float:
        return self._ram_percent

    @property
    def gpu_percent(self) -> float:
        return self._gpu_percent

    @property
    def vram_percent(self) -> float:
        return self._vram_percent

    @property
    def temperature(self) -> float:
        return self._temperature

    def snapshot(self) -> dict:
        """Return a dict of all current readings."""
        return {
            "cpu_usage_percent": self._cpu_percent,
            "ram_usage_percent": self._ram_percent,
            "gpu_usage_percent": self._gpu_percent,
            "vram_usage_percent": self._vram_percent,
            "temperature": self._temperature,
        }
