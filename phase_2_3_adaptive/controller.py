"""
controller.py — Heuristic Controller (Phase 3)

Rule-based mode selector that chooses an inference mode based on the
current SystemState.  This is the **strong baseline** that the
learned RL policy (Phase 5) must beat.

Hardware-aware:
    - On NVIDIA GPU machines: uses GPU% and GPU temp thresholds
    - On CPU-only / Apple Silicon: uses CPU% and RAM% as proxies
"""

from __future__ import annotations

from state import SystemState


# ======================================================================
# MODE DEFINITIONS  (action space — shared across all phases)
# ======================================================================
MODES = {
    "M0": {"imgsz": 640, "skip": 0},   # highest accuracy, highest load
    "M1": {"imgsz": 480, "skip": 1},   # balanced
    "M2": {"imgsz": 320, "skip": 2},   # lowest load, lower accuracy
}

MODE_NAMES = list(MODES.keys())


# ======================================================================
# HEURISTIC CONTROLLER
# ======================================================================
class HeuristicController:
    """Rule-based mode selector that adapts to available hardware sensors.

    When GPU metrics are available (gpu_percent > 0 at some point),
    uses GPU-based thresholds.  Otherwise, falls back to CPU-based
    thresholds so the controller works on any hardware (MacBook,
    Raspberry Pi, etc.)
    """

    # GPU-based thresholds (used when NVIDIA GPU is present)
    GPU_HIGH = 85.0
    TEMP_HIGH = 80.0
    GPU_MEDIUM = 60.0

    # CPU-based fallback thresholds (used on Mac / CPU-only machines)
    CPU_HIGH = 80.0
    RAM_HIGH = 85.0
    CPU_MEDIUM = 50.0

    def __init__(self, initial_mode: str = "M0"):
        self.current_mode: str = initial_mode
        self.switch_count: int = 0
        self._prev_mode: str = initial_mode
        self._gpu_ever_seen: bool = False  # tracks if we ever got GPU readings

    def select_mode(self, state: SystemState) -> str:
        """Pick a mode based on current system state.

        Automatically detects whether GPU metrics are available and
        uses the appropriate thresholds.
        """
        # Track if GPU metrics are ever non-zero
        if state.gpu_percent > 0:
            self._gpu_ever_seen = True

        if self._gpu_ever_seen:
            mode = self._select_gpu_based(state)
        else:
            mode = self._select_cpu_based(state)

        self._prev_mode = self.current_mode
        self.current_mode = mode

        if mode != self._prev_mode:
            self.switch_count += 1

        return mode

    def _select_gpu_based(self, state: SystemState) -> str:
        """GPU-based rules (for NVIDIA GPU machines)."""
        if state.gpu_percent > self.GPU_HIGH or state.temperature > self.TEMP_HIGH:
            return "M2"
        elif state.gpu_percent > self.GPU_MEDIUM:
            return "M1"
        return "M0"

    def _select_cpu_based(self, state: SystemState) -> str:
        """CPU-based fallback rules (for Mac / CPU-only machines)."""
        if state.cpu_percent > self.CPU_HIGH or state.ram_percent > self.RAM_HIGH:
            return "M2"
        elif state.cpu_percent > self.CPU_MEDIUM:
            return "M1"
        return "M0"

    def get_config(self) -> dict:
        """Return the imgsz/skip config for the current mode."""
        return MODES[self.current_mode]

    def describe_decision(self, state: SystemState) -> str:
        """Human-readable explanation of why a mode was chosen."""
        if self._gpu_ever_seen:
            return self._describe_gpu(state)
        return self._describe_cpu(state)

    def _describe_gpu(self, state: SystemState) -> str:
        if state.gpu_percent > self.GPU_HIGH:
            reason = f"GPU {state.gpu_percent:.0f}% > {self.GPU_HIGH}%"
        elif state.temperature > self.TEMP_HIGH:
            reason = f"Temp {state.temperature:.0f}C > {self.TEMP_HIGH}C"
        elif state.gpu_percent > self.GPU_MEDIUM:
            reason = f"GPU {state.gpu_percent:.0f}% > {self.GPU_MEDIUM}%"
        else:
            reason = f"GPU {state.gpu_percent:.0f}% <= {self.GPU_MEDIUM}% (low)"
        return f"{self.current_mode} <- {reason}"

    def _describe_cpu(self, state: SystemState) -> str:
        if state.cpu_percent > self.CPU_HIGH:
            reason = f"CPU {state.cpu_percent:.0f}% > {self.CPU_HIGH}%"
        elif state.ram_percent > self.RAM_HIGH:
            reason = f"RAM {state.ram_percent:.0f}% > {self.RAM_HIGH}%"
        elif state.cpu_percent > self.CPU_MEDIUM:
            reason = f"CPU {state.cpu_percent:.0f}% > {self.CPU_MEDIUM}%"
        else:
            reason = f"CPU {state.cpu_percent:.0f}% <= {self.CPU_MEDIUM}% (low)"
        return f"{self.current_mode} <- {reason}"
