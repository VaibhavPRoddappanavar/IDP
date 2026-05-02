"""
state.py — RL State-Space Definition

Defines the SystemState dataclass that formalizes the observation vector
the adaptive controller (and later, the RL agent) uses to pick an
inference mode.

State vector: [gpu%, vram%, cpu%, temperature, fps]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class SystemState:
    """Snapshot of system resources at a single point in time.

    This is the **state space** for the adaptive inference system.
    Every controller — heuristic or learned — receives one of these
    each decision step.
    """

    gpu_percent: float = 0.0       # GPU core utilization  (0–100)
    vram_percent: float = 0.0      # GPU memory utilization (0–100)
    cpu_percent: float = 0.0       # CPU utilization        (0–100)
    ram_percent: float = 0.0       # System RAM utilization  (0–100)
    temperature: float = 0.0       # Primary temp sensor    (°C)
    fps: float = 0.0               # Smoothed inference FPS

    # ------------------------------------------------------------------
    #  Conversions
    # ------------------------------------------------------------------
    def to_vector(self) -> List[float]:
        """Flat feature vector for ML model input.

        Order: [gpu%, vram%, cpu%, temperature, fps]
        (matches the README Phase 5 input spec)
        """
        return [
            self.gpu_percent,
            self.vram_percent,
            self.cpu_percent,
            self.temperature,
            self.fps,
        ]

    def to_dict(self) -> dict:
        """Full dict representation for CSV logging."""
        return {
            "state_gpu_percent": self.gpu_percent,
            "state_vram_percent": self.vram_percent,
            "state_cpu_percent": self.cpu_percent,
            "state_ram_percent": self.ram_percent,
            "state_temperature": self.temperature,
            "state_fps": self.fps,
        }

    # ------------------------------------------------------------------
    #  Pretty-print
    # ------------------------------------------------------------------
    def summary(self) -> str:
        return (
            f"GPU={self.gpu_percent:.1f}% | VRAM={self.vram_percent:.1f}% | "
            f"CPU={self.cpu_percent:.1f}% | RAM={self.ram_percent:.1f}% | "
            f"Temp={self.temperature:.1f}°C | FPS={self.fps:.1f}"
        )
