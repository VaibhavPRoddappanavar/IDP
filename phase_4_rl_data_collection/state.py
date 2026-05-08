from dataclasses import dataclass

@dataclass
class SystemState:
    """Represents the system hardware state at a given frame.
    Explicitly removes GPU/VRAM for CPU-only environments (e.g., Raspberry Pi).
    """
    cpu_percent: float
    ram_percent: float
    temperature: float
    fps: float
    latency_ms: float

    def to_dict(self) -> dict:
        return {
            "cpu_percent": self.cpu_percent,
            "ram_percent": self.ram_percent,
            "temperature": self.temperature,
            "fps": self.fps,
            "latency_ms": self.latency_ms,
        }
