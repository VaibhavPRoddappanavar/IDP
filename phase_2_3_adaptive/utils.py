"""
utils.py — Shared utilities for Phase 2+3 adaptive pipeline.

Contains:
  - Smoothed FPS calculator (exponential moving average)
  - Drawing helpers for annotated frames
"""

import cv2
import numpy as np


class FPSCounter:
    """Exponential-moving-average FPS calculator."""

    def __init__(self, smoothing: float = 0.9):
        self._alpha = smoothing
        self._avg_dt: float | None = None

    def update(self, dt: float) -> float:
        if dt <= 0:
            return 0.0
        if self._avg_dt is None:
            self._avg_dt = dt
        else:
            self._avg_dt = self._alpha * self._avg_dt + (1 - self._alpha) * dt
        return 1.0 / self._avg_dt

    def reset(self) -> None:
        self._avg_dt = None


def draw_detections(frame, results, fps, latency_ms, mode_name="", hw=None):
    """Annotate frame with boxes, labels, FPS, and mode overlay."""
    boxes = results.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = f"{results.names[cls_id]} {conf:.2f}"
        color = _class_color(cls_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    info = f"FPS: {fps:.1f}  |  Latency: {latency_ms:.1f} ms  |  Objects: {len(boxes)}"
    cv2.putText(frame, info, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    if mode_name and hw:
        overlay = (f"Mode: {mode_name} | GPU: {hw.get('gpu_usage_percent', 0):.1f}% | "
                   f"VRAM: {hw.get('vram_usage_percent', 0):.1f}% | "
                   f"Temp: {hw.get('temperature', 0):.1f}C")
        cv2.putText(frame, overlay, (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
    return frame


def _class_color(cls_id: int) -> tuple:
    """Deterministic BGR colour for a class id."""
    np.random.seed(cls_id + 42)
    return tuple(int(c) for c in np.random.randint(80, 255, size=3))
