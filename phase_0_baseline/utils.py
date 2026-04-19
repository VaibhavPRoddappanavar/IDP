"""
utils.py — Shared utilities for Phase 0 baseline.

Contains:
  - Smoothed FPS calculator (exponential moving average)
  - Drawing helpers for annotated frames
"""

import cv2
import numpy as np


class FPSCounter:
    """
    Exponential-moving-average FPS calculator.

    Using an EMA avoids the noisy "instantaneous" FPS that results from
    simply taking 1/dt each frame while still reacting quickly to real
    performance changes.
    """

    def __init__(self, smoothing: float = 0.9):
        """
        Args:
            smoothing: EMA weight for the previous estimate (0–1).
                       Higher = smoother but slower to react.
        """
        self._alpha = smoothing
        self._avg_dt: float | None = None  # seconds

    def update(self, dt: float) -> float:
        """
        Feed a new frame-time and return the smoothed FPS.

        Args:
            dt: Elapsed time for the last frame (seconds).

        Returns:
            Smoothed FPS value.
        """
        if dt <= 0:
            return 0.0

        if self._avg_dt is None:
            self._avg_dt = dt
        else:
            self._avg_dt = self._alpha * self._avg_dt + (1 - self._alpha) * dt

        return 1.0 / self._avg_dt

    def reset(self) -> None:
        self._avg_dt = None


def draw_detections(frame: np.ndarray, results, fps: float, latency_ms: float) -> np.ndarray:
    """
    Annotate *frame* in-place with bounding boxes, class labels, and an
    info overlay showing FPS and latency.

    Args:
        frame:      BGR image (numpy array).
        results:    A single ultralytics Results object.
        fps:        Current smoothed FPS.
        latency_ms: Inference latency for this frame (ms).

    Returns:
        The same frame (modified in-place) for convenience.
    """
    # --- Draw bounding boxes ---
    boxes = results.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = f"{results.names[cls_id]} {conf:.2f}"

        color = _class_color(cls_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # --- Info overlay (top-left) ---
    info = f"FPS: {fps:.1f}  |  Latency: {latency_ms:.1f} ms  |  Objects: {len(boxes)}"
    cv2.putText(frame, info, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    return frame


def _class_color(cls_id: int) -> tuple:
    """Deterministic BGR colour for a class id."""
    np.random.seed(cls_id + 42)
    return tuple(int(c) for c in np.random.randint(80, 255, size=3))
