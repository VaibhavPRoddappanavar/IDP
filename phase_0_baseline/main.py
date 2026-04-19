"""
main.py — Phase 0: Baseline Inference Pipeline

Runs YOLOv8n inference on every frame of a video (or webcam) with FIXED
settings (imgsz=640, FP32, no frame skipping) and logs per-frame metrics
to a CSV file.

Usage:
    python main.py
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

from monitor import SystemMonitor
from logger import CSVLogger
from utils import FPSCounter, draw_detections

# ======================================================================
# CONFIGURATION — all paths hard-coded for one-command execution
# ======================================================================
MODEL_PATH = str(Path(__file__).resolve().parent.parent / "BaseModelTraining" / "test" / "best.pt")
VIDEO_PATH = str(Path(__file__).resolve().parent.parent / "BaseModelTraining" / "test" / "test_video_1.mp4")
OUTPUT_CSV = str(Path(__file__).resolve().parent / "output" / "logs.csv")

# Inference settings (FIXED — baseline)
IMG_SIZE = 640
USE_WEBCAM = False          # Set True to use webcam (index 0) instead of video file
SHOW_DISPLAY = True         # Set False for headless / server runs
CONF_THRESHOLD = 0.25       # YOLOv8 default confidence threshold
# ======================================================================


def run_pipeline() -> None:
    """Main inference + logging loop."""

    # ----- Load model -----
    print(f"[INFO] Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"[INFO] Model loaded successfully. Classes: {list(model.names.values())}")

    # ----- Open video source -----
    source = 0 if USE_WEBCAM else VIDEO_PATH
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not USE_WEBCAM else "∞"
    print(f"[INFO] Video source opened. Total frames: {total_frames}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = None

    # ----- Initialise modules -----
    monitor = SystemMonitor(poll_interval=0.5)
    fps_counter = FPSCounter(smoothing=0.9)
    logger = CSVLogger(output_path=OUTPUT_CSV, flush_every=10)

    monitor.start()
    logger.open()

    frame_idx = 0
    run_start_time = time.perf_counter()

    try:
        while True:
            t_start = time.perf_counter()

            # --- Read frame ---
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video stream.")
                break

            # --- Inference (FP32, fixed imgsz, every frame) ---
            results = model.predict(
                source=frame,
                imgsz=IMG_SIZE,
                conf=CONF_THRESHOLD,
                verbose=False,          # suppress per-frame YOLO logs
            )
            result = results[0]

            # --- Timing ---
            t_end = time.perf_counter()
            latency_s = t_end - t_start
            latency_ms = latency_s * 1000.0
            fps = fps_counter.update(latency_s)

            # --- System snapshot (non-blocking, reads cached values) ---
            hw = monitor.snapshot()

            # --- Detections ---
            num_detections = len(result.boxes)

            confidence_score_avg = 0.0
            if num_detections > 0 and result.boxes.conf is not None:
                confidence_score_avg = float(result.boxes.conf.mean())

            frame_drop_rate_percent = 0.0
            if source_fps is not None:
                elapsed_s = max(t_end - run_start_time, 1e-9)
                expected_frames = source_fps * elapsed_s
                processed_frames = frame_idx + 1
                dropped_frames = max(expected_frames - processed_frames, 0.0)
                if expected_frames > 0:
                    frame_drop_rate_percent = (dropped_frames / expected_frames) * 100.0

            # --- Build log row ---
            row = {
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "fps": round(fps, 2),
                "latency_ms": round(latency_ms, 2),
                "ram_usage_percent": round(hw["ram_usage_percent"], 1),
                "cpu_usage_percent": round(hw["cpu_usage_percent"], 1),
                "temperature": round(hw["temperature"], 1),
                "num_detections": num_detections,
                "confidence_score_avg": round(confidence_score_avg, 4),
                "frame_drop_rate_percent": round(frame_drop_rate_percent, 2),
            }
            logger.log(row)

            # --- Display (optional) ---
            if SHOW_DISPLAY:
                annotated = draw_detections(frame, result, fps, latency_ms)
                cv2.imshow("Phase 0 — Baseline Inference", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Quit requested by user.")
                    break

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"[INFO] Processed {frame_idx} frames  |  FPS: {fps:.1f}  |  Latency: {latency_ms:.1f} ms")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C).")
    finally:
        # ----- Cleanup -----
        monitor.stop()
        logger.close()
        cap.release()
        if SHOW_DISPLAY:
            cv2.destroyAllWindows()
        print(f"[INFO] Pipeline finished. {frame_idx} frames logged to: {OUTPUT_CSV}")


if __name__ == "__main__":
    run_pipeline()
