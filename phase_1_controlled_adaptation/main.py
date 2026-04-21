"""
main.py - Phase 1: Controlled Adaptation (no ML policy)

Runs YOLO inference with fixed, manually switched modes to characterize
trade-offs before training an adaptive policy.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

from logger import CSVLogger
from monitor import SystemMonitor
from utils import FPSCounter, draw_detections

# ======================================================================
# CONFIGURATION
# ======================================================================
MODEL_PATH = str(Path(__file__).resolve().parent.parent / "BaseModelTraining" / "test" / "best.pt")
VIDEO_PATH = str(Path(__file__).resolve().parent.parent / "BaseModelTraining" / "test" / "test_video_1.mp4")
OUTPUT_CSV = str(Path(__file__).resolve().parent / "output" / "logs.csv")

USE_WEBCAM = False
SHOW_DISPLAY = False
CONF_THRESHOLD = 0.25
MODE_SWITCH_INTERVAL_SEC = 5.0

MODES = {
    "M0": {"imgsz": 640, "skip": 0},
    "M1": {"imgsz": 480, "skip": 1},
    "M2": {"imgsz": 320, "skip": 2},
}
MODE_SEQUENCE = ["M0", "M1", "M2"]
# ======================================================================


def _get_active_mode(elapsed_s: float) -> tuple[str, dict]:
    idx = int(elapsed_s // MODE_SWITCH_INTERVAL_SEC) % len(MODE_SEQUENCE)
    mode_name = MODE_SEQUENCE[idx]
    return mode_name, MODES[mode_name]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]

    rank = (len(sorted_vals) - 1) * (p / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_vals) - 1)
    if lower == upper:
        return sorted_vals[lower]

    weight = rank - lower
    return sorted_vals[lower] * (1.0 - weight) + sorted_vals[upper] * weight


def run_pipeline() -> None:
    """Main controlled-adaptation loop."""
    print(f"[INFO] Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"[INFO] Model loaded successfully. Classes: {list(model.names.values())}")

    source = 0 if USE_WEBCAM else VIDEO_PATH
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not USE_WEBCAM else "inf"
    print(f"[INFO] Video source opened. Total frames: {total_frames}")
    print(f"[INFO] Mode sequence: {MODE_SEQUENCE} (switch every {MODE_SWITCH_INTERVAL_SEC:.0f}s)")

    monitor = SystemMonitor(poll_interval=0.5)
    fps_counter = FPSCounter(smoothing=0.9)
    logger = CSVLogger(output_path=OUTPUT_CSV, flush_every=10)

    monitor.start()
    logger.open()

    run_start_time = time.perf_counter()
    active_mode_name = ""
    frames_to_skip = 0

    source_frames_seen = 0
    processed_frames = 0
    intentional_skips_total = 0

    mode_stats = {
        mode_name: {
            "count": 0,
            "latency_ms": [],
            "fps_sum": 0.0,
            "gpu_sum": 0.0,
            "temp_sum": 0.0,
            "acc_sum": 0.0,
        }
        for mode_name in MODE_SEQUENCE
    }

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video stream.")
                break
            source_frames_seen += 1

            elapsed_s = time.perf_counter() - run_start_time
            if total_frames != "inf" and total_frames > 0:
                frames_per_mode = max(1, total_frames // len(MODE_SEQUENCE))
                idx = min((source_frames_seen - 1) // frames_per_mode, len(MODE_SEQUENCE) - 1)
                mode_name = MODE_SEQUENCE[idx]
                mode_cfg = MODES[mode_name]
            else:
                mode_name, mode_cfg = _get_active_mode(elapsed_s)
            if mode_name != active_mode_name:
                active_mode_name = mode_name
                frames_to_skip = 0
                print(
                    f"[MODE] -> {mode_name} | imgsz={mode_cfg['imgsz']} | skip={mode_cfg['skip']}"
                )

            if frames_to_skip > 0:
                frames_to_skip -= 1
                intentional_skips_total += 1
                continue

            t_start = time.perf_counter()
            results = model.predict(
                source=frame,
                imgsz=mode_cfg["imgsz"],
                conf=CONF_THRESHOLD,
                verbose=False,
            )
            result = results[0]
            t_end = time.perf_counter()

            latency_s = t_end - t_start
            latency_ms = latency_s * 1000.0
            fps = fps_counter.update(latency_s)
            hw = monitor.snapshot()

            num_detections = len(result.boxes)
            confidence_score_avg = 0.0
            accuracy = 0.0
            if num_detections > 0 and result.boxes.conf is not None:
                confidence_score_avg = float(result.boxes.conf.mean())
                accuracy = float(result.boxes.conf.max()) * 100.0

            processed_frames += 1
            frames_to_skip = int(mode_cfg["skip"])

            intentional_drop_rate_percent = (
                intentional_skips_total / max(source_frames_seen, 1)
            ) * 100.0
            frame_drop_rate_percent = (
                (source_frames_seen - processed_frames) / max(source_frames_seen, 1)
            ) * 100.0

            row = {
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "mode": mode_name,
                "mode_imgsz": mode_cfg["imgsz"],
                "mode_skip": mode_cfg["skip"],
                "fps": round(fps, 2),
                "latency_ms": round(latency_ms, 2),
                "ram_usage_percent": round(hw["ram_usage_percent"], 1),
                "cpu_usage_percent": round(hw["cpu_usage_percent"], 1),
                "gpu_usage_percent": round(hw["gpu_usage_percent"], 1),
                "temperature": round(hw["temperature"], 1),
                "num_detections": num_detections,
                "confidence_score_avg": round(confidence_score_avg, 4),
                "accuracy": round(accuracy, 2),
                "source_frames_seen": source_frames_seen,
                "processed_frames": processed_frames,
                "intentional_skips_total": intentional_skips_total,
                "intentional_drop_rate_percent": round(intentional_drop_rate_percent, 2),
                "frame_drop_rate_percent": round(frame_drop_rate_percent, 2),
            }
            logger.log(row)

            stats = mode_stats[mode_name]
            stats["count"] += 1
            stats["latency_ms"].append(latency_ms)
            stats["fps_sum"] += fps
            stats["gpu_sum"] += hw["gpu_usage_percent"]
            stats["temp_sum"] += hw["temperature"]
            stats["acc_sum"] += accuracy

            if SHOW_DISPLAY:
                annotated = draw_detections(frame, result, fps, latency_ms)
                overlay = (
                    f"Mode: {mode_name} | imgsz: {mode_cfg['imgsz']} | skip: {mode_cfg['skip']} "
                    f"| GPU: {hw['gpu_usage_percent']:.1f}% | Temp: {hw['temperature']:.1f}C"
                )
                cv2.putText(
                    annotated,
                    overlay,
                    (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Phase 1 - Controlled Adaptation", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Quit requested by user.")
                    break

            if processed_frames % 100 == 0:
                print(
                    f"[INFO] Processed={processed_frames} | Mode={mode_name} "
                    f"| FPS={fps:.1f} | Latency={latency_ms:.1f}ms "
                    f"| GPU={hw['gpu_usage_percent']:.1f}% | Temp={hw['temperature']:.1f}C"
                )

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C).")
    finally:
        monitor.stop()
        logger.close()
        cap.release()
        if SHOW_DISPLAY:
            cv2.destroyAllWindows()

        print("\n[SUMMARY] Mode-wise performance")
        for mode_name in MODE_SEQUENCE:
            stats = mode_stats[mode_name]
            count = stats["count"]
            if count == 0:
                print(f"  {mode_name}: no processed frames")
                continue

            avg_fps = stats["fps_sum"] / count
            avg_gpu = stats["gpu_sum"] / count
            avg_temp = stats["temp_sum"] / count
            avg_acc = stats["acc_sum"] / count
            p95_latency = _percentile(stats["latency_ms"], 95.0)

            print(
                f"  {mode_name}: frames={count} | avg_fps={avg_fps:.2f} "
                f"| p95_latency_ms={p95_latency:.2f} | avg_gpu={avg_gpu:.1f}% "
                f"| avg_temp={avg_temp:.1f}C | avg_acc={avg_acc:.2f}%"
            )

        print(f"\n[INFO] CSV saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    run_pipeline()
