"""
main.py — Phase 2+3: Adaptive Inference with Heuristic Controller

Combines real-time system monitoring (Phase 2) with a rule-based
controller (Phase 3) that dynamically selects the inference mode
based on live hardware state.

Unlike Phase 1 (time-based switching), modes are now chosen by the
HeuristicController reading the SystemState every frame.

Usage:
    python main.py
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

from controller import HeuristicController, MODES
from logger import CSVLogger
from monitor import SystemMonitor
from state import SystemState
from utils import FPSCounter, draw_detections

# ======================================================================
# CONFIGURATION
# ======================================================================
MODEL_PATH = str(
    Path(__file__).resolve().parent.parent
    / "BaseModelTraining" / "test" / "best.pt"
)
VIDEO_PATH = str(
    Path(__file__).resolve().parent.parent
    / "BaseModelTraining" / "test" / "test_video_1.mp4"
)
OUTPUT_CSV = str(
    Path(__file__).resolve().parent / "output" / "logs.csv"
)

USE_WEBCAM = False
SHOW_DISPLAY = False
CONF_THRESHOLD = 0.25
# ======================================================================


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (len(s) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    w = rank - lo
    return s[lo] * (1.0 - w) + s[hi] * w


def run_pipeline() -> None:
    """Main adaptive inference loop."""

    # ----- Load model -----
    print(f"[INFO] Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"[INFO] Model loaded. Classes: {list(model.names.values())}")

    # ----- Open video source -----
    source = 0 if USE_WEBCAM else VIDEO_PATH
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not USE_WEBCAM else "inf"
    print(f"[INFO] Video opened. Total frames: {total_frames}")

    # ----- Initialise components -----
    monitor = SystemMonitor(poll_interval=0.5)
    fps_counter = FPSCounter(smoothing=0.9)
    logger = CSVLogger(output_path=OUTPUT_CSV, flush_every=10)
    controller = HeuristicController(initial_mode="M0")

    monitor.start()
    logger.open()

    source_frames_seen = 0
    processed_frames = 0
    intentional_skips_total = 0
    frames_to_skip = 0

    # Per-mode aggregation for final summary
    mode_stats: dict[str, dict] = {
        m: {"count": 0, "latency_ms": [], "fps_sum": 0.0,
            "gpu_sum": 0.0, "temp_sum": 0.0, "acc_sum": 0.0}
        for m in MODES
    }

    print("[INFO] Starting adaptive inference (heuristic controller)...\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video stream.")
                break
            source_frames_seen += 1

            # --- Skip frames if current mode requires it ---
            if frames_to_skip > 0:
                frames_to_skip -= 1
                intentional_skips_total += 1
                continue

            # --- Build system state (Phase 2: state space) ---
            hw = monitor.snapshot()
            current_fps = fps_counter.update(0.001) if processed_frames == 0 else 0.0

            state = SystemState(
                gpu_percent=hw["gpu_usage_percent"],
                vram_percent=hw["vram_usage_percent"],
                cpu_percent=hw["cpu_usage_percent"],
                ram_percent=hw["ram_usage_percent"],
                temperature=hw["temperature"],
                fps=current_fps,
            )

            # --- Controller decides mode (Phase 3: heuristic) ---
            prev_mode = controller.current_mode
            mode_name = controller.select_mode(state)
            mode_cfg = controller.get_config()

            if mode_name != prev_mode:
                reason = controller.describe_decision(state)
                print(f"[SWITCH] {reason} (imgsz={mode_cfg['imgsz']}, skip={mode_cfg['skip']})")

            # --- Inference ---
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

            # Update state FPS for logging (now we have real FPS)
            state.fps = fps

            # --- Detections ---
            num_det = len(result.boxes)
            conf_avg = 0.0
            accuracy = 0.0
            if num_det > 0 and result.boxes.conf is not None:
                conf_avg = float(result.boxes.conf.mean())
                accuracy = float(result.boxes.conf.max()) * 100.0

            processed_frames += 1
            frames_to_skip = int(mode_cfg["skip"])

            # --- Frame accounting ---
            intent_drop = (intentional_skips_total / max(source_frames_seen, 1)) * 100.0
            total_drop = ((source_frames_seen - processed_frames) / max(source_frames_seen, 1)) * 100.0

            # --- Build log row ---
            reason_str = controller.describe_decision(state)
            state_dict = state.to_dict()

            row = {
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "mode": mode_name,
                "mode_imgsz": mode_cfg["imgsz"],
                "mode_skip": mode_cfg["skip"],
                "controller_reason": reason_str,
                "switch_count": controller.switch_count,
                **state_dict,
                "fps": round(fps, 2),
                "latency_ms": round(latency_ms, 2),
                "num_detections": num_det,
                "confidence_score_avg": round(conf_avg, 4),
                "accuracy": round(accuracy, 2),
                "gpu_usage_percent": round(hw["gpu_usage_percent"], 1),
                "vram_usage_percent": round(hw["vram_usage_percent"], 1),
                "cpu_usage_percent": round(hw["cpu_usage_percent"], 1),
                "ram_usage_percent": round(hw["ram_usage_percent"], 1),
                "temperature": round(hw["temperature"], 1),
                "source_frames_seen": source_frames_seen,
                "processed_frames": processed_frames,
                "intentional_skips_total": intentional_skips_total,
                "intentional_drop_rate_percent": round(intent_drop, 2),
                "frame_drop_rate_percent": round(total_drop, 2),
            }
            logger.log(row)

            # --- Per-mode stats ---
            s = mode_stats[mode_name]
            s["count"] += 1
            s["latency_ms"].append(latency_ms)
            s["fps_sum"] += fps
            s["gpu_sum"] += hw["gpu_usage_percent"]
            s["temp_sum"] += hw["temperature"]
            s["acc_sum"] += accuracy

            # --- Display ---
            if SHOW_DISPLAY:
                annotated = draw_detections(frame, result, fps, latency_ms, mode_name, hw)
                cv2.imshow("Phase 2+3 — Adaptive Inference", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Quit requested.")
                    break

            if processed_frames % 100 == 0:
                print(
                    f"[INFO] frame={processed_frames} | {mode_name} "
                    f"| FPS={fps:.1f} | Lat={latency_ms:.1f}ms "
                    f"| GPU={hw['gpu_usage_percent']:.1f}% "
                    f"| Temp={hw['temperature']:.1f}C "
                    f"| switches={controller.switch_count}"
                )

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted (Ctrl+C).")
    finally:
        monitor.stop()
        logger.close()
        cap.release()
        if SHOW_DISPLAY:
            cv2.destroyAllWindows()

        # --- Final summary ---
        print("\n" + "=" * 65)
        print(" PHASE 2+3 — ADAPTIVE INFERENCE SUMMARY ".center(65, "="))
        print("=" * 65)
        print(f"  Total source frames: {source_frames_seen}")
        print(f"  Processed frames:    {processed_frames}")
        print(f"  Mode switches:       {controller.switch_count}")
        print()

        for mn in MODES:
            st = mode_stats[mn]
            c = st["count"]
            if c == 0:
                print(f"  {mn}: no frames processed")
                continue
            avg_fps = st["fps_sum"] / c
            avg_gpu = st["gpu_sum"] / c
            avg_temp = st["temp_sum"] / c
            avg_acc = st["acc_sum"] / c
            p95_lat = _percentile(st["latency_ms"], 95.0)
            pct = (c / processed_frames) * 100 if processed_frames else 0
            print(
                f"  {mn}: {c} frames ({pct:.1f}%) | "
                f"avg_fps={avg_fps:.1f} | p95_lat={p95_lat:.1f}ms | "
                f"avg_gpu={avg_gpu:.1f}% | avg_temp={avg_temp:.1f}C | "
                f"avg_acc={avg_acc:.1f}%"
            )

        print(f"\n[INFO] CSV → {OUTPUT_CSV}")
        print("=" * 65)


if __name__ == "__main__":
    run_pipeline()
