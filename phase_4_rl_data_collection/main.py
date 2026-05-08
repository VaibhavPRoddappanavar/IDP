"""
main.py - Phase 4: Continuous RL Data Collection

Runs YOLOv8 inference with an Exploration Controller that randomly
selects continuous `imgsz` and `skip` values. It logs the resulting
State -> Action -> Reward -> NextState transitions to build the RL dataset.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

from monitor import SystemMonitor
from state import SystemState
from exploration import ExplorationController
from logger import TransitionLogger

# ======================================================================
# CONFIGURATION
# ======================================================================
MODEL_PATH = str(Path(__file__).resolve().parent.parent / "BaseModelTraining" / "test" / "best.pt")
VIDEO_PATH = str(Path(__file__).resolve().parent.parent / "BaseModelTraining" / "test" / "test_video_1.mp4")
OUTPUT_DIR = str(Path(__file__).resolve().parent / "output")

USE_WEBCAM = False
SHOW_DISPLAY = True
CONF_THRESHOLD = 0.25
ACTION_INTERVAL_FRAMES = 5
MAX_RUNTIME_MINUTES = 60
MAX_TRANSITIONS = 20000
# ======================================================================

class FPSCounter:
    def __init__(self, smoothing: float = 0.9):
        self.smoothing = smoothing
        self.fps = 0.0

    def update(self, latency_s: float) -> float:
        if latency_s > 0:
            current_fps = 1.0 / latency_s
            if self.fps == 0.0:
                self.fps = current_fps
            else:
                self.fps = (self.fps * self.smoothing) + (current_fps * (1.0 - self.smoothing))
        return self.fps


def run_pipeline():
    print(f"[INFO] Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    source = 0 if USE_WEBCAM else VIDEO_PATH
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not USE_WEBCAM else "inf"
    print(f"[INFO] Video opened. Total frames: {total_frames}")

    monitor = SystemMonitor(poll_interval=0.5)
    fps_counter = FPSCounter(smoothing=0.9)
    logger = TransitionLogger(output_dir=OUTPUT_DIR)
    controller = ExplorationController(action_interval_frames=ACTION_INTERVAL_FRAMES)

    monitor.start()
    time.sleep(1.0) # Let monitor populate

    source_frames_seen = 0
    processed_frames = 0
    
    # RL Buffers
    prev_state = None
    prev_action = None
    accumulated_accuracy = 0.0
    accumulated_frames = 0
    transitions_logged = 0
    run_start_time = time.perf_counter()

    print("[INFO] Starting Continuous RL Data Collection...")

    try:
        while True:
            elapsed_minutes = (time.perf_counter() - run_start_time) / 60.0
            if elapsed_minutes >= MAX_RUNTIME_MINUTES:
                print(f"\n[INFO] Auto-stopping: Reached max runtime of {MAX_RUNTIME_MINUTES} minutes.")
                break
            if transitions_logged >= MAX_TRANSITIONS:
                print(f"\n[INFO] Auto-stopping: Reached max transitions of {MAX_TRANSITIONS}.")
                break

            ret, frame = cap.read()
            if not ret:
                if not USE_WEBCAM:
                    # Loop the video indefinitely for long data collection
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        break # Safety break if looping fails
                else:
                    print("[INFO] End of video stream.")
                    break
            source_frames_seen += 1

            # Decide if we skip the frame based on continuous probability
            import random
            if random.random() < controller.current_skip_rate:
                continue

            # Evaluate if it's time to sample a new action and log the previous transition
            if controller.should_update():
                # 1. Measure Next State (S_t+1)
                hw = monitor.snapshot()
                next_state = SystemState(
                    cpu_percent=hw["cpu_usage_percent"],
                    ram_percent=hw["ram_usage_percent"],
                    temperature=hw["temperature"],
                    fps=fps_counter.fps,
                    latency_ms=1000.0 / fps_counter.fps if fps_counter.fps > 0 else 0.0
                )

                # 2. Compute Reward for the prev action, log transition
                if prev_state is not None and prev_action is not None:
                    proxy_acc = accumulated_accuracy / max(accumulated_frames, 1)
                    reward = controller.compute_reward(prev_state, prev_action, next_state, proxy_acc)
                    
                    transition = {
                        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                        
                        # S_t
                        "state_cpu_percent": prev_state.cpu_percent,
                        "state_ram_percent": prev_state.ram_percent,
                        "state_temperature": prev_state.temperature,
                        "state_fps": prev_state.fps,
                        "state_latency_ms": prev_state.latency_ms,
                        
                        # A_t
                        "action_imgsz_raw": prev_action["action_imgsz_raw"],
                        "action_skip_raw": prev_action["action_skip_raw"],
                        "action_imgsz_applied": prev_action["imgsz"],
                        "action_skip_applied": prev_action["skip_prob"],
                        
                        # R_t
                        "reward": reward,
                        
                        # S_t+1
                        "next_state_cpu_percent": next_state.cpu_percent,
                        "next_state_ram_percent": next_state.ram_percent,
                        "next_state_temperature": next_state.temperature,
                        "next_state_fps": next_state.fps,
                        "next_state_latency_ms": next_state.latency_ms,
                        
                        "done": False
                    }
                    logger.log_transition(transition)
                    transitions_logged += 1
                    
                    print(f"[TRANSITION] {transitions_logged}/{MAX_TRANSITIONS} Action(imgsz={prev_action['imgsz']}, skip={prev_action['skip_prob']:.2f}) -> Reward: {reward:.2f}")

                # 3. Sample New Action (A_t)
                prev_action = controller.sample_action()
                prev_state = next_state
                
                # Reset accumulators for the new step
                accumulated_accuracy = 0.0
                accumulated_frames = 0

            # --- Inference with current action parameters ---
            t_start = time.perf_counter()
            results = model.predict(
                source=frame,
                imgsz=controller.current_imgsz,
                conf=CONF_THRESHOLD,
                verbose=False,
            )
            result = results[0]
            t_end = time.perf_counter()

            latency_s = t_end - t_start
            latency_ms = latency_s * 1000.0
            fps_counter.update(latency_s)
            
            # --- Tracking metrics ---
            accuracy = 0.0
            if len(result.boxes) > 0 and result.boxes.conf is not None:
                accuracy = float(result.boxes.conf.max()) * 100.0
            
            accumulated_accuracy += accuracy
            accumulated_frames += 1
            processed_frames += 1

            if SHOW_DISPLAY:
                annotated = result.plot()
                cv2.putText(annotated, f"RL DATA COLLECTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(annotated, f"imgsz: {controller.current_imgsz} skip: {controller.current_skip_rate:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.imshow("Phase 4", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C).")
    finally:
        # Log terminal done transition if possible
        if prev_state is not None and prev_action is not None:
            hw = monitor.snapshot()
            next_state = SystemState(
                cpu_percent=hw["cpu_usage_percent"],
                ram_percent=hw["ram_usage_percent"],
                temperature=hw["temperature"],
                fps=fps_counter.fps,
                latency_ms=0.0
            )
            transition = {
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "state_cpu_percent": prev_state.cpu_percent,
                "state_ram_percent": prev_state.ram_percent,
                "state_temperature": prev_state.temperature,
                "state_fps": prev_state.fps,
                "state_latency_ms": prev_state.latency_ms,
                "action_imgsz_raw": prev_action["action_imgsz_raw"],
                "action_skip_raw": prev_action["action_skip_raw"],
                "action_imgsz_applied": prev_action["imgsz"],
                "action_skip_applied": prev_action["skip_prob"],
                "reward": 0.0,
                "next_state_cpu_percent": next_state.cpu_percent,
                "next_state_ram_percent": next_state.ram_percent,
                "next_state_temperature": next_state.temperature,
                "next_state_fps": next_state.fps,
                "next_state_latency_ms": next_state.latency_ms,
                "done": True
            }
            logger.log_transition(transition)

        monitor.stop()
        logger.close()
        cap.release()
        if SHOW_DISPLAY:
            cv2.destroyAllWindows()
            
        print(f"\n[INFO] Phase 4 complete. Data saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_pipeline()
