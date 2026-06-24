# Phase 6 — Final Evaluation & Paper-Ready Comparison

## Background & Repository Brush-Up

This is a **Pothole Detection System** built around YOLOv8 with an Offline Reinforcement Learning (RL) adaptive inference pipeline. Here's what every phase did:

| Phase | Description |
|---|---|
| **Phase 0** | Baseline — YOLOv8 at fixed imgsz=640, FP32, every frame. Logs FPS, latency, CPU, RAM, temperature to CSV. |
| **Phase 1** | Controlled adaptation — time-based switching between M0/M1/M2 modes. |
| **Phase 2+3** | Heuristic adaptive — real-time SystemMonitor drives a rule-based HeuristicController that chooses mode (M0=640, M1=480, M2=320) based on CPU/GPU/Temp thresholds. |
| **Phase 4** | RL Data Collection — ExplorationController randomly samples continuous (imgsz ∈ [256,640], skip_prob ∈ [0,1]) actions, computes reward = w_acc×accuracy + w_fps×fps − w_lat×latency − thermal_penalty, saves as parquet. |
| **Phase 5** | Offline RL Training — Behavioral Cloning on top 30% of rewards. PyTorch MLP ContinuousActor (5→128→128→2, Tanh) trained for 150 epochs. Outputs `rl_adaptive_policy.pt` + `state_mean.npy` + `state_std.npy`. The RL model decides imgsz ∈ [256,640] and skip_prob dynamically per frame based on system state. |
| **Phase 6** | **Evaluation** (to be built now) — Two scripts: accuracy comparison + system parameters comparison. |

### Key Labels Note
The YOLO labels in `phase_6_evaluation/yolo/labels/` are in **OBB (Oriented Bounding Box) / polygon format**, NOT standard `class cx cy w h`. Each line has `class x1 y1 x2 y2 ... xN yN` (variable number of polygon vertices). This needs custom IoU computation (polygon IoU or axis-aligned bounding rect IoU).

The two models being compared:
- **Baseline (Normal)**: `best_new_model.pt` run at fixed **imgsz=640**
- **RL Model**: same `best_new_model.pt` but run at the **dynamically decided imgsz** from the trained RL policy (`rl_adaptive_policy.pt`)

---

## Proposed Changes

### Script 1 — Accuracy Comparison [`phase_6_evaluation/evaluate_accuracy.py`] [NEW]

**Purpose**: For each of the 51 labelled images, run YOLOv8 inference twice — at full resolution (Baseline) and at RL-determined resolution — then compare predicted boxes vs. ground-truth boxes using IoU to compute per-image detection metrics.

**Algorithm**:
1. Load `rl_adaptive_policy.pt`, `state_mean.npy`, `state_std.npy` from Phase 5 output
2. Construct a representative system state (simulated: 50% CPU, 40% RAM, 55°C temp, 20 FPS, 30ms latency) to let the RL policy decide a single inference size for evaluation
3. For each image:
   - Parse its `.txt` polygon label → convert polygon vertices to axis-aligned bounding boxes (AABB) for IoU computation
   - Run YOLO at **imgsz=640** → get predictions → compute IoU vs GT → derive Precision/Recall/F1
   - Run YOLO at **RL-decided imgsz** → get predictions → compute IoU vs GT → derive Precision/Recall/F1
4. Aggregate across all 51 images → mAP@0.5, Precision, Recall, F1 for both modes
5. **Terminal output**: Formatted table with per-metric comparison
6. **Visual outputs** (saved to `phase_6_evaluation/output/`):
   - Bar chart: mAP@0.5, Precision, Recall, F1 side-by-side (Baseline vs RL)
   - Scatter plot: per-image F1 Baseline vs RL (shows consistency)
   - Example detection visualizations: 3 sample images with GT boxes, Baseline predictions, RL predictions overlaid
   - Summary table PNG (rendered as matplotlib table)

> [!IMPORTANT]
> The IoU threshold used is 0.5 (standard mAP@0.5). TP = IoU ≥ 0.5 with a GT box. FP = prediction with no GT match. FN = GT box with no prediction match.

> [!NOTE]
> The RL policy's imgsz decision is made once with a representative "moderate load" state (not per-frame live state, since these are static test images). The imgsz selected by RL will typically be in the 320–576 range, demonstrating the RL model voluntarily choosing a smaller resolution to save compute.

---

### Script 2 — System Parameters Comparison [`phase_6_evaluation/evaluate_system_params.py`] [NEW]

**Purpose**: Run a live inference loop on all 51 test images (looped to simulate sustained video-like load) under two modes and measure real-time system parameters.

**Algorithm**:
1. **Baseline run**: Process all 51 images at imgsz=640, repeat for N iterations, collect per-frame: latency_ms, FPS, CPU%, RAM%, temperature, num_detections
2. **RL run**: Process all 51 images at RL-decided imgsz (dynamic, re-queried from policy per-image using current live hardware state), collect same metrics
3. Compute averages and improvement deltas for all metrics
4. **Terminal output**: Rich benchmark table (similar to `compare_benchmarks.py` but richer)
5. **Visual outputs** (saved to `phase_6_evaluation/output/`):
   - Time-series plot: FPS over iterations (Baseline vs RL)
   - Time-series plot: CPU% over iterations (Baseline vs RL)
   - Time-series plot: Latency over iterations (Baseline vs RL)
   - Bar chart: Mean values of all 5 metrics compared (2×5 grouped bars)
   - Box plots: Distribution spread of latency and FPS
   - Radar/Spider chart: Multi-metric comparison normalized to baseline
   - Summary comparison table PNG

> [!IMPORTANT]
> On Windows, `psutil.sensors_temperatures()` returns empty — temperature will be logged as 0.0. The scripts handle this gracefully (skip temperature plot if all-zero, report "N/A" in terminal table).

---

## Verification Plan

### Automated
- Scripts auto-create output dir if missing
- Graceful error if model files not found (clear message telling what to run first)
- Both scripts end with a printed summary of all saved files

### Manual Verification
- Run `python phase_6_evaluation/evaluate_accuracy.py` → check terminal table + 4 output PNGs
- Run `python phase_6_evaluation/evaluate_system_params.py` → check terminal table + 6 output PNGs

### Requirements (all already in venv)
```
ultralytics
torch
numpy
matplotlib
psutil
```

---

## Open Questions

> [!IMPORTANT]
> **Q1: IoU format** — The YOLO labels are in **polygon/OBB format** (variable-length vertex lists), not standard `cx cy w h`. My plan is to convert polygons → axis-aligned bounding rectangles (AABB) for IoU, which is standard practice when full OBB IoU is overkill. Should I use AABB IoU or implement full polygon IoU?

> [!IMPORTANT]
> **Q2: RL imgsz per-image or fixed?** — For accuracy evaluation (static images), the RL policy will be queried with a fixed representative hardware state since there's no live system stress. This means the RL model will choose ONE imgsz for all 51 images. Alternatively I can use varying simulated states (escalating thermal stress) per image batch. Which is preferred?

> [!NOTE]
> **Q3: Model path** — `best_new_model.pt` is in `BaseModelTraining/test/`. This is what the baseline (phase 0) uses. The RL policy uses the same detection model but with dynamic imgsz. Confirming this is the correct model for evaluation.

> [!NOTE]
> **Q4: Number of loops for system params** — To get stable averages, the 51 images will be looped ~5 times (255 total inference passes per mode). Increase if you want more stable readings, but this should be sufficient.
