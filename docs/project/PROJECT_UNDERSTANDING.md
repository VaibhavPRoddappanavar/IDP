# Project Understanding

## 1. Purpose

This repository develops and evaluates an adaptive YOLOv8 pothole-detection pipeline for resource-constrained edge devices. The project investigates whether inference workload can be reduced by changing image resolution and frame-processing behavior in response to hardware telemetry.

The repository is organized as a sequence of experimental phases. It is a research prototype rather than a single packaged application.

## 2. High-Level Pipeline

```text
YOLOv8 model and pothole dataset
          |
          v
Phase 0: fixed-resolution baseline
          |
          v
Phase 1: manually controlled resolution/frame skipping
          |
          v
Phase 2-3: telemetry-driven heuristic controller
          |
          v
Phase 4: random transition-data collection
          |
          v
Phase 5: offline behavioral-cloning policy
          |
          v
Phase 6: accuracy and runtime evaluations
```

The intended feedback loop is:

```text
Video frame -> YOLO inference -> performance metrics
     ^                                  |
     |                                  v
     +------ selected image size <- system state
```

The current learned-policy implementation controls image size during Phase 6. Although Phase 4 also samples a frame-skip probability, Phase 6 does not apply the learned skip output.

## 3. Repository Structure

### Model and dataset

- `BaseModelTraining/` contains YOLO training outputs, detector weights, datasets, validation artifacts, and training configuration.
- `best.pt` and the weights under `BaseModelTraining/test/` are detector models, not adaptive policies.
- `BaseModelTraining/detect/adaptive_system/baseline_yolov8n/args.yaml` records a YOLOv8n training run with 50 epochs and 640-pixel training images.
- `model_accuracy/` contains a separate Ultralytics validation utility and dataset configuration.

### Experimental phases

- `phase_0_baseline/`: fixed inference and per-frame logging.
- `phase_1_controlled_adaptation/`: manually sequenced inference modes.
- `phase_2_3_adaptive/`: live telemetry plus heuristic mode selection.
- `phase_4_rl_data_collection/`: random continuous action sampling and transition logging.
- `phase_5_offline_rl_training/`: policy training and normalization-statistic generation.
- `phase_6_evaluation/`: accuracy and runtime comparison scripts plus generated plots.

### Documentation and paper material

- `README.md`: project setup, phase overview, and headline benchmark claims.
- `PROGRESS_REPORT.md`: week-by-week project diary.
- `POSTER_CONTENT.md`: poster narrative and architecture claims.
- `paper_format.tex` and `adaptive_inference_paper.tex`: IEEE-style manuscript drafts.
- `reference_paper.tex`: unrelated/reference paper material used for formatting or comparison.

## 4. Phase-by-Phase Behavior

### Phase 0: Static baseline

Entry point: `phase_0_baseline/main.py`

The pipeline:

1. Loads a YOLO model.
2. Opens a video file or webcam.
3. Runs inference on every frame at `imgsz=640`.
4. Measures latency and smoothed FPS.
5. Reads cached CPU, RAM, and temperature values from a background monitor.
6. Logs detections and runtime metrics to CSV.

The baseline is fixed-resolution inference. It does not implement frame skipping, adaptive control, TensorRT, or FP16 configuration in this phase.

The monitor in `phase_0_baseline/monitor.py` uses `psutil` and returns temperature `0.0` when the operating system does not expose a sensor.

### Phase 1: Controlled adaptation

Entry point: `phase_1_controlled_adaptation/main.py`

The declared modes are:

```text
M0 = (imgsz 640, skip 0)
M1 = (imgsz 480, skip 1)
M2 = (imgsz 320, skip 2)
```

For a finite video, the implementation divides the source frames into three segments and assigns one mode to each segment. The `MODE_SWITCH_INTERVAL_SEC` path is used for webcam-like operation, but the finite-video behavior is frame-position based.

Skipped frames are read from the video but are not sent through YOLO inference. Mode-specific statistics are collected for processed frames.

### Phase 2 and 3: Heuristic adaptation

Entry point: `phase_2_3_adaptive/main.py`

The `SystemMonitor` in `phase_2_3_adaptive/monitor.py` can collect:

```text
GPU utilization
VRAM utilization
CPU utilization
RAM utilization
temperature
```

GPU and VRAM values depend on NVIDIA tooling (`pynvml` or `nvidia-smi`). On systems without those tools, values remain zero and the monitor falls back to CPU-oriented behavior.

The `HeuristicController` in `controller.py` selects M0, M1, or M2 using threshold rules. It uses GPU utilization and temperature when GPU readings have been observed; otherwise it uses CPU and RAM thresholds.

The controller does not implement the hysteresis described in the paper. It directly recomputes the mode from current threshold values. The code tracks switch counts but does not impose separate upgrade and downgrade thresholds.

### Phase 4: Transition collection

Entry point: `phase_4_rl_data_collection/main.py`

Phase 4 is the data-generation path used by the learned policy. It uses a CPU-oriented state:

```text
state = [CPU utilization, RAM utilization, temperature, FPS, latency]
```

Actions are continuous:

```text
raw image-size action in [-1, 1]
raw skip action in [-1, 1]
```

The raw image-size action is mapped to a YOLO image size between 256 and 640 pixels, rounded to a multiple of 32. The raw skip action is mapped to a skip probability between 0 and 1.

The collector periodically samples a new action and probabilistically skips frames using the current skip probability. It records transitions containing:

```text
state
raw and applied action
reward
next state
done flag
```

The reward implemented in `exploration.py` is:

```text
10 * proxy accuracy
+ 0.5 * next-state FPS
- 0.1 * next-state latency
- 50 if temperature > 80 C
- 10 if CPU utilization > 90 percent
```

The proxy accuracy is the maximum confidence score among detections, not ground-truth accuracy or mAP.

`TransitionLogger` writes CSV and Parquet files. Existing transition files are present under `phase_4_rl_data_collection/output/`.

### Phase 5: Offline policy training

Entry point: `phase_5_offline_rl_training/train.py`

The training procedure is supervised behavioral cloning over the top 30 percent of transitions by reward:

1. Load the Phase 4 Parquet dataset.
2. Keep rows at or above the 70th reward percentile.
3. Extract the five state features.
4. Standardize them using saved means and standard deviations.
5. Train a PyTorch `ContinuousActor` with MSE loss.
6. Export the actor as TorchScript.

The network is:

```text
5 inputs -> Linear(128) -> ReLU -> Linear(128) -> ReLU
         -> 2 outputs -> Tanh
```

The two outputs represent continuous image-size and skip actions. The training script does not implement:

- Q-values
- Bellman or temporal-difference targets
- a target network
- experience replay
- epsilon-greedy exploration
- a discrete action head
- online policy updates

The Phase 5 output directory currently contains normalization arrays and plots, but the required `rl_adaptive_policy.pt` artifact is not present in this checkout.

The `d3rlpy_logs/` directories contain parameter files, but the repository's active `train.py` does not import or execute d3rlpy training.

### Phase 6: Evaluation

#### Accuracy evaluation

Entry point: `phase_6_evaluation/evaluate_accuracy.py`

The script compares:

```text
Baseline: YOLO at imgsz=640
Adaptive: YOLO at the image size returned by the TorchScript policy
```

It parses labelled pothole images, performs polygon or AABB matching at IoU 0.5, and reports precision, recall, F1, and a value labelled `mAP@0.5`.

The hardware state used to query the policy is synthetically generated as a three-stage ramp. It is not read from live telemetry during this evaluation. The stored report explicitly identifies the scenario as a simulated Raspberry Pi deployment.

The aggregate value labelled `mAP@0.5` is calculated as the arithmetic mean of per-image precision values. It is not standard average precision computed from a precision-recall curve.

The stored result shows the learned policy selecting `imgsz=320` for all 51 images. This is a constant selected resolution in that run, not evidence of dynamic three-mode switching.

#### Runtime/system-parameter evaluation

Entry point: `phase_6_evaluation/evaluate_system_params.py`

This script runs baseline and adaptive YOLO inference over a video and collects FPS, latency, CPU, RAM, temperature, image size, and detection counts. It generates plots and summary tables.

The current setting is `VIDEO_LOOPS = 1`, despite an outdated comment referring to four loops. Raw records are held in memory and are not saved as a reproducible CSV artifact by this script.

Temperature can be unavailable on Windows and some other systems. In that case, the monitor reports zero and temperature statistics are omitted.

## 5. Implemented State, Action, and Reward Versus Manuscript

| Concept      | Implemented repository                               | Manuscript claim                                      |
| ------------ | ---------------------------------------------------- | ----------------------------------------------------- |
| State        | 5 values: CPU, RAM, temperature, FPS, latency        | 6 values including GPU, VRAM, and previous mode       |
| Action       | Continuous image size and skip probability           | Discrete M0, M1, M2 modes                             |
| Learner      | Top-reward behavioral cloning                        | DQN-lite/contextual bandit                            |
| Network      | 5 -> 128 -> 128 -> 2 actor                           | 6-input Q-network with 32/32/16 layers                |
| Reward       | Confidence, FPS, latency, CPU/temperature thresholds | Exponential temperature penalty and switching penalty |
| Telemetry    | Python `psutil`, optional NVML in heuristic phase    | Claimed C HAL and `ctypes` integration                |
| Evaluation   | Partly simulated telemetry and custom metrics        | Claimed sustained real thermal experiments            |
| Optimization | No verified TensorRT implementation                  | TensorRT FP16 results claimed                         |

## 6. Evidence-Supported Contributions

The repository supports these claims:

- A YOLOv8 pothole detector was trained and used in inference pipelines.
- A fixed-resolution baseline was implemented with runtime telemetry logging.
- Resolution and frame skipping were explored as workload-control variables.
- A heuristic controller adapts inference using available system telemetry.
- A transition dataset containing state, action, reward, and next-state values was collected.
- A lightweight offline behavioral-cloning policy was trained from high-reward transitions.
- Accuracy and runtime comparison tooling was created.

## 7. Reproducibility and Risk Notes

1. The learned policy artifact required by Phase 6 is missing from the current checkout.
2. The accuracy evaluation uses synthetic hardware states.
3. The stored accuracy report, README, poster, progress report, and paper contain different metric values and method descriptions.
4. The reported mAP calculation is non-standard and should be renamed or replaced.
5. The paper's DQN, TensorRT, C telemetry HAL, online adaptation, and 15,000-frame thermal claims are not supported by the current source tree.
6. Several existing outputs were generated on another machine or environment, as shown by paths in stored reports.
7. The Phase 2/3 state class contains GPU/VRAM/RAM fields, but its `to_vector()` method omits RAM. This state definition should not be treated as the Phase 5 learner's input contract.
8. The Phase 4 skip action is trained as an output, but Phase 6 only uses the image-size output when running adaptive inference.

## 8. Recommended Basis for Paper Revision

The paper should describe the implemented method as an offline behavioral-cloning policy for continuous image-size adaptation, unless the missing DQN/TensorRT/C-HAL components are implemented and validated separately.

Before reporting final quantitative results, the project should:

- regenerate the policy artifact from a documented dataset;
- preserve raw evaluation records;
- use standard detection metrics, preferably Ultralytics validation or a clearly defined AP implementation;
- distinguish live hardware measurements from synthetic simulations;
- report the actual hardware, operating system, model file, video, number of frames, and runtime configuration;
- reconcile all README, poster, progress-report, and manuscript numbers against the same raw results.
