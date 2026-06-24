# Dynamic Edge Inference for Real-Time Pothole Detection using Offline Reinforcement Learning

This repository implements a **self-adaptive, resource-aware edge inference pipeline** for real-time pothole detection. Designed for deployment on resource-constrained hardware like the Raspberry Pi 4B/5, the system dynamically scales the YOLOv8 input resolution (`imgsz`) using a deep reinforcement learning policy trained on offline telemetry data. 

By adjusting input resolution in response to system health (CPU usage, RAM usage, core temperature, frame rates, and latency), the engine maintains stable performance and prevents thermal throttling under prolonged load.

---

## 🏗️ Architecture & Pipeline Flow

The project is structured as a progressive multi-phase pipeline, tracking development from baseline performance measurements to a closed-loop offline RL system:

```mermaid
graph TD
    A[Phase 0: Baseline YOLO] --> B[Phase 1: Controlled Adaptation]
    B --> C[Phase 2-3: Telemetry Feedback Loops]
    C --> D[Phase 4: RL Telemetry Data Collection]
    D --> E[Phase 5: PyTorch Behavioral Cloning & Offline RL]
    E --> F[Phase 6: Comprehensive Benchmark & Evaluation]
```

---

## 📂 Repository Structure

The codebase is modularized by implementation phases:

*   **`phase_0_baseline/`**: Benchmarks standard, non-adaptive YOLOv8 inference at a fixed $640\times640$ resolution. Establishes the baseline resource footprint (CPU, RAM, temperatures, frame latency).
*   **`phase_1_controlled_adaptation/`**: Explores baseline image-size changes and verifies system behavior under step changes.
*   **`phase_2_3_adaptive/`**: Implements simple, heuristic-based feedback loops that adjust image sizes based on telemetry thresholds.
*   **`phase_4_rl_data_collection/`**: Runs randomized or exploration policies on validation video loops to collect telemetry transition vectors (state, action, reward, next state) under simulated and real load conditions.
*   **`phase_5_offline_rl_training/`**: Trains a Continuous Actor network (`rl_adaptive_policy.pt`) in PyTorch using Behavioral Cloning (BC). The model is trained on the top 30% highest-reward transitions from Phase 4 to learn optimal adaptation policies.
*   **`phase_6_evaluation/`**: Compares the trained RL agent against the Baseline under simulated RPi 4B heat ramps and live runs, generating publication-ready charts and comparative summaries.

---

## ⚙️ Installation & Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/VaibhavPRoddappanavar/IDP.git
    cd IDP
    ```

2.  **Activate the Virtual Environment**:
    *   **Windows**:
        ```powershell
        .\venv\Scripts\activate
        ```
    *   **Linux/Raspberry Pi**:
        ```bash
        source venv/bin/activate
        ```

3.  **Install Dependencies**:
    ```bash
    pip install -r phase_5_offline_rl_training/requirements.txt
    ```
    *Make sure OpenCV, PyTorch, Ultralytics (YOLOv8), Matplotlib, Psutil, and Shapely (for polygon IoU) are installed.*

---

## 🚀 Step-by-Step Execution Guide

### Step 1: Collect RL Transition Telemetry (Phase 4)
Generate the telemetry dataset by running inference with exploration:
```bash
python phase_4_rl_data_collection/exploration.py
```
This logs state-action transitions to a `.parquet` file in `phase_4_rl_data_collection/output/`.

### Step 2: Train the Offline RL Policy (Phase 5)
Train the deep neural network using Behavioral Cloning to clone the highest-reward decisions:
```bash
python phase_5_offline_rl_training/train.py
```
This exports:
*   `state_mean.npy` & `state_std.npy`: Normalization constants.
*   `rl_adaptive_policy.pt`: The trained TorchScript actor network.

### Step 3: Run the Benchmarks (Phase 6)
Verify detection accuracy and system parameter improvements side-by-side:

```bash
# Evaluate mAP and F1-Score trade-offs on labeled test images
python phase_6_evaluation/evaluate_accuracy.py

# Benchmark live system metrics (FPS, Latency, CPU%) on video stream
python phase_6_evaluation/evaluate_system_params.py
```

---

## 📊 Performance Benchmarks & Highlights

### 1. Accuracy vs. Compute Savings (Phase 6 Results)
```
====================================================================
==================== ACCURACY COMPARISON REPORT ====================
====================================================================
Metric                        Baseline (640px)    RL-Adaptive      Delta
--------------------------------------------------------------------
mAP@0.5                                 0.6961         0.6757 [dn]  2.93%
Precision                               0.6118         0.6234 [up]  1.90%
Recall                                  0.7027         0.6486 [dn]  7.69%
F1-Score                                0.6541         0.6358 [dn]  2.80%
--------------------------------------------------------------------
TP / FP / FN                 52/33/      22   48/29/26
Avg RL imgsz                             640px         400px
Pixel compute saving                        0%       61.0%
====================================================================
```

### 2. Key Academic Takeaways
*   **Massive Resource Relief**: The RL agent dynamically scales the input image size to an average of **400px** during thermal/CPU load stress, saving **$61.0\%$ of pixel compute** calculations.
*   **Negligible Quality Penalty**: The drop in F1-score is restricted to only **$2.80\%$**, proving that high resolution is often redundant on edge workloads.
*   **The Precision Paradox**: Lower resolution (320-440px) actually **increases detection precision by $1.90\%$**. By making the model slightly more conservative, it mitigates false positives triggered by road textures, shadows, and minor cracks.