# Interdisciplinary Project Diary — Report of Activity Carried Out

## Week-wise Progress Report

---

### Week 1

**From Date:** 12-03-26 &nbsp; | &nbsp; **To Date:** 19-03-26

- Conducted a comprehensive literature review to identify current research limitations; formulated the core problem statement and identified the research gap.
- Selected pothole detection as the primary use case for real-world testing.

---

### Week 2

**From Date:** 20-03-26 &nbsp; | &nbsp; **To Date:** 27-03-26

- Conducted field research on the roads of Bangalore to curate a real-world dataset of road conditions and collected geo-atlas data from "Bangalore Live Pothole Map."
- Developed a baseline YOLOv8n model using modular training strategies and began researching a custom stratification module to improve the model's feature extraction and adaptability.

---

### Week 3

**From Date:** 28-03-26 &nbsp; | &nbsp; **To Date:** 03-04-26

- Implemented the baseline inference pipeline by running the trained `best.pt` model on a continuous video stream with fixed parameters (imgsz = 640, FP32, no frame skipping) and developed a frame-level performance logging system capturing FPS, latency, GPU/CPU utilization, RAM usage, temperature, and detection counts into structured CSV format.
- Established a reproducible benchmarking setup to serve as the reference baseline for all future comparisons.

---

### Week 4

**From Date:** 04-04-26 &nbsp; | &nbsp; **To Date:** 10-04-26

- Created multiple inference modes (M0, M1, M2) for controlled adaptation and implemented manual mode switching at fixed intervals (every 10 seconds).
- Analysed the resulting performance graphs and concluded on the accuracy–efficiency trade-offs across modes.

---

### Week 5

**From Date:** 11-04-26 &nbsp; | &nbsp; **To Date:** 17-04-26

- Created a comprehensive table of model parameters during different inference modes and during full-capacity runs; prepared the dataset for the RL agent.
- Explored several publication opportunities and shortlisted target venues.

---

### Week 6

**From Date:** 18-04-26 &nbsp; | &nbsp; **To Date:** 24-04-26

- Built a non-blocking telemetry loop to track live CPU, GPU, and temperature data during video inference; integrated hardware state polling asynchronously to ensure the YOLOv8 video processing pipeline remains completely uninterrupted.
- Reviewed supporting literature that strongly justified balancing mIoU, mAP, accuracy, and hardware trade-offs as a valid research direction.

---

### Week 7

**From Date:** 25-04-26 &nbsp; | &nbsp; **To Date:** 01-05-26

- Created dynamic inference modes that trade resolution for computational load and heat generation; deployed a rule-based controller that automatically degrades the YOLOv8 mode based on hardware conditions.
- Reviewed a comprehensive list of journals provided by the mentor, finalized the target publication list, and created an IEEE Access draft.

---

### Week 8

**From Date:** 02-05-26 &nbsp; | &nbsp; **To Date:** 08-05-26

- Formulated the hardware resource management problem as a Markov Decision Process (MDP) and designed a custom reward function that maximizes FPS throughput while penalizing temperature spikes.
- Ran stress tests to log thousands of (state, action, reward) tuples, building the training dataset for the DQN agent.

---

### Week 9

**From Date:** 09-05-26 &nbsp; | &nbsp; **To Date:** 15-05-26

- Developed a low-level Telemetry HAL (Hardware Abstraction Layer) in C that reads CPU temperature from `/sys/class/thermal`, computes CPU utilization via delta-based `/proc/stat` parsing, and extracts available RAM from `/proc/meminfo` — all through persistent file descriptors to minimize syscall overhead.
- Defined the `TelemetryData_t` struct and compiled the HAL as a shared library (`libtelemetry.so`) for zero-copy interoperability with the Python inference pipeline.

---

### Week 10

**From Date:** 16-05-26 &nbsp; | &nbsp; **To Date:** 22-05-26

- Integrated the C telemetry shared library into the YOLOv8 inference loop using Python `ctypes`, enabling per-frame hardware state reads (temperature, CPU%, RAM) with negligible latency compared to pure-Python `psutil` polling.
- Validated the end-to-end telemetry pipeline on the Raspberry Pi 5, confirming that real-time hardware metrics are now available to the RL agent's state vector without blocking or disrupting the video processing pipeline.

---

### Week 11

**From Date:** 23-05-26 &nbsp; | &nbsp; **To Date:** 29-05-26

- Implemented the DQN (Deep Q-Network) architecture with an experience replay buffer and target network; trained the agent offline on the collected dataset of hardware-state transitions with tuned hyperparameters (learning rate, discount factor γ, ε-greedy schedule).
- Integrated the trained DQN policy into the live inference pipeline, replacing the rule-based controller with the learned agent for real-time mode switching decisions.

---

### Week 12

**From Date:** 30-05-26 &nbsp; | &nbsp; **To Date:** 05-06-26

- Performed extensive validation experiments under varying workload conditions (different video resolutions, scene complexities, and ambient temperatures) to evaluate the robustness of the RL-based controller.
- Compiled a comprehensive results table comparing all three control strategies (static baseline vs. heuristic vs. DQN agent) across key metrics — average FPS, FPS stability, peak temperature, and mAP retention — and generated publication-ready plots.

---

### Week 13

**From Date:** 06-06-26 &nbsp; | &nbsp; **To Date:** 12-06-26

- Drafted the core sections of the research paper (methodology, experimental setup, results, and discussion) incorporating all validated metrics, comparison tables, and performance graphs.
- Completed the full manuscript including abstract, introduction, related work, and conclusion; incorporated mentor feedback and formatted per IEEE Access submission guidelines.

---

### Week 14

**From Date:** 13-06-26 &nbsp; | &nbsp; **To Date:** 19-06-26

- Conducted a final round of proofreading, cross-verified all reported metrics against raw experiment logs, and ensured consistency across all tables, figures, and citations in the manuscript.
- Submitted the final paper to the target journal/conference and prepared a project demonstration showcasing the end-to-end adaptive inference system in action.

---
