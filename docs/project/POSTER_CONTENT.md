# IDP Project Poster Content

## Project Title

**A Learned Adaptive Inference System for Real-Time YOLOv8 Pothole Detection on Edge Devices**

**Guide:** Dr. Rajeswara Rao K V S, Dept. of IE&M, RVCE

**Team:** Vishal K. Bhat, Vaibhav P. Rodappanavar, Naren, Hardik Paneru — Dept. of CSE, RVCE

---

## 1. Abstract

Static YOLOv8 pipelines on edge devices overheat and crash under sustained workloads. This work proposes a **Learned Adaptive Inference System** that monitors live hardware telemetry and dynamically switches between inference modes using a **DQN-based RL controller**. The learned policy achieves **48.2 FPS** with **75% lower variance** than the heuristic baseline, zero throttle events, and only **3.8% mAP drop**.

---

## 2. Problem Statement

Deploying YOLOv8 on edge devices for continuous road monitoring causes thermal throttling — the OS underclocks the CPU/GPU at ~85°C, leading to unpredictable frame drops and crashes. Static pipelines cannot adapt to the device's physical state, making sustained real-time pothole detection unreliable.

---

## 3. Objectives

- Develop a **hardware-aware adaptive inference system** that dynamically adjusts YOLOv8 parameters (resolution, frame skip) based on real-time telemetry.
- Formulate the resource management as an **MDP** and train a lightweight **DQN agent** to learn the optimal mode-switching policy.
- Build a **low-overhead C-based Telemetry HAL** for zero-blocking hardware state reads from Linux sysfs/procfs.

---

## 4. Methodology

### Diagram Description

The block diagram shows a closed-loop adaptive system with three layers:

- **Top layer — Inference Pipeline:** Dashboard camera video stream feeds into the YOLOv8 model, which outputs pothole detections (bounding boxes). The pipeline accepts dynamic parameters — input resolution (640/480/320) and frame skip count — from the controller.
- **Middle layer — Adaptive Controller (DQN-lite):** Receives a 6-dimensional state vector [GPU%, CPU%, Temp, FPS, VRAM%, Previous Mode] from the telemetry layer. Outputs one of three actions: M0 (full quality), M1 (balanced), or M2 (low load). The controller's 3-layer MLP (32→32→16) runs in <1ms on CPU.
- **Bottom layer — Telemetry HAL:** A C shared library (`libtelemetry.so`) that reads CPU temperature, CPU utilization, and RAM from `/sys/class/thermal`, `/proc/stat`, and `/proc/meminfo` using persistent file descriptors. Integrated into Python via `ctypes`.

---

## 5. Experimentation / Hardware / Software Model

*[Picture of circuit/model to be added]*

---

## 6. Results

*[Graphs / quantitative output to be added]*

---

## 7. Outcome

- Developed an **end-to-end adaptive inference system** that transforms static YOLOv8 into a dynamic, hardware-aware pipeline with zero thermal throttle events.
- The **RL controller learns thermal momentum** and makes anticipatory mode switches — achieving 75% lower FPS variance than rule-based heuristics.
- Built a **C Telemetry HAL** with <1ms overhead, enabling real-time hardware monitoring without blocking the inference pipeline.
- Research paper titled *"A Learned Adaptive Inference System for Real-Time YOLOv8 Pothole Detection on Edge Devices"* prepared for **IEEE Access** submission.

---

## 8. References

1. M. Ren et al., "An annotated street view image dataset for automated road damage detection," *Nature Scientific Data*, 2024.
2. S. Zhang et al., "OBC-YOLOV8n: An improved road damage detection model," *PeerJ Computer Science*, 2025.
3. S. Teerapittayanon et al., "BranchyNet: Fast inference via early exiting from deep neural networks," *ICPR*, 2016.
4. B. Taylor et al., "Adaptive deep learning model selection on embedded systems," *ACM VEE*, 2018.
5. M. Garcia et al., "Edge computing in IoT for smart cities," *All Multidisciplinary Journal*, 2025.

---
