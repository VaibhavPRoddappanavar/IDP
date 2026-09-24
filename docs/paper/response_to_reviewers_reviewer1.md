# Response to Reviewers

**Original Manuscript ID:** Access-2026-34151  
**Original Article Title:** “A Learned Adaptive Inference System for Real-Time YOLOv8 Pothole Detection on Edge Devices”  
**Revised Article Title:** “A Learned Adaptive Inference System for Real-Time YOLOv8 Object Detection on Edge Devices with Pothole Detection as an Application”

**To:** IEEE Access Editor  
**Re:** Response to Reviewers (Reviewer 1)

Dear Editor,

Thank you for allowing us to resubmit our manuscript and for giving us the opportunity to address the reviewers’ valuable comments.

We have thoroughly revised the manuscript to incorporate all recommendations and clarify our methodology, experimental setup, camera acquisition geometry, and baseline justifications. We are uploading (a) our point-by-point response to the comments below under “Author’s Response Files,” (b) an updated manuscript with yellow highlighting indicating the changes as the “Highlighted PDF,” and (c) a clean updated manuscript without highlights as the “Main Manuscript.”

Best regards,  
Vishal K. Bhat et al.

---

## Reviewer 1

### Concern 1: Missing references and insufficient justification

**Reviewer’s concern:**  
*There are missing references and the number of references is not sufficient for a journal publication. As a result, the justification of the proposed method is poor.*

**Author response:**  
We thank the reviewer for identifying this weakness. We agree that the original manuscript did not provide sufficient context for our design choices or adequately position the proposed system within the existing literature. In the revised manuscript, we significantly expanded the literature review and grounded the proposed method across four foundational research areas:
1. **Pothole and Road-Damage Detection on Edge Devices:** Reviewing lightweight convolutional and transformer-based architectures for road surface inspection.
2. **Dynamic Neural Networks and Adaptive Inference:** Discussing early-exit networks, dynamic resolution scaling, and frame-skipping algorithms.
3. **Thermal Management in Edge Computing:** Examining hardware-level throttling mechanisms, DVFS, and thermal inertia in embedded systems.
4. **Reinforcement Learning for Edge Resource Management:** Reviewing MDP formulations and learned control for runtime trade-offs.

Furthermore, the revised manuscript explicitly differentiates our contribution from prior work. Existing studies predominantly focus on modifying detector architectures, designing static pruning/early-exit schemes, or delegating control to OS-level hardware governors (e.g., DVFS). In contrast, our approach introduces an application-level, continuous controller applied to an unmodified YOLOv8n detector. By continuously modulating input resolution and frame-skip probabilities based on live multi-variable hardware telemetry, our system proactively manages computational and thermal loads without requiring model retraining or architectural modifications.

**Author action:**  
We have updated the manuscript as follows:
1. **Restructured Section II (Related Work):** Created dedicated, structured subsections covering *Pothole Detection on Edge Devices*, *Dynamic Neural Networks and Adaptive Inference*, *Thermal Management in Edge Computing*, and *Reinforcement Learning and Edge-Cloud Resource Optimization*.
2. **Expanded Bibliography (References [1]–[24]):** Expanded the reference list from 15 to 24 high-quality journal and conference publications covering YOLO variants for road inspection, dynamic inference frameworks, embedded thermal management, and RL-based resource allocation.
3. **Methodological Justification:** Expanded Sections I (Introduction), III (System Architecture), IV (Methodology), and VII (Discussion) to provide comprehensive theoretical and empirical justification for every component (5D telemetry state, dual continuous action space, 5-frame decision interval, FBC offline learning, and TorchScript deployment).

---

### Concern 2: Camera placement, camera angle, camera height, field of view, and motion blur

**Reviewer’s concern:**  
*There should be discussion on camera placement such as camera angle, camera height and camera field of view so that the proposed method can make sense. (Additional Question 2: how they coped with motion blur).*

**Author response:**  
We completely agree with the reviewer that camera placement and acquisition geometry are crucial for interpreting road-inspection vision systems. Camera height, pitch angle, and field of view directly dictate the visible road surface area, perspective distortion, and the scale (pixel footprint) of potholes across distance. Furthermore, vehicle velocity and camera shutter speed determine motion blur characteristics.

In the revised manuscript, we have added comprehensive camera and acquisition specifications to **Section V-A (Experimental Setup)**:
- **Mounting Position & Height:** Front-facing dashboard camera mounted centrally behind the vehicle's front windshield at an approximate height of $1.3\,\text{m}$ ($1.2\text{--}1.5\,\text{m}$ operational range) above the road surface.
- **Pitch Angle:** Downward pitch angle of approximately $12^\circ\text{--}15^\circ$ toward the roadway horizon.
- **Field of View (FOV):** Wide-angle lens with a horizontal FOV of $\approx 120^\circ$, providing complete coverage of the active travel lane and adjacent road shoulders.
- **Visible Road Surface:** Captures the asphalt surface spanning $5\text{--}30\,\text{m}$ in front of the vehicle. Potholes in the near-to-mid field ($5\text{--}15\,\text{m}$) occupy sufficient spatial resolution even under downscaled input dimensions ($480\times 480$), whereas potholes at the far horizon ($>20\,\text{m}$) become small-scale targets sensitive to aggressive downscaling ($320\times 320$).
- **Video Format & Motion Blur Handling:** 1080p Full HD ($1920\times 1080$ at 30 FPS, H.264 MP4). Under typical urban inspection speeds ($30\text{--}50\,\text{km/h}$), daylight auto-exposure maintains fast shutter speeds ($\ge 1/500\,\text{s}$), preventing severe motion blur. We also explicitly discuss the impact of high-speed motion blur and frame skipping as an operational limitation and designate multi-camera geometry and speed-dependent ablation studies as future research directions.

**Author action:**  
1. Updated **Section V-A (Experimental Setup)** under the paragraph *“Camera Setup, Geometry, and Video Acquisition”* detailing exact camera height ($1.3\,\text{m}$), pitch angle ($12^\circ\text{--}15^\circ$), horizontal FOV ($120^\circ$), visible range ($5\text{--}30\,\text{m}$), 1080p @ 30 FPS video acquisition, and daylight shutter speed motion blur mitigation.
2. Updated **Section VII (Discussion)** under *“State Design and Future Context-Aware Extensions”* and *“Future Research Directions”* to explicitly characterize camera geometry dependencies, vehicle speed integration, and high-speed motion-blur evaluations.

---

### Concern 3: Rationale for two processing modes and alternatives such as 320x320, 240x240, and 160x160

**Reviewer’s concern:**  
*Why are there two modes of processing? What if we process frames at 320x320 resolution all the time? Or, what if we process frames at 160x160 or 240x240 resolutions?*

**Author response:**  
We thank the reviewer for this insightful question and welcome the opportunity to clarify the multi-tier workload control architecture:

1. **Complementary Roles of Two Control Knobs:**  
   The system modulates two orthogonal parameters:
   - *Input Resolution Scaling ($imgsz$):* Directly scales per-frame computational complexity (floating-point operations scale quadratically, $\mathcal{O}(W \times H)$). Downscaling from $640\times 640$ to $320\times 320$ yields a $75\%$ reduction in processed pixel area and FLOPs.
   - *Frame Skipping ($p_{skip}$):* Regulates temporal processing frequency and pipeline queue pressure. Bypassing frames entirely avoids model execution while keeping the system responsive to incoming sensor streams.
   
2. **Why Not 320x320 All the Time?:**  
   Processing frames at $320\times 320$ continuously reduces computational load and lowers latency, but it causes significant degradation in detection accuracy—particularly for small, distant potholes. As detailed in our accuracy evaluations, $480\times 480$ retains high fidelity (less than 2% drop in F1-score relative to $640\times 640$), whereas $320\times 320$ suffers substantial recall and precision losses on small objects. Continuous execution at $320\times 320$ is sub-optimal; the controller should dynamically operate at higher resolutions whenever thermal and CPU headroom permit, falling back to lower resolutions only under severe resource contention.

3. **Continuous Action Space vs. Discrete Heuristic Modes:**  
   The discrete modes ($M_0=(640,0)$, $M_1=(480,1)$, $M_2=(320,2)$) in Phase 1 and Phase 2/3 serve solely as interpretable characterization points. The learned controller in Phase 4/5/6 operates over a **continuous action space**:
   $$imgsz_t \in \{256, 288, 320, \ldots, 640\} \quad (\text{quantized in multiples of 32}), \quad p_{skip,t} \in [0, 1]$$
   
4. **Why Not 240x240 or 160x160?:**  
   The lower bound of $256\times 256$ is enforced because YOLOv8's feature extraction backbone utilizes a total downsampling stride of 32 across its 5 pyramid stages ($P_1\text{--}P_5$). At $160\times 160$, the deepest feature map shrinks to a mere $5\times 5$ grid, destroying spatial features necessary to detect small asphalt cracks and potholes. Empirical tests confirmed negligible detection capability below $256\times 256$. We have added this architectural justification and documented a wider resolution ablation as future work.

**Author action:**  
1. Revised **Section IV-B (Phase 1: Action Space Definition and Controlled Adaptation)** to provide mathematical FLOP derivations for resolution scaling and explain the complementary role of frame skipping.
2. Revised **Section IV-D (Phase 4: MDP Formulation)** to define the continuous action space mapping ($imgsz \in [256, 640]$, $p_{skip} \in [0, 1]$) and justify the lower bound of 256 based on YOLOv8 stride-32 backbone constraints.
3. Updated **Section V-D (Model Accuracy and Fidelity Trade-offs)** and **Section VII (Discussion)** to discuss the accuracy penalties of aggressive downscaling ($320\times 320$) and outline comprehensive resolution sweeps ($160\times 160$, $240\times 240$) as future work.

---

### Concern 4: Justification of every component of the proposed method

**Reviewer’s concern:**  
*Justify every bit of your proposed method and provide relevant references on them.*

**Author response:**  
We have systematically reviewed and justified every stage of the proposed end-to-end architecture with theoretical rationales, empirical measurements on the physical Raspberry Pi 5 platform, and supporting citations:

- **Baseline Profiling (Phase 0):** Justifies the need for adaptive control by measuring the failure mode of static $640\times 640$ inference on Raspberry Pi 5 (achieving only $3.31\,\text{FPS}$ and suffering an $87.04\%$ cumulative frame-drop rate).
- **5-Dimensional Telemetry State ($s_t = [U_{cpu}, U_{ram}, T_{core}, FPS_{avg}, L_t]$):** Combines system resource utilization ($U_{cpu}, U_{ram}$), thermal state ($T_{core}$ capturing physical thermal inertia), and perceptual throughput ($FPS_{avg}, L_t$) collected asynchronously at $2\,\text{Hz}$ via `psutil` kernel interface without GIL overhead (Section III-A).
- **Decision Interval (5 frames):** Stabilizes telemetry signals while ensuring sub-second control agility without adding runtime overhead to the inference loop (Section IV-D).
- **Reward Function Formulation:** Formulates priorities reflecting real-world edge deployment ($R_t = 10\hat{a}_t + 0.5 FPS_t - 0.1 L_t - 50\cdot\mathbf{1}[T_{core}>80] - 10\cdot\mathbf{1}[U_{cpu}>90]$), heavily penalizing thermal runaway ($>80^\circ\text{C}$) and CPU saturation while rewarding detection confidence and throughput (Section IV-D).
- **Reward-Filtered Behavioral Cloning (Phase 5):** Collects offline transitions ($N=3,033$) via uniform random exploration and filters the top 30% highest-reward transitions. Trains a compact $5\to 128\to 128\to 2$ MLP via MSE loss, preventing low-reward / thermal-overload state imitation while guaranteeing safe, offline policy development (Section IV-E).
- **TorchScript Frozen Deployment (Phase 6):** Eliminates online exploration risk and learning overhead during vehicle operation; forward pass executes in $<1\,\text{ms}$ deterministically on CPU (Section IV-F).

**Author action:**  
We updated the entire manuscript across Sections I, II, III, IV, V, and VII to ensure every single design decision, state variable, action bound, reward coefficient, and architecture layer is explicitly justified and supported by citations [1]–[24].

---

### Additional Questions (Reviewer 1)

**Question 1: Does the paper contribute to the body of knowledge?**  
*Reviewer comment: Yes, the paper proposes a solution to thermal throttling of edge devices when they were running YOLOv8 model to detect potholes on roads. However, there are missing references on the topic.*  
**Author response & action:** As detailed under Concern 1, we expanded the literature review to 24 relevant references covering edge road damage detection, adaptive inference, thermal management, and reinforcement learning.

**Question 2: Is the paper technically sound?**  
*Reviewer comment: Yes, the paper is technically sound, however, the authors should justify why they chose such a method by discussing camera angle, camera field of view, camera height relative to the road surface, and how they coped with motion blur.*  
**Author response & action:** As detailed under Concern 2, we added complete camera geometry specifications (mounting height $1.3\,\text{m}$, downward pitch $12^\circ\text{--}15^\circ$, FOV $120^\circ$, road surface $5\text{--}30\,\text{m}$, daylight exposure for motion blur mitigation) to Section V-A.

**Question 3: Is the subject matter presented in a comprehensive manner?**  
*Reviewer comment: Yes, the subject matter is presented in a comprehensive manner, however, more references and justification over the design is required.*  
**Author response & action:** As detailed under Concern 1 and Concern 4, we added in-depth justifications for every architectural and algorithmic component with 24 supporting citations across all sections.

**Question 4: Are the references provided applicable and sufficient?**  
*Reviewer comment: Yes.*  
**Author response:** We thank the reviewer for confirming the applicability of our references.

**Question 5: Are there references that are not appropriate for the topic being discussed?**  
*Reviewer comment: No.*  
**Author response:** We thank the reviewer for this confirmation.

---

## Note on Suggested References

All references included in the revised manuscript were evaluated strictly for technical merit and relevance to pothole detection, dynamic resolution scaling, embedded thermal management, edge computing, and offline reinforcement learning.
