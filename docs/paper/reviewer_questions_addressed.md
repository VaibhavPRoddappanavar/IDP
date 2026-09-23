# Reviewer Questions Addressed

This file records reviewer questions addressed by revisions to the manuscript. Each entry identifies the corresponding manuscript change and the evidence used.

## Reviewer 1, Concern 3: Rationale for Multiple Processing Configurations

**Reviewer question:**

Why are there two modes of processing? What would happen if all frames were processed at 320x320, 160x160, or 240x240 resolution?

**Answer in the revised manuscript:**

The system uses two complementary workload controls: input-resolution scaling reduces the computational cost of each processed frame, while frame skipping reduces the number of frames sent through inference. The Phase 1 characterization evaluates three interpretable operating points: $M_0=(640,0)$, $M_1=(480,1)$, and $M_2=(320,2)$. The measured results show increasing throughput and decreasing latency as the workload is reduced, but also fewer detections and lower average confidence. The learned controller is subsequently allowed to select continuous resolution and frame-skipping values rather than being restricted to these three points. The current experiments do not evaluate fixed 160x160 or 240x240 modes; these are identified as a limitation and follow-up experiment.

**Manuscript locations:**

- Methodology, Phase 1: Action Space Definition and Controlled Adaptation
- Experimental Results, Experimental Setup
- Experimental Results, Baseline vs. Heuristic Performance
- Discussion, Interpretation of Baseline and Deterministic Control Results

## Reviewer 1, Concern 4: Justification of the Proposed Method

**Reviewer question:**

Please justify every component of the proposed method and provide relevant references.

**Answer in the revised manuscript:**

The revised baseline-versus-controller comparison establishes the engineering motivation for adaptive workload control. The fixed $640 \times 640$ pipeline achieved only $3.31$ FPS and ended with an $87.04\%$ frame-drop rate in the recorded Raspberry Pi baseline run. The deterministic controller improved the measured rate to $5.35$ FPS and reduced the final frame-drop rate to $41.91\%$, while the logged core temperature remained below $60^\circ$C. These results justify adapting resolution and processing frequency at the application layer. The revision also states the limitation of the deterministic policy: it is reactive and stateless, which motivates the learned controller.

**Manuscript locations:**

- Experimental Results, Baseline vs. Heuristic Performance
- Discussion, Interpretation of Baseline and Deterministic Control Results
- Related Work and Methodology sections for the supporting literature and design rationale

## Reviewer 2, Concern 1: Frame Skipping and Edge-Cloud Offloading

**Reviewer question:**

Why was local adaptive inference used instead of deterministic edge-cloud offloading, and how can the system avoid unsafe loss of perception under thermal stress?

**Answer in the revised manuscript:**

The system is scoped to passive road-surface monitoring and infrastructure-condition logging rather than safety-critical autonomous vehicle control. The local architecture was selected to remain operational without depending on network availability or remote-server latency. The revised results now avoid claiming hard real-time guarantees: the static and deterministic runs are presented as runtime characterization, and the remaining frame loss is explicitly acknowledged. Edge-cloud offloading remains a valid complementary architecture when connectivity and bounded communication latency are available.

**Manuscript locations:**

- Related Work, Reinforcement Learning and Edge-Cloud Resource Optimization
- Discussion, Local Adaptive Inference Versus Edge-Cloud Offloading
- Discussion, Interpretation of Baseline and Deterministic Control Results
