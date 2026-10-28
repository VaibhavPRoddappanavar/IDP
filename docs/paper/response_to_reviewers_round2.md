# Response to Reviewers

**Original Manuscript ID:** Access-2026-48720
**Original Article Title:** “A Learned Adaptive Inference System for Real-Time YOLOv8 Object Detection on Edge Devices with Pothole Detection as an Application”

**To:** IEEE Access Editor
**Re:** Response to reviewers

Dear Editor,

Thank you for allowing us to resubmit our manuscript and for the opportunity to address the reviewers' comments.

We are uploading (a) this point-by-point response, (b) an updated manuscript with the revisions highlighted, and (c) a clean manuscript. We also audited the reported values against the committed logs and policy artifacts. Claims that were not reproducible from those artifacts were withdrawn or explicitly classified as exploratory or future work.

Best regards, Vishal K. Bhat et al.

## Reviewer 1

Reviewer 1 indicated that the previous revision had addressed the earlier concerns and considered the manuscript suitable for consideration. We thank the reviewer for this assessment. The present revision preserves the expanded literature discussion and further improves result traceability in response to Reviewer 2.

## Reviewer 2, Concern 1: Resolve inconsistent results

**Reviewer concern:** Reported learned-controller improvements conflict with statements that hardware validation is incomplete and artifacts are missing or simulated.

**Author response:** We agree that the earlier presentation was inconsistent. In particular, it combined measurements from different experimental phases with an offline policy analysis whose telemetry schedule was simulated. It also presented learned-controller improvements as if they were validated closed-loop hardware results. We therefore re-audited the source logs, evaluation scripts, and policy artifacts and assigned every reported quantity one of four classifications: measured, derived, exploratory artifact, or simulated/unavailable. This audit showed that the Raspberry Pi measurements support fixed operating-point and deterministic-controller observations, but do not support a claim of learned-controller superiority on hardware.

**Author action:** We replaced the conflicting runtime and accuracy values in the abstract, experimental results, discussion, and conclusion. The revised manuscript now reports the available Raspberry Pi evidence directly: the Phase 0 run lasted 213 s with 3.31 processed FPS, 301.8 ms mean latency, 388.5 ms P95 latency, 53.2% mean CPU utilization, 29.5% mean RAM utilization, 56.6 °C mean temperature, and 60.6 °C maximum temperature. The fixed Phase 1 operating points achieved 3.08, 4.98, and 7.09 processed FPS for M0, M1, and M2, respectively. These values are not presented as a learned-policy comparison. The archived policy-condition analysis is explicitly identified as exploratory: it used a simulated state schedule and reported 61.18/70.27/65.41% baseline precision/recall/F1 and 67.24/52.70/59.09% policy-condition precision/recall/F1 on 51 labeled images. The revised manuscript states that the offline actor remains an analysis artifact until a dedicated hardware-in-the-loop rollout is completed.

## Reviewer 2, Concern 2: Support thermal and real-time claims

**Reviewer concern:** Provide sustained-load tests, throttle records, consistent FPS definitions, and explicit timing requirements.

**Author response:** We agree that a short run without a throttle event cannot establish sustained thermal stability or a hard real-time guarantee. The available Raspberry Pi baseline run lasted 213 s. During that recorded interval, the mean temperature was 56.6 °C and the maximum was 60.6 °C, with no observed throttle event in the available artifact. This is evidence about that particular run only; it does not demonstrate that throttling would not occur during longer operation, under different ambient conditions, or after repeated workload transitions. We also distinguish processed FPS from source coverage because frame skipping changes the number of source frames represented by each processed inference.

**Author action:** We define processed FPS as completed detector inferences per second and source coverage as the corresponding source-frame rate after accounting for the skip setting. For the fixed operating points, the measured processed FPS values are 3.08, 4.98, and 7.09 for M0, M1, and M2, while the derived source-coverage values are 3.08, 9.96, and 21.28 FPS. We report mean and P95 latency where available and explicitly limit the temperature statement to the recorded 213 s interval. We removed wording that implied thermal-throttling prevention or hard real-time operation. Repeated sustained-load experiments, ambient-condition reporting, explicit throttle-state logging, and an application-level timing requirement are identified as necessary follow-up validation.

## Reviewer 2, Concern 3: Strengthen comparisons and ablations

**Reviewer concern:** Compare fixed resolutions, heuristic control, adaptive methods, and resolution/frame-skipping ablations.

**Author response:** We agree that a learned method should not be judged against only one operating point. The available evidence does include measured fixed operating points and a deterministic heuristic controller, but it does not include a reproducible learned closed-loop rollout on the Raspberry Pi or a controlled comparison with an external adaptive framework. It also does not isolate the effects of resolution scaling and frame skipping in a complete factorial ablation. We therefore do not claim that the learned controller is superior to these alternatives.

**Author action:** We revised the comparison discussion to keep the evidence categories separate. The manuscript reports the Raspberry Pi-only M0/M1/M2 results and the deterministic-controller measurements as deployment evidence. It describes the offline policy condition separately and labels it exploratory because its telemetry schedule was simulated. We added an explicit limitation stating that resolution-only, skip-only, combined resolution-and-skipping, and external adaptive-method comparisons remain incomplete. These are framed as future experiments rather than being represented by fabricated or incomparable values.

## Reviewer 2, Concern 4: Clarify the learning contribution

**Reviewer concern:** Reward-filtered behavioral cloning does not establish long-term optimization or anticipatory thermal control.

**Author response:** We agree with this distinction. The implemented learner filters collected transitions by reward and trains a continuous actor by behavioral cloning. The actor has a five-dimensional input state consisting of CPU utilization, RAM utilization, temperature, processed FPS, and latency, and it is frozen for deployment. There is no Bellman target, replay-based value update, online policy improvement, or demonstrated long-horizon closed-loop rollout. Consequently, the method should be described as an offline policy-learning component, not as DQN or Q-learning, and the available evidence cannot establish anticipatory thermal control, long-term optimization, or deployment superiority.

**Author action:** We removed DQN-style and unsupported optimization wording from the abstract, contribution claims, methodology interpretation, results discussion, and conclusion. We describe the top-30%-filtered behavioral-cloning procedure and the frozen actor explicitly, and renamed the relevant discussion to “Limits on Learning and Thermal Claims.” The revised manuscript now states that a hardware-in-the-loop rollout with sequential state transitions, action execution, and measured thermal response is required before making closed-loop or anticipatory-control claims.

## Reviewer 2, Concern 5: Validate the reward function

**Reviewer concern:** Justify maximum detection confidence as a proxy and assess reward-weight, filtering-threshold, and control-interval sensitivity.

**Author response:** We agree that maximum detection confidence is an imperfect quality signal. It is convenient for transition collection because it is available without labels at inference time, but it measures the model's confidence rather than localization correctness. A detector can assign high confidence to an incorrect or poorly localized box, and the maximum over detections can also ignore missed objects. Thus, the reward can be used as a training-time proxy for prioritizing candidate transitions, but it cannot by itself validate detection quality or establish an optimal control policy.

**Author action:** We now label maximum confidence as a proxy throughout the methodology and results, state its calibration and missed-detection limitations, and document the reward construction and 5-frame control interval. We also state that the available artifacts do not contain a completed sensitivity study over reward weights, top-k filtering thresholds, or control intervals. The revised manuscript therefore presents those settings as design choices used in the prototype, not as validated optima, and identifies calibration-aware rewards, label-based quality checks, and sensitivity experiments as required follow-up work.

## Reviewer 2, Concern 6: Expand accuracy evaluation

**Reviewer concern:** Use a larger and more diverse dataset, report small-object performance, uncertainty, and video-level missed detections.

**Author response:** We agree that the archived evaluation is too limited to support broad accuracy claims. It contains 51 labeled images and uses a simulated telemetry schedule rather than a recorded hardware trajectory. The artifact does not provide standard COCO-style AP computation, a predefined small-object subset, confidence intervals or repeated-trial uncertainty estimates, or video-level labels for missed detections. In addition, the historical value called “mAP@0.5” was actually a mean of per-image precision values and was therefore not standard average precision.

**Author action:** We relabeled the archived output as a preliminary IoU-thresholded analysis and withdrew the historical pseudo-mAP interpretation. The revised manuscript reports the available precision/recall/F1 values transparently: 61.18/70.27/65.41% for the baseline condition and 67.24/52.70/59.09% for the policy condition. It also states that the policy selected 320 pixels in that archived analysis and that the corresponding 75% pixel-area reduction is a derived computational quantity, not an accuracy or speed improvement. We added explicit limitations and future-work requirements for a larger and more diverse dataset, standard AP and per-class metrics, small-object stratification, confidence intervals or repeated-run uncertainty, and frame-level/video-level missed-detection analysis.

Reviewer-suggested references have been included only where relevant to the topic and are not treated as a condition of acceptance.
