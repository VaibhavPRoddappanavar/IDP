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

**Author response:** We agree. The earlier manuscript mixed values from different runs and used unsupported learned-controller claims. We performed an evidence audit and separated Raspberry Pi measurements, derived values, and simulated policy analyses.

**Author action:** We replaced the conflicting runtime and accuracy values in the abstract, experimental results, discussion, and conclusion. A canonical results ledger is provided at `phase_6_evaluation/output/canonical_numbers.md`. The revised manuscript states that the offline actor is an analysis artifact until a dedicated hardware-in-the-loop rollout is completed.

## Reviewer 2, Concern 2: Support thermal and real-time claims

**Reviewer concern:** Provide sustained-load tests, throttle records, consistent FPS definitions, and explicit timing requirements.

**Author response:** The available Raspberry Pi run is 213 seconds and contains no observed throttle event, but it is not a long-term validation. We therefore do not claim thermal-throttling prevention or hard real-time guarantees.

**Author action:** We define processed FPS, source coverage, and local inference latency in the Results discussion. We report mean and P95 latency where available, bound the thermal observation to the recorded duration, and identify repeated sustained-load testing with explicit throttle-state logging as required future work.

## Reviewer 2, Concern 3: Strengthen comparisons and ablations

**Reviewer concern:** Compare fixed resolutions, heuristic control, adaptive methods, and resolution/frame-skipping ablations.

**Author response:** The repository contains measured fixed operating points and a deterministic heuristic controller, but it does not contain a reproducible completed comparison against a learned closed-loop rollout or external adaptive framework.

**Author action:** We report the Raspberry Pi-only M0/M1/M2 operating points, the deterministic controller, and the available offline policy condition without conflating them. We explicitly identify resolution-only, skip-only, combined, and external-framework comparisons as future experiments rather than inventing results.

## Reviewer 2, Concern 4: Clarify the learning contribution

**Reviewer concern:** Reward-filtered behavioral cloning does not establish long-term optimization or anticipatory thermal control.

**Author response:** We agree. The implemented learner is reward-filtered behavioral cloning with a frozen continuous actor; it is not a DQN and does not prove anticipatory control, long-term optimization, or deployment superiority.

**Author action:** We removed DQN-style wording from the contribution claims, renamed the relevant discussion to “Limits on Learning and Thermal Claims,” and state that a hardware-in-the-loop rollout is required before making closed-loop policy claims.

## Reviewer 2, Concern 5: Validate the reward function

**Reviewer concern:** Justify maximum detection confidence as a proxy and assess reward-weight, filtering-threshold, and control-interval sensitivity.

**Author response:** Maximum confidence is only a proxy used during transition collection. It is not ground-truth accuracy and can reward miscalibrated confidence.

**Author action:** We now label the proxy explicitly, state its limitations, and document the reward formula and 5-frame interval. The revised manuscript identifies weight, top-k filtering, and interval sensitivity as required validation. No unsupported optimality claim is made.

## Reviewer 2, Concern 6: Expand accuracy evaluation

**Reviewer concern:** Use a larger and more diverse dataset, report small-object performance, uncertainty, and video-level missed detections.

**Author response:** The available archived evaluation contains 51 labeled images and a simulated telemetry schedule. It does not provide standard AP, small-object metrics, confidence intervals, or video-level missed-detection labels.

**Author action:** We relabel the archived output as a preliminary IoU-thresholded analysis, withdraw the historical pseudo-mAP interpretation, report the actual precision/recall/F1 values, and identify standard AP, small-object stratification, uncertainty intervals, and video-level labels as necessary follow-up evaluation.

Reviewer-suggested references have been included only where relevant to the topic and are not treated as a condition of acceptance.
