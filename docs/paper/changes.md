# access.tex — Change Log

---

| # | Section | What Changed | Summary of Change | Reason |
| - | - | - | - | - |
| 1 | Title | Changed the paper title | Clarified that the system targets general YOLOv8 object detection on edge devices, with pothole detection as the application. | Aligns the paper scope with the proposed system. |
| 2 | Abstract | Rewrote all 3 paragraphs and corrected metrics | Updated the thermal-inference motivation, telemetry description, application context, and reported F1 and precision values. | Corrects scope and aligns claims with the evaluation. |
| 3 | Keywords | Updated and expanded list | Added terms for offline reinforcement learning, behavioral cloning, Raspberry Pi, dynamic resolution scaling, frame skipping, and telemetry-driven control. | Improves discoverability and reflects the implemented techniques. |
| 4 | Introduction | Revised all 3 paragraphs | Reframed pothole detection as an application of adaptive YOLOv8 inference on resource-constrained edge devices. | Aligns the motivation with the revised paper scope. |
| 5 | References and compile compatibility | Expanded bibliography and added caption guard | Added references covering pothole detection, edge inference, offloading, thermal management, adaptive inference, and early exits; added the `\\xfigwd` compatibility guard. | Strengthens literature support and fixes IEEE Access caption compatibility. |
| 6 | System Architecture | Revised telemetry, controller, and inference pipeline descriptions | Documented the five telemetry signals, continuous `imgsz` and `p_skip` control, probabilistic frame skipping, dynamic resolution, and feedback measurements. | Reflects the current Raspberry Pi implementation. |
| 7 | Related Work | Replaced the literature review | Added focused subsections covering pothole detection, dynamic neural networks, thermal management, and reinforcement learning, with distinctions from the proposed pipeline-level controller. | Provides a current, structured comparison with related approaches and clarifies the contribution. |

---

_Last updated: 2026-09-22_
