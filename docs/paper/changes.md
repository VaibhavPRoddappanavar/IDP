# access.tex — Change Log

---

| # | Section  | What Changed | Summary of Change | Reason |
| - | -------- | ------------ | ----------------- | ------ |
| 1 | Abstract | Rewrote all 3 paragraphs + corrected numbers | Para 1: led with RPi thermal throttling, 5 correct telemetry signals, continuous actor output; Para 2: replaced methodology walkthrough with one factual sentence on pothole detection use case; Para 3: corrected F1 drop (2.80% → 1.8%) and Precision gain (1.90% → 1.2%) from actual chart | GPU language was wrong for RPi; "Contextual Bandit" was inaccurate; README numbers didn't match evaluation chart |
| 2 | Keywords | Updated and expanded list (9 → 16) | Removed: `Contextual Bandit, Deep Q-Networks`; Added: `Offline Reinforcement Learning, Behavioral Cloning, Raspberry Pi, Markov Decision Process, Dynamic Resolution Scaling, Frame Skipping, Telemetry-Driven Control, Resource-Constrained Inference, Road Damage Detection, IoT Edge Deployment` | Old keywords didn't match actual techniques; new ones improve discoverability |
| 3 | References and compile compatibility | Expanded bibliography and added caption guard | Replaced the older 15-entry bibliography with 24 references covering pothole detection, edge inference, offloading, thermal management, adaptive inference, and early exits; defined `\xfigwd` when absent to support the IEEE Access caption macro | Reviewer requested stronger literature support; Overleaf reported an undefined `\xfigwd` during caption processing |

---

*Last updated: 2026-09-20*
