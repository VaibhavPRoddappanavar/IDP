"""
evaluate_system_params.py — Phase 6: System Parameters Comparison
==================================================================
Runs YOLOv8 inference on test_video_1.mp4 (looped 4 times) under two regimes:

  1. BASELINE  — fixed imgsz=640, every frame
  2. RL-ADAPTIVE — imgsz decided each frame by the trained offline-RL policy
                   based on live hardware state (CPU, RAM, temp, FPS, latency)

Metrics collected per frame:
    fps, latency_ms, cpu_usage_percent, ram_usage_percent, temperature

Outputs (saved to phase_6_evaluation/output/):
  1. sysparams_timeseries.png      — FPS + CPU + Latency over time
  2. sysparams_bar_comparison.png  — Mean values grouped bar chart
  3. sysparams_boxplots.png        — Distribution box plots
  4. sysparams_radar.png           — Radar / spider chart (normalised)
  5. sysparams_summary_table.png   — Paper-ready summary table

Usage:
    python evaluate_system_params.py

Notes:
  * Temperature may read 0.0 on Windows (psutil limitation) — handled gracefully.
  * Designed for CPU-only machines (Windows first, then Raspberry Pi).
"""

import sys
import time
import threading
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import psutil
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

from ultralytics import YOLO

# ---- PATHS ----
BASE_DIR   = Path(__file__).resolve().parent
ROOT_DIR   = BASE_DIR.parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_PATH = ROOT_DIR / "BaseModelTraining" / "test" / "test_video_1.mp4"

_MODEL_CANDIDATES = [
    ROOT_DIR / "BaseModelTraining" / "test" / "best_new_model.pt",
    ROOT_DIR / "BaseModelTraining" / "test" / "best.pt",
    ROOT_DIR / "BaseModelTraining" / "best.pt",
]
MODEL_PATH = next((p for p in _MODEL_CANDIDATES if p.exists()), None)

RL_DIR          = ROOT_DIR / "phase_5_offline_rl_training" / "output"
RL_POLICY_PATH  = RL_DIR / "rl_adaptive_policy.pt"
STATE_MEAN_PATH = RL_DIR / "state_mean.npy"
STATE_STD_PATH  = RL_DIR / "state_std.npy"

CONF_THRESHOLD = 0.25
VIDEO_LOOPS    = 1        # loop the 27s video 4 times per run
MAX_FRAMES     = 9999999  # safety cap (set low during debugging if needed)

# ---- STYLE ----
DARK_BG      = "#0f1117"
PANEL_BG     = "#1a1d2e"
BASELINE_CLR = "#4fc3f7"
RL_CLR       = "#81c784"
ACCENT_CLR   = "#ffcc02"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": DARK_BG, "axes.facecolor": PANEL_BG,
    "axes.labelcolor": "#e0e0e0", "axes.edgecolor": "#3a3d4e",
    "text.color": "#e0e0e0", "xtick.color": "#c0c0c0",
    "ytick.color": "#c0c0c0", "grid.color": "#2a2d3e",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.framealpha": 0.2, "legend.edgecolor": "#3a3d4e",
})

# ============================================================
# SYSTEM MONITOR  (non-blocking background thread)
# ============================================================
class SystemMonitor:
    def __init__(self, poll_interval=0.5):
        self._interval  = poll_interval
        self._cpu       = 0.0
        self._ram       = 0.0
        self._temp      = 0.0
        self._running   = False
        self._thread    = None

    def start(self):
        if self._running: return
        self._running = True
        psutil.cpu_percent(interval=None)          # prime
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread: self._thread.join(timeout=2.0)

    def _loop(self):
        while self._running:
            self._cpu  = psutil.cpu_percent(interval=None)
            self._ram  = psutil.virtual_memory().percent
            self._temp = self._read_temp()
            time.sleep(self._interval)

    @staticmethod
    def _read_temp():
        try:
            temps = psutil.sensors_temperatures()
            if not temps: return 0.0
            for _, entries in temps.items():
                if entries: return entries[0].current
        except (AttributeError, OSError): pass
        return 0.0

    def snapshot(self):
        return {"cpu": self._cpu, "ram": self._ram, "temp": self._temp}

# ============================================================
# RL POLICY
# ============================================================
def load_rl_policy():
    for p in [RL_POLICY_PATH, STATE_MEAN_PATH, STATE_STD_PATH]:
        if not p.exists():
            print(f"[ERROR] Not found: {p}")
            print("        Run phase_5_offline_rl_training/train.py first.")
            sys.exit(1)
    policy = torch.jit.load(str(RL_POLICY_PATH))
    policy.eval()
    mean = np.load(STATE_MEAN_PATH)
    std  = np.load(STATE_STD_PATH)
    print(f"[INFO] RL policy loaded ({RL_POLICY_PATH.name})")
    return policy, mean, std

def query_rl_imgsz(policy, mean, std, cpu, ram, temp, fps, latency_ms):
    try:
        state = np.array([[cpu, ram, temp, fps, latency_ms]], dtype=np.float32)
        norm  = (state - mean) / std
        with torch.no_grad():
            action = policy(torch.from_numpy(norm)).numpy()[0]
        
        # Map raw action output from [-1.0, 1.0] to imgsz range [256, 640]
        mapped_imgsz = 256 + (((action[0] + 1.0) / 2.0) * (640 - 256))
        # Round to the nearest multiple of 32 for YOLO compliance
        return int(round(mapped_imgsz / 32) * 32)
    except Exception:
        # Fallback in case of runtime evaluation failure
        return 640

# ============================================================
# INFERENCE LOOP
# ============================================================
class FPSCounter:
    def __init__(self, smoothing=0.9):
        self.smoothing = smoothing
        self.fps = 0.0
    def update(self, latency_s):
        if latency_s > 0:
            cur = 1.0 / latency_s
            self.fps = cur if self.fps == 0 else (self.fps*self.smoothing + cur*(1-self.smoothing))
        return self.fps

def run_inference_loop(model, monitor, label, imgsz_fixed=None,
                       policy=None, state_mean=None, state_std=None):
    """
    Run inference on VIDEO_PATH looped VIDEO_LOOPS times.
    imgsz_fixed: int  → use that size every frame (baseline)
    policy      : not None → query RL policy for imgsz each frame (RL mode)
    Returns list of per-frame metric dicts.
    """
    if not VIDEO_PATH.exists():
        print(f"[ERROR] Video not found: {VIDEO_PATH}"); sys.exit(1)

    fps_ctr    = FPSCounter(smoothing=0.9)
    records    = []
    frame_idx  = 0
    loop_count = 0
    imgsz_now  = imgsz_fixed if imgsz_fixed else 640

    print(f"\n[{label}] Starting inference  (loops={VIDEO_LOOPS}) ...")
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {VIDEO_PATH}"); sys.exit(1)

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps_native  = cap.get(cv2.CAP_PROP_FPS) or 30
    target_total = VIDEO_LOOPS * total_video_frames
    report_every = max(1, total_video_frames // 5)

    try:
        while loop_count < VIDEO_LOOPS:
            ret, frame = cap.read()
            if not ret:
                # end of video — restart for next loop
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                loop_count += 1
                if loop_count >= VIDEO_LOOPS:
                    break
                continue

            # hardware snapshot (non-blocking cached values)
            hw  = monitor.snapshot()
            cur_fps     = fps_ctr.fps
            cur_latency = 1000.0 / cur_fps if cur_fps > 0 else 999.0

            # decide imgsz
            if policy is not None:
                imgsz_now = query_rl_imgsz(policy, state_mean, state_std,
                                            hw["cpu"], hw["ram"], hw["temp"],
                                            cur_fps, cur_latency)
            else:
                imgsz_now = imgsz_fixed

            # inference
            t_s     = time.perf_counter()
            result  = model.predict(frame, imgsz=imgsz_now,
                                    conf=CONF_THRESHOLD, verbose=False)[0]
            t_e     = time.perf_counter()
            lat_ms  = (t_e - t_s) * 1000.0
            fps_val = fps_ctr.update(t_e - t_s)

            records.append({
                "frame"       : frame_idx,
                "fps"         : fps_val,
                "latency_ms"  : lat_ms,
                "cpu_pct"     : hw["cpu"],
                "ram_pct"     : hw["ram"],
                "temp"        : hw["temp"],
                "imgsz_used"  : imgsz_now,
                "num_det"     : len(result.boxes),
            })

            frame_idx += 1
            if frame_idx % report_every == 0:
                pct = 100 * frame_idx / target_total
                print(f"  [{label}] {frame_idx}/{target_total} frames "
                      f"({pct:.0f}%)  FPS={fps_val:.1f}  "
                      f"lat={lat_ms:.1f}ms  CPU={hw['cpu']:.1f}%  "
                      f"imgsz={imgsz_now}")
    except KeyboardInterrupt:
        print(f"\n[{label}] Interrupted.")
    finally:
        cap.release()

    print(f"  [{label}] Finished: {len(records)} frames processed.")
    return records

# ============================================================
# STATISTICS HELPERS
# ============================================================
def summarise(records, key):
    vals = [r[key] for r in records if r[key] > 0]
    if not vals: return {"mean":0,"median":0,"std":0,"p5":0,"p95":0}
    arr = np.array(vals)
    return {"mean"  : float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std"   : float(np.std(arr)),
            "p5"    : float(np.percentile(arr,5)),
            "p95"   : float(np.percentile(arr,95))}

def improvement(base_val, rl_val, lower_is_better=False):
    """Return % change from base to rl. Positive = improvement in the desired direction."""
    if base_val == 0: return 0.0
    raw = (rl_val - base_val) / abs(base_val) * 100
    return -raw if lower_is_better else raw

# ============================================================
# PLOT 1 — Time-Series (FPS, CPU, Latency)
# ============================================================
def plot_timeseries(base_records, rl_records):
    # subsample for readability
    STEP = max(1, len(base_records)//400)
    b_fps  = [r["fps"]        for r in base_records[::STEP]]
    r_fps  = [r["fps"]        for r in rl_records[::STEP]]
    b_cpu  = [r["cpu_pct"]    for r in base_records[::STEP]]
    r_cpu  = [r["cpu_pct"]    for r in rl_records[::STEP]]
    b_lat  = [r["latency_ms"] for r in base_records[::STEP]]
    r_lat  = [r["latency_ms"] for r in rl_records[::STEP]]
    b_sz   = None
    r_sz   = [r["imgsz_used"] for r in rl_records[::STEP]]

    max_len = max(len(b_fps), len(r_fps))
    xb = np.linspace(0, 100, len(b_fps))
    xr = np.linspace(0, 100, len(r_fps))

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=False)
    fig.suptitle("System Parameters Over Time: Baseline vs RL-Adaptive\n"
                 f"(test_video_1.mp4  x{VIDEO_LOOPS} loops)",
                 fontsize=14, fontweight="bold", y=0.98)

    # FPS
    ax=axes[0]
    ax.plot(xb, b_fps, color=BASELINE_CLR, linewidth=1.2, alpha=0.85, label="Baseline FPS")
    ax.plot(xr, r_fps, color=RL_CLR,       linewidth=1.2, alpha=0.85, label="RL-Adaptive FPS")
    ax.fill_between(xr, r_fps, alpha=0.12, color=RL_CLR)
    ax.set_ylabel("FPS", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.25, linestyle="--", zorder=0)
    ax.set_title("Frames Per Second", fontsize=11, pad=6)

    # CPU
    ax=axes[1]
    ax.plot(xb, b_cpu, color=BASELINE_CLR, linewidth=1.2, alpha=0.85, label="Baseline CPU %")
    ax.plot(xr, r_cpu, color=RL_CLR,       linewidth=1.2, alpha=0.85, label="RL-Adaptive CPU %")
    ax.fill_between(xr, r_cpu, alpha=0.12, color=RL_CLR)
    # RL imgsz on twin axis
    ax2t=ax.twinx()
    ax2t.plot(xr, r_sz, color="#ffb300", linewidth=1.0, alpha=0.6, linestyle=":", label="RL imgsz")
    ax2t.set_ylabel("RL imgsz (px)", fontsize=9, color="#ffb300")
    ax2t.tick_params(axis="y", colors="#ffb300")
    ax2t.legend(fontsize=8, loc="upper right")
    ax.set_ylabel("CPU Usage (%)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.25, linestyle="--", zorder=0)
    ax.set_title("CPU Utilisation (with RL image size overlay)", fontsize=11, pad=6)

    # Latency
    ax=axes[2]
    ax.plot(xb, b_lat, color=BASELINE_CLR, linewidth=1.2, alpha=0.85, label="Baseline latency")
    ax.plot(xr, r_lat, color=RL_CLR,       linewidth=1.2, alpha=0.85, label="RL latency")
    ax.set_ylabel("Latency (ms)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Normalised Frame Progress (%)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.25, linestyle="--", zorder=0)
    ax.set_title("Inference Latency per Frame", fontsize=11, pad=6)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out=OUTPUT_DIR/"sysparams_timeseries.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig); print(f"[SAVED] {out.name}")

# ============================================================
# PLOT 2 — Grouped Bar Chart (Mean Metrics)
# ============================================================
def plot_bar_comparison(b_stats, r_stats, temp_available):
    metrics = ["fps", "latency_ms", "cpu_pct", "ram_pct"]
    labels  = ["FPS", "Latency (ms)", "CPU (%)", "RAM (%)"]
    if temp_available:
        metrics.append("temp"); labels.append("Temp (C)")

    b_means = [b_stats[m]["mean"] for m in metrics]
    r_means = [r_stats[m]["mean"] for m in metrics]

    x = np.arange(len(metrics)); w = 0.35
    fig, ax = plt.subplots(figsize=(13, 7))

    bars_b = ax.bar(x-w/2, b_means, w, label="Baseline (640px)",
                    color=BASELINE_CLR, alpha=0.92, edgecolor="white", linewidth=0.6, zorder=3)
    bars_r = ax.bar(x+w/2, r_means, w, label="RL-Adaptive",
                    color=RL_CLR, alpha=0.92, edgecolor="white", linewidth=0.6, zorder=3)

    for bar in [*bars_b, *bars_r]:
        hv = bar.get_height()
        if hv > 0:
            ax.text(bar.get_x()+bar.get_width()/2, hv+0.3, f"{hv:.1f}",
                    ha="center", va="bottom", fontsize=9, color="white", fontweight="bold")

    # Delta annotation
    lower_better = {"fps":False,"latency_ms":True,"cpu_pct":True,"ram_pct":True,"temp":True}
    units = {"fps": "FPS", "latency_ms": "ms", "cpu_pct": "%", "ram_pct": "%", "temp": "C"}
    for xi,(m,bv,rv) in enumerate(zip(metrics,b_means,r_means)):
        if bv == 0: continue
        lb  = lower_better.get(m, False)
        imp = improvement(bv, rv, lower_is_better=lb)
        colour = RL_CLR if imp >= 0 else "#ef5350"
        diff = rv - bv
        unit = units.get(m, "")
        unit_str = f" {unit}" if unit != "%" else "%"
        y_pos  = max(bv, rv) + max(bv,rv)*0.08
        ax.text(xi, y_pos, f"{diff:+.1f}{unit_str}",
                ha="center", va="bottom", fontsize=9, color=colour, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Mean Value", fontsize=12, fontweight="bold")
    ax.set_title("System Parameters: Baseline vs RL-Adaptive (Mean Values)\n"
                 f"test_video_1.mp4 x{VIDEO_LOOPS} loops",
                 fontsize=13, fontweight="bold", pad=18)
    ax.legend(fontsize=11); ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)

    fig.tight_layout()
    out=OUTPUT_DIR/"sysparams_bar_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig); print(f"[SAVED] {out.name}")

# ============================================================
# PLOT 3 — Box Plots (Distribution Spread)
# ============================================================
def plot_boxplots(base_records, rl_records, temp_available):
    keys = ["fps","latency_ms","cpu_pct","ram_pct"]
    lbls = ["FPS","Latency (ms)","CPU (%)","RAM (%)"]
    if temp_available:
        keys.append("temp"); lbls.append("Temp (C)")

    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 7))
    if n == 1: axes = [axes]

    bprops = dict(color=BASELINE_CLR, linewidth=1.5)
    rprops = dict(color=RL_CLR,       linewidth=1.5)

    for ax, key, lbl in zip(axes, keys, lbls):
        b_vals = [r[key] for r in base_records if r[key]>0]
        r_vals = [r[key] for r in rl_records   if r[key]>0]
        if not b_vals or not r_vals: continue

        bp1 = ax.boxplot([b_vals], positions=[1], widths=0.5,
                         boxprops=bprops.copy(), whiskerprops=bprops.copy(),
                         medianprops=dict(color="white",linewidth=2),
                         capprops=bprops.copy(), flierprops=dict(marker=".",
                         markerfacecolor=BASELINE_CLR, markersize=3, alpha=0.3),
                         patch_artist=True)
        for patch in bp1["boxes"]:
            patch.set_facecolor(BASELINE_CLR); patch.set_alpha(0.3)

        bp2 = ax.boxplot([r_vals], positions=[2], widths=0.5,
                         boxprops=rprops.copy(), whiskerprops=rprops.copy(),
                         medianprops=dict(color="white",linewidth=2),
                         capprops=rprops.copy(), flierprops=dict(marker=".",
                         markerfacecolor=RL_CLR, markersize=3, alpha=0.3),
                         patch_artist=True)
        for patch in bp2["boxes"]:
            patch.set_facecolor(RL_CLR); patch.set_alpha(0.3)

        ax.set_xticks([1,2]); ax.set_xticklabels(["Baseline","RL"], fontsize=10)
        ax.set_title(lbl, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)

    fig.suptitle("Distribution of System Parameters: Baseline vs RL-Adaptive",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out=OUTPUT_DIR/"sysparams_boxplots.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig); print(f"[SAVED] {out.name}")

# ============================================================
# PLOT 4 — Radar Chart
# ============================================================
def plot_radar(b_stats, r_stats, temp_available):
    # Normalise all metrics to [0,1] relative to baseline
    # For "lower-is-better" metrics, invert so larger = better on radar
    metric_names = ["FPS", "Latency\n(lower=better)", "CPU\n(lower=better)",
                    "RAM\n(lower=better)"]
    b_raw = [b_stats["fps"]["mean"], b_stats["latency_ms"]["mean"],
             b_stats["cpu_pct"]["mean"], b_stats["ram_pct"]["mean"]]
    r_raw = [r_stats["fps"]["mean"], r_stats["latency_ms"]["mean"],
             r_stats["cpu_pct"]["mean"], r_stats["ram_pct"]["mean"]]

    if temp_available and b_stats["temp"]["mean"] > 0:
        metric_names.append("Temp\n(lower=better)")
        b_raw.append(b_stats["temp"]["mean"])
        r_raw.append(r_stats["temp"]["mean"])

    # Normalise: baseline=1.0, RL relative
    lower_better_flags = [False, True, True, True, True][:len(b_raw)]

    b_norm = [1.0] * len(b_raw)
    r_norm = []
    for i,(bv,rv,lb) in enumerate(zip(b_raw, r_raw, lower_better_flags)):
        if bv == 0:
            r_norm.append(1.0); continue
        ratio = rv / bv
        if lb:
            r_norm.append(2.0 - ratio)   # invert: lower rv → higher radar value
        else:
            r_norm.append(ratio)

    N   = len(metric_names)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    b_vals = b_norm + b_norm[:1]
    r_vals = r_norm + r_norm[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_facecolor(PANEL_BG)
    ax.plot(angles, b_vals, "o-", color=BASELINE_CLR, linewidth=2.5, label="Baseline")
    ax.fill(angles, b_vals, color=BASELINE_CLR, alpha=0.12)
    ax.plot(angles, r_vals, "s-", color=RL_CLR,       linewidth=2.5, label="RL-Adaptive")
    ax.fill(angles, r_vals, color=RL_CLR,       alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=10, color="white")
    ax.set_yticks([0.5, 1.0, 1.5])
    ax.set_yticklabels(["0.5x", "1x (baseline)", "1.5x"], fontsize=8, color="#a0a0a0")
    ax.set_ylim(0, 2.0)
    ax.grid(color="#3a3d4e", linestyle="--", alpha=0.5)
    ax.spines["polar"].set_color("#3a3d4e")

    ax.set_title("System Performance Radar Chart\n"
                 "(Baseline = 1.0, RL values normalised relative to baseline;\n"
                 " for lower-is-better metrics, higher radar = better RL)",
                 fontsize=11, fontweight="bold", pad=25, color="white")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=11)

    fig.tight_layout()
    out=OUTPUT_DIR/"sysparams_radar.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig); print(f"[SAVED] {out.name}")

# ============================================================
# PLOT 5 — Summary Table
# ============================================================
def plot_summary_table(b_stats, r_stats, temp_available, avg_rl_imgsz, total_frames):
    def delta_str(bv, rv, lower_better=False):
        if bv == 0: return "N/A"
        imp = improvement(bv, rv, lower_is_better=lower_better)
        sign = "+" if imp >= 0 else ""
        return f"{sign}{imp:.1f}%"

    rows = [
        ["Metric", "Baseline (640px)", "RL-Adaptive", "Mean Δ", "Interpretation"],
        ["Avg FPS",
         f"{b_stats['fps']['mean']:.2f}",
         f"{r_stats['fps']['mean']:.2f}",
         delta_str(b_stats['fps']['mean'], r_stats['fps']['mean'], lower_better=False),
         "Higher = faster throughput"],
        ["Avg Latency (ms)",
         f"{b_stats['latency_ms']['mean']:.1f}",
         f"{r_stats['latency_ms']['mean']:.1f}",
         delta_str(b_stats['latency_ms']['mean'], r_stats['latency_ms']['mean'], lower_better=True),
         "Lower = less delay per frame"],
        ["P95 Latency (ms)",
         f"{b_stats['latency_ms']['p95']:.1f}",
         f"{r_stats['latency_ms']['p95']:.1f}",
         delta_str(b_stats['latency_ms']['p95'], r_stats['latency_ms']['p95'], lower_better=True),
         "Tail latency reduction"],
        ["Avg CPU (%)",
         f"{b_stats['cpu_pct']['mean']:.1f}",
         f"{r_stats['cpu_pct']['mean']:.1f}",
         delta_str(b_stats['cpu_pct']['mean'], r_stats['cpu_pct']['mean'], lower_better=True),
         "Lower = less CPU pressure"],
        ["Avg RAM (%)",
         f"{b_stats['ram_pct']['mean']:.1f}",
         f"{r_stats['ram_pct']['mean']:.1f}",
         delta_str(b_stats['ram_pct']['mean'], r_stats['ram_pct']['mean'], lower_better=True),
         "Lower = less memory usage"],
    ]
    if temp_available and b_stats["temp"]["mean"] > 0:
        rows.append([
            "Avg Temp (C)",
            f"{b_stats['temp']['mean']:.1f}",
            f"{r_stats['temp']['mean']:.1f}",
            delta_str(b_stats['temp']['mean'], r_stats['temp']['mean'], lower_better=True),
            "Lower = cooler operation"])

    rows.append(["Avg RL imgsz", "640 px", f"{avg_rl_imgsz:.0f} px",
                 f"{avg_rl_imgsz-640:.0f} px", "Adaptive size chosen by RL"])
    rows.append(["Frames Processed", str(total_frames), str(total_frames),
                 "—", f"{VIDEO_LOOPS}x video loops"])

    n_rows = len(rows) - 1
    fig, ax = plt.subplots(figsize=(17, 0.55*n_rows + 2.5))
    ax.axis("off")

    col_c  = ["#1e3a5f","#1a3555","#1a4f30","#1f2f1f","#28281f"]
    row_c  = [["#1c2740","#0d2137","#0d2d1a","#1c2d1c","#1c1c10"]
               for _ in range(n_rows)]

    tbl = ax.table(
        cellText=[r for r in rows[1:]],
        colLabels=rows[0],
        cellLoc="center", loc="center",
        colColours=col_c, cellColours=row_c)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1.0, 2.1)
    for j in range(5):
        tbl[0,j].set_text_props(fontweight="bold", color="white")

    ax.set_title("Phase 6 — System Parameters Summary Table",
                 fontsize=14, fontweight="bold", pad=22, color="white")
    fig.tight_layout()
    out=OUTPUT_DIR/"sysparams_summary_table.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig); print(f"[SAVED] {out.name}")

# ============================================================
# TERMINAL REPORT
# ============================================================
def print_terminal_report(b_stats, r_stats, temp_available, avg_rl_imgsz, n_frames):
    W = 80
    print("\n" + "="*W)
    print(" PHASE 6 — SYSTEM PARAMETERS COMPARISON REPORT ".center(W,"="))
    print("="*W)
    print(f"{'Metric':<26} {'Baseline':>12} {'RL-Adaptive':>13} {'Delta':>10} {'Better?':>8}")
    print("-"*W)

    def row(label, key, lb=False, unit=""):
        bv=b_stats[key]["mean"]; rv=r_stats[key]["mean"]
        if bv==0: d="N/A"; better="?"
        else:
            imp = improvement(bv,rv,lower_is_better=lb)
            sign="+" if imp>=0 else ""; d=f"{sign}{imp:.1f}%"
            better = "RL [YES]" if imp>=0 else "BASE"
        print(f"{label+unit:<26} {bv:>12.2f} {rv:>13.2f} {d:>10} {better:>8}")

    row("Avg FPS",             "fps",         lb=False)
    row("Avg Latency",         "latency_ms",  lb=True,  unit=" (ms)")
    row("P95 Latency",         "latency_ms",  lb=True,  unit=" (ms)")
    row("Avg CPU Usage",       "cpu_pct",     lb=True,  unit=" (%)")
    row("Avg RAM Usage",       "ram_pct",     lb=True,  unit=" (%)")
    if temp_available and b_stats["temp"]["mean"]>0:
        row("Avg Temperature","temp",         lb=True,  unit=" (C)")

    print("-"*W)
    avg_sz_save = (1-(avg_rl_imgsz/640)**2)*100
    print(f"{'Avg RL imgsz selected':<26} {'640':>12} {avg_rl_imgsz:>13.0f} "
          f"{'('+str(int(avg_rl_imgsz-640))+'px)':>10} {'—':>8}")
    print(f"{'Pixel compute saving':<26} {'0%':>12} {avg_sz_save:>12.1f}% "
          f"{'':>10} {'RL [YES]':>8}")
    print(f"{'Frames (each run)':<26} {n_frames:>12} {n_frames:>13} {'—':>10} {'—':>8}")
    print("="*W)

# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "="*70)
    print(" PHASE 6 — SYSTEM PARAMETERS EVALUATION ".center(70,"="))
    print("="*70)
    print(f"  Video       : {VIDEO_PATH.name}  (loops={VIDEO_LOOPS})")
    print(f"  Model       : {MODEL_PATH.name if MODEL_PATH else 'NOT FOUND'}")
    print()

    if MODEL_PATH is None:
        print("[ERROR] No model .pt found."); sys.exit(1)
    if not VIDEO_PATH.exists():
        print(f"[ERROR] Video not found: {VIDEO_PATH}"); sys.exit(1)

    model  = YOLO(str(MODEL_PATH))
    policy, state_mean, state_std = load_rl_policy()
    monitor = SystemMonitor(poll_interval=0.4)

    # ── BASELINE RUN ─────────────────────────────────────────────────────────
    monitor.start()
    time.sleep(1.0)   # let monitor populate
    base_records = run_inference_loop(
        model, monitor, "BASELINE", imgsz_fixed=640)
    monitor.stop()

    time.sleep(2.0)   # cooldown between runs

    # ── RL RUN ───────────────────────────────────────────────────────────────
    monitor.start()
    time.sleep(1.0)
    rl_records = run_inference_loop(
        model, monitor, "RL-ADAPTIVE",
        policy=policy, state_mean=state_mean, state_std=state_std)
    monitor.stop()

    # ── STATISTICS ───────────────────────────────────────────────────────────
    keys = ["fps","latency_ms","cpu_pct","ram_pct","temp"]
    b_stats = {k: summarise(base_records, k) for k in keys}
    r_stats = {k: summarise(rl_records,   k) for k in keys}

    avg_rl_imgsz   = float(np.mean([r["imgsz_used"] for r in rl_records]))
    temp_available = b_stats["temp"]["mean"] > 0 or r_stats["temp"]["mean"] > 0
    n_frames       = len(base_records)

    # ── TERMINAL REPORT ──────────────────────────────────────────────────────
    print_terminal_report(b_stats, r_stats, temp_available, avg_rl_imgsz, n_frames)

    # ── PLOTS ────────────────────────────────────────────────────────────────
    print("\n[INFO] Generating plots...")
    plot_timeseries(base_records, rl_records)
    plot_bar_comparison(b_stats, r_stats, temp_available)
    plot_boxplots(base_records, rl_records, temp_available)
    plot_radar(b_stats, r_stats, temp_available)
    plot_summary_table(b_stats, r_stats, temp_available, avg_rl_imgsz, n_frames)

    print(f"\n[SUCCESS] All 5 plots saved to: {OUTPUT_DIR}\n")

if __name__ == "__main__":
    main()
