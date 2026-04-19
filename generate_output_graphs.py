"""Generate visualization images for baseline and controlled-adaptation CSV runs."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIRS = [
    ROOT_DIR / "phase_0_baseline" / "output",
    ROOT_DIR / "phase_1_controlled_adaptation" / "output",
]


def _to_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if text == "":
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _series(rows: list[dict[str, str]], key: str) -> list[float]:
    return [_to_float(row.get(key)) for row in rows]


def _has_finite(values: list[float]) -> bool:
    return any(math.isfinite(v) for v in values)


def _percentile(values: list[float], p: float) -> float:
    valid = sorted(v for v in values if math.isfinite(v))
    if not valid:
        return float("nan")
    if len(valid) == 1:
        return valid[0]

    rank = (len(valid) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(valid) - 1)
    if lo == hi:
        return valid[lo]
    w = rank - lo
    return valid[lo] * (1.0 - w) + valid[hi] * w


def _mean(values: list[float]) -> float:
    valid = [v for v in values if math.isfinite(v)]
    if not valid:
        return float("nan")
    return sum(valid) / len(valid)


def _read_csv(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _plot_overview(csv_path: Path, rows: list[dict[str, str]]) -> Path:
    has_mode = "mode" in rows[0] if rows else False
    nrows = 4 if has_mode else 3

    x = list(range(1, len(rows) + 1))
    fps = _series(rows, "fps")
    latency = _series(rows, "latency_ms")
    cpu = _series(rows, "cpu_usage_percent")
    ram = _series(rows, "ram_usage_percent")
    gpu = _series(rows, "gpu_usage_percent")
    temp = _series(rows, "temperature")
    detections = _series(rows, "num_detections")
    confidence = _series(rows, "confidence_score_avg")
    frame_drop = _series(rows, "frame_drop_rate_percent")

    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(14, 3.5 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    fig.suptitle(f"Run Overview: {csv_path.name}", fontsize=14)

    ax0 = axes[0]
    ax0.plot(x, fps, label="FPS", color="#2c7fb8", linewidth=1.5)
    ax0.set_ylabel("FPS")
    ax0.grid(alpha=0.25)
    ax0b = ax0.twinx()
    ax0b.plot(x, latency, label="Latency (ms)", color="#d95f0e", linewidth=1.2)
    ax0b.set_ylabel("Latency (ms)")
    ax0.legend(loc="upper left")
    ax0b.legend(loc="upper right")

    ax1 = axes[1]
    ax1.plot(x, cpu, label="CPU %", color="#41ab5d", linewidth=1.2)
    ax1.plot(x, ram, label="RAM %", color="#756bb1", linewidth=1.2)
    if _has_finite(gpu):
        ax1.plot(x, gpu, label="GPU %", color="#e7298a", linewidth=1.2)
    if _has_finite(temp):
        ax1.plot(x, temp, label="Temp C", color="#636363", linewidth=1.0)
    ax1.set_ylabel("System")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper right")

    ax2 = axes[2]
    ax2.plot(x, detections, label="Detections", color="#1b9e77", linewidth=1.2)
    if _has_finite(confidence):
        ax2b = ax2.twinx()
        ax2b.plot(x, confidence, label="Avg confidence", color="#e6ab02", linewidth=1.0)
        ax2b.set_ylabel("Confidence")
        ax2b.legend(loc="upper right")
    if _has_finite(frame_drop):
        ax2.plot(x, frame_drop, label="Frame drop %", color="#e31a1c", linewidth=1.0)
    ax2.set_ylabel("Detections/Drop")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="upper left")

    if has_mode:
        ax3 = axes[3]
        modes = [row.get("mode", "") for row in rows]
        unique_modes = []
        for mode in modes:
            if mode not in unique_modes:
                unique_modes.append(mode)
        mode_to_idx = {mode: idx for idx, mode in enumerate(unique_modes)}
        mode_idx = [mode_to_idx.get(mode, math.nan) for mode in modes]
        ax3.step(x, mode_idx, where="post", color="#252525", linewidth=1.2)
        ax3.set_yticks(list(mode_to_idx.values()))
        ax3.set_yticklabels(list(mode_to_idx.keys()))
        ax3.set_ylabel("Mode")
        ax3.grid(alpha=0.25)

    axes[-1].set_xlabel("Processed frame index")

    out_path = csv_path.with_name(f"{csv_path.stem}_overview.png")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_mode_summary(csv_path: Path, rows: list[dict[str, str]]) -> Path | None:
    if not rows or "mode" not in rows[0]:
        return None

    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {
        "fps": [],
        "latency": [],
        "detections": [],
        "confidence": [],
    })

    for row in rows:
        mode = row.get("mode", "")
        grouped[mode]["fps"].append(_to_float(row.get("fps")))
        grouped[mode]["latency"].append(_to_float(row.get("latency_ms")))
        grouped[mode]["detections"].append(_to_float(row.get("num_detections")))
        grouped[mode]["confidence"].append(_to_float(row.get("confidence_score_avg")))

    modes = list(grouped.keys())
    avg_fps = [_mean(grouped[m]["fps"]) for m in modes]
    p95_latency = [_percentile(grouped[m]["latency"], 95.0) for m in modes]
    avg_det = [_mean(grouped[m]["detections"]) for m in modes]
    avg_conf = [_mean(grouped[m]["confidence"]) for m in modes]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    fig.suptitle(f"Mode Summary: {csv_path.name}", fontsize=14)

    axes[0, 0].bar(modes, avg_fps, color="#2c7fb8")
    axes[0, 0].set_title("Average FPS")
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].bar(modes, p95_latency, color="#d95f0e")
    axes[0, 1].set_title("P95 Latency (ms)")
    axes[0, 1].grid(axis="y", alpha=0.25)

    axes[1, 0].bar(modes, avg_det, color="#1b9e77")
    axes[1, 0].set_title("Average Detections")
    axes[1, 0].grid(axis="y", alpha=0.25)

    axes[1, 1].bar(modes, avg_conf, color="#e6ab02")
    axes[1, 1].set_title("Average Confidence")
    axes[1, 1].grid(axis="y", alpha=0.25)

    out_path = csv_path.with_name(f"{csv_path.stem}_mode_summary.png")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_folder_run_comparison(folder: Path, csv_paths: list[Path]) -> Path | None:
    if len(csv_paths) < 2:
        return None

    labels: list[str] = []
    mean_fps: list[float] = []
    p95_latency: list[float] = []
    mean_conf: list[float] = []

    for csv_path in csv_paths:
        rows = _read_csv(csv_path)
        if not rows:
            continue
        labels.append(csv_path.stem)
        mean_fps.append(_mean(_series(rows, "fps")))
        p95_latency.append(_percentile(_series(rows, "latency_ms"), 95.0))
        mean_conf.append(_mean(_series(rows, "confidence_score_avg")))

    if len(labels) < 2:
        return None

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 4.8))
    fig.suptitle(f"Run Comparison: {folder.parent.name}", fontsize=14)

    axes[0].bar(labels, mean_fps, color="#2c7fb8")
    axes[0].set_title("Mean FPS")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(labels, p95_latency, color="#d95f0e")
    axes[1].set_title("P95 Latency (ms)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(labels, mean_conf, color="#e6ab02")
    axes[2].set_title("Mean Confidence")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].grid(axis="y", alpha=0.25)

    out_path = folder / "runs_comparison_summary.png"
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def generate_all_graphs() -> list[Path]:
    generated: list[Path] = []

    for output_dir in OUTPUT_DIRS:
        if not output_dir.exists():
            continue

        csv_paths = sorted(output_dir.glob("*.csv"))
        for csv_path in csv_paths:
            rows = _read_csv(csv_path)
            if not rows:
                continue

            generated.append(_plot_overview(csv_path, rows))
            mode_plot = _plot_mode_summary(csv_path, rows)
            if mode_plot is not None:
                generated.append(mode_plot)

        compare_plot = _plot_folder_run_comparison(output_dir, csv_paths)
        if compare_plot is not None:
            generated.append(compare_plot)

    return generated


def main() -> None:
    generated = generate_all_graphs()
    if not generated:
        print("No graphs generated. Ensure CSV files exist in output folders.")
        return

    print("Generated graph files:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
