import csv
import argparse
from pathlib import Path


def load_csv(filepath: str) -> dict:
    """Reads a CSV and computes average metrics across all frames."""
    data = {
        "fps": [],
        "latency_ms": [],
        "ram_usage_percent": [],
        "cpu_usage_percent": [],
        "confidence_pct": [],
        "frame_count": 0
    }
    
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["frame_count"] += 1
            
            # Normalize all CSVs onto the same metric: average confidence per frame.
            # Prefer confidence_score_avg when available, and only fall back to the
            # legacy accuracy field if a CSV does not expose confidence_score_avg.
            if "confidence_score_avg" in row:
                conf = row.get("confidence_score_avg", "0")
                try:
                    row["confidence_pct"] = float(conf) * 100.0
                except ValueError:
                    row["confidence_pct"] = 0.0
            elif "accuracy" in row:
                conf = row.get("accuracy", "0")
                try:
                    row["confidence_pct"] = float(conf)
                except ValueError:
                    row["confidence_pct"] = 0.0

            # Safely cast and append. If empty/missing fallback to 0.0
            for key in ["fps", "latency_ms", "ram_usage_percent", "cpu_usage_percent", "confidence_pct"]:
                val = row.get(key, "0")
                try:
                    data[key].append(float(val) if val else 0.0)
                except ValueError:
                    data[key].append(0.0)

    # Calculate averages
    averages = {"frame_count": data["frame_count"]}
    for key in ["fps", "latency_ms", "ram_usage_percent", "cpu_usage_percent", "confidence_pct"]:
        if data[key]:
            averages[key] = sum(data[key]) / len(data[key])
        else:
            averages[key] = 0.0

    return averages


def print_comparison(baseline: dict, targets: list, b_name: str):
    """Prints a structured terminal report mapping the differences."""
    
    num_t = len(targets)
    width = 36 + (27 * num_t)
    print("\n" + "="*width)
    print(f" BENCHMARK COMPARISON REPORT ".center(width, "="))
    print("="*width)
    
    header = f"{'Metric':<20} | {b_name[:12]:<12}"
    for t_name, _ in targets:
        header += f" | {t_name[:12]:<12} | {'Difference':<10}"
    print(header)
    print("-" * width)

    def diff_str(bl, tg, absolute=False):
        if bl == 0: return "N/A"
        if absolute:
            diff = tg - bl
            return f"{'+' if diff > 0 else ''}{diff:.1f}"
        percent = ((tg - bl) / bl) * 100
        return f"{'+' if percent > 0 else ''}{percent:.1f}%"

    def row_str(m_name, b_val, key, absolute=False, prec=1):
        row = f"{m_name:<20} | {b_val:<12.{prec}f}"
        for _, t_data in targets:
            t_val = t_data[key]
            d_str = diff_str(b_val, t_val, absolute)
            row += f" | {t_val:<12.{prec}f} | {d_str:<10}"
        return row

    # Print rows
    print(row_str('Average FPS', baseline['fps'], 'fps', False, 1))
    print(row_str('Avg Latency (ms)', baseline['latency_ms'], 'latency_ms', False, 1))
    print(row_str('Avg CPU Usage (%)', baseline['cpu_usage_percent'], 'cpu_usage_percent', True, 1))
    print(row_str('Avg RAM Usage (%)', baseline['ram_usage_percent'], 'ram_usage_percent', True, 1))

    print("-" * width)
    # Normalized confidence (Higher is better)
    print(row_str('Avg Confidence (%)', baseline['confidence_pct'], 'confidence_pct', False, 2))
    print("="*width)
    
    # Textual Conclusion
    for t_name, t_data in targets:
        print(f"\n[CONCLUSION FOR {t_name}]")
        t_acc = t_data['confidence_pct']
        b_acc = baseline['confidence_pct']
        t_fps = t_data['fps']
        b_fps = baseline['fps']
        
        if t_acc < (b_acc - 1.0):
            acc_drop = b_acc - t_acc
            print(f"⚠️ Average confidence dropped by {acc_drop:.1f}% compared to baseline.")
            print("   (Expected behavior if targeting an edge device using quantization/smaller models).")
        elif t_acc > (b_acc + 1.0):
            print("✅ Target achieved higher average confidence overall.")
        else:
            print("✅ Average confidence is identical/negligible change (Same mathematical operations).")

        if t_fps < b_fps:
            fps_drop = b_fps - t_fps
            print(f"⚠️ Target runs {fps_drop:.1f} FPS slower than the baseline ({(1 - t_fps/b_fps)*100:.1f}% speed reduction).")
        else:
            print(f"🚀 Target runs FASTER than the baseline by {t_fps - b_fps:.1f} FPS!")

    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare YOLO benchmark CSV logs.")
    parser.add_argument("--baseline", type=str, default="output/logs_macbook.csv", help="Path to baseline CSV (e.g., MacBook)")
    parser.add_argument("--target", type=str, nargs='+', default=["output/logs.csv"], help="Path to one or more target CSVs")
    
    args = parser.parse_args()

    try:
        baseline_stats = load_csv(args.baseline)
        
        targets = []
        for t_path in args.target:
            t_stats = load_csv(t_path)
            t_name = Path(t_path).stem
            targets.append((t_name, t_stats))
            
        b_name = Path(args.baseline).stem
        
        print_comparison(baseline_stats, targets, b_name)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("Make sure all CSV files exist before running.")
