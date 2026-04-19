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
        "accuracy": [],
        "frame_count": 0
    }
    
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["frame_count"] += 1
            
            # Backward compatibility: if you use a CSV generated before we added 'accuracy'
            # (like logs_macbook.csv), we estimate it from confidence_score_avg * 100
            if "accuracy" not in row and "confidence_score_avg" in row:
                conf = row.get("confidence_score_avg", "0")
                try:
                    row["accuracy"] = float(conf) * 100.0
                except ValueError:
                    row["accuracy"] = 0.0

            # Safely cast and append. If empty/missing fallback to 0.0
            for key in ["fps", "latency_ms", "ram_usage_percent", "cpu_usage_percent", "accuracy"]:
                val = row.get(key, "0")
                try:
                    data[key].append(float(val) if val else 0.0)
                except ValueError:
                    data[key].append(0.0)

    # Calculate averages
    averages = {"frame_count": data["frame_count"]}
    for key in ["fps", "latency_ms", "ram_usage_percent", "cpu_usage_percent", "accuracy"]:
        if data[key]:
            averages[key] = sum(data[key]) / len(data[key])
        else:
            averages[key] = 0.0

    return averages


def print_comparison(baseline: dict, target: dict, b_name: str, t_name: str):
    """Prints a structured terminal report mapping the differences."""
    
    print("\n" + "="*65)
    print(f" BENCHMARK COMPARISON REPORT ".center(65, "="))
    print("="*65)
    print(f"{'Metric':<20} | {b_name[:12]:<12} | {t_name[:12]:<12} | {'Difference':<12}")
    print("-" * 65)

    def diff_str(bl, tg, higher_is_better=True, absolute=False):
        if bl == 0: return "N/A"
        
        if absolute:
            diff = tg - bl
            prefix = "+" if diff > 0 else ""
            return f"{prefix}{diff:.1f}"

        percent = ((tg - bl) / bl) * 100
        prefix = "+" if percent > 0 else ""
        return f"{prefix}{percent:.1f}%"

    # FPS (Higher = Better)
    b_fps = baseline['fps']
    t_fps = target['fps']
    print(f"{'Average FPS':<20} | {b_fps:<12.1f} | {t_fps:<12.1f} | {diff_str(b_fps, t_fps, True)}")

    # Latency (Lower = Better)
    b_lat = baseline['latency_ms']
    t_lat = target['latency_ms']
    print(f"{'Avg Latency (ms)':<20} | {b_lat:<12.1f} | {t_lat:<12.1f} | {diff_str(b_lat, t_lat, False)}")

    # CPU/RAM Usage (Absolute difference is usually more readable for percentages)
    b_cpu = baseline['cpu_usage_percent']
    t_cpu = target['cpu_usage_percent']
    print(f"{'Avg CPU Usage (%)':<20} | {b_cpu:<12.1f} | {t_cpu:<12.1f} | {diff_str(b_cpu, t_cpu, absolute=True)}%")
    
    b_ram = baseline['ram_usage_percent']
    t_ram = target['ram_usage_percent']
    print(f"{'Avg RAM Usage (%)':<20} | {b_ram:<12.1f} | {t_ram:<12.1f} | {diff_str(b_ram, t_ram, absolute=True)}%")

    print("-" * 65)
    # Accuracy (Higher is better)
    b_acc = baseline['accuracy']
    t_acc = target['accuracy']
    print(f"{'Precision Acc (%)':<20} | {b_acc:<12.2f} | {t_acc:<12.2f} | {diff_str(b_acc, t_acc, True)}")
    print("="*65)
    
    # Textual Conclusion
    print("\n[CONCLUSION]")
    if t_acc < (b_acc - 1.0):
        acc_drop = b_acc - t_acc
        print(f"⚠️ Model accuracy dropped by {acc_drop:.1f}% compared to baseline.")
        print("   (Expected behavior if targeting an edge device using quantization/smaller models).")
    elif t_acc > (b_acc + 1.0):
        print("✅ Target achieved higher confidence scores overall.")
    else:
        print("✅ Detection Accuracy is identical/negligible change (Same mathematical operations).")

    if t_fps < b_fps:
        fps_drop = b_fps - t_fps
        print(f"⚠️ Target runs {fps_drop:.1f} FPS slower than the baseline ({(1 - t_fps/b_fps)*100:.1f}% speed reduction).")
    else:
        print(f"🚀 Target runs FASTER than the baseline by {t_fps - b_fps:.1f} FPS!")

    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare YOLO benchmark CSV logs.")
    parser.add_argument("--baseline", type=str, default="output/logs_macbook.csv", help="Path to baseline CSV (e.g., MacBook)")
    parser.add_argument("--target", type=str, default="output/logs.csv", help="Path to target CSV (e.g., RasPi)")
    
    args = parser.parse_args()

    try:
        baseline_stats = load_csv(args.baseline)
        target_stats = load_csv(args.target)
        
        b_name = Path(args.baseline).stem
        t_name = Path(args.target).stem
        
        print_comparison(baseline_stats, target_stats, b_name, t_name)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("Make sure both CSV files exist before running.")
