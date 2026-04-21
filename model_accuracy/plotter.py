import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def main():
    # Paths
    logs_path = Path("/Users/vishalbhat/RVCE/SEM 6/IDP project/IDP/phase_1_controlled_adaptation/output/logs.csv")
    output_dir = Path("/Users/vishalbhat/RVCE/SEM 6/IDP project/IDP/model_accuracy")
    
    if not logs_path.exists():
        print(f"Error: {logs_path} does not exist.")
        return

    print(f"Reading data from {logs_path}...")
    df = pd.read_csv(logs_path)
    
    # Convert timestamp string to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Add elapsed time column in seconds starting from 0
    df['elapsed_time'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
    
    print("Generating comprehensive plots...")
    sns.set_theme(style="whitegrid")

    # 1. TIME SERIES PLOT
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    sns.lineplot(data=df, x='elapsed_time', y='fps', hue='mode', ax=axes[0], palette='viridis', marker='o', markersize=3)
    axes[0].set_title('FPS over Time across Modes (0, 1, 2)')
    axes[0].set_ylabel('FPS')
    
    sns.lineplot(data=df, x='elapsed_time', y='latency_ms', hue='mode', ax=axes[1], palette='flare', marker='o', markersize=3)
    axes[1].set_title('Inference Latency over Time')
    axes[1].set_ylabel('Latency (ms)')

    sns.lineplot(data=df, x='elapsed_time', y='cpu_usage_percent', label='CPU', ax=axes[2], color='blue')
    axes[2].set_title('System Utilization over Time')
    axes[2].set_ylabel('Usage (%)')
    axes[2].set_xlabel('Elapsed Time (seconds)')
    
    plt.tight_layout()
    timeseries_path = output_dir / "timeseries_metrics.png"
    plt.savefig(timeseries_path, dpi=300)
    print(f"Saved time series plot to {timeseries_path}")

    # 2. BAR CHARTS FOR AGGREGATES
    fig2, axes2 = plt.subplots(1, 4, figsize=(18, 5))
    
    # Calculate means
    mode_stats = df.groupby('mode').agg({
        'fps': 'mean',
        'latency_ms': 'mean',
        'cpu_usage_percent': 'mean',
        'accuracy': 'mean'
    }).reset_index()

    sns.barplot(data=mode_stats, x='mode', y='fps', ax=axes2[0], palette='Blues')
    axes2[0].set_title('Average FPS')

    sns.barplot(data=mode_stats, x='mode', y='latency_ms', ax=axes2[1], palette='Reds')
    axes2[1].set_title('Average Latency (ms)')

    sns.barplot(data=mode_stats, x='mode', y='cpu_usage_percent', ax=axes2[2], palette='Greens')
    axes2[2].set_title('Average CPU Usage (%)')
    
    sns.barplot(data=mode_stats, x='mode', y='accuracy', ax=axes2[3], palette='Purples')
    axes2[3].set_title('Average Accuracy Score')

    plt.tight_layout()
    bar_path = output_dir / "aggregate_metrics.png"
    plt.savefig(bar_path, dpi=300)
    print(f"Saved bar plots to {bar_path}")

if __name__ == "__main__":
    main()
