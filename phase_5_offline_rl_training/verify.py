import torch
import numpy as np
from pathlib import Path

def verify_model():
    output_dir = Path(__file__).resolve().parent / "output"
    model_path = output_dir / "rl_adaptive_policy.pt"
    mean_path = output_dir / "state_mean.npy"
    std_path = output_dir / "state_std.npy"
    
    if not model_path.exists():
        print(f"[ERROR] Could not find the exported model at {model_path}")
        print("Please run train.py first to generate the policy.")
        return

    print(f"[INFO] Loading TorchScript model from {model_path}...")
    # Load the pure PyTorch model
    policy = torch.jit.load(str(model_path))
    policy.eval() # Set to evaluation mode
    
    print("[INFO] Loading Normalization constants...")
    state_mean = np.load(mean_path)
    state_std = np.load(std_path)
    print("[SUCCESS] Model and normalizers loaded successfully!\n")

    import matplotlib.pyplot as plt

    # --- SEQUENTIAL THERMAL HEATING TEST ---
    temps = np.linspace(35, 95, 10) # Test 10 temps from 35C to 95C
    
    results_imgsz = []
    results_skip = []
    
    print("--- SIMULATING GRADUAL SYSTEM HEATING ---")
    for temp in temps:
        # Simulate hardware state worsening as temp rises
        cpu = min(98.0, 10 + (temp - 35) * 1.5) # CPU rises
        ram = 40.0 # RAM stays mostly static
        fps = max(5.0, 30.0 - (temp - 35) * 0.4) # FPS drops
        latency = min(300.0, 20.0 + (temp - 35) * 4) # Latency rises
        
        state = np.array([[cpu, ram, temp, fps, latency]], dtype=np.float32)
        norm_state = (state - state_mean) / state_std
        tensor_state = torch.from_numpy(norm_state)
        
        with torch.no_grad():
            action = policy(tensor_state).numpy()[0]
            
        mapped_imgsz = 256 + (((action[0] + 1.0) / 2.0) * (640 - 256))
        final_imgsz = int(round(mapped_imgsz / 32) * 32)
        final_skip = (action[1] + 1.0) / 2.0
        
        results_imgsz.append(final_imgsz)
        results_skip.append(final_skip)
        
        print(f"State: Temp={temp:2.0f}C, CPU={cpu:2.0f}%, FPS={fps:2.0f} | AI Decision -> imgsz={final_imgsz:3d}, skip={final_skip:.2f}")

    # --- PLOTTING ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('System Temperature (°C)', fontweight='bold')
    ax1.set_ylabel('Selected Image Size (px)', color=color, fontweight='bold')
    ax1.plot(temps, results_imgsz, color=color, marker='o', linewidth=3)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Frame Skip Rate (0.0 - 1.0)', color=color, fontweight='bold')
    ax2.plot(temps, results_skip, color=color, marker='s', linewidth=3, linestyle='dashed')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('AI Dynamic Inference Adaptation vs Thermal Stress', fontsize=14, fontweight='bold')
    
    plot_path = output_dir / "adaptation_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n[SUCCESS] Graphical representation saved to: {plot_path}")

if __name__ == "__main__":
    verify_model()
