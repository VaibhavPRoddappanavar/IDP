import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# 1. Define the Continuous Policy Network (The Brain)
class ContinuousActor(nn.Module):
    def __init__(self, state_dim=5, action_dim=2):
        super(ContinuousActor, self).__init__()
        # Standard MLP Architecture
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh() # Tanh ensures the output is between -1.0 and 1.0 (raw action space)
        )
        
    def forward(self, state):
        return self.net(state)

def main():
    # 2. Setup paths
    workspace_dir = Path(__file__).resolve().parent.parent
    data_path = workspace_dir / "phase_4_rl_data_collection" / "output" / "rl_transitions_rpi_1.parquet"
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Loading offline dataset from: {data_path}")
    df = pd.read_parquet(data_path)
    
    # 3. Filter for Behavioral Cloning (Offline RL variant)
    # We only want the AI to learn from the TOP 30% of highest reward actions!
    # This prevents it from learning bad habits.
    reward_threshold = df["reward"].quantile(0.70)
    best_df = df[df["reward"] >= reward_threshold]
    print(f"[INFO] Filtered dataset for top 30% rewards (> {reward_threshold:.2f}). Rows remaining: {len(best_df)}")

    # 4. Extract and Standardize Arrays
    state_cols = ["state_cpu_percent", "state_ram_percent", "state_temperature", "state_fps", "state_latency_ms"]
    states = best_df[state_cols].values.astype(np.float32)
    
    action_cols = ["action_imgsz_raw", "action_skip_raw"]
    actions = best_df[action_cols].values.astype(np.float32)

    # Normalize States to prevent exploding gradients (Mean=0, Std=1)
    state_mean = states.mean(axis=0)
    state_std = states.std(axis=0) + 1e-8 # add epsilon to avoid div by zero
    states_normalized = (states - state_mean) / state_std
    
    # Save the normalization stats so Phase 6 can use them!
    np.save(output_dir / "state_mean.npy", state_mean)
    np.save(output_dir / "state_std.npy", state_std)

    # 5. Create PyTorch DataLoader
    dataset = TensorDataset(torch.from_numpy(states_normalized), torch.from_numpy(actions))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 6. Initialize the Network and Optimizer
    print("[INFO] Initializing PyTorch Continuous Actor Network...")
    actor = ContinuousActor(state_dim=5, action_dim=2)
    optimizer = optim.Adam(actor.parameters(), lr=1e-3)
    criterion = nn.MSELoss() # We want the network's output to match the best actions

    # 7. Train the Neural Network Offline
    epochs = 150
    print(f"[INFO] Commencing Offline Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_states, batch_actions in dataloader:
            optimizer.zero_grad()
            
            # Predict action from state
            pred_actions = actor(batch_states)
            
            # Calculate loss (how far off from the "golden" actions)
            loss = criterion(pred_actions, batch_actions)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 25 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss/len(dataloader):.4f}")

    # 8. Export the Model as TorchScript
    policy_path = output_dir / "rl_adaptive_policy.pt"
    print(f"\n[INFO] Training finished! Exporting Actor network to: {policy_path}")
    
    # Export as TorchScript for universal loading
    actor.eval() # Set to inference mode
    dummy_input = torch.zeros(1, 5) # Create a fake input tensor
    traced_script_module = torch.jit.trace(actor, dummy_input)
    traced_script_module.save(str(policy_path))
    
    print("[SUCCESS] Phase 5 is complete. You now have a custom PyTorch RL brain ready for Phase 6!")

if __name__ == "__main__":
    main()
