import os
import pandas as pd
from pathlib import Path

class TransitionLogger:
    """Logs MDP transitions to both CSV and Parquet formats."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_path = self.output_dir / "rl_transitions.csv"
        self.parquet_path = self.output_dir / "rl_transitions.parquet"
        
        self.buffer = []
        self.flush_every = 20

    def log_transition(self, transition: dict):
        self.buffer.append(transition)
        if len(self.buffer) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
            
        df_new = pd.DataFrame(self.buffer)
        
        # Append to CSV
        if not self.csv_path.exists():
            df_new.to_csv(self.csv_path, index=False)
        else:
            df_new.to_csv(self.csv_path, mode='a', header=False, index=False)
            
        # Append to Parquet
        if not self.parquet_path.exists():
            df_new.to_parquet(self.parquet_path, engine='pyarrow', index=False)
        else:
            # For parquet, we have to read existing, concat, and rewrite.
            # In production, writing to a new partition is better, but this works for local data collection.
            try:
                df_existing = pd.read_parquet(self.parquet_path, engine='pyarrow')
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined.to_parquet(self.parquet_path, engine='pyarrow', index=False)
            except Exception as e:
                print(f"[ERROR] Failed to update parquet: {e}")
                
        self.buffer.clear()

    def close(self):
        self.flush()
