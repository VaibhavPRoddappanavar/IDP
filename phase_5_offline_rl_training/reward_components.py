"""
reward_components.py — recover and recompute reward components offline (Phase 5)
================================================================================
The Phase 4 collector logs one scalar `reward` per transition. Because the
reward formula is linear in its observable inputs, every component can be
recovered exactly from the logged `reward` plus the logged next state:

    R = w_acc * proxy_acc + w_fps * FPS' - w_lat * L'
        - 50 * 1[T' > 80 C] - 10 * 1[U_cpu' > 90]

with (w_acc, w_fps, w_lat) = (10, 0.5, 0.1) at collection time. Therefore

    proxy_acc = (R + w_lat*L' - w_fps*FPS' + 50*1[T'>80] + 10*1[U_cpu'>90]) / w_acc

Recovered `comp_acc` must lie in [0, 100] (it is max detection confidence x 100
averaged over the action window); `validate_recovery` asserts this and that the
residual of the reconstruction is below a small tolerance.

This module lets sensitivity.py re-derive rewards under ANY weight combination
without re-running hardware data collection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Weights used at collection time (must match phase_4 exploration.py v1).
COLLECTION_WEIGHTS = {"w_acc": 10.0, "w_fps": 0.5, "w_lat": 0.1}

COMP_COLS = ["comp_acc", "comp_fps", "comp_lat", "comp_pen_temp", "comp_pen_cpu"]


def recover_components(df: pd.DataFrame, tol: float = 0.05) -> pd.DataFrame:
    """Recover per-transition reward components from a Phase 4 parquet frame.

    Terminal rows (done=True) are dropped: the collector writes reward=0.0 for
    them, which is not a real transition reward and breaks the inversion.
    Adds columns comp_acc, comp_fps, comp_lat, comp_pen_temp, comp_pen_cpu and
    validates that reward == sum of weighted components within `tol`.
    """
    df = df[~df["done"].astype(bool)].copy()
    w = COLLECTION_WEIGHTS
    next_cpu = df["next_state_cpu_percent"].to_numpy(dtype=float)
    next_temp = df["next_state_temperature"].to_numpy(dtype=float)
    next_fps = df["next_state_fps"].to_numpy(dtype=float)
    next_lat = df["next_state_latency_ms"].to_numpy(dtype=float)
    reward = df["reward"].to_numpy(dtype=float)

    pen_temp = 50.0 * (next_temp > 80.0)
    pen_cpu = 10.0 * (next_cpu > 90.0)
    # Invert for proxy accuracy (the only unobserved component).
    comp_acc = (reward - w["w_fps"] * next_fps + w["w_lat"] * next_lat
                + pen_temp + pen_cpu) / w["w_acc"]

    # Clamp float32 rounding noise (must be tiny; assert rather than hide).
    lo_viol = float(np.max(-comp_acc)) if np.any(comp_acc < 0) else 0.0
    hi_viol = float(np.max(comp_acc - 100.0)) if np.any(comp_acc > 100.0) else 0.0
    if max(lo_viol, hi_viol) > 0.01:
        raise ValueError(f"recovered comp_acc outside [0,100] by {max(lo_viol, hi_viol):.4f}")
    comp_acc = np.clip(comp_acc, 0.0, 100.0)

    out = pd.DataFrame({
        "comp_acc": comp_acc,
        "comp_fps": next_fps,
        "comp_lat": next_lat,
        "comp_pen_temp": pen_temp,
        "comp_pen_cpu": pen_cpu,
    }, index=df.index)

    # --- validation ---
    recon = (w["w_acc"] * out["comp_acc"].to_numpy() + w["w_fps"] * out["comp_fps"].to_numpy()
             - w["w_lat"] * out["comp_lat"].to_numpy() - out["comp_pen_temp"].to_numpy()
             - out["comp_pen_cpu"].to_numpy())
    max_resid = float(np.max(np.abs(recon - reward)))
    if max_resid > tol:
        raise ValueError(f"reward reconstruction residual {max_resid:.4f} > tol {tol}")
    acc = out["comp_acc"].to_numpy()
    if acc.min() < -1e-6 or acc.max() > 100.0 + 1e-6:
        raise ValueError(f"recovered comp_acc out of [0,100]: [{acc.min():.2f}, {acc.max():.2f}]")
    return out


def recompute_reward(comp: pd.DataFrame, w_acc: float, w_fps: float, w_lat: float) -> np.ndarray:
    """Recompute rewards under an arbitrary weight combination."""
    return (w_acc * comp["comp_acc"].to_numpy()
            + w_fps * comp["comp_fps"].to_numpy()
            - w_lat * comp["comp_lat"].to_numpy()
            + comp["comp_pen_temp"].to_numpy()
            + comp["comp_pen_cpu"].to_numpy())


def _selftest() -> None:
    """Round-trip test: fabricate components -> reward -> recovery -> compare."""
    rng = np.random.default_rng(0)
    n = 500
    acc = rng.uniform(0, 100, n)
    fps = rng.uniform(2, 14, n)
    lat = rng.uniform(70, 340, n)
    temp = rng.uniform(45, 95, n)
    cpu = rng.uniform(30, 99, n)
    w = COLLECTION_WEIGHTS
    reward = (w["w_acc"] * acc + w["w_fps"] * fps - w["w_lat"] * lat
              - 50.0 * (temp > 80.0) - 10.0 * (cpu > 90.0))
    df = pd.DataFrame({
        "reward": reward,
        "next_state_cpu_percent": cpu,
        "next_state_temperature": temp,
        "next_state_fps": fps,
        "next_state_latency_ms": lat,
        "done": False,
    })
    rec = recover_components(df)
    err = float(np.max(np.abs(rec["comp_acc"].to_numpy() - acc)))
    assert err < 1e-6, f"selftest failed: max component error {err}"
    print(f"[SELFTEST OK] reward_components recovery, max error {err:.2e}")


if __name__ == "__main__":
    _selftest()
