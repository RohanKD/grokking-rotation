"""Experiment 3: Grokking Dynamics.

Research question
-----------------
Does generalization emerge suddenly after a long period of memorization
(a "grokking" phase transition), or does it improve gradually?

We look for the signature: training loss converges early, but test LPIPS
on complex objects continues to improve for much longer — suggesting the
model moves from a memorized solution to a more structured representation.

Protocol
--------
1. Train with N=40 for 500,000 steps (2.5× the baseline).
2. Every 5,000 steps log: train_loss, test_lpips_simple, test_lpips_complex.
3. Plot train loss and test LPIPS on the same figure to see if test LPIPS
   "groks" long after train loss plateaus.

Output:
  grokking_log.csv      – [step, train_loss, test_lpips_simple, test_lpips_complex]
  exp3_grokking.png     – Figure 4 equivalent

Usage
-----
    python experiments/exp3_grokking.py \
        --config configs/base.yaml \
        --data_root /path/to/coil-100 \
        --output_dir results/exp3
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.coil100 import get_train_test_split, build_dataloader
from models.ddpm import ConditionalDDPM, EMAHelper
from metrics.perceptual import PerceptualMetrics
from complexity import compute_all_complexities, get_quartile_splits
from train import (
    load_config, build_model, load_checkpoint,
    save_checkpoint, set_seed, get_lr_scheduler
)


# ---------------------------------------------------------------------------
# Inline training with dense logging
# ---------------------------------------------------------------------------


def train_with_grokking_log(
    cfg: Dict,
    data_root: str,
    output_dir: Path,
    device: torch.device,
    n_train_objects: int,
    seed: int,
    total_steps: int,
    eval_every: int,
    test_simple: List[int],
    test_complex: List[int],
    resume_log: bool = True,
) -> Path:
    """Train and log eval metrics densely to detect grokking.

    Returns path to grokking_log.csv.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    train_80, _ = get_train_test_split(n_train=80, seed=cfg["data"]["seed"])
    rng = random.Random(seed)
    shuffled = train_80.copy()
    rng.shuffle(shuffled)
    train_ids = sorted(shuffled[:n_train_objects])

    model = build_model(cfg, device)
    ema = EMAHelper(model, decay=cfg["training"].get("ema_decay", 0.9999))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    scheduler = get_lr_scheduler(optimizer, total_steps, cfg["training"]["cosine_schedule"])

    # --- Resume if checkpoint exists ---------------------------------------
    ckpt_files = sorted((output_dir / "checkpoints").glob("checkpoint_*.pt")) if (output_dir / "checkpoints").exists() else []
    start_step = 0
    if ckpt_files and resume_log:
        latest = ckpt_files[-1]
        start_step = load_checkpoint(str(latest), model, ema, optimizer, scheduler, device=str(device))
        print(f"  Resumed from step {start_step}")

    loader = build_dataloader(
        root=data_root,
        object_ids=train_ids,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        length=(total_steps - start_step) * cfg["training"]["batch_size"],
    )
    loader_iter = iter(loader)

    log_csv = output_dir / "grokking_log.csv"
    fields = ["step", "train_loss", "test_lpips_simple", "test_lpips_complex"]
    if start_step == 0:
        with open(log_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    metrics = PerceptualMetrics(device=str(device))

    def _quick_eval_lpips(obj_ids: List[int], n_eval: int = 8) -> float:
        ids = obj_ids[:min(len(obj_ids), n_eval)]
        loader_e = build_dataloader(
            root=data_root, object_ids=ids, batch_size=n_eval,
            shuffle=False, num_workers=0, length=n_eval,
        )
        batch = next(iter(loader_e))
        src = batch["source"].to(device) * 2.0 - 1.0
        tgt = batch["target"].to(device)
        delta = batch["delta_theta"].float().to(device)
        # Use EMA weights.
        original = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        ema.copy_to(model)
        model.eval()
        with torch.no_grad():
            pred_n = model.ddim_sample(src, delta, n_steps=cfg["diffusion"]["ddim_steps"])
        pred_01 = ((pred_n + 1.0) / 2.0).clamp(0.0, 1.0)
        val = metrics.compute_lpips(pred_01, tgt)
        for name, param in model.named_parameters():
            if name in original:
                param.data.copy_(original[name])
        model.train()
        return val

    model.train()
    loss_sum = 0.0
    loss_count = 0
    print(f"  Training from step {start_step} to {total_steps} ...")

    for step in range(start_step, total_steps):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        source_n = batch["source"].to(device) * 2.0 - 1.0
        target_n = batch["target"].to(device) * 2.0 - 1.0
        delta = batch["delta_theta"].float().to(device)

        optimizer.zero_grad()
        loss = model.p_losses(target_n, source_n, delta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"].get("grad_clip", 1.0))
        optimizer.step()
        scheduler.step()
        ema.update(model)
        loss_sum += loss.item()
        loss_count += 1

        if (step + 1) % eval_every == 0:
            avg_loss = loss_sum / loss_count
            loss_sum = 0.0
            loss_count = 0

            lpips_s = _quick_eval_lpips(test_simple)
            lpips_c = _quick_eval_lpips(test_complex)

            row = {
                "step": step + 1,
                "train_loss": avg_loss,
                "test_lpips_simple": lpips_s,
                "test_lpips_complex": lpips_c,
            }
            with open(log_csv, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)

            print(
                f"  step={step+1}/{total_steps}  loss={avg_loss:.4f}  "
                f"lpips_simple={lpips_s:.4f}  lpips_complex={lpips_c:.4f}"
            )

            # Save checkpoint for resume.
            ckpt_dir = output_dir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            save_checkpoint(ckpt_dir, step + 1, model, ema, optimizer, scheduler, cfg)

    return log_csv


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_grokking(log_csv: Path, output_dir: Path) -> None:
    """Plot Figure 4 equivalent: train loss and test LPIPS over steps."""
    import pandas as pd

    df = pd.read_csv(log_csv)
    steps = df["step"].values / 1000  # k-steps

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Train loss --------------------------------------------------------
    ax1.plot(steps, df["train_loss"].values, color="black", linewidth=2, label="Train loss")
    ax1.set_ylabel("Training loss (MSE)")
    ax1.set_title("Exp 3: Grokking Dynamics (N=40 objects)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # --- Test LPIPS --------------------------------------------------------
    ax2.plot(steps, df["test_lpips_simple"].values, color="steelblue", linewidth=2,
             label="Test LPIPS — simple (Q1)")
    ax2.plot(steps, df["test_lpips_complex"].values, color="tomato", linewidth=2,
             label="Test LPIPS — complex (Q4)")
    ax2.set_xlabel("Training steps (×1000)")
    ax2.set_ylabel("LPIPS (↓ better)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Annotate potential grokking transition: find where LPIPS drops fastest.
    for col, color in [("test_lpips_simple", "steelblue"), ("test_lpips_complex", "tomato")]:
        vals = df[col].values
        if len(vals) > 2:
            diffs = np.diff(vals)
            grok_idx = int(np.argmin(diffs)) + 1  # step of steepest descent
            ax2.axvline(steps[grok_idx], color=color, linestyle="--", alpha=0.5,
                        label=f"Max drop: {steps[grok_idx]:.0f}k")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out_path = output_dir / "exp3_grokking.png"
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")

    # Compute and print grokking statistics.
    print("\nGrokking statistics:")
    for col in ["train_loss", "test_lpips_simple", "test_lpips_complex"]:
        vals = df[col].values
        steps_k = df["step"].values / 1000
        min_idx = np.argmin(vals)
        plateau_idx = np.argmin(np.abs(vals - vals[-1]))
        print(f"  {col}: min={vals[min_idx]:.4f} @ {steps_k[min_idx]:.0f}k, "
              f"final={vals[-1]:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    """Execute Experiment 3."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    N = 40
    seed = 42
    total_steps = args.steps if args.steps else 500_000
    eval_every = args.eval_every if args.eval_every else 5_000

    print("=" * 60)
    print("Experiment 3: Grokking Dynamics")
    print(f"  N={N}, seed={seed}, total_steps={total_steps}, eval_every={eval_every}")
    print("=" * 60)

    # --- Test complexity ---------------------------------------------------
    _, test_ids = get_train_test_split(n_train=80, seed=cfg["data"]["seed"])
    complexity_path = output_dir / "test_complexity.json"
    if complexity_path.exists():
        with open(complexity_path) as f:
            raw = json.load(f)
        complexity_scores = {int(k): v for k, v in raw.items()}
        print("Loaded cached complexity scores.")
    else:
        print("Computing complexity scores ...")
        complexity_scores = compute_all_complexities(
            args.data_root, test_ids,
            alpha_deg=cfg["complexity"]["alpha_deg"],
            device=str(device), verbose=True,
        )
        with open(complexity_path, "w") as f:
            json.dump({str(k): v for k, v in complexity_scores.items()}, f, indent=2)

    qs = get_quartile_splits(complexity_scores)
    test_simple = qs["Q1"]
    test_complex = qs["Q4"]
    print(f"Test simple (Q1): {test_simple}")
    print(f"Test complex (Q4): {test_complex}")

    # --- Train with dense logging ------------------------------------------
    log_csv = train_with_grokking_log(
        cfg=cfg,
        data_root=args.data_root,
        output_dir=output_dir / "model",
        device=device,
        n_train_objects=N,
        seed=seed,
        total_steps=total_steps,
        eval_every=eval_every,
        test_simple=test_simple,
        test_complex=test_complex,
        resume_log=not args.restart,
    )

    # Copy log to top-level output_dir.
    import shutil
    dest = output_dir / "grokking_log.csv"
    shutil.copy(str(log_csv), str(dest))
    print(f"\nFinal log saved to {dest}")

    # --- Plot --------------------------------------------------------------
    plot_grokking(dest, output_dir)

    print(f"\nExperiment 3 complete. Results saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Exp 3: Grokking dynamics.")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--data_root", required=True)
    p.add_argument("--output_dir", default="results/exp3")
    p.add_argument("--steps", type=int, default=None, help="Total training steps (default: 500k).")
    p.add_argument("--eval_every", type=int, default=None, help="Eval interval (default: 5k).")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--restart", action="store_true", help="Restart training from scratch.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
