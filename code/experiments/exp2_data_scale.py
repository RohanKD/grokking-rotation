"""Experiment 2: N × Complexity Interaction (Data Scale).

Research question
-----------------
Does the number of training objects N interact with object complexity?
Specifically: does adding more training data help simple objects more
than complex ones, or does complexity remain a persistent bottleneck?

Protocol
--------
For N in [10, 20, 40, 60, 80] and 5 seeds each:
  1. Train a model on N objects.
  2. Evaluate on a fixed set of simple (Q1) and complex (Q4) test objects.
  3. Record lpips_simple and lpips_complex.

Output:
  results_exp2.csv     – [N, seed, lpips_simple, lpips_complex]
  exp2_figure.png      – Figure 3 equivalent

Usage
-----
    python experiments/exp2_data_scale.py \
        --config configs/base.yaml \
        --data_root /path/to/coil-100 \
        --output_dir results/exp2
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.coil100 import get_train_test_split, build_dataloader
from models.unet import UNet
from models.ddpm import ConditionalDDPM, EMAHelper
from metrics.perceptual import PerceptualMetrics
from complexity import compute_all_complexities, get_quartile_splits
from train import (
    load_config, build_model, load_checkpoint, set_seed, get_lr_scheduler,
    save_checkpoint, evaluate_test_set
)
from evaluate import evaluate_model, copy_source_baseline


# ---------------------------------------------------------------------------
# Training helper (inline, without subprocess)
# ---------------------------------------------------------------------------


def train_model(
    cfg: Dict,
    data_root: str,
    output_dir: Path,
    device: torch.device,
    n_train_objects: int,
    seed: int,
    total_steps: int,
) -> Path:
    """Train a model and save checkpoint; returns path to final.pt."""
    import math
    import time
    from torch.utils.data import DataLoader
    from data.coil100 import build_dataloader

    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    # Split objects: use the first n_train_objects from a fixed 80-object train set,
    # but with a seed-specific shuffle within those 80 so different seeds vary object composition.
    train_80, _ = get_train_test_split(n_train=80, seed=cfg["data"]["seed"])
    rng = random.Random(seed)
    shuffled = train_80.copy()
    rng.shuffle(shuffled)
    train_ids = sorted(shuffled[:n_train_objects])

    model = build_model(cfg, device)
    ema = EMAHelper(model, decay=cfg["training"].get("ema_decay", 0.9999))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    scheduler = get_lr_scheduler(
        optimizer, total_steps=total_steps, use_cosine=cfg["training"]["cosine_schedule"]
    )

    loader = build_dataloader(
        root=data_root,
        object_ids=train_ids,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        length=total_steps * cfg["training"]["batch_size"],
    )
    loader_iter = iter(loader)

    model.train()
    loss_sum = 0.0
    for step in range(total_steps):
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

        if (step + 1) % 1000 == 0:
            print(f"    N={n_train_objects} seed={seed} step={step+1}/{total_steps} "
                  f"loss={loss_sum/1000:.4f}")
            loss_sum = 0.0

    ckpt_path = output_dir / "final.pt"
    torch.save(
        {
            "step": total_steps,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "config": cfg,
        },
        ckpt_path,
    )
    return ckpt_path


def eval_on_fixed_test(
    ckpt_path: Path,
    test_simple: List[int],
    test_complex: List[int],
    data_root: str,
    device: torch.device,
    cfg: Dict,
    n_eval: int = 8,
) -> Tuple[float, float]:
    """Load checkpoint and evaluate LPIPS on simple and complex objects."""
    ckpt = torch.load(str(ckpt_path), map_location=str(device))
    cfg_ckpt = ckpt.get("config", cfg)
    model = build_model(cfg_ckpt, device)
    ema = EMAHelper(model)
    load_checkpoint(str(ckpt_path), model, ema, device=str(device))
    ema.copy_to(model)
    model.eval()

    metrics = PerceptualMetrics(device=str(device))
    n_ddim = cfg_ckpt["diffusion"]["ddim_steps"]

    def _eval_group(obj_ids: List[int]) -> float:
        ids = obj_ids[:min(len(obj_ids), n_eval)]
        loader = build_dataloader(
            root=data_root,
            object_ids=ids,
            batch_size=n_eval,
            shuffle=False,
            num_workers=0,
            length=n_eval,
        )
        batch = next(iter(loader))
        src = batch["source"].to(device) * 2.0 - 1.0
        tgt = batch["target"].to(device)
        delta = batch["delta_theta"].float().to(device)
        with torch.no_grad():
            pred_n = model.ddim_sample(src, delta, n_steps=n_ddim)
        pred_01 = ((pred_n + 1.0) / 2.0).clamp(0.0, 1.0)
        return metrics.compute_lpips(pred_01, tgt)

    lpips_s = _eval_group(test_simple)
    lpips_c = _eval_group(test_complex)
    return lpips_s, lpips_c


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_figure3(results_csv: Path, output_dir: Path) -> None:
    """Reproduce Figure 3: LPIPS vs. N, separated by complexity."""
    import pandas as pd

    df = pd.read_csv(results_csv)
    Ns = sorted(df["N"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (col, ylabel, title) in zip(
        axes,
        [
            ("lpips_simple", "LPIPS (↓ better)", "Simple objects (Q1)"),
            ("lpips_complex", "LPIPS (↓ better)", "Complex objects (Q4)"),
        ],
    ):
        means = [df[df["N"] == n][col].mean() for n in Ns]
        stds = [df[df["N"] == n][col].std() for n in Ns]
        ax.errorbar(Ns, means, yerr=stds, marker="o", linewidth=2, capsize=4,
                    color="steelblue" if "simple" in col else "tomato",
                    label="Mean ± std (5 seeds)")
        ax.set_xlabel("N training objects")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(Ns)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Exp 2: N × Complexity Interaction", fontsize=13)
    plt.tight_layout()
    out_path = output_dir / "exp2_figure3.png"
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")

    # Also plot both on the same axis for direct comparison.
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for col, color, label in [
        ("lpips_simple", "steelblue", "Simple (Q1)"),
        ("lpips_complex", "tomato", "Complex (Q4)"),
    ]:
        means = [df[df["N"] == n][col].mean() for n in Ns]
        stds = [df[df["N"] == n][col].std() for n in Ns]
        ax2.errorbar(Ns, means, yerr=stds, marker="o", linewidth=2, capsize=4,
                     color=color, label=label)
    ax2.set_xlabel("N training objects")
    ax2.set_ylabel("LPIPS (↓ better)")
    ax2.set_title("Exp 2: LPIPS vs. Training Scale by Complexity")
    ax2.set_xticks(Ns)
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    out_path2 = output_dir / "exp2_combined.png"
    fig2.savefig(out_path2, dpi=150)
    plt.close()
    print(f"  Saved {out_path2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    """Execute Experiment 2."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    total_steps = args.steps if args.steps else cfg["training"]["total_steps"]

    Ns = [10, 20, 40, 60, 80]
    seeds = [42, 43, 44, 45, 46]

    print("=" * 60)
    print("Experiment 2: N × Complexity Interaction")
    print(f"  N values: {Ns}")
    print(f"  Seeds:    {seeds}")
    print(f"  Steps:    {total_steps}")
    print("=" * 60)

    # --- Fixed test set & complexity scores --------------------------------
    _, test_ids = get_train_test_split(n_train=80, seed=cfg["data"]["seed"])
    print(f"\nComputing complexity scores for {len(test_ids)} test objects ...")
    complexity_path = output_dir / "test_complexity.json"
    if complexity_path.exists():
        with open(complexity_path) as f:
            raw = json.load(f)
        complexity_scores = {int(k): v for k, v in raw.items()}
        print("  Loaded cached complexity scores.")
    else:
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
    print(f"  Simple (Q1): {test_simple}")
    print(f"  Complex (Q4): {test_complex}")

    # --- Results CSV -------------------------------------------------------
    results_csv = output_dir / "results_exp2.csv"
    fields = ["N", "seed", "lpips_simple", "lpips_complex"]
    existing_keys = set()
    if results_csv.exists():
        import pandas as pd
        df_existing = pd.read_csv(results_csv)
        for _, row in df_existing.iterrows():
            existing_keys.add((int(row["N"]), int(row["seed"])))
        print(f"\n  Resuming: found {len(existing_keys)} completed runs.")

    # --- Main loop ---------------------------------------------------------
    total_runs = len(Ns) * len(seeds)
    run_idx = 0
    for N in Ns:
        for seed in seeds:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] N={N}, seed={seed}")

            if (N, seed) in existing_keys:
                print(f"  Already computed — skipping.")
                continue

            run_dir = output_dir / f"N{N}_seed{seed}"
            ckpt_path = run_dir / "final.pt"

            if not ckpt_path.exists():
                print(f"  Training ...")
                ckpt_path = train_model(cfg, args.data_root, run_dir, device, N, seed, total_steps)

            print(f"  Evaluating ...")
            lpips_s, lpips_c = eval_on_fixed_test(
                ckpt_path, test_simple, test_complex, args.data_root, device, cfg
            )
            print(f"  lpips_simple={lpips_s:.4f}  lpips_complex={lpips_c:.4f}")

            with open(results_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerow({"N": N, "seed": seed, "lpips_simple": lpips_s, "lpips_complex": lpips_c})
            existing_keys.add((N, seed))

    # Ensure header exists even if we wrote rows incrementally.
    import pandas as pd
    df = pd.read_csv(results_csv) if results_csv.exists() else pd.DataFrame(columns=fields)
    # Re-save cleanly with header.
    df.to_csv(results_csv, index=False)

    # --- Print summary table -----------------------------------------------
    print("\n" + "=" * 60)
    print("Experiment 2 summary")
    print("=" * 60)
    print(f"{'N':>6} {'LPIPS simple':>14} {'LPIPS complex':>14}")
    print("-" * 36)
    for N in Ns:
        sub = df[df["N"] == N]
        if len(sub) == 0:
            continue
        ms = sub["lpips_simple"].mean()
        ss = sub["lpips_simple"].std()
        mc = sub["lpips_complex"].mean()
        sc = sub["lpips_complex"].std()
        print(f"{N:>6} {ms:>7.4f}±{ss:.4f} {mc:>7.4f}±{sc:.4f}")

    # --- Plot --------------------------------------------------------------
    plot_figure3(results_csv, output_dir)

    print(f"\nExperiment 2 complete. Results saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Exp 2: N × complexity interaction.")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--data_root", required=True)
    p.add_argument("--output_dir", default="results/exp2")
    p.add_argument("--steps", type=int, default=None, help="Override training steps.")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
