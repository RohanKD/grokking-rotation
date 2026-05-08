"""Experiment 4: Interpolation vs. Extrapolation Angle Generalization.

Research question
-----------------
If we train on only Δθ=90° pairs, can the model generalize to other angles?
  - Interpolation angles (within 0°–90° range): [45°, 60°, 75°]
  - Extrapolation angles (beyond 90°):            [120°, 135°, 180°]

We also test whether generalization interacts with N (data scale).

Protocol
--------
For N in [10, 20, 40, 60, 80]:
  1. Train on ONLY Δθ=90° pairs.
  2. Evaluate on angles [45, 60, 75, 90, 120, 135, 180].
  3. Record LPIPS for simple (Q1) and complex (Q4) test objects.

Output:
  angle_gen_results.csv   – [N, angle, type, complexity, lpips]
  exp4_heatmaps.png       – Figure 5 heatmaps

Usage
-----
    python experiments/exp4_angle_generalization.py \
        --config configs/base.yaml \
        --data_root /path/to/coil-100 \
        --output_dir results/exp4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
# Training: fixed delta = 90 degrees
# ---------------------------------------------------------------------------

TRAIN_ANGLE_DEG: int = 90
TRAIN_ANGLE_IDX: int = TRAIN_ANGLE_DEG // 5  # = 18


def train_fixed_delta(
    cfg: Dict,
    data_root: str,
    output_dir: Path,
    device: torch.device,
    n_train_objects: int,
    seed: int,
    total_steps: int,
) -> Path:
    """Train using only Δθ=90° pairs.  Returns path to final.pt."""
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

    # Fix angle_delta=18 (90 degrees) for ALL training batches.
    loader = build_dataloader(
        root=data_root,
        object_ids=train_ids,
        batch_size=cfg["training"]["batch_size"],
        angle_delta=TRAIN_ANGLE_IDX,  # <-- fixed delta
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

        if (step + 1) % 2000 == 0:
            print(
                f"    [N={n_train_objects} seed={seed}] "
                f"step={step+1}/{total_steps} loss={loss_sum/2000:.4f}"
            )
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


# ---------------------------------------------------------------------------
# Evaluation at a specific angle
# ---------------------------------------------------------------------------


def eval_at_angle(
    ckpt_path: Path,
    obj_ids: List[int],
    angle_deg: int,
    data_root: str,
    device: torch.device,
    cfg: Dict,
    n_eval: int = 8,
) -> float:
    """Load checkpoint and evaluate LPIPS at a fixed angle delta."""
    ckpt = torch.load(str(ckpt_path), map_location=str(device))
    cfg_ckpt = ckpt.get("config", cfg)
    model = build_model(cfg_ckpt, device)
    ema = EMAHelper(model)
    load_checkpoint(str(ckpt_path), model, ema, device=str(device))
    ema.copy_to(model)
    model.eval()

    metrics = PerceptualMetrics(device=str(device))
    n_ddim = cfg_ckpt["diffusion"]["ddim_steps"]

    angle_idx = (angle_deg // 5) % 72
    ids = obj_ids[:min(len(obj_ids), n_eval)]
    loader = build_dataloader(
        root=data_root,
        object_ids=ids,
        batch_size=n_eval,
        angle_delta=angle_idx,
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


# ---------------------------------------------------------------------------
# Plotting: heatmaps
# ---------------------------------------------------------------------------

EVAL_ANGLES = [45, 60, 75, 90, 120, 135, 180]
ANGLE_TYPES = {45: "interp", 60: "interp", 75: "interp", 90: "train",
               120: "extrap", 135: "extrap", 180: "extrap"}
NS = [10, 20, 40, 60, 80]


def plot_heatmaps(results_csv: Path, output_dir: Path) -> None:
    """Plot Figure 5: N × angle LPIPS heatmaps for simple and complex objects."""
    import pandas as pd

    df = pd.read_csv(results_csv)
    angles = EVAL_ANGLES
    Ns = NS

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, complexity_label in zip(axes, ["simple", "complex"]):
        sub = df[df["complexity"] == complexity_label]
        # Build N x angle matrix.
        mat = np.full((len(Ns), len(angles)), float("nan"))
        for i, N in enumerate(Ns):
            for j, angle in enumerate(angles):
                row = sub[(sub["N"] == N) & (sub["angle"] == angle)]
                if len(row) > 0:
                    mat[i, j] = row["lpips"].mean()

        vmin = np.nanmin(mat)
        vmax = np.nanmax(mat)
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=vmin, vmax=vmax,
                       interpolation="nearest")

        # Axis labels.
        angle_labels = []
        for a in angles:
            t = ANGLE_TYPES.get(a, "?")
            marker = "★" if t == "train" else ("↔" if t == "interp" else "→")
            angle_labels.append(f"{a}°{marker}")
        ax.set_xticks(range(len(angles)))
        ax.set_xticklabels(angle_labels, fontsize=9)
        ax.set_yticks(range(len(Ns)))
        ax.set_yticklabels([f"N={n}" for n in Ns])
        ax.set_xlabel("Evaluation angle Δθ")
        ax.set_ylabel("N training objects")
        ax.set_title(f"LPIPS — {complexity_label.capitalize()} objects (Q{'1' if complexity_label == 'simple' else '4'})")

        # Add colorbar.
        plt.colorbar(im, ax=ax, label="LPIPS (↓ better)")

        # Annotate cells with values.
        for i in range(len(Ns)):
            for j in range(len(angles)):
                val = mat[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=7, color="black" if val < (vmax + vmin) / 2 else "white")

        # Draw vertical line separating interp/extrap.
        # Train angle (90°) is at index 3.
        ax.axvline(3.5, color="navy", linewidth=2, linestyle="--", label="← interp | extrap →")
        ax.legend(fontsize=8, loc="upper right")

    plt.suptitle(
        "Exp 4: Angle Generalization (trained on Δθ=90° only)\n"
        "★ = train angle   ↔ = interpolation   → = extrapolation",
        fontsize=11,
    )
    plt.tight_layout()
    out_path = output_dir / "exp4_heatmaps.png"
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")

    # Line plot: LPIPS vs. angle for each N.
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(Ns)))
    for ax, complexity_label in zip(axes2, ["simple", "complex"]):
        sub = df[df["complexity"] == complexity_label]
        for ni, (N, c) in enumerate(zip(Ns, colors)):
            ys = [sub[(sub["N"] == N) & (sub["angle"] == a)]["lpips"].mean()
                  if len(sub[(sub["N"] == N) & (sub["angle"] == a)]) > 0 else float("nan")
                  for a in angles]
            ax.plot(angles, ys, marker="o", color=c, label=f"N={N}", linewidth=2)

        ax.axvline(90, color="navy", linestyle="--", alpha=0.5, label="Train angle")
        ax.axvspan(0, 90, alpha=0.05, color="steelblue", label="Interp zone")
        ax.axvspan(90, 200, alpha=0.05, color="tomato", label="Extrap zone")
        ax.set_xticks(angles)
        ax.set_xticklabels([f"{a}°" for a in angles])
        ax.set_xlabel("Evaluation angle Δθ")
        ax.set_ylabel("LPIPS (↓ better)")
        ax.set_title(f"{complexity_label.capitalize()} objects (Q{'1' if complexity_label == 'simple' else '4'})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Exp 4: LPIPS vs. Evaluation Angle", fontsize=13)
    plt.tight_layout()
    out_path2 = output_dir / "exp4_line_plots.png"
    fig2.savefig(out_path2, dpi=150)
    plt.close()
    print(f"  Saved {out_path2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    """Execute Experiment 4."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    total_steps = args.steps if args.steps else cfg["training"]["total_steps"]
    seed = 42

    print("=" * 60)
    print("Experiment 4: Interpolation vs. Extrapolation Generalization")
    print(f"  Train angle: {TRAIN_ANGLE_DEG}°  (index {TRAIN_ANGLE_IDX})")
    print(f"  Eval angles: {EVAL_ANGLES}")
    print(f"  N values:    {NS}")
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

    # --- Results CSV -------------------------------------------------------
    results_csv = output_dir / "angle_gen_results.csv"
    fields = ["N", "angle", "type", "complexity", "lpips"]
    existing_keys = set()
    if results_csv.exists():
        import pandas as pd
        df_ex = pd.read_csv(results_csv)
        for _, row in df_ex.iterrows():
            existing_keys.add((int(row["N"]), int(row["angle"]), str(row["complexity"])))
        print(f"  Resuming: found {len(existing_keys)} completed (N, angle, complexity) entries.")

    # --- Main loop ---------------------------------------------------------
    total_runs = len(NS)
    for run_idx, N in enumerate(NS):
        print(f"\n[{run_idx+1}/{total_runs}] N={N}")
        run_dir = output_dir / f"N{N}"
        ckpt_path = run_dir / "final.pt"

        if not ckpt_path.exists():
            print(f"  Training on Δθ=90° only ...")
            ckpt_path = train_fixed_delta(cfg, args.data_root, run_dir, device, N, seed, total_steps)
        else:
            print(f"  Found existing checkpoint.")

        for angle in EVAL_ANGLES:
            atype = ANGLE_TYPES.get(angle, "unknown")
            for complexity_label, obj_ids in [("simple", test_simple), ("complex", test_complex)]:
                key = (N, angle, complexity_label)
                if key in existing_keys:
                    print(f"    angle={angle}° ({atype}) {complexity_label}: already computed.")
                    continue

                print(f"    Evaluating angle={angle}° ({atype}) on {complexity_label} objects ...")
                lpips_val = eval_at_angle(ckpt_path, obj_ids, angle, args.data_root, device, cfg)
                print(f"    LPIPS={lpips_val:.4f}")

                row = {
                    "N": N,
                    "angle": angle,
                    "type": atype,
                    "complexity": complexity_label,
                    "lpips": lpips_val,
                }
                with open(results_csv, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fields)
                    if f.tell() == 0:
                        w.writeheader()
                    w.writerow(row)
                existing_keys.add(key)

    # Re-save cleanly.
    import pandas as pd
    df = pd.read_csv(results_csv)
    df.to_csv(results_csv, index=False)

    # --- Print summary table -----------------------------------------------
    print("\n" + "=" * 70)
    print("Experiment 4 summary")
    print("=" * 70)
    print(f"{'N':>6} {'Angle':>7} {'Type':>8} {'Complexity':>10} {'LPIPS':>8}")
    print("-" * 45)
    for N in NS:
        for angle in EVAL_ANGLES:
            for comp in ["simple", "complex"]:
                sub = df[(df["N"] == N) & (df["angle"] == angle) & (df["complexity"] == comp)]
                if len(sub) == 0:
                    continue
                atype = ANGLE_TYPES.get(angle, "?")
                print(f"{N:>6} {angle:>6}° {atype:>8} {comp:>10} {sub['lpips'].mean():>8.4f}")
        print()

    # --- Plot heatmaps -----------------------------------------------------
    plot_heatmaps(results_csv, output_dir)

    print(f"\nExperiment 4 complete. Results saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Exp 4: Angle interpolation vs. extrapolation.")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--data_root", required=True)
    p.add_argument("--output_dir", default="results/exp4")
    p.add_argument("--steps", type=int, default=None, help="Override training steps.")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
