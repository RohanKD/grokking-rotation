#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=str, default="data/abo_processed/manifests/pairs_train.csv")
    parser.add_argument("--out", type=str, default="data/abo_processed/logs/pair_sanity_grid.png")
    parser.add_argument("--n", type=int, default=12)
    args = parser.parse_args()

    df = pd.read_csv(args.pairs).sample(args.n, random_state=0).reset_index(drop=True)

    fig, axes = plt.subplots(args.n, 2, figsize=(4, 2 * args.n))

    for i, r in df.iterrows():
        src = Image.open(r["source_image"]).convert("RGB")
        tgt = Image.open(r["target_image"]).convert("RGB")

        axes[i, 0].imshow(src)
        axes[i, 0].set_title(f"src {r['source_angle']:.0f}°")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(tgt)
        axes[i, 1].set_title(
            f"tgt {r['target_angle']:.0f}° | Δ {r['delta_angle']:.0f}° | Q{int(r['complexity_quartile'])}"
        )
        axes[i, 1].axis("off")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()