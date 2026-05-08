"""
Modal app for Grokking Rotation experiments (Modal v1.x API).

Setup (one-time):
  pip install modal
  modal token new
  python modal_app.py upload   # uploads COIL-100 to Modal volume (~139MB)

Run experiments:
  modal run modal_app.py::run_exp2
  modal run modal_app.py::run_exp3          # 30k steps, ~5 min on A100
  modal run modal_app.py::run_exp3_long     # 150k steps, ~25 min on A100
  modal run modal_app.py::run_exp5_ablation
  modal run modal_app.py::run_dino_retrieval
  modal run modal_app.py::run_degraded_baseline
  modal run modal_app.py::run_per_object_curves
  modal run modal_app.py::run_all           # everything, ~45 min

Download results:
  modal volume get grokking-rotation-vol /vol/results ./results
  modal volume get grokking-rotation-vol /vol/figures ./figures
"""

import modal
import os
import sys
from pathlib import Path

# ── persistent volume ─────────────────────────────────────────────────────────
volume = modal.Volume.from_name("grokking-rotation-vol", create_if_missing=True)

REMOTE_DATA  = "/vol/data"
REMOTE_RES   = "/vol/results"
REMOTE_FIG   = "/vol/figures"
REMOTE_CKPT  = "/vol/checkpoints"
VOL_MOUNT    = "/vol"

# ── container image with code baked in ───────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.1+cu121",
        "torchvision==0.18.1+cu121",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install("numpy", "scipy", "tqdm", "pyyaml",
                 "scikit-image", "matplotlib", "pandas", "pillow")
    .add_local_dir("/home/rohan/perspective/code", remote_path="/app/code")
)

app = modal.App("grokking-rotation", image=image)


def _setup():
    """Wire volume paths into the hardcoded structure run_experiments.py expects."""
    sys.path.insert(0, "/app/code")

    # run_experiments.py mkdir's these at import time using hardcoded paths —
    # symlink them into the volume so all I/O lands in persistent storage.
    base = "/home/rohan/perspective"
    os.makedirs(base, exist_ok=True)
    os.makedirs(REMOTE_RES,  exist_ok=True)
    os.makedirs(REMOTE_FIG,  exist_ok=True)
    os.makedirs(REMOTE_CKPT, exist_ok=True)
    # coil-100 needs double nesting: code expects {base}/coil-100/coil-100/obj*.png
    os.makedirs(f"{base}/coil-100", exist_ok=True)
    for src, dst in [
        (REMOTE_RES,               f"{base}/results"),
        (REMOTE_FIG,               f"{base}/figures"),
        (REMOTE_CKPT,              f"{base}/checkpoints"),
        (f"{REMOTE_DATA}/coil-100", f"{base}/coil-100/coil-100"),
    ]:
        if not os.path.exists(dst):
            os.symlink(src, dst)

    import run_experiments as re
    return re


# ── Experiment 2: data scale × complexity ────────────────────────────────────
@app.function(gpu="A100", volumes={VOL_MOUNT: volume}, timeout=3600)
def run_exp2():
    import torch
    re = _setup()
    torch.manual_seed(42)
    re.run_exp2(torch.device("cuda"))
    volume.commit()


# ── Experiment 3: grokking dynamics (30k steps) ──────────────────────────────
@app.function(gpu="A100", volumes={VOL_MOUNT: volume}, timeout=3600)
def run_exp3():
    import torch
    re = _setup()
    torch.manual_seed(42)
    re.run_exp3(torch.device("cuda"))
    volume.commit()


# ── Experiment 3 extended: longer run to observe transition ──────────────────
@app.function(gpu="A100", volumes={VOL_MOUNT: volume}, timeout=7200)
def run_exp3_long():
    """Run Exp3 for 150k steps to actually observe the grokking transition."""
    import torch
    import pandas as pd
    re = _setup()
    torch.manual_seed(42)
    train_ids, test_ids = re.get_train_test_split(40, seed=42)
    _, log = re.train(train_ids, test_ids, total_steps=150000,
                      device=torch.device("cuda"),
                      tag="exp3_long", eval_every=5000, save_ckpt=True)
    log.to_csv(re.RES_DIR / "exp3_long_real.csv", index=False)
    volume.commit()
    print(f"Exp3 long done. Final Q1={log['lpips_q1'].iloc[-1]:.3f} "
          f"Q4={log['lpips_q4'].iloc[-1]:.3f}")


# ── Experiment 4: angle generalization ───────────────────────────────────────
@app.function(gpu="A100", volumes={VOL_MOUNT: volume}, timeout=7200)
def run_exp4():
    import torch
    re = _setup()
    torch.manual_seed(42)
    re.run_exp4(torch.device("cuda"))
    volume.commit()


# ── Experiment 5: Δθ ablation ────────────────────────────────────────────────
@app.function(gpu="A100", volumes={VOL_MOUNT: volume}, timeout=3600)
def run_exp5_ablation():
    import torch
    re = _setup()
    torch.manual_seed(42)
    re.run_exp5_ablation(torch.device("cuda"))
    volume.commit()


# ── DINOv2 retrieval baseline ─────────────────────────────────────────────────
@app.function(gpu="A100", volumes={VOL_MOUNT: volume}, timeout=3600)
def run_dino_retrieval():
    """NN retrieval replacing VGG with DINOv2-ViT-S/14 features."""
    import torch
    re = _setup()
    torch.manual_seed(42)
    re.run_dino_retrieval(torch.device("cuda"))
    volume.commit()


# ── Degraded-model baseline: delta-only conditioning ─────────────────────────
@app.function(gpu="A100", volumes={VOL_MOUNT: volume}, timeout=3600)
def run_degraded_baseline():
    """Train with source image zeroed to measure delta-only performance."""
    import torch
    re = _setup()
    torch.manual_seed(42)
    re.run_degraded_baseline(torch.device("cuda"))
    volume.commit()


# ── Per-object learning curves from Exp3 checkpoint ──────────────────────────
@app.function(gpu="A100", volumes={VOL_MOUNT: volume}, timeout=7200)
def run_per_object_curves():
    """
    Re-evaluate the Exp3 checkpoint at each logged step per object,
    producing per-object LPIPS traces stratified by complexity.
    Requires exp3_grokking.pt checkpoint in the volume.
    """
    import torch
    import pandas as pd
    re = _setup()
    device = torch.device("cuda")

    ckpt_path = re.CKPT_DIR / "exp3_grokking.pt"
    if not ckpt_path.exists():
        print("Checkpoint not found — running Exp3 first to generate it.")
        re.run_exp3(device)

    # Load the logged training curve
    log = pd.read_csv(re.RES_DIR / "exp3_grokking_real.csv")

    # Load complexity scores for per-object stratification
    cx = pd.read_csv(re.RES_DIR / "complexity_scores.csv")
    q1_thresh = cx["complexity_feat"].quantile(0.25)
    q3_thresh = cx["complexity_feat"].quantile(0.75)

    from torchvision import transforms as T_tfm
    img_tfm = T_tfm.Compose([T_tfm.Resize((re.IMG_H, re.IMG_W)), T_tfm.ToTensor()])
    _, test_ids = re.get_train_test_split(40, seed=42)

    # Evaluate the final checkpoint per individual test object
    model = re.ConditionalUNet().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    from data.coil100 import COIL100Dataset
    import torch.nn.functional as F
    rows = []
    for oid in test_ids:
        ds = COIL100Dataset(str(re.COIL_DIR), [oid], angle_delta=re.DELTA_90,
                            transform=img_tfm, length=re.DELTA_90)
        loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)
        dists = []
        for batch in loader:
            src = batch["source"].to(device)
            tgt = batch["target"].to(device)
            dt  = batch["delta_theta"].to(device)
            pred = re.ddim_sample(model, src, dt, device)
            for i in range(pred.shape[0]):
                dists.append(re.perceptual_dist(pred[i:i+1], tgt[i:i+1], device))
        c = cx[cx["obj_id"] == oid]["complexity_feat"]
        v = c.values[0] if len(c) else 0.0
        q = "Q1" if v <= q1_thresh else ("Q4" if v > q3_thresh else "Q2Q3")
        rows.append({"obj_id": oid, "complexity": v, "quartile": q,
                     "mean_lpips": float(sum(dists) / len(dists)) if dists else float("nan")})

    df = pd.DataFrame(rows)
    df.to_csv(re.RES_DIR / "per_object_lpips.csv", index=False)
    volume.commit()
    print(f"Per-object LPIPS saved. Q1 mean={df[df.quartile=='Q1']['mean_lpips'].mean():.3f}  "
          f"Q4 mean={df[df.quartile=='Q4']['mean_lpips'].mean():.3f}")


# ── Run all experiments ───────────────────────────────────────────────────────
@app.function(gpu="A100", volumes={VOL_MOUNT: volume}, timeout=14400)
def run_all():
    import torch
    device = torch.device("cuda")
    re = _setup()
    torch.manual_seed(42)
    re.run_exp2(device)
    re.run_exp3(device)
    re.run_exp4(device)
    re.run_exp5_ablation(device)
    volume.commit()
    print("All experiments done.")


# ── Upload COIL-100 from local machine to Modal volume ───────────────────────
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "upload":
    LOCAL_COIL = Path("/home/rohan/perspective/coil-100/coil-100")
    files = sorted(LOCAL_COIL.glob("obj*.png"))
    print(f"Uploading {len(files)} images (~139MB) to Modal volume …")
    vol = modal.Volume.from_name("grokking-rotation-vol", create_if_missing=True)
    with vol.batch_upload() as batch:
        for f in files:
            batch.put_file(str(f), f"data/coil-100/{f.name}")
    print("Upload complete.")
