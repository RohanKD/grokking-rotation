"""
Modal app for Grokking Rotation experiments.

Setup (one-time):
  pip install modal
  modal token new
  python modal_app.py upload   # uploads COIL-100 to Modal volume (~139MB)

Run experiments:
  modal run modal_app.py::run_exp2
  modal run modal_app.py::run_exp3
  modal run modal_app.py::run_exp5_ablation

Download results:
  modal volume get grokking-rotation-vol /results ./results
  modal volume get grokking-rotation-vol /figures ./figures
"""

import modal
import sys
import os

# ── persistent volume (dataset + results survive across runs) ─────────────────
volume = modal.Volume.from_name("grokking-rotation-vol", create_if_missing=True)
REMOTE_DATA   = "/data"       # coil-100 images live here
REMOTE_CODE   = "/code"       # experiment code
REMOTE_RES    = "/results"    # CSVs written here
REMOTE_FIG    = "/figures"    # PNGs written here
REMOTE_CKPT   = "/checkpoints"

# ── container image ───────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "numpy",
        "scipy",
        "tqdm",
        "pyyaml",
        "scikit-image",
        "matplotlib",
        "pandas",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install("pillow")
)

app = modal.App("grokking-rotation", image=image)

# ── mount local code into the container ──────────────────────────────────────
code_mount = modal.Mount.from_local_dir(
    "/home/rohan/perspective/code",
    remote_path=REMOTE_CODE,
)


# ── helper: configure paths inside container ─────────────────────────────────
def _setup():
    import sys
    sys.path.insert(0, REMOTE_CODE)
    os.makedirs(REMOTE_RES,  exist_ok=True)
    os.makedirs(REMOTE_FIG,  exist_ok=True)
    os.makedirs(REMOTE_CKPT, exist_ok=True)

    # Patch run_experiments paths to point at volume
    import run_experiments as re
    from pathlib import Path
    re.COIL_DIR = Path(f"{REMOTE_DATA}/coil-100")
    re.RES_DIR  = Path(REMOTE_RES)
    re.FIG_DIR  = Path(REMOTE_FIG)
    re.CKPT_DIR = Path(REMOTE_CKPT)
    return re


# ── Experiment 2: data scale ─────────────────────────────────────────────────
@app.function(
    gpu="A100",
    volumes={REMOTE_DATA: volume, REMOTE_RES: volume,
             REMOTE_FIG: volume, REMOTE_CKPT: volume},
    mounts=[code_mount],
    timeout=3600,
)
def run_exp2():
    _setup()
    import run_experiments as re
    import torch
    torch.manual_seed(42)
    re.run_exp2(torch.device("cuda"))
    volume.commit()
    print("Exp2 done. Results in volume:/results/exp2_data_scale_real.csv")


# ── Experiment 3: grokking dynamics ──────────────────────────────────────────
@app.function(
    gpu="A100",
    volumes={REMOTE_DATA: volume, REMOTE_RES: volume,
             REMOTE_FIG: volume, REMOTE_CKPT: volume},
    mounts=[code_mount],
    timeout=3600,
)
def run_exp3():
    _setup()
    import run_experiments as re
    import torch
    torch.manual_seed(42)
    re.run_exp3(torch.device("cuda"))
    volume.commit()
    print("Exp3 done. Results in volume:/results/exp3_grokking_real.csv")


# ── Experiment 4: angle generalization ───────────────────────────────────────
@app.function(
    gpu="A100",
    volumes={REMOTE_DATA: volume, REMOTE_RES: volume,
             REMOTE_FIG: volume, REMOTE_CKPT: volume},
    mounts=[code_mount],
    timeout=7200,
)
def run_exp4():
    _setup()
    import run_experiments as re
    import torch
    torch.manual_seed(42)
    re.run_exp4(torch.device("cuda"))
    volume.commit()
    print("Exp4 done. Results in volume:/results/exp4_angle_gen_real.csv")


# ── Experiment 5: Δθ ablation ────────────────────────────────────────────────
@app.function(
    gpu="A100",
    volumes={REMOTE_DATA: volume, REMOTE_RES: volume,
             REMOTE_FIG: volume, REMOTE_CKPT: volume},
    mounts=[code_mount],
    timeout=3600,
)
def run_exp5_ablation():
    _setup()
    import run_experiments as re
    import torch
    torch.manual_seed(42)
    re.run_exp5_ablation(torch.device("cuda"))
    volume.commit()
    print("Exp5 done. Results in volume:/results/exp5_ablation_real.csv")


# ── Run all experiments sequentially ─────────────────────────────────────────
@app.function(
    gpu="A100",
    volumes={REMOTE_DATA: volume, REMOTE_RES: volume,
             REMOTE_FIG: volume, REMOTE_CKPT: volume},
    mounts=[code_mount],
    timeout=14400,
)
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


# ── Upload COIL-100 to the volume ─────────────────────────────────────────────
# Run locally: python modal_app.py upload
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "upload":
    import pathlib

    LOCAL_COIL = pathlib.Path("/home/rohan/perspective/coil-100/coil-100")
    files = sorted(LOCAL_COIL.glob("obj*.png"))
    print(f"Uploading {len(files)} images to Modal volume …")

    with modal.Volume.from_name("grokking-rotation-vol", create_if_missing=True).batch_upload() as batch:
        for f in files:
            batch.put_file(str(f), f"coil-100/{f.name}")

    print("Upload complete. COIL-100 is in volume:/coil-100/")
