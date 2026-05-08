"""
Conditional VAE with disentangled identity/rotation latents for COIL-100 novel-view synthesis.

Architecture
------------
  IdentityEncoder  : x_s (3×64×64) → (μ, logvar) ∈ ℝ^{Z_ID_DIM=128}
  RotationEmbedder : Δθ (rad, scalar) → z_rot ∈ ℝ^{Z_ROT_DIM=32}
                     via MLP([sin Δθ, cos Δθ])
  Decoder          : [z_id ‖ z_rot] → x̂_t ∈ [0,1]^{3×64×64}

Losses (per step)
-----------------
  L_recon   = MSE(x̂_t, x_t)
  L_KL      = β · KL( q(z_id|x_s) ‖ 𝒩(0,I) )
  L_dis     = λ_dis · (1 − cos_sim(μ(x_s), μ(x_t)))   ← view-invariance on μ
  L_perc    = λ_perc · VGG-cosine-dist(x̂_t, x_t)
  L_total   = L_recon + L_KL + L_dis + L_perc

Latent diagnostic
-----------------
  log_z_rot_circle(): evaluate RotationEmbedder on all 72 COIL-100 angles;
  project to 2D via PCA; smooth circle ≡ rotation structure learned.
  Also track z_id consistency (std across views of same object ↓ with training).
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torchvision.models as tvm

sys.path.insert(0, str(Path(__file__).parent))
from data.coil100 import COIL100Dataset, get_train_test_split

# ─── paths ────────────────────────────────────────────────────────────────────
ROOT     = Path('/home/rohan/perspective')
COIL_DIR = ROOT / 'coil-100' / 'coil-100'
RES_DIR  = ROOT / 'results'
FIG_DIR  = ROOT / 'figures'
CKPT_DIR = ROOT / 'checkpoints'
for _d in [RES_DIR, FIG_DIR, CKPT_DIR]: _d.mkdir(exist_ok=True)

# ─── hyperparameters ─────────────────────────────────────────────────────────
IMG_H = IMG_W = 64
Z_ID_DIM   = 128    # identity latent dim
Z_ROT_DIM  = 32     # rotation embedding dim
BATCH      = 64
LR         = 2e-4
BETA_KL    = 1e-3   # KL weight (small → keep reconstruction quality)
LAMBDA_DIS = 1.0    # view-invariance loss weight
LAMBDA_PERC = 0.1   # perceptual loss weight
DELTA_90   = 18     # angle_idx for 90° (18 × 5°)


# ═══════════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════════

def _enc_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
        nn.GroupNorm(8, out_ch),
        nn.SiLU(),
    )


def _dec_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
        nn.GroupNorm(8, out_ch),
        nn.SiLU(),
    )


class IdentityEncoder(nn.Module):
    """CNN: 3×64×64 → (μ, logvar) each ∈ ℝ^{Z_ID_DIM}. ~3M params."""
    def __init__(self, z_dim: int = Z_ID_DIM):
        super().__init__()
        self.cnn = nn.Sequential(
            _enc_block(3,   32),   # 64→32
            _enc_block(32,  64),   # 32→16
            _enc_block(64,  128),  # 16→8
            _enc_block(128, 256),  # 8→4
        )
        self.fc       = nn.Linear(256 * 4 * 4, 512)
        self.mu_head  = nn.Linear(512, z_dim)
        self.lv_head  = nn.Linear(512, z_dim)

    def forward(self, x: torch.Tensor):
        h = F.silu(self.fc(self.cnn(x).flatten(1)))
        return self.mu_head(h), self.lv_head(h)


class RotationEmbedder(nn.Module):
    """Δθ (radians, shape B) → z_rot ∈ ℝ^{Z_ROT_DIM} via MLP([sin Δθ, cos Δθ])."""
    def __init__(self, z_dim: int = Z_ROT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, z_dim),
        )

    def forward(self, delta_theta: torch.Tensor) -> torch.Tensor:
        sc = torch.stack([delta_theta.sin(), delta_theta.cos()], dim=-1).float()
        return self.net(sc)


class Decoder(nn.Module):
    """[z_id ‖ z_rot] → 3×64×64 image in [0,1]. ~2M params."""
    def __init__(self, z_id_dim: int = Z_ID_DIM, z_rot_dim: int = Z_ROT_DIM):
        super().__init__()
        z_in = z_id_dim + z_rot_dim
        self.fc = nn.Sequential(
            nn.Linear(z_in, 512), nn.SiLU(),
            nn.Linear(512, 256 * 4 * 4), nn.SiLU(),
        )
        self.net = nn.Sequential(
            _dec_block(256, 128),   # 4→8
            _dec_block(128, 64),    # 8→16
            _dec_block(64,  32),    # 16→32
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  # 32→64
            nn.Sigmoid(),
        )

    def forward(self, z_id: torch.Tensor, z_rot: torch.Tensor) -> torch.Tensor:
        h = self.fc(torch.cat([z_id, z_rot], dim=-1)).reshape(-1, 256, 4, 4)
        return self.net(h)


class ConditionalVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = IdentityEncoder()
        self.rot_emb  = RotationEmbedder()
        self.decoder  = Decoder()

    def forward(self, x_s: torch.Tensor, delta_theta: torch.Tensor):
        mu, logvar = self.encoder(x_s)
        z_id  = _reparameterize(mu, logvar) if self.training else mu
        z_rot = self.rot_emb(delta_theta)
        return self.decoder(z_id, z_rot), mu, logvar

    @torch.no_grad()
    def reconstruct(self, x_s: torch.Tensor, delta_theta: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encoder(x_s)
        return self.decoder(mu, self.rot_emb(delta_theta))


# ═══════════════════════════════════════════════════════════════════════════════
# Loss helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = (0.5 * logvar).exp()
    return mu + std * torch.randn_like(std)


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()


# ─── VGG perceptual distance (same as run_experiments.py) ────────────────────
_vgg_net  = None
_VGG_MEAN = torch.tensor([0.485, 0.456, 0.406])
_VGG_STD  = torch.tensor([0.229, 0.224, 0.225])

def _get_vgg(device: torch.device) -> nn.Module:
    global _vgg_net
    if _vgg_net is None:
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT)
        _vgg_net = nn.Sequential(*list(vgg.features.children())[:18]).eval()
        for p in _vgg_net.parameters():
            p.requires_grad_(False)
    return _vgg_net.to(device)


def perceptual_dist(pred: torch.Tensor, target: torch.Tensor,
                    device: torch.device) -> float:
    """Mean VGG-16 cosine distance. pred/target: (B,3,H,W) in [0,1]."""
    vgg  = _get_vgg(device)
    mean = _VGG_MEAN.to(device)[None, :, None, None]
    std  = _VGG_STD.to(device)[None, :, None, None]
    def _feat(x):
        f = vgg((x - mean) / std).flatten(1)
        return F.normalize(f, dim=1)
    return (1 - (_feat(pred) * _feat(target)).sum(dim=1)).mean().item()


def perceptual_loss(pred: torch.Tensor, target: torch.Tensor,
                    device: torch.device) -> torch.Tensor:
    """Differentiable version of perceptual_dist."""
    vgg  = _get_vgg(device)
    mean = _VGG_MEAN.to(device)[None, :, None, None]
    std  = _VGG_STD.to(device)[None, :, None, None]
    def _feat(x):
        return F.normalize(vgg((x - mean) / std).flatten(1), dim=1)
    return (1 - (_feat(pred) * _feat(target)).sum(dim=1)).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def _quartile_fn(cx: pd.DataFrame):
    q1 = cx['complexity_feat'].quantile(0.25)
    q3 = cx['complexity_feat'].quantile(0.75)
    q2 = cx['complexity_feat'].quantile(0.50)
    def cat(oid):
        v = cx[cx['obj_id'] == oid]['complexity_feat'].values
        if len(v) == 0: return 'Q2'
        v = v[0]
        if v <= q1: return 'Q1'
        if v <= q2: return 'Q2'
        if v <= q3: return 'Q3'
        return 'Q4'
    return cat


@torch.no_grad()
def evaluate_vae(model: ConditionalVAE, test_ids, device: torch.device,
                 n_samples: int = 80, delta_idx: int = DELTA_90) -> dict:
    """LPIPS by complexity quartile, matching run_experiments.evaluate() signature."""
    model.eval()
    from torchvision import transforms as T_tfm
    img_tfm = T_tfm.Compose([T_tfm.Resize((IMG_H, IMG_W)), T_tfm.ToTensor()])
    cx = pd.read_csv(RES_DIR / 'complexity_scores.csv')
    qcat = _quartile_fn(cx)

    ds = COIL100Dataset(str(COIL_DIR), test_ids, angle_delta=delta_idx,
                        transform=img_tfm, length=n_samples)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)

    results = {'Q1': [], 'Q2': [], 'Q3': [], 'Q4': []}
    for batch in loader:
        src = batch['source'].to(device)
        tgt = batch['target'].to(device)
        dt  = batch['delta_theta'].to(device)
        pred = model.reconstruct(src, dt)
        for i, oid in enumerate(batch['object_id'].tolist()):
            d = perceptual_dist(pred[i:i+1], tgt[i:i+1], device)
            results[qcat(oid)].append(d)
    model.train()
    return {q: float(np.mean(v)) if v else float('nan') for q, v in results.items()}


@torch.no_grad()
def evaluate_id_consistency(model: ConditionalVAE, test_ids,
                             device: torch.device, n_views: int = 8) -> float:
    """
    Mean std of z_id (μ) across views of the same object.
    Lower = more view-invariant identity encoding.
    """
    model.eval()
    from torchvision import transforms as T_tfm
    from PIL import Image
    img_tfm = T_tfm.Compose([T_tfm.Resize((IMG_H, IMG_W)), T_tfm.ToTensor()])

    step = 72 // n_views
    stds = []
    for oid in test_ids[:20]:
        mus = []
        for ang_idx in range(0, 72, step):
            path = COIL_DIR / f'obj{oid}__{ang_idx * 5}.png'
            if not path.exists():
                continue
            img = img_tfm(Image.open(path).convert('RGB')).unsqueeze(0).to(device)
            mu, _ = model.encoder(img)
            mus.append(mu.squeeze(0).cpu().float().numpy())
        if len(mus) > 1:
            stds.append(np.stack(mus).std(axis=0).mean())
    model.train()
    return float(np.mean(stds)) if stds else float('nan')


# ═══════════════════════════════════════════════════════════════════════════════
# Latent circle diagnostic
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def log_z_rot_circle(model: ConditionalVAE, device: torch.device) -> np.ndarray:
    """
    Evaluate RotationEmbedder at all 72 COIL-100 angles (0°, 5°, …, 355°).
    Returns array of shape (72, Z_ROT_DIM).
    """
    model.eval()
    angles_deg = np.arange(0, 360, 5)   # 72 values
    z_rots = []
    for ang in angles_deg:
        rad = torch.tensor([ang * math.pi / 180.0], device=device, dtype=torch.float32)
        z_rots.append(model.rot_emb(rad).squeeze(0).cpu().float().numpy())
    model.train()
    return np.stack(z_rots)   # (72, Z_ROT_DIM)


def circle_score(z_rots: np.ndarray) -> float:
    """
    PCA circularity: ratio of variance in PC1+PC2 to total variance.
    Approaches 1.0 when z_rot lies on a 2D plane (circular manifold).
    """
    from sklearn.decomposition import PCA
    pca = PCA().fit(z_rots)
    return float(pca.explained_variance_ratio_[:2].sum())


def plot_z_rot_evolution(latent_snapshots: list, save_path: Path):
    """
    latent_snapshots: list of {'step': int, 'z_rots': (72, Z_ROT_DIM)}
    Panels show the PCA projection at each logged step.
    """
    from sklearn.decomposition import PCA
    n = len(latent_snapshots)
    ncols = min(n, 5)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.8 * nrows))
    axes = np.array(axes).reshape(-1)
    angles = np.linspace(0, 360, 72, endpoint=False)
    cmap   = plt.cm.hsv

    for i, snap in enumerate(latent_snapshots):
        ax = axes[i]
        z  = snap['z_rots']
        pca = PCA(n_components=2).fit(z)
        z2  = pca.transform(z)
        sc  = circle_score(z)
        ax.scatter(z2[:, 0], z2[:, 1], c=angles, cmap=cmap, s=20, vmin=0, vmax=360)
        ax.set_title(f"step {snap['step']//1000}k\ncircle={sc:.2f}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle('z_rot PCA projection across training (hue = Δθ)', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved latent circle plot → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════════

def train_vae(train_ids, test_ids, total_steps: int, device: torch.device,
              tag: str, eval_every: int = 1000, log_latent_every: int = 5000,
              save_ckpt: bool = False):
    from torchvision import transforms as T_tfm
    img_tfm = T_tfm.Compose([T_tfm.Resize((IMG_H, IMG_W)), T_tfm.ToTensor()])

    model = ConditionalVAE().to(device)
    opt   = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=total_steps, eta_min=LR * 0.1)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"ConditionalVAE: {n_params/1e6:.2f}M params")

    ds = COIL100Dataset(str(COIL_DIR), train_ids, angle_delta=None,
                        transform=img_tfm,
                        length=max(len(train_ids) * 72, BATCH * 4))
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True)

    log: list          = []
    latent_snapshots: list = []
    step               = 0
    pbar = tqdm(total=total_steps, desc=tag, dynamic_ncols=True)

    while step < total_steps:
        for batch in loader:
            if step >= total_steps:
                break

            x_s = batch['source'].to(device)
            x_t = batch['target'].to(device)
            dt  = batch['delta_theta'].float().to(device)

            # ── forward ──────────────────────────────────────────────────────
            x_hat, mu, logvar = model(x_s, dt)

            # ── losses ───────────────────────────────────────────────────────
            L_recon = F.mse_loss(x_hat, x_t)
            L_kl    = BETA_KL * kl_loss(mu, logvar)

            # Disentanglement: encode x_t with same identity encoder
            mu_tgt, _ = model.encoder(x_t)
            L_dis = LAMBDA_DIS * (1.0 - F.cosine_similarity(mu, mu_tgt, dim=1).mean())

            # Perceptual loss (no grad through VGG weights)
            L_perc = LAMBDA_PERC * perceptual_loss(x_hat, x_t, device)

            loss = L_recon + L_kl + L_dis + L_perc

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            pbar.update(1)
            pbar.set_postfix(recon=f'{L_recon.item():.4f}',
                             kl=f'{L_kl.item():.4f}',
                             dis=f'{L_dis.item():.3f}')

            # ── eval ─────────────────────────────────────────────────────────
            if step % eval_every == 0:
                scores  = evaluate_vae(model, test_ids, device)
                id_cons = evaluate_id_consistency(model, test_ids, device)
                row = {
                    'step':      step,
                    'recon':     L_recon.item(),
                    'kl':        L_kl.item(),
                    'dis':       L_dis.item(),
                    'perc':      L_perc.item(),
                    'id_cons':   id_cons,
                    **{f'lpips_{k.lower()}': v for k, v in scores.items()},
                }
                log.append(row)
                tqdm.write(
                    f"[{tag}] step={step}  recon={L_recon.item():.4f} "
                    f"kl={L_kl.item():.4f} dis={L_dis.item():.3f} "
                    f"Q1={scores['Q1']:.3f} Q4={scores['Q4']:.3f} "
                    f"id_cons={id_cons:.4f}"
                )

            # ── latent circle snapshot ────────────────────────────────────────
            if step % log_latent_every == 0:
                z_rots = log_z_rot_circle(model, device)
                cs     = circle_score(z_rots)
                latent_snapshots.append({'step': step, 'z_rots': z_rots,
                                         'circle_score': cs})
                tqdm.write(f"  [latent] step={step} circle_score={cs:.3f}")

    pbar.close()

    if save_ckpt:
        ckpt = CKPT_DIR / f'{tag}.pt'
        torch.save(model.state_dict(), ckpt)
        print(f"Saved checkpoint: {ckpt}")

    return model, pd.DataFrame(log), latent_snapshots


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_vae_dynamics(log: pd.DataFrame, tag: str):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    steps = log['step'] / 1000

    ax = axes[0]
    ax.plot(steps, log['recon'], color='#555', lw=1.2, label='recon (MSE)')
    ax.plot(steps, log['perc'],  color='#888', lw=1.0, ls='--', label='perceptual')
    ax.set_xlabel('Steps (×10³)'); ax.set_ylabel('Loss'); ax.legend(fontsize=7)
    ax.set_title('Reconstruction', fontsize=9)

    ax = axes[1]
    ax.plot(steps, log['lpips_q1'], color='#2166AC', lw=1.5, label='Q1 (simple)')
    ax.plot(steps, log['lpips_q4'], color='#D6604D', lw=1.5, ls='--', label='Q4 (complex)')
    ax.set_xlabel('Steps (×10³)'); ax.set_ylabel('Test LPIPS ↓')
    ax.legend(fontsize=7); ax.set_title('LPIPS by quartile', fontsize=9)

    ax = axes[2]
    ax.plot(steps, log['id_cons'], color='#4DAC26', lw=1.5)
    ax.set_xlabel('Steps (×10³)'); ax.set_ylabel('z_id std across views ↓')
    ax.set_title('Identity disentanglement', fontsize=9)

    for a in axes:
        a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    plt.suptitle(f'VAE training dynamics ({tag})', fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / f'vae_dynamics_{tag}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out}")


def _plot_vae_exp2(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.yaxis.grid(True, color='#EEEEEE'); ax.set_axisbelow(True)
    for col, label, color, marker in [
        ('lpips_simple',  'Simple (Q1)', '#2166AC', 'o'),
        ('lpips_complex', 'Complex (Q4)', '#D6604D', 's'),
    ]:
        grp = df.groupby('N')[col]
        m, s = grp.mean(), grp.std().fillna(0)
        ax.fill_between(m.index, m - s, m + s, alpha=0.15, color=color)
        ax.plot(m.index, m, f'{marker}-', color=color, label=label)
    ax.set_xlabel('Training objects (N)'); ax.set_ylabel('Test LPIPS ↓')
    ax.set_xticks([10, 20, 40, 60, 80]); ax.legend(framealpha=0.9)
    ax.set_title('VAE: data scale × complexity', fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'vae_exp2.png', dpi=180, bbox_inches='tight')
    plt.close()
    print("vae_exp2.png saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment runners
# ═══════════════════════════════════════════════════════════════════════════════

def run_vae_exp2(device: torch.device):
    """Data scale × complexity: N ∈ {10,40,80}, 3 seeds, 15k steps."""
    print("\n=== VAE Experiment 2: Data Scale × Complexity ===")
    _, test_ids = get_train_test_split(80, seed=42)
    Ns    = [10, 40, 80]
    seeds = [42, 7, 123]
    rows  = []
    csv_path = RES_DIR / 'vae_exp2_data_scale.csv'
    done = set()
    if csv_path.exists():
        df_done = pd.read_csv(csv_path)
        done = set(zip(df_done['N'], df_done['seed']))

    for N in Ns:
        for seed in seeds:
            if (N, seed) in done:
                continue
            torch.manual_seed(seed)
            train_ids, _ = get_train_test_split(N, seed=seed)
            tag = f'vae_exp2_N{N}_s{seed}'
            _, log, _ = train_vae(train_ids, test_ids, total_steps=15000,
                                  device=device, tag=tag, eval_every=5000,
                                  log_latent_every=15000)
            if len(log):
                last = log.iloc[-1]
                rows.append({'N': N, 'seed': seed,
                             'lpips_simple':  last['lpips_q1'],
                             'lpips_complex': last['lpips_q4'],
                             'id_cons':       last['id_cons']})
            if rows:
                existing = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
                pd.concat([existing, pd.DataFrame(rows[-1:])],
                          ignore_index=True).to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\nVAE Exp2:\n{df.groupby('N')[['lpips_simple','lpips_complex']].mean().round(3)}")
    _plot_vae_exp2(df)
    return df


def run_vae_exp3(device: torch.device):
    """
    Grokking dynamics: N=40, 30k steps, eval + latent logging every 1k steps.
    Key output: vae_exp3_dynamics.csv, vae_latent_circles.png
    """
    print("\n=== VAE Experiment 3: Grokking Dynamics + Latent Circles ===")
    train_ids, test_ids = get_train_test_split(40, seed=42)
    model, log, snaps = train_vae(
        train_ids, test_ids,
        total_steps=30000, device=device,
        tag='vae_exp3', eval_every=1000,
        log_latent_every=3000, save_ckpt=True,
    )
    log.to_csv(RES_DIR / 'vae_exp3_dynamics.csv', index=False)
    _plot_vae_dynamics(log, 'exp3')

    if snaps:
        plot_z_rot_evolution(snaps, FIG_DIR / 'vae_latent_circles.png')
        # Save circle scores to CSV for paper figure
        cs_df = pd.DataFrame([
            {'step': s['step'], 'circle_score': s['circle_score']} for s in snaps
        ])
        cs_df.to_csv(RES_DIR / 'vae_circle_scores.csv', index=False)

    print(f"\nFinal  Q1={log['lpips_q1'].iloc[-1]:.3f}  "
          f"Q4={log['lpips_q4'].iloc[-1]:.3f}  "
          f"id_cons={log['id_cons'].iloc[-1]:.4f}")
    return model, log, snaps


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--exp', type=int, default=3,
                        help='2=data scale, 3=grokking+latents')
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if args.exp in (0, 2): run_vae_exp2(device)
    if args.exp in (0, 3): run_vae_exp3(device)
