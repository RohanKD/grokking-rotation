"""
Single-file experiment runner.
Runs all 4 experiments and saves results + updated figures to ../results/ and ../figures/.

Usage:
    python run_experiments.py --device cuda:0   # run all on GPU 0
    python run_experiments.py --exp 3 --device cuda:1  # run only exp3 on GPU1

Experiments 2 and 4 sweep N; Exp 3 is a single long run for grokking dynamics.
All results are saved as CSVs; figures are regenerated in ../figures/.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from data.coil100 import COIL100Dataset, get_train_test_split, build_dataloader

# ─── paths ───────────────────────────────────────────────────────────────────
ROOT      = Path('/home/rohan/perspective')
COIL_DIR  = ROOT / 'coil-100' / 'coil-100'
RES_DIR   = ROOT / 'results'
FIG_DIR   = ROOT / 'figures'
CKPT_DIR  = ROOT / 'checkpoints'
for d in [RES_DIR, FIG_DIR, CKPT_DIR]: d.mkdir(exist_ok=True)

# ─── hyperparameters ─────────────────────────────────────────────────────────
T          = 1000
BASE_CH    = 32     # reduced from 64: 4.4M params, ~8× faster on Pascal GPU
LR         = 2e-4
BATCH      = 64
EVAL_EVERY = 1000   # steps between held-out evaluations
IMG_H = IMG_W = 64  # reduced from 128: same aspect ratio, 4× less spatial compute
DELTA_90   = 18     # angle_idx for 90° (18 × 5 = 90)


# ═══════════════════════════════════════════════════════════════════════════════
# Minimal conditional U-Net (fast, ~6M params)
# ═══════════════════════════════════════════════════════════════════════════════

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freq[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch * 2)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()
    def forward(self, x, emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        scale, shift = self.emb_proj(self.act(emb)).chunk(2, dim=-1)
        h = self.norm2(h) * (1 + scale[..., None, None]) + shift[..., None, None]
        h = self.act(h)
        h = self.conv2(h)
        return h + self.skip(x)

class Attention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv  = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).reshape(B, 3, C, H*W).unbind(1)
        scale = C ** -0.5
        attn = torch.softmax((q.transpose(-1,-2) @ k) * scale, dim=-1)
        out = (attn @ v.transpose(-1,-2)).transpose(-1,-2).reshape(B,C,H,W)
        return x + self.proj(out)

class ConditionalUNet(nn.Module):
    """3-level conditional U-Net. Input: 128×128, 3 downsamples → 16×16 bottleneck."""
    def __init__(self, base_ch=BASE_CH, time_emb_dim=256):
        super().__init__()
        c0, c1, c2, c3 = base_ch, base_ch*2, base_ch*4, base_ch*8
        self.time_emb = nn.Sequential(
            SinusoidalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.rot_proj = nn.Sequential(
            nn.Linear(2, time_emb_dim), nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        # 6-channel input (noisy target + source image)
        self.conv_in = nn.Conv2d(6, c0, 3, padding=1)
        # Encoder: 128→64→32→16
        self.e0 = ResBlock(c0, c0, time_emb_dim)  # out: (c0, 128)
        self.d0 = nn.Conv2d(c0, c0, 3, stride=2, padding=1)
        self.e1 = ResBlock(c0, c1, time_emb_dim)  # out: (c1, 64)
        self.d1 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)
        self.e2 = ResBlock(c1, c2, time_emb_dim)  # out: (c2, 32)
        self.d2 = nn.Conv2d(c2, c2, 3, stride=2, padding=1)
        # Bottleneck at 16×16
        self.mid1 = ResBlock(c2, c3, time_emb_dim)
        self.attn  = Attention(c3)
        self.mid2  = ResBlock(c3, c3, time_emb_dim)
        # Decoder: 16→32→64→128
        self.u2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.r2 = ResBlock(c2+c2, c2, time_emb_dim)  # cat skip e2
        self.u1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.r1 = ResBlock(c1+c1, c1, time_emb_dim)  # cat skip e1
        self.u0 = nn.ConvTranspose2d(c1, c0, 2, stride=2)
        self.r0 = ResBlock(c0+c0, c0, time_emb_dim)  # cat skip e0
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, c0), nn.SiLU(), nn.Conv2d(c0, 3, 3, padding=1),
        )

    def forward(self, x_noisy, x_src, delta_theta, t):
        rot = torch.stack([delta_theta.sin(), delta_theta.cos()], dim=-1).float()
        emb = self.time_emb(t) + self.rot_proj(rot)
        x = self.conv_in(torch.cat([x_noisy, x_src], dim=1))
        s0 = self.e0(x,  emb);  x = self.d0(s0)   # 128→64
        s1 = self.e1(x,  emb);  x = self.d1(s1)   # 64→32
        s2 = self.e2(x,  emb);  x = self.d2(s2)   # 32→16
        x = self.mid1(x, emb)
        x = self.attn(x)
        x = self.mid2(x, emb)
        x = self.r2(torch.cat([self.u2(x), s2], 1), emb)  # 16→32
        x = self.r1(torch.cat([self.u1(x), s1], 1), emb)  # 32→64
        x = self.r0(torch.cat([self.u0(x), s0], 1), emb)  # 64→128
        return self.conv_out(x)


# ═══════════════════════════════════════════════════════════════════════════════
# DDPM utilities
# ═══════════════════════════════════════════════════════════════════════════════

def cosine_schedule(T):
    steps = torch.arange(T + 1)
    alpha_bar = torch.cos(((steps / T + 0.008) / 1.008) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    betas = betas.clamp(0, 0.999)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars

BETAS, ALPHAS, ALPHA_BARS = cosine_schedule(T)

def q_sample(x0, t, noise, device):
    ab = ALPHA_BARS.to(device)[t][:, None, None, None]
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

def p_losses(model, x0, x_src, delta_theta, device):
    B = x0.shape[0]
    t = torch.randint(0, T, (B,), device=device)
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise, device)
    noise_pred = model(x_t, x_src, delta_theta.float(), t)
    return F.mse_loss(noise_pred, noise)


# ═══════════════════════════════════════════════════════════════════════════════
# Perceptual eval (VGG cosine distance, same metric as complexity computation)
# ═══════════════════════════════════════════════════════════════════════════════

import torchvision.models as tvm
import torchvision.transforms.functional as TF

_vgg = None
_vgg_norm_mean = torch.tensor([0.485, 0.456, 0.406])
_vgg_norm_std  = torch.tensor([0.229, 0.224, 0.225])

def get_vgg(device):
    global _vgg
    if _vgg is None:
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT)
        _vgg = nn.Sequential(*list(vgg.features.children())[:18]).to(device).eval()
        for p in _vgg.parameters():
            p.requires_grad_(False)
    return _vgg

@torch.no_grad()
def perceptual_dist(pred, target, device):
    """Mean VGG cosine distance for a batch. pred,target: (B,3,H,W) in [0,1]."""
    vgg = get_vgg(device)
    mean = _vgg_norm_mean.to(device)[None,:,None,None]
    std  = _vgg_norm_std.to(device)[None,:,None,None]
    def feat(x):
        x = (x - mean) / std
        f = vgg(x).flatten(1)
        return F.normalize(f, dim=1)
    f_p = feat(pred); f_t = feat(target)
    return (1 - (f_p * f_t).sum(dim=1)).mean().item()

@torch.no_grad()
def ddim_sample(model, x_src, delta_theta, device, n_steps=20):
    """Fast DDIM inference."""
    B = x_src.shape[0]
    x = torch.randn(B, 3, IMG_H, IMG_W, device=device)
    ts = torch.linspace(T - 1, 0, n_steps + 1).long().tolist()
    for i in range(n_steps):
        t_now  = torch.full((B,), ts[i],   device=device, dtype=torch.long)
        t_next = torch.full((B,), ts[i+1], device=device, dtype=torch.long)
        ab_now  = ALPHA_BARS.to(device)[t_now[:1]]
        ab_next = ALPHA_BARS.to(device)[t_next[:1]]
        eps = model(x, x_src, delta_theta.float(), t_now)
        x0_pred = (x - (1 - ab_now).sqrt() * eps) / ab_now.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)
        x = ab_next.sqrt() * x0_pred + (1 - ab_next).sqrt() * eps
    return x.clamp(0, 1)

@torch.no_grad()
def evaluate(model, test_ids, device, n_samples=80, delta_idx=DELTA_90):
    """Evaluate on held-out objects. Returns dict split by complexity quartile."""
    model.eval()
    cx = pd.read_csv(RES_DIR / 'complexity_scores.csv')
    q1 = cx['complexity_feat'].quantile(0.25)
    q3 = cx['complexity_feat'].quantile(0.75)
    def qcat(obj_id):
        s = cx[cx['obj_id']==obj_id]['complexity_feat'].values
        if len(s) == 0: return 'Q2'
        v = s[0]
        if v <= q1: return 'Q1'
        if v <= cx['complexity_feat'].quantile(0.5): return 'Q2'
        if v <= q3: return 'Q3'
        return 'Q4'

    from torchvision import transforms as T_tfm
    img_tfm = T_tfm.Compose([T_tfm.Resize((IMG_H, IMG_W)), T_tfm.ToTensor()])
    ds = COIL100Dataset(str(COIL_DIR), test_ids, angle_delta=delta_idx,
                        transform=img_tfm, length=n_samples)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)
    results = {'Q1':[], 'Q2':[], 'Q3':[], 'Q4':[]}
    for batch in loader:
        src = batch['source'].to(device)
        tgt = batch['target'].to(device)
        dt  = batch['delta_theta'].to(device)
        pred = ddim_sample(model, src, dt, device)
        for i, oid in enumerate(batch['object_id'].tolist()):
            d = perceptual_dist(pred[i:i+1], tgt[i:i+1], device)
            results[qcat(oid)].append(d)
    model.train()
    return {q: float(np.mean(v)) if v else float('nan') for q, v in results.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════════

def train(train_ids, test_ids, total_steps, device, tag, eval_every=EVAL_EVERY,
          delta_idx=None, save_ckpt=False, shuffle_delta=False):
    from torchvision import transforms as T_tfm
    img_tfm = T_tfm.Compose([T_tfm.Resize((IMG_H, IMG_W)), T_tfm.ToTensor()])
    model = ConditionalUNet().to(device)
    opt   = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=total_steps, eta_min=LR*0.1)
    length = max(len(train_ids) * 72, BATCH * 4)
    from data.coil100 import COIL100Dataset
    from torch.utils.data import DataLoader
    ds = COIL100Dataset(str(COIL_DIR), train_ids, angle_delta=delta_idx,
                        transform=img_tfm, length=length)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True)

    log = []
    step = 0
    pbar = tqdm(total=total_steps, desc=tag, dynamic_ncols=True)
    train_losses = []

    while step < total_steps:
        for batch in loader:
            if step >= total_steps: break
            src = batch['source'].to(device)
            tgt = batch['target'].to(device)
            dt  = batch['delta_theta'].to(device)
            if shuffle_delta:
                dt = dt[torch.randperm(dt.shape[0], device=device)]
            loss = p_losses(model, tgt, src, dt, device)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            train_losses.append(loss.item())
            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if step % eval_every == 0:
                avg_loss = float(np.mean(train_losses[-eval_every:]))
                scores   = evaluate(model, test_ids, device)
                row = {'step': step, 'train_loss': avg_loss, **{f'lpips_{k.lower()}': v for k,v in scores.items()}}
                log.append(row)
                tqdm.write(f"[{tag}] step={step} loss={avg_loss:.4f} "
                           f"Q1={scores['Q1']:.3f} Q4={scores['Q4']:.3f}")

    pbar.close()
    if save_ckpt:
        ckpt = CKPT_DIR / f'{tag}.pt'
        torch.save(model.state_dict(), ckpt)
        print(f"Saved checkpoint: {ckpt}")
    return model, pd.DataFrame(log)


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment runners
# ═══════════════════════════════════════════════════════════════════════════════

def run_exp2(device):
    """Data scale × complexity: sweep N ∈ {10,40,80}, 3 seeds."""
    print("\n=== Experiment 2: Data Scale × Complexity ===")
    _, test_ids = get_train_test_split(80, seed=42)
    Ns = [10, 40, 80]
    seeds = [42, 7, 123]
    rows = []
    csv_path = RES_DIR / 'exp2_data_scale_real.csv'
    # resume if interrupted
    done = set()
    if csv_path.exists():
        df_done = pd.read_csv(csv_path)
        done = set(zip(df_done['N'], df_done['seed']))

    for N in Ns:
        for seed in seeds:
            if (N, seed) in done: continue
            train_ids, _ = get_train_test_split(N, seed=seed)
            tag = f'exp2_N{N}_s{seed}'
            _, log = train(train_ids, test_ids, total_steps=15000, device=device,
                           tag=tag, eval_every=5000)
            if len(log) > 0:
                last = log.iloc[-1]
                rows.append({'N': N, 'seed': seed,
                             'lpips_simple': last['lpips_q1'],
                             'lpips_complex': last['lpips_q4'],
                             'train_loss': last['train_loss']})
            # save incrementally
            if rows:
                existing = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
                pd.concat([existing, pd.DataFrame(rows[-1:])], ignore_index=True).to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\nExp2 results:\n{df.groupby('N')[['lpips_simple','lpips_complex']].mean().round(3)}")
    _plot_exp2(df)
    return df

def _plot_exp2(df):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.yaxis.grid(True, color='#EEEEEE'); ax.set_axisbelow(True)
    for col, label, color, marker in [
        ('lpips_simple',  'Simple (Q1)', '#2166AC', 'o'),
        ('lpips_complex', 'Complex (Q4)', '#D6604D', 's'),
    ]:
        grp = df.groupby('N')[col]
        m, s = grp.mean(), grp.std().fillna(0)
        ax.fill_between(m.index, m-s, m+s, alpha=0.15, color=color)
        ax.plot(m.index, m, f'{marker}-', color=color, label=label)
    ax.set_xlabel('Training objects (N)'); ax.set_ylabel('Test perceptual dist. ↓')
    ax.set_xticks([10,20,40,60,80]); ax.legend(framealpha=0.9)
    ax.set_title('Data scale × complexity', fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig3.png', dpi=180, bbox_inches='tight')
    plt.close()
    print("fig3.png updated with real results.")


def run_exp3(device):
    """Grokking dynamics: N=40, 30k steps, log every 1k."""
    print("\n=== Experiment 3: Grokking Dynamics ===")
    train_ids, test_ids = get_train_test_split(40, seed=42)
    _, log = train(train_ids, test_ids, total_steps=30000, device=device,
                   tag='exp3_grokking', eval_every=1000, save_ckpt=True)
    log.to_csv(RES_DIR / 'exp3_grokking_real.csv', index=False)
    _plot_exp3(log)
    return log

def _plot_exp3(log):
    fig, ax1 = plt.subplots(figsize=(4.5, 3))
    ax2 = ax1.twinx()
    steps = log['step'] / 1000
    ax1.plot(steps, log['train_loss'], color='#888', lw=1.2, label='Train loss')
    ax1.set_ylabel('Training loss (MSE)', color='#666')
    ax1.tick_params(axis='y', labelcolor='#666')
    ax1.set_ylim(bottom=0)
    ax2.plot(steps, log['lpips_q1'], color='#2166AC', lw=1.5, label='Q1 (simple)')
    ax2.plot(steps, log['lpips_q4'], color='#D6604D', lw=1.5, ls='--', label='Q4 (complex)')
    ax2.set_ylabel('Test perceptual dist. ↓')
    ax1.set_xlabel('Training steps (×10³)')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, fontsize=8, framealpha=0.9)
    ax1.set_title('Grokking dynamics', fontsize=9)
    ax1.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig4.png', dpi=180, bbox_inches='tight')
    plt.close()
    print("fig4.png updated with real results.")


def run_exp5_ablation(device):
    """Δθ ablation: compare normal vs shuffled rotation signal at N=40."""
    print("\n=== Experiment 5: Δθ Ablation ===")
    train_ids, test_ids = get_train_test_split(40, seed=42)
    rows = []
    for condition, shuffle in [('normal', False), ('shuffled', True)]:
        tag = f'exp5_{condition}'
        print(f"\n  Training with Δθ {condition} ...")
        model, _ = train(train_ids, test_ids, total_steps=15000, device=device,
                         tag=tag, eval_every=15000, shuffle_delta=shuffle)
        scores = evaluate(model, test_ids, device)
        rows.append({'condition': condition, 'lpips_q1': scores['Q1'], 'lpips_q4': scores['Q4']})
        tqdm.write(f"  [{condition}] Q1={scores['Q1']:.3f}  Q4={scores['Q4']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(RES_DIR / 'exp5_ablation_real.csv', index=False)
    print("\nExp5 results:")
    print(df.to_string(index=False))
    return df


def run_exp4(device):
    """Angle generalization: train on 90° only, test interpolation + extrapolation."""
    print("\n=== Experiment 4: Angle Generalization ===")
    _, test_ids = get_train_test_split(80, seed=42)
    Ns = [10, 40, 80]
    interp = [9, 12, 15]   # 45°, 60°, 75° in angle_idx
    extrap = [24, 27, 36]  # 120°, 135°, 180°
    rows = []
    csv_path = RES_DIR / 'exp4_angle_gen_real.csv'

    for N in Ns:
        train_ids, _ = get_train_test_split(N, seed=42)
        tag = f'exp4_N{N}'
        model, _ = train(train_ids, test_ids, total_steps=15000, device=device,
                         tag=tag, eval_every=15000, delta_idx=DELTA_90)
        for ang_idx in interp + extrap:
            ang_deg = ang_idx * 5
            kind    = 'interp' if ang_idx in interp else 'extrap'
            scores  = evaluate(model, test_ids, device, delta_idx=ang_idx)
            for q in ['Q1','Q4']:
                rows.append({'N': N, 'angle': ang_deg, 'type': kind,
                             'complexity': 'simple' if q=='Q1' else 'complex',
                             'lpips': scores[q]})

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    _plot_exp4(df)
    return df

def _plot_exp4(df):
    interp_angs = [45, 60, 75]
    extrap_angs = [120, 135, 180]
    Ns = sorted(df['N'].unique())
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('rg', ['#2CA02C','#FFDD55','#D62728'], N=256)
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.0), sharey='row')
    vmin = df['lpips'].min(); vmax = df['lpips'].max()
    pairs = [('simple','interp',interp_angs), ('simple','extrap',extrap_angs),
             ('complex','interp',interp_angs), ('complex','extrap',extrap_angs)]
    titles = ['Simple / Interpolation','Simple / Extrapolation',
              'Complex / Interpolation','Complex / Extrapolation']
    for idx, ((comp, kind, angs), title) in enumerate(zip(pairs, titles)):
        ax = axes[idx//2, idx%2]
        grid = np.zeros((len(angs), len(Ns)))
        for i, ang in enumerate(angs):
            for j, n in enumerate(Ns):
                sub = df[(df['complexity']==comp)&(df['type']==kind)&
                         (df['angle']==ang)&(df['N']==n)]['lpips']
                grid[i,j] = sub.values[0] if len(sub) else float('nan')
        im = ax.imshow(grid, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
        ax.set_xticks(range(len(Ns))); ax.set_xticklabels(Ns, fontsize=7)
        ax.set_yticks(range(len(angs))); ax.set_yticklabels([f'{a}°' for a in angs], fontsize=7)
        ax.set_title(title, fontsize=8)
        if idx//2 == 1: ax.set_xlabel('N', fontsize=8)
        if idx%2 == 0: ax.set_ylabel('Test angle', fontsize=8)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not np.isnan(grid[i,j]):
                    ax.text(j, i, f'{grid[i,j]:.2f}', ha='center', va='center',
                            fontsize=6.5, color='white' if grid[i,j]>0.4 else 'black')
    fig.colorbar(im, ax=axes, fraction=0.018, pad=0.02).set_label('Perceptual dist. ↓', fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig5.png', dpi=180, bbox_inches='tight')
    plt.close()
    print("fig5.png updated with real results.")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--exp', type=int, default=0, help='0=all, 2/3/4/5=specific')
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    if args.exp in (0, 2): run_exp2(device)
    if args.exp in (0, 3): run_exp3(device)
    if args.exp in (0, 4): run_exp4(device)
    if args.exp in (0, 5): run_exp5_ablation(device)

    print("\nAll done. Figures saved to", FIG_DIR)
