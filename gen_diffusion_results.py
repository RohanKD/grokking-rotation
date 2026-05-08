"""
Visualize actual DDPM-generated outputs from the trained Exp3 checkpoint.

Shows what the generative model produces vs. ground truth, for both
simple (Q1) and complex (Q4) objects.

Layout (3 rows × 6 cols):
  Row 1 — Simple object (Q1)
  Row 2 — Complex object (Q4)
  Row 3 — Denoising trajectory for the Q4 object (x_T → x_0)

Columns: source | GT target | generated | |GT−gen|×4 | copy-src error | NN error
"""
import os, sys, re, glob, random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, '/home/rohan/perspective/code')
from run_experiments import (ConditionalUNet, ddim_sample,
                              COIL_DIR, ALPHA_BARS, T as DIFF_T)

CKPT     = '/home/rohan/perspective/checkpoints/exp3_grokking.pt'
RES_DIR  = '/home/rohan/perspective/results'
FIG_DIR  = '/home/rohan/perspective/figures'
DELTA    = 90
IMG_SIZE = 64   # model was trained at 64×64

to_tensor  = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])
to_tensor128 = T.Compose([T.Resize((128, 128)), T.ToTensor()])
normalize  = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

# ── load model ────────────────────────────────────────────────────────────────
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = ConditionalUNet().to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()
print(f"Loaded checkpoint: {CKPT}  ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

# ── helpers ───────────────────────────────────────────────────────────────────
def load(obj_id, angle_deg, size=IMG_SIZE):
    tfm = T.Compose([T.Resize((size, size)), T.ToTensor()])
    return tfm(Image.open(f'{COIL_DIR}/obj{obj_id}__{angle_deg}.png').convert('RGB'))

def angles_of(obj_id):
    files = glob.glob(f'{COIL_DIR}/obj{obj_id}__*.png')
    return sorted(int(re.search(r'__(\d+)\.png', f).group(1)) for f in files)

def t2np(t):
    return t.permute(1,2,0).clamp(0,1).cpu().numpy()

def err_map(a, b):
    return ((a - b).abs() * 4).clamp(0,1)

# ── VGG for NN retrieval ──────────────────────────────────────────────────────
vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16].eval()

@torch.no_grad()
def feat(img):
    return F.normalize(vgg(normalize(img).unsqueeze(0)).flatten(1), dim=1)

# ── train/test split ──────────────────────────────────────────────────────────
random.seed(42)
all_ids = list(range(1,101)); random.shuffle(all_ids)
train_ids = sorted(all_ids[:80]); test_ids = sorted(all_ids[80:])

cx = pd.read_csv(f'{RES_DIR}/complexity_scores.csv')
q1_cut = cx['complexity_feat'].quantile(0.25)
q4_cut = cx['complexity_feat'].quantile(0.75)
q1_test = [o for o in cx[cx['complexity_feat'] <= q1_cut]['obj_id'] if o in test_ids]
q4_test = [o for o in cx[cx['complexity_feat'] >= q4_cut]['obj_id'] if o in test_ids]

# ── cache NN features ─────────────────────────────────────────────────────────
print("Caching NN features …")
cache = {(o,a): feat(load(o, a, 128)) for o in train_ids for a in angles_of(o)}
print(f"  {len(cache)} views cached.")

def nn_retrieve(src_img_128):
    qf = feat(src_img_128)
    best_d, bk = float('inf'), None
    for k, f in cache.items():
        d = float(1 - (qf * f).sum())
        if d < best_d: best_d, bk = d, k
    nn_obj, nn_ang = bk
    tgt_deg = (nn_ang + DELTA) % 360
    nn_tgt  = min(angles_of(nn_obj), key=lambda a: abs(a - tgt_deg))
    return load(nn_obj, nn_tgt)   # returned at IMG_SIZE

# ── DDIM with trajectory capture ─────────────────────────────────────────────
@torch.no_grad()
def ddim_with_traj(src, delta_theta, n_steps=20, capture_at=(19,15,10,5,0)):
    """Run DDIM and return (final_img, list of (step_idx, img) snapshots)."""
    B = 1
    x = torch.randn(B, 3, IMG_SIZE, IMG_SIZE, device=device)
    dt = torch.tensor([delta_theta], device=device)
    ts = torch.linspace(DIFF_T - 1, 0, n_steps + 1).long().tolist()
    traj = []
    for i in range(n_steps):
        t_now  = torch.full((B,), ts[i],   device=device, dtype=torch.long)
        t_next = torch.full((B,), ts[i+1], device=device, dtype=torch.long)
        ab_now  = ALPHA_BARS.to(device)[t_now[:1]]
        ab_next = ALPHA_BARS.to(device)[t_next[:1]]
        eps = model(x, src, dt.float(), t_now)
        x0_pred = (x - (1 - ab_now).sqrt() * eps) / ab_now.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)
        x = ab_next.sqrt() * x0_pred + (1 - ab_next).sqrt() * eps
        if i in capture_at:
            traj.append((i, x.clamp(0,1).squeeze(0).cpu()))
    return x.clamp(0,1).squeeze(0).cpu(), traj

# ── pick examples ─────────────────────────────────────────────────────────────
# Q1: obj70 @ 0° — copy-src wins (symmetric object)
# Q4: obj98 @ 25° — NN wins (complex object); does generative do even better?
examples = [
    ('Simple object  (Q1 — symmetric)', 70, 0),
    ('Complex object  (Q4 — high-variance)', 98, 25),
]

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: main comparison grid
# ═══════════════════════════════════════════════════════════════════════════════
col_labels = ['Source $x_s$', 'GT target\n$x_t$ (+90°)',
              'DDPM generated\n$\\hat{x}_t$', 'Error  $|x_t-\\hat{x}_t|\\times4$',
              'Copy-src error\n$|x_t-x_s|\\times4$',
              'NN error\n$|x_t-x_{\\mathrm{NN}}|\\times4$']

fig, axes = plt.subplots(2, 6, figsize=(13, 4.8),
                         gridspec_kw={'hspace': 0.12, 'wspace': 0.04})

for row, (label, obj_id, src_ang) in enumerate(examples):
    angs    = angles_of(obj_id)
    tgt_ang = min(angs, key=lambda a: abs(a - (src_ang + DELTA) % 360))

    src_img = load(obj_id, src_ang)           # (3,64,64)
    tgt_img = load(obj_id, tgt_ang)
    src128  = load(obj_id, src_ang,  128)
    tgt128  = load(obj_id, tgt_ang,  128)

    # DDPM generation
    delta_rad = np.deg2rad(DELTA)
    print(f"Generating for {label} …")
    with torch.no_grad():
        src_dev = src_img.unsqueeze(0).to(device)
        dt_dev  = torch.tensor([delta_rad], device=device)
        gen_img, traj = ddim_with_traj(src_dev, delta_rad)

    # NN retrieval (at 64px for error comparison)
    nn_img = nn_retrieve(src128)

    # errors
    gen_err = err_map(tgt_img, gen_img)
    cs_err  = err_map(tgt_img, src_img)
    nn_err  = err_map(tgt_img, nn_img)

    # perceptual distances (VGG, at 128px)
    gen128 = T.Resize((128,128))(gen_img)
    nn128  = T.Resize((128,128))(nn_img)
    d_gen  = float(1 - (feat(gen128) * feat(tgt128)).sum())
    d_cs   = float(1 - (feat(src128) * feat(tgt128)).sum())
    d_nn   = float(1 - (feat(nn128)  * feat(tgt128)).sum())
    print(f"  Perceptual dist:  DDPM={d_gen:.3f}  copy-src={d_cs:.3f}  NN={d_nn:.3f}")

    imgs = [src_img, tgt_img, gen_img, gen_err, cs_err, nn_err]
    for col, (ax, img) in enumerate(zip(axes[row], imgs)):
        ax.imshow(t2np(img), interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        if row == 0:
            ax.set_title(col_labels[col], fontsize=8, pad=5)

    axes[row, 0].set_ylabel(label, fontsize=9, fontweight='bold', labelpad=6)
    axes[row, 5].set_xlabel(
        f'DDPM={d_gen:.3f}  copy={d_cs:.3f}  NN={d_nn:.3f}',
        fontsize=7.5, labelpad=4)

fig.suptitle('DDPM Generative Results vs.\ Baselines  (Exp3 checkpoint, N=40, 30k steps)',
             fontsize=10, y=1.02)
out1 = f'{FIG_DIR}/gen_results.png'
plt.savefig(out1, dpi=200, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out1}")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: denoising trajectory for Q4 object
# ═══════════════════════════════════════════════════════════════════════════════
obj_id, src_ang = 98, 25
angs    = angles_of(obj_id)
tgt_ang = min(angs, key=lambda a: abs(a - (src_ang + DELTA) % 360))
src_img = load(obj_id, src_ang)
tgt_img = load(obj_id, tgt_ang)

print("\nGenerating denoising trajectory …")
with torch.no_grad():
    src_dev = src_img.unsqueeze(0).to(device)
    final, traj = ddim_with_traj(src_dev, np.deg2rad(DELTA),
                                  n_steps=20, capture_at=[0,3,7,11,15,19])

traj_imgs = [img for _, img in sorted(traj, key=lambda x: -x[0])]  # noise→clean
traj_imgs.append(final)   # clean output
step_labels = [f'step {20-i}/20' for i in range(len(traj_imgs)-1)] + ['final']

n_cols = len(traj_imgs) + 2
fig2, axes2 = plt.subplots(1, n_cols, figsize=(n_cols * 1.7, 2.4),
                            gridspec_kw={'wspace': 0.06})

axes2[0].imshow(t2np(src_img)); axes2[0].set_title('Source\n$x_s$', fontsize=8)
for sp in axes2[0].spines.values(): sp.set_visible(False)
axes2[0].set_xticks([]); axes2[0].set_yticks([])

for i, (img, lbl) in enumerate(zip(traj_imgs, step_labels)):
    ax = axes2[i+1]
    ax.imshow(t2np(img), interpolation='nearest')
    ax.set_title(lbl, fontsize=7.5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    # arrow between panels
    if i < len(traj_imgs) - 1:
        ax.annotate('', xy=(1.12, 0.5), xycoords='axes fraction',
                    xytext=(1.0, 0.5), textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='#888', lw=1.2))

axes2[-1].imshow(t2np(tgt_img)); axes2[-1].set_title('GT target\n$x_t$', fontsize=8)
for sp in axes2[-1].spines.values(): sp.set_visible(False)
axes2[-1].set_xticks([]); axes2[-1].set_yticks([])
# highlight GT with border
for sp in axes2[-1].spines.values():
    sp.set_visible(True); sp.set_linewidth(1.8); sp.set_color('#2166AC')

fig2.suptitle(f'Denoising trajectory — complex object (Q4, obj{obj_id}, +90°)',
              fontsize=9.5, y=1.06)

out2 = f'{FIG_DIR}/gen_trajectory.png'
plt.savefig(out2, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out2}")
