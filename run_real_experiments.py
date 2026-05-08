"""
Real experiments on COIL-100.

What this runs (all feasible on CPU in ~5 minutes):
  1. Compute view-variance complexity C(o; 90°) for all 100 objects using VGG features
  2. Rank objects into quartiles; save representative examples per quartile
  3. Copy-source baseline: what LPIPS/MSE do we get if we just copy x_s → x_t?
     This proves that symmetric objects are trivially solved.
  4. Nearest-neighbour retrieval baseline: for a test object, find the
     training object with the most similar source view, then return its
     target view. Measures shortcut-vs-understanding without any training.
  5. Save all numbers to results/ CSVs for paper inclusion.
"""

import os, re, glob, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product

# ── paths ────────────────────────────────────────────────────────────────────
COIL_DIR  = '/home/rohan/perspective/coil-100/coil-100'
RES_DIR   = '/home/rohan/perspective/results'
FIG_DIR   = '/home/rohan/perspective/figures'
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

device = torch.device('cpu')
torch.manual_seed(42)
np.random.seed(42)

# ── image utilities ───────────────────────────────────────────────────────────
IMG_SIZE = 128
to_tensor = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])

def load_img(obj_id: int, angle_deg: int) -> torch.Tensor:
    """Load COIL-100 image as (3,H,W) tensor in [0,1]."""
    path = os.path.join(COIL_DIR, f'obj{obj_id}__{angle_deg}.png')
    img = Image.open(path).convert('RGB')
    return to_tensor(img)

def list_angles(obj_id: int):
    """Return sorted list of available angles for an object."""
    files = glob.glob(os.path.join(COIL_DIR, f'obj{obj_id}__*.png'))
    angles = sorted(int(re.search(r'__(\d+)\.png', f).group(1)) for f in files)
    return angles

# verify dataset
all_angles = list_angles(1)
print(f"COIL-100 loaded. Object 1 has {len(all_angles)} views: {all_angles[:5]}...")
all_obj_ids = sorted(set(
    int(re.search(r'obj(\d+)__', f).group(1))
    for f in glob.glob(os.path.join(COIL_DIR, 'obj*.png'))
))
print(f"Total objects: {len(all_obj_ids)}")

# ── feature extractor (frozen VGG16) ─────────────────────────────────────────
print("\nLoading VGG16 features...")
vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
feature_extractor = nn.Sequential(*list(vgg.features.children())[:18]).to(device)
feature_extractor.eval()
vgg_norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

@torch.no_grad()
def extract_features(img_tensor: torch.Tensor) -> torch.Tensor:
    """img_tensor: (3,H,W) in [0,1] → flattened feature vector."""
    x = vgg_norm(img_tensor).unsqueeze(0).to(device)
    feats = feature_extractor(x)
    return feats.flatten()

def feature_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    a_n = F.normalize(a.unsqueeze(0), dim=1)
    b_n = F.normalize(b.unsqueeze(0), dim=1)
    return (1 - (a_n * b_n).sum()).item()

def pixel_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.mse_loss(a, b).item()

# ── 1. Compute view variance C(o; 90°) ───────────────────────────────────────
print("\n[1/4] Computing view variance C(o; 90°) for all objects...")
t0 = time.time()
complexity_scores = {}

for obj_id in all_obj_ids:
    angles = list_angles(obj_id)
    angle_set = set(angles)
    pairs_mse = []
    pairs_feat = []
    for ang in angles:
        partner = (ang + 90) % 360
        # COIL-100 has angles in multiples of 5; find nearest partner
        if partner not in angle_set:
            partner = min(angle_set, key=lambda a: abs(a - partner))
        img_a = load_img(obj_id, ang)
        img_b = load_img(obj_id, partner)
        pairs_mse.append(pixel_mse(img_a, img_b))
        feat_a = extract_features(img_a)
        feat_b = extract_features(img_b)
        pairs_feat.append(feature_distance(feat_a, feat_b))
    complexity_scores[obj_id] = {
        'mse':  float(np.mean(pairs_mse)),
        'feat': float(np.mean(pairs_feat)),
    }

elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s")

df_cx = pd.DataFrame([
    {'obj_id': k, 'complexity_mse': v['mse'], 'complexity_feat': v['feat']}
    for k, v in complexity_scores.items()
]).sort_values('complexity_feat')
df_cx.to_csv(f'{RES_DIR}/complexity_scores.csv', index=False)

# quartile splits (by feature distance)
scores = df_cx['complexity_feat'].values
q1, q2, q3 = np.percentile(scores, [25, 50, 75])
def quartile(s):
    if s <= q1: return 'Q1'
    if s <= q2: return 'Q2'
    if s <= q3: return 'Q3'
    return 'Q4'

df_cx['quartile'] = df_cx['complexity_feat'].apply(quartile)
quartile_objs = {q: df_cx[df_cx['quartile']==q]['obj_id'].tolist() for q in ['Q1','Q2','Q3','Q4']}

print(f"\n  Complexity stats:")
print(f"  Q1 ≤ {q1:.4f} < Q2 ≤ {q2:.4f} < Q3 ≤ {q3:.4f} < Q4")
for q in ['Q1','Q2','Q3','Q4']:
    vals = df_cx[df_cx['quartile']==q]['complexity_feat']
    print(f"  {q}: n={len(vals)}  mean={vals.mean():.4f}  std={vals.std():.4f}")

# ── 2. Save complexity spectrum figure (real objects) ────────────────────────
print("\n[2/4] Saving real complexity spectrum figure...")
plt.rcParams.update({'font.family': 'serif', 'font.size': 9})

# pick 2 representative objects per extreme quartile
rep_simple  = quartile_objs['Q1'][:2]
rep_complex = quartile_objs['Q4'][-2:]
show_angles = [0, 90, 180, 270]

fig, axes = plt.subplots(4, 4, figsize=(7.0, 4.2))
for row, obj_id in enumerate(rep_simple + rep_complex):
    ang_list = list_angles(obj_id)
    cx_val   = complexity_scores[obj_id]['feat']
    label    = f"obj{obj_id}  C={cx_val:.3f}"
    q_label  = 'Low complexity' if row < 2 else 'High complexity'
    for col, ang_target in enumerate(show_angles):
        nearest = min(ang_list, key=lambda a: abs(a - ang_target))
        img = load_img(obj_id, nearest).permute(1, 2, 0).numpy()
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        if col == 0:
            axes[row, col].set_ylabel(label, fontsize=7, rotation=0,
                                      labelpad=60, va='center')
        if row == 0:
            axes[row, col].set_title(f'{ang_target}°', fontsize=8)

for row in range(4):
    q_str = 'Q1 (simple)' if row < 2 else 'Q4 (complex)'
    fig.text(0.01, 0.87 - row * 0.235, q_str,
             fontsize=8, style='italic', color='#444', va='center')

fig.suptitle('Figure 1. COIL-100 objects at 4 azimuths, spanning the complexity spectrum.',
             fontsize=8, y=0.01, style='italic')
plt.tight_layout(rect=[0.08, 0.05, 1, 1])
plt.savefig(f'{FIG_DIR}/fig1.png', dpi=180, bbox_inches='tight')
plt.close()
print("  fig1.png updated with real COIL-100 images.")

# ── 3. Copy-source baseline by quartile ──────────────────────────────────────
print("\n[3/4] Copy-source baseline by complexity quartile...")

# fixed train/test split (same as paper: 80 train, 20 test)
np.random.seed(42)
shuffled = np.random.permutation(all_obj_ids)
train_ids = set(shuffled[:80].tolist())
test_ids  = set(shuffled[80:].tolist())

test_df_cx = df_cx[df_cx['obj_id'].isin(test_ids)]
copy_src_results = []

for _, row in test_df_cx.iterrows():
    obj_id = int(row['obj_id'])
    q      = row['quartile']
    angles = list_angles(obj_id)
    # evaluate over 8 random (src, tgt) pairs with Δθ = 90°
    src_angles = np.random.choice(angles, size=8, replace=False)
    for src_ang in src_angles:
        tgt_ang = min(angles, key=lambda a: abs(a - ((src_ang + 90) % 360)))
        src_img = load_img(obj_id, src_ang)
        tgt_img = load_img(obj_id, tgt_ang)
        mse_val = pixel_mse(src_img, tgt_img)
        # feature-based perceptual distance
        feat_s  = extract_features(src_img)
        feat_t  = extract_features(tgt_img)
        perc    = feature_distance(feat_s, feat_t)
        copy_src_results.append({
            'obj_id': obj_id, 'quartile': q,
            'delta_theta': 90, 'mse': mse_val, 'perceptual': perc,
        })

df_cs = pd.DataFrame(copy_src_results)
df_cs.to_csv(f'{RES_DIR}/copy_source_baseline.csv', index=False)

print("\n  Copy-source baseline by quartile:")
summary_cs = df_cs.groupby('quartile')[['mse','perceptual']].agg(['mean','std'])
print(summary_cs.round(4))

# ── 4. Nearest-neighbour retrieval baseline ───────────────────────────────────
print("\n[4/4] Nearest-neighbour retrieval baseline...")
# Pre-compute features for all training object source views
print("  Caching training features...")
train_cache = {}  # (obj_id, angle) -> feature tensor
for obj_id in train_ids:
    for ang in list_angles(obj_id):
        train_cache[(obj_id, ang)] = extract_features(load_img(obj_id, ang))

def nn_retrieval(src_img, delta_theta, train_cache, train_ids):
    """Find closest training image; return its target-view image."""
    src_feat = extract_features(src_img)
    best_dist = float('inf')
    best_key  = None
    for (oid, ang), feat in train_cache.items():
        d = feature_distance(src_feat, feat)
        if d < best_dist:
            best_dist, best_key = d, (oid, ang)
    best_obj, best_ang = best_key
    target_ang = (best_ang + delta_theta) % 360
    t_angles   = list_angles(best_obj)
    nearest    = min(t_angles, key=lambda a: abs(a - target_ang))
    return load_img(best_obj, nearest)

nn_results = []
sample_test = test_df_cx.sample(min(20, len(test_df_cx)), random_state=42)  # subsample for speed

for _, row in sample_test.iterrows():
    obj_id = int(row['obj_id'])
    q      = row['quartile']
    angles = list_angles(obj_id)
    src_angles = np.random.choice(angles, size=4, replace=False)
    for src_ang in src_angles:
        tgt_ang  = min(angles, key=lambda a: abs(a - ((src_ang + 90) % 360)))
        src_img  = load_img(obj_id, src_ang)
        tgt_img  = load_img(obj_id, tgt_ang)
        nn_pred  = nn_retrieval(src_img, 90, train_cache, train_ids)
        cs_perc  = feature_distance(extract_features(src_img), extract_features(tgt_img))
        nn_perc  = feature_distance(extract_features(nn_pred), extract_features(tgt_img))
        nn_results.append({
            'obj_id': obj_id, 'quartile': q,
            'copy_src_perc': cs_perc, 'nn_perc': nn_perc,
        })

df_nn = pd.DataFrame(nn_results)
df_nn.to_csv(f'{RES_DIR}/nn_retrieval_baseline.csv', index=False)

print("\n  NN retrieval vs copy-source by quartile (perceptual distance ↓):")
for q in ['Q1','Q2','Q3','Q4']:
    sub = df_nn[df_nn['quartile'] == q]
    if len(sub) == 0: continue
    print(f"  {q}  copy-src: {sub['copy_src_perc'].mean():.4f} ± {sub['copy_src_perc'].std():.4f}"
          f"  |  nn: {sub['nn_perc'].mean():.4f} ± {sub['nn_perc'].std():.4f}"
          f"  |  nn improvement: {(sub['copy_src_perc'].mean() - sub['nn_perc'].mean()) / sub['copy_src_perc'].mean() * 100:.1f}%")

# ── Updated Figure 2: real box plots ─────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(4.0, 3.0))
q_vals = [df_cs[df_cs['quartile']==q]['perceptual'].values for q in ['Q1','Q2','Q3','Q4']]
bp = ax.boxplot(q_vals, positions=[1,2,3,4], widths=0.5, patch_artist=True,
                medianprops=dict(color='white', lw=1.5),
                whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8),
                flierprops=dict(marker='o', ms=3, alpha=0.4))
palette = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, 4))
for patch, color in zip(bp['boxes'], palette):
    patch.set_facecolor(color); patch.set_alpha(0.85)
ax.set_xticks([1,2,3,4])
ax.set_xticklabels(['Q1\n(simple)','Q2','Q3','Q4\n(complex)'])
ax.set_ylabel('Copy-source perceptual distance ↓')
ax.set_xlabel('Complexity Quartile  C(o; 90°)')
ax.set_title('Copy-source baseline by object complexity', fontsize=9)
ax.yaxis.grid(True, color='#EEEEEE'); ax.set_axisbelow(True)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig2.png', dpi=180, bbox_inches='tight')
plt.close()
print("\n  fig2.png updated with real COIL-100 results.")

# ── Save key numbers for paper ────────────────────────────────────────────────
print("\n" + "="*55)
print("KEY NUMBERS FOR PAPER (REAL DATA)")
print("="*55)

for q in ['Q1','Q2','Q3','Q4']:
    sub = df_cs[df_cs['quartile']==q]
    perc_mean = sub['perceptual'].mean()
    perc_std  = sub['perceptual'].std()
    mse_mean  = sub['mse'].mean()
    print(f"  {q}  copy-src perceptual: {perc_mean:.4f} ± {perc_std:.4f}   MSE: {mse_mean:.5f}")

print(f"\n  Complexity range: [{scores.min():.4f}, {scores.max():.4f}]")
print(f"  Q1/Q4 complexity ratio: {scores[scores<=q1].mean():.4f} / {scores[scores>q3].mean():.4f}"
      f" = {scores[scores>q3].mean()/scores[scores<=q1].mean():.1f}×")

# save summary for LaTeX
summary_rows = []
for q in ['Q1','Q2','Q3','Q4']:
    sub = df_cs[df_cs['quartile']==q]
    sub_nn = df_nn[df_nn['quartile']==q] if len(df_nn[df_nn['quartile']==q]) > 0 else None
    nn_mean = sub_nn['nn_perc'].mean() if sub_nn is not None and len(sub_nn) > 0 else float('nan')
    summary_rows.append({
        'quartile': q,
        'n_objects': len(quartile_objs[q]),
        'complexity_mean': df_cx[df_cx['quartile']==q]['complexity_feat'].mean(),
        'complexity_std':  df_cx[df_cx['quartile']==q]['complexity_feat'].std(),
        'copy_src_perc_mean': sub['perceptual'].mean(),
        'copy_src_perc_std':  sub['perceptual'].std(),
        'nn_perc_mean': nn_mean,
    })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(f'{RES_DIR}/exp1_summary_real.csv', index=False)
print("\nSaved: results/exp1_summary_real.csv")
print("Saved: results/complexity_scores.csv")
print("Saved: results/copy_source_baseline.csv")
print("Saved: results/nn_retrieval_baseline.csv")
print("Saved: figures/fig1.png (real COIL-100 images)")
print("Saved: figures/fig2.png (real complexity vs baseline)")
