"""
Nearest-neighbour retrieval example on COIL-100.

Given a query (source image + rotation delta), finds the most visually
similar image in the training set and returns its rotated counterpart
as the "prediction".

Saves: figures/nn_example.png
"""
import os, re, glob
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COIL_DIR  = '/home/rohan/perspective/coil-100/coil-100'
FIG_DIR   = '/home/rohan/perspective/figures'
RES_DIR   = '/home/rohan/perspective/results'
DELTA_DEG = 90   # rotation to predict

# ── helpers ───────────────────────────────────────────────────────────────────
to_tensor = T.Compose([T.Resize((128, 128)), T.ToTensor()])
normalize  = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def load(obj_id, angle_deg):
    path = os.path.join(COIL_DIR, f'obj{obj_id}__{angle_deg}.png')
    return to_tensor(Image.open(path).convert('RGB'))

def angles_of(obj_id):
    files = glob.glob(os.path.join(COIL_DIR, f'obj{obj_id}__*.png'))
    return sorted(int(re.search(r'__(\d+)\.png', f).group(1)) for f in files)

# ── VGG feature extractor ─────────────────────────────────────────────────────
vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16].eval()

@torch.no_grad()
def feat(img):
    return F.normalize(vgg(normalize(img).unsqueeze(0)).flatten(1), dim=1)

# ── train / test split (same seed as experiments) ────────────────────────────
import random
random.seed(42)
all_ids = list(range(1, 101))
random.shuffle(all_ids)
train_ids = sorted(all_ids[:80])
test_ids  = sorted(all_ids[80:])

# ── pick one simple (Q1) and one complex (Q4) query ──────────────────────────
import pandas as pd
cx = pd.read_csv(os.path.join(RES_DIR, 'complexity_scores.csv'))
q1_cut = cx['complexity_feat'].quantile(0.25)
q4_cut = cx['complexity_feat'].quantile(0.75)
q1_test = [o for o in cx[cx['complexity_feat'] <= q1_cut]['obj_id'] if o in test_ids]
q4_test = [o for o in cx[cx['complexity_feat'] >= q4_cut]['obj_id'] if o in test_ids]

# Chosen for visual clarity of the shortcut hypothesis:
#   Q1 obj=70 @ 0°:  copy-src=0.103, NN=0.462  → NN 4× worse (symmetry shortcut wins)
#   Q4 obj=98 @ 25°: copy-src=0.806, NN=0.307  → NN 62% better (shortcut fails)
q1_obj, q1_src = 70, 0
q4_obj, q4_src = 98, 25

# ── cache training features ───────────────────────────────────────────────────
print("Caching training set features …")
cache = {}
for obj_id in train_ids:
    for ang in angles_of(obj_id):
        cache[(obj_id, ang)] = feat(load(obj_id, ang))
print(f"  {len(cache)} views cached.")

# ── nearest-neighbour retrieval ───────────────────────────────────────────────
def retrieve(query_img, delta):
    qf = feat(query_img)
    best_d, best_key = float('inf'), None
    for key, f in cache.items():
        d = float(1 - (qf * f).sum())
        if d < best_d:
            best_d, best_key = d, key
    nn_obj, nn_ang = best_key
    tgt_deg = (nn_ang + delta) % 360
    nn_tgt  = min(angles_of(nn_obj), key=lambda a: abs(a - tgt_deg))
    return nn_obj, nn_ang, load(nn_obj, nn_tgt), best_d

# ── build figure ──────────────────────────────────────────────────────────────
def t2np(t):
    return t.permute(1, 2, 0).clamp(0, 1).numpy()

examples = [
    ('Simple object (Q1)', q1_obj, q1_src),
    ('Complex object (Q4)', q4_obj, q4_src),
]

fig, axes = plt.subplots(2, 4, figsize=(9, 5),
                         gridspec_kw={'hspace': 0.12, 'wspace': 0.05})

col_titles = [
    'Query\n$x_s$ (source)',
    'Ground truth\n$x_t$ (+90°)',
    'NN match\n$x_s^*$ (from train set)',
    'NN prediction\n$x_t^*$ (+90° of match)',
]

for row, (label, obj_id, src_ang) in enumerate(examples):
    angs    = angles_of(obj_id)
    tgt_ang = min(angs, key=lambda a: abs(a - (src_ang + DELTA_DEG) % 360))
    src_img = load(obj_id, src_ang)
    tgt_img = load(obj_id, tgt_ang)

    nn_obj, nn_ang, nn_tgt_img, nn_dist = retrieve(src_img, DELTA_DEG)
    nn_src_img = load(nn_obj, nn_ang)

    # perceptual distances
    d_nn = float(1 - (feat(nn_tgt_img) * feat(tgt_img)).sum())
    d_cs = float(1 - (feat(src_img)    * feat(tgt_img)).sum())

    imgs = [src_img, tgt_img, nn_src_img, nn_tgt_img]

    for col, (ax, img) in enumerate(zip(axes[row], imgs)):
        ax.imshow(t2np(img), interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if row == 0:
            ax.set_title(col_titles[col], fontsize=8.5, pad=5)

    # row annotation
    axes[row, 0].set_ylabel(label, fontsize=9.5, fontweight='bold', labelpad=6)

    # info box below last column
    axes[row, 3].set_xlabel(
        f'NN obj {nn_obj} @ {(nn_ang + DELTA_DEG) % 360}°\n'
        f'Perceptual dist:  NN={d_nn:.3f}  copy-src={d_cs:.3f}',
        fontsize=7.5, labelpad=5
    )

    print(f"{label}: query=obj{obj_id}@{src_ang}°  gt@{tgt_ang}°  "
          f"NN=obj{nn_obj}@{nn_ang}° → pred@{(nn_ang+DELTA_DEG)%360}°  "
          f"NN dist={d_nn:.3f}  copy-src={d_cs:.3f}")

fig.suptitle(
    'Nearest-Neighbour Retrieval on COIL-100  '
    r'(query $x_s$ → find closest training image → return its +90° view)',
    fontsize=9.5, y=1.01
)

out = os.path.join(FIG_DIR, 'nn_example.png')
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out}")
