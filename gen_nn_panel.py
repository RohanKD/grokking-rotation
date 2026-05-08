"""
Generate Figure 6: NN retrieval panel.

For one simple (Q1) and one complex (Q4) test object, shows:
  Col 1  – query source image (x_s)
  Col 2  – ground-truth target at +90° (x_t)
  Col 3  – NN retrieved source from training set (x_s^*)
  Col 4  – NN retrieved target at +90° (the "prediction")
  Col 5  – pixel difference |x_t - NN_pred| × 4

This makes the shortcut failure visible: for Q1 objects the NN source
looks like the query and the retrieved target is close to ground truth,
but for Q4 the NN source is perceptually different and the retrieved
target diverges badly.
"""
import os, re, glob, random
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

COIL_DIR = '/home/rohan/perspective/coil-100/coil-100'
RES_DIR  = '/home/rohan/perspective/results'
FIG_DIR  = '/home/rohan/perspective/figures'
DELTA    = 90   # degrees

IMG_SIZE = 128
to_tensor = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])


def load_img(obj_id: int, angle_deg: int) -> torch.Tensor:
    path = os.path.join(COIL_DIR, f'obj{obj_id}__{angle_deg}.png')
    return to_tensor(Image.open(path).convert('RGB'))


def list_angles(obj_id: int):
    files = glob.glob(os.path.join(COIL_DIR, f'obj{obj_id}__*.png'))
    return sorted(int(re.search(r'__(\d+)\.png', f).group(1)) for f in files)


# ── VGG features ─────────────────────────────────────────────────────────────
device = torch.device('cpu')
vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
normalize = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

@torch.no_grad()
def feat(img: torch.Tensor) -> torch.Tensor:
    return F.normalize(vgg(normalize(img).unsqueeze(0)).flatten(1), dim=1)


# ── Load complexity quartile assignments ──────────────────────────────────────
cx = pd.read_csv(os.path.join(RES_DIR, 'complexity_scores.csv'))
q1_cut = cx['complexity_feat'].quantile(0.25)
q4_cut = cx['complexity_feat'].quantile(0.75)

q1_ids = sorted(cx[cx['complexity_feat'] <= q1_cut]['obj_id'].tolist())
q4_ids = sorted(cx[cx['complexity_feat'] >= q4_cut]['obj_id'].tolist())

# train/test split (same seed as experiments)
random.seed(42)
all_ids = list(range(1, 101))
random.shuffle(all_ids)
train_ids = sorted(all_ids[:80])
test_ids  = sorted(all_ids[80:])

q1_test = [o for o in q1_ids if o in test_ids]
q4_test = [o for o in q4_ids if o in test_ids]

print(f"Q1 test objects: {q1_test}")
print(f"Q4 test objects: {q4_test}")

# ── Pre-cache training features ───────────────────────────────────────────────
print("Caching training features …")
train_cache = {}
for obj_id in train_ids:
    for ang in list_angles(obj_id):
        train_cache[(obj_id, ang)] = feat(load_img(obj_id, ang))
print(f"  cached {len(train_cache)} views")


def nn_retrieve(query_img: torch.Tensor, delta: int):
    """Return (best_obj, src_ang, tgt_img_tensor, best_dist)."""
    qf = feat(query_img)
    best_d, best_key = float('inf'), None
    for (oid, ang), f in train_cache.items():
        d = float(1 - (qf * f).sum())
        if d < best_d:
            best_d, best_key = d, (oid, ang)
    best_obj, best_ang = best_key
    angles = list_angles(best_obj)
    tgt_deg = (best_ang + delta) % 360
    nearest = min(angles, key=lambda a: abs(a - tgt_deg))
    return best_obj, best_ang, load_img(best_obj, nearest), best_d


def pick_example(test_obj_ids, seed=0):
    """Pick a fixed source angle from one test object."""
    rng = random.Random(seed)
    obj = rng.choice(test_obj_ids)
    ang = rng.choice(list_angles(obj))
    return obj, ang


# ── Build panel ───────────────────────────────────────────────────────────────
random.seed(7)
q1_obj, q1_ang  = pick_example(q1_test, seed=3)
q4_obj, q4_ang  = pick_example(q4_test, seed=5)

examples = [
    ('Simple (Q1)', q1_obj, q1_ang),
    ('Complex (Q4)', q4_obj, q4_ang),
]

def t2img(t):
    return t.permute(1,2,0).clamp(0,1).numpy()

def diff_img(a, b):
    d = ((a - b).abs() * 4).clamp(0, 1)
    return d.permute(1,2,0).numpy()

cols = ['Query\n$x_s$', 'Ground truth\n$x_t$ (+90°)',
        'NN match\n$x_s^*$ (train)', 'NN target\n$x_t^*$ (+90°)',
        'Error\n$|x_t - x_t^*|$']

fig, axes = plt.subplots(2, 5, figsize=(10.5, 4.6),
                         gridspec_kw={'hspace': 0.08, 'wspace': 0.04})

for row, (label, obj_id, src_ang) in enumerate(examples):
    angles   = list_angles(obj_id)
    tgt_deg  = (src_ang + DELTA) % 360
    tgt_ang  = min(angles, key=lambda a: abs(a - tgt_deg))
    src_img  = load_img(obj_id, src_ang)
    tgt_img  = load_img(obj_id, tgt_ang)
    nn_obj, nn_ang, nn_tgt, nn_dist = nn_retrieve(src_img, DELTA)
    nn_src = load_img(nn_obj, nn_ang)

    # VGG distances for annotation
    d_nn  = float(1 - (feat(nn_tgt) * feat(tgt_img)).sum())
    d_cs  = float(1 - (feat(src_img) * feat(tgt_img)).sum())

    imgs = [t2img(src_img), t2img(tgt_img), t2img(nn_src), t2img(nn_tgt),
            diff_img(tgt_img, nn_tgt)]

    for col, (ax, img) in enumerate(zip(axes[row], imgs)):
        ax.imshow(img, interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title(cols[col], fontsize=8.5, pad=4)
        for spine in ax.spines.values():
            spine.set_linewidth(0)

    # row label on leftmost
    axes[row, 0].set_ylabel(label, fontsize=9, fontweight='bold', labelpad=6)
    # annotation on error panel
    axes[row, 4].set_title(f'Error\n$|x_t - x_t^*|$\nNN dist={d_nn:.3f}  CS={d_cs:.3f}',
                           fontsize=7.5, pad=4)
    print(f"{label}  obj={obj_id} src={src_ang}°→{tgt_ang}°  "
          f"NN match: obj={nn_obj} ang={nn_ang}°  "
          f"NN dist={d_nn:.4f}  copy-src dist={d_cs:.4f}")

fig.suptitle('Nearest-Neighbour Retrieval Baseline  '
             '(source → +90° target)',
             fontsize=10, y=1.01)
plt.savefig(os.path.join(FIG_DIR, 'fig_nn_panel.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print(f"\nSaved: {FIG_DIR}/fig_nn_panel.png")
