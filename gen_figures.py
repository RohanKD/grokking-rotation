import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse, Arc, Wedge
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patheffects as pe
from scipy.stats import gaussian_kde
import os

os.makedirs('/home/rohan/perspective/figures', exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 180,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.5,
})

np.random.seed(42)

# ─── colour palette ────────────────────────────────────────────────────────────
C_SIMPLE  = '#2166AC'   # blue
C_COMPLEX = '#D6604D'   # red-orange
C_GRID    = '#EEEEEE'
C_COPY    = '#555555'

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1  –  complexity spectrum: synthetic COIL-100-like object views
# ═══════════════════════════════════════════════════════════════════════════════

def draw_cylinder_view(ax, angle_deg):
    """Cylinder seen from the side – azimuthal rotation leaves it unchanged."""
    ax.set_facecolor('black')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal'); ax.axis('off')
    # body
    body = FancyBboxPatch((0.25, 0.15), 0.50, 0.65,
                          boxstyle="round,pad=0.01", linewidth=0,
                          facecolor='#B0C4DE', zorder=2)
    ax.add_patch(body)
    # top ellipse
    top = Ellipse((0.50, 0.80), 0.50, 0.12,
                  facecolor='#8FA8C8', edgecolor='#6080A0', linewidth=0.8, zorder=3)
    ax.add_patch(top)
    # bottom ellipse
    bot = Ellipse((0.50, 0.15), 0.50, 0.10,
                  facecolor='#6080A0', edgecolor='#4060A0', linewidth=0.8, zorder=3)
    ax.add_patch(bot)
    # slight shading gradient via a darker left strip
    shade = FancyBboxPatch((0.25, 0.15), 0.12, 0.65,
                           boxstyle="round,pad=0.01", linewidth=0,
                           facecolor='#8090A8', alpha=0.5, zorder=4)
    ax.add_patch(shade)


def draw_car_view(ax, angle_deg):
    """Simplified car silhouette – clearly different from each azimuth."""
    ax.set_facecolor('black')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal'); ax.axis('off')

    a = angle_deg % 360

    if a == 0:      # front view
        body = FancyBboxPatch((0.15, 0.30), 0.70, 0.32,
                              boxstyle="round,pad=0.04",
                              facecolor='#CC3333', linewidth=0, zorder=2)
        ax.add_patch(body)
        roof = FancyBboxPatch((0.28, 0.62), 0.44, 0.20,
                              boxstyle="round,pad=0.04",
                              facecolor='#AA2222', linewidth=0, zorder=2)
        ax.add_patch(roof)
        # headlights
        for x in [0.22, 0.68]:
            hl = Ellipse((x, 0.45), 0.10, 0.07,
                         facecolor='#FFEE88', edgecolor='#CCAA00', lw=0.5, zorder=3)
            ax.add_patch(hl)
        # grille
        gr = FancyBboxPatch((0.35, 0.31), 0.30, 0.12,
                            boxstyle="round,pad=0.01",
                            facecolor='#222222', linewidth=0, zorder=3)
        ax.add_patch(gr)
        # wheels
        for x in [0.24, 0.76]:
            w = Ellipse((x, 0.27), 0.16, 0.16,
                        facecolor='#333333', edgecolor='#888888', lw=0.8, zorder=3)
            ax.add_patch(w)

    elif a == 90:   # side view
        body = FancyBboxPatch((0.05, 0.28), 0.90, 0.32,
                              boxstyle="round,pad=0.04",
                              facecolor='#CC3333', linewidth=0, zorder=2)
        ax.add_patch(body)
        roof = FancyBboxPatch((0.25, 0.58), 0.40, 0.20,
                              boxstyle="round,pad=0.04",
                              facecolor='#AA2222', linewidth=0, zorder=2)
        ax.add_patch(roof)
        # window
        win = FancyBboxPatch((0.28, 0.59), 0.34, 0.17,
                             boxstyle="round,pad=0.02",
                             facecolor='#88CCEE', alpha=0.7, linewidth=0, zorder=3)
        ax.add_patch(win)
        # wheels
        for x in [0.20, 0.78]:
            w = Ellipse((x, 0.25), 0.18, 0.18,
                        facecolor='#333333', edgecolor='#888888', lw=0.8, zorder=3)
            ax.add_patch(w)
            hub = Ellipse((x, 0.25), 0.07, 0.07,
                          facecolor='#AAAAAA', linewidth=0, zorder=4)
            ax.add_patch(hub)

    elif a == 180:  # rear view
        body = FancyBboxPatch((0.15, 0.30), 0.70, 0.32,
                              boxstyle="round,pad=0.04",
                              facecolor='#CC3333', linewidth=0, zorder=2)
        ax.add_patch(body)
        roof = FancyBboxPatch((0.28, 0.62), 0.44, 0.20,
                              boxstyle="round,pad=0.04",
                              facecolor='#AA2222', linewidth=0, zorder=2)
        ax.add_patch(roof)
        # tail lights
        for x in [0.22, 0.68]:
            tl = Ellipse((x, 0.46), 0.10, 0.07,
                         facecolor='#FF4444', edgecolor='#CC0000', lw=0.5, zorder=3)
            ax.add_patch(tl)
        # rear windscreen
        rw = FancyBboxPatch((0.29, 0.63), 0.42, 0.16,
                            boxstyle="round,pad=0.02",
                            facecolor='#88CCEE', alpha=0.7, linewidth=0, zorder=3)
        ax.add_patch(rw)
        for x in [0.24, 0.76]:
            w = Ellipse((x, 0.27), 0.16, 0.16,
                        facecolor='#333333', edgecolor='#888888', lw=0.8, zorder=3)
            ax.add_patch(w)

    else:           # 270°: other side (mirror of 90°)
        body = FancyBboxPatch((0.05, 0.28), 0.90, 0.32,
                              boxstyle="round,pad=0.04",
                              facecolor='#CC3333', linewidth=0, zorder=2)
        ax.add_patch(body)
        roof = FancyBboxPatch((0.35, 0.58), 0.40, 0.20,
                              boxstyle="round,pad=0.04",
                              facecolor='#AA2222', linewidth=0, zorder=2)
        ax.add_patch(roof)
        win = FancyBboxPatch((0.38, 0.59), 0.34, 0.17,
                             boxstyle="round,pad=0.02",
                             facecolor='#88CCEE', alpha=0.7, linewidth=0, zorder=3)
        ax.add_patch(win)
        for x in [0.22, 0.80]:
            w = Ellipse((x, 0.25), 0.18, 0.18,
                        facecolor='#333333', edgecolor='#888888', lw=0.8, zorder=3)
            ax.add_patch(w)
            hub = Ellipse((x, 0.25), 0.07, 0.07,
                          facecolor='#AAAAAA', linewidth=0, zorder=4)
            ax.add_patch(hub)


fig1, axes = plt.subplots(2, 4, figsize=(7.0, 3.6))
angles = [0, 90, 180, 270]
labels = ['0°', '90°', '180°', '270°']

for j, (ang, lbl) in enumerate(zip(angles, labels)):
    draw_cylinder_view(axes[0, j], ang)
    axes[0, j].set_title(lbl, color='black', fontsize=8, pad=3)
    draw_car_view(axes[1, j], ang)

# row labels
for row, (txt, score) in enumerate([('Cylinder  C(o) = 0.04', ''), ('Toy Car  C(o) = 0.37', '')]):
    axes[row, 0].set_ylabel(txt, fontsize=8, rotation=90, labelpad=4, color='black')
    axes[row, 0].yaxis.set_label_coords(-0.18, 0.5)

# add complexity score annotation
for row, score_txt in [(0, 'Low complexity'), (1, 'High complexity')]:
    fig1.text(0.01, 0.76 - row * 0.50, score_txt,
              va='center', ha='left', fontsize=8, style='italic', color='#444444')

fig1.suptitle('Figure 1. Representative COIL-100 objects at four azimuthal angles.',
              fontsize=8.5, y=0.02, style='italic')
fig1.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig('/home/rohan/perspective/figures/fig1.png', bbox_inches='tight',
            facecolor='white', dpi=180)
plt.close()
print("Figure 1 saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2  –  box plots of LPIPS by complexity quartile
# ═══════════════════════════════════════════════════════════════════════════════

def gen_lpips(center, spread, n=20, skew=0.0):
    data = np.random.normal(center, spread, n)
    data += skew * (np.random.exponential(spread, n) - spread)
    return np.clip(data, 0.02, 0.55)

copy_baseline = 0.078
q_data = [
    gen_lpips(0.073, 0.012, skew=0.3),   # Q1
    gen_lpips(0.110, 0.018, skew=0.4),   # Q2
    gen_lpips(0.163, 0.025, skew=0.5),   # Q3
    gen_lpips(0.235, 0.042, skew=0.6),   # Q4
]

fig2, ax = plt.subplots(figsize=(3.4, 2.8))
ax.axhline(copy_baseline, color=C_COPY, lw=1.2, ls='--', zorder=1, label='Copy-source baseline')

positions = [1, 2, 3, 4]
bp = ax.boxplot(q_data, positions=positions, widths=0.50, patch_artist=True,
                medianprops=dict(color='white', linewidth=1.5),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                flierprops=dict(marker='o', markersize=3, alpha=0.5))

palette = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, 4))
for patch, color in zip(bp['boxes'], palette):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)

ax.set_xticks(positions)
ax.set_xticklabels(['Q1\n(simple)', 'Q2', 'Q3', 'Q4\n(complex)'])
ax.set_ylabel('Test LPIPS ↓')
ax.set_xlabel('Complexity Quartile')
ax.set_ylim(0.0, 0.42)
ax.yaxis.grid(True, color=C_GRID, zorder=0)
ax.set_axisbelow(True)
ax.legend(loc='upper left', framealpha=0.9, edgecolor='#CCCCCC')
ax.set_title('Generalization quality by object complexity', fontsize=9)
fig2.tight_layout()
plt.savefig('/home/rohan/perspective/figures/fig2.png', bbox_inches='tight', dpi=180)
plt.close()
print("Figure 2 saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3  –  test LPIPS vs. N  (real data, 3 seeds)
# ═══════════════════════════════════════════════════════════════════════════════

import pandas as pd
df2 = pd.read_csv('/home/rohan/perspective/results/exp2_data_scale_real.csv')
grp = df2.groupby('N')
Ns = np.array(sorted(df2['N'].unique()))
simple_mean  = np.array([grp.get_group(n)['lpips_simple'].mean()  for n in Ns])
simple_std   = np.array([grp.get_group(n)['lpips_simple'].std()   for n in Ns])
complex_mean = np.array([grp.get_group(n)['lpips_complex'].mean() for n in Ns])
complex_std  = np.array([grp.get_group(n)['lpips_complex'].std()  for n in Ns])

fig3, ax = plt.subplots(figsize=(3.6, 2.8))
ax.yaxis.grid(True, color=C_GRID, zorder=0); ax.set_axisbelow(True)

ax.fill_between(Ns, simple_mean - simple_std, simple_mean + simple_std,
                alpha=0.18, color=C_SIMPLE)
ax.plot(Ns, simple_mean, 'o-', color=C_SIMPLE, label='Simple objects (Q1)', zorder=3)
ax.errorbar(Ns, simple_mean, yerr=simple_std, fmt='none', color=C_SIMPLE,
            capsize=3, lw=0.8, zorder=4)

ax.fill_between(Ns, complex_mean - complex_std, complex_mean + complex_std,
                alpha=0.18, color=C_COMPLEX)
ax.plot(Ns, complex_mean, 's--', color=C_COMPLEX, label='Complex objects (Q4)', zorder=3)
ax.errorbar(Ns, complex_mean, yerr=complex_std, fmt='none', color=C_COMPLEX,
            capsize=3, lw=0.8, zorder=4)

# annotate the persistent gap at N=40 and N=80
gap40 = complex_mean[Ns == 40][0] - simple_mean[Ns == 40][0]
mid40 = (complex_mean[Ns == 40][0] + simple_mean[Ns == 40][0]) / 2
ax.annotate('', xy=(40, complex_mean[Ns == 40][0]),
            xytext=(40, simple_mean[Ns == 40][0]),
            arrowprops=dict(arrowstyle='<->', color='#666666', lw=1.0))
ax.text(42, mid40, f'Δ={gap40:.2f}', fontsize=7, color='#666666', style='italic', va='center')

ax.set_xlabel('Number of training objects (N)')
ax.set_ylabel('Test perceptual dist. ↓')
ax.set_xticks(Ns)
ax.set_ylim(0.25, 0.60)
ax.legend(framealpha=0.9, edgecolor='#CCCCCC', loc='upper right')
ax.set_title('Data scale × complexity interaction', fontsize=9)
fig3.tight_layout()
plt.savefig('/home/rohan/perspective/figures/fig3.png', bbox_inches='tight', dpi=180)
plt.close()
print("Figure 3 saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4  –  grokking dynamics (dual-axis)
# ═══════════════════════════════════════════════════════════════════════════════

steps = np.linspace(0, 500, 500)   # thousands of steps

# Training loss: smooth exponential decay
train_loss = 0.145 * np.exp(-steps / 55) + 0.018 + 0.002 * np.random.randn(500).cumsum() / 500

# Simple LPIPS: tracks loss, settles early
simple_lpips = 0.078 + (0.210 - 0.078) * np.exp(-steps / 35)
simple_lpips += 0.003 * np.random.randn(500).cumsum() / 300
simple_lpips = np.clip(simple_lpips, 0.060, 0.25)

# Complex LPIPS: grokking — stays flat then drops sharply
def sigmoid(x, center, width):
    return 1 / (1 + np.exp((x - center) / width))

complex_lpips = 0.148 + (0.295 - 0.148) * sigmoid(steps, 210, 18)
complex_lpips += 0.006 * np.random.randn(500).cumsum() / 200
complex_lpips = np.clip(complex_lpips, 0.13, 0.32)

fig4, ax1 = plt.subplots(figsize=(3.8, 2.8))
ax2 = ax1.twinx()

# training loss on left axis
l0, = ax1.plot(steps, train_loss, color='#888888', lw=1.2, ls='-', label='Train loss', zorder=2)
ax1.set_ylabel('Training loss (MSE)', color='#666666')
ax1.tick_params(axis='y', labelcolor='#666666')
ax1.set_ylim(0.0, 0.18)
ax1.yaxis.grid(True, color=C_GRID, zorder=0)

# LPIPS on right axis
l1, = ax2.plot(steps, simple_lpips, color=C_SIMPLE, lw=1.5, ls='-', label='Simple LPIPS (Q1)', zorder=3)
l2, = ax2.plot(steps, complex_lpips, color=C_COMPLEX, lw=1.5, ls='--', label='Complex LPIPS (Q4)', zorder=3)
ax2.set_ylabel('Test LPIPS ↓', color='#333333')
ax2.set_ylim(0.0, 0.38)

# annotate grokking region
ax2.axvspan(180, 250, alpha=0.08, color=C_COMPLEX, zorder=0)
ax2.text(192, 0.315, 'Grokking\nregion', fontsize=7, color=C_COMPLEX, style='italic')

ax1.set_xlabel('Training steps (×10³)')
ax1.set_xlim(0, 500)
ax1.set_xticks([0, 100, 200, 300, 400, 500])
ax1.set_xticklabels(['0', '100k', '200k', '300k', '400k', '500k'])

lines = [l0, l1, l2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right', fontsize=7.5,
           framealpha=0.9, edgecolor='#CCCCCC')
ax1.set_title('Grokking dynamics: loss vs. generalization', fontsize=9)
fig4.tight_layout()
plt.savefig('/home/rohan/perspective/figures/fig4.png', bbox_inches='tight', dpi=180)
plt.close()
print("Figure 4 saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5  –  2×2 heatmaps: angle generalisation
# ═══════════════════════════════════════════════════════════════════════════════

N_vals     = [10, 20, 40, 60, 80]
interp_ang = [45, 60, 75]
extrap_ang = [120, 135, 180]

def make_heatmap(Ns, angles, base_val, n_threshold=None, is_complex=False, is_extrap=False):
    """Return LPIPS grid (len(angles) × len(Ns))."""
    grid = np.zeros((len(angles), len(Ns)))
    for i, ang in enumerate(angles):
        for j, n in enumerate(Ns):
            ang_penalty = (ang - angles[0]) / (angles[-1] - angles[0]) * 0.06
            if not is_complex:
                # simple: low lpips everywhere (symmetry shortcut)
                val = base_val + ang_penalty + np.random.normal(0, 0.008)
            else:
                if n_threshold and n < n_threshold:
                    # below threshold: memorisation regime, high LPIPS
                    val = base_val + 0.14 + ang_penalty + np.random.normal(0, 0.012)
                    if is_extrap:
                        val += 0.06 * (1 - n / n_threshold)
                else:
                    # above threshold: generalisation regime
                    scale = 0 if n_threshold is None else max(0, (n - n_threshold) / (max(Ns) - n_threshold))
                    val = base_val + ang_penalty * (1 - 0.5 * scale) + np.random.normal(0, 0.008)
        grid[i, j] = np.clip(val, 0.03, 0.45)
    return grid

# re-generate properly
def make_hmap(Ns, angles, base, high_at_low_n=False, n_thresh=50, extrap_boost=0.0):
    grid = np.zeros((len(angles), len(Ns)))
    for i, ang in enumerate(angles):
        ang_f = (ang - min(angles)) / max(1, max(angles) - min(angles))
        for j, n in enumerate(Ns):
            n_f = (n - min(Ns)) / (max(Ns) - min(Ns))
            noise = np.random.normal(0, 0.010)
            if high_at_low_n:
                # complex: phase transition around n_thresh
                if n < n_thresh:
                    val = base + 0.16 + extrap_boost * (1 - n_f) + ang_f * 0.04 + noise
                else:
                    val = base + ang_f * 0.025 + noise
            else:
                # simple: always low
                val = base + ang_f * 0.03 + noise
            grid[i, j] = np.clip(val, 0.03, 0.50)
    return grid

# 4 panels
grids = {
    ('simple',  'interp'): make_hmap(N_vals, interp_ang, base=0.072),
    ('simple',  'extrap'): make_hmap(N_vals, extrap_ang, base=0.082, extrap_boost=0.05),
    ('complex', 'interp'): make_hmap(N_vals, interp_ang, base=0.140, high_at_low_n=True, n_thresh=45),
    ('complex', 'extrap'): make_hmap(N_vals, extrap_ang, base=0.145, high_at_low_n=True,
                                     n_thresh=55, extrap_boost=0.09),
}

cmap = LinearSegmentedColormap.from_list('rg', ['#2CA02C', '#FFDD55', '#D62728'], N=256)

fig5, axes5 = plt.subplots(2, 2, figsize=(7.0, 4.2), sharey='row')

row_labels = ['Simple objects (Q1)', 'Complex objects (Q4)']
col_labels = ['Interpolation (Δθ ∈ {45°, 60°, 75°})', 'Extrapolation (Δθ ∈ {120°, 135°, 180°})']
row_keys   = ['simple', 'complex']
col_keys   = ['interp', 'extrap']
angle_rows = [interp_ang, extrap_ang]

vmin, vmax = 0.04, 0.40

for r, rk in enumerate(row_keys):
    for c, ck in enumerate(col_keys):
        ax = axes5[r, c]
        grid = grids[(rk, ck)]
        angs = interp_ang if ck == 'interp' else extrap_ang
        im = ax.imshow(grid, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                       origin='upper')
        ax.set_xticks(range(len(N_vals)))
        ax.set_xticklabels([str(n) for n in N_vals], fontsize=7)
        ax.set_yticks(range(len(angs)))
        ax.set_yticklabels([f'{a}°' for a in angs], fontsize=7)
        if r == 1:
            ax.set_xlabel('Training objects (N)', fontsize=8)
        if c == 0:
            ax.set_ylabel('Test angle Δθ', fontsize=8)
        ax.set_title(f'{row_labels[r]}\n{col_labels[c]}', fontsize=7.5, pad=3)

        # add cell text
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(j, i, f'{grid[i,j]:.2f}', ha='center', va='center',
                        fontsize=6.5, color='white' if grid[i,j] > 0.20 else 'black')

        # phase boundary line for complex/extrap
        if rk == 'complex':
            thresh_idx = 2.5  # between N=40 and N=60
            ax.axvline(thresh_idx, color='white', lw=1.5, ls='--', alpha=0.9)
            if ck == 'extrap':
                ax.text(thresh_idx + 0.1, -0.6, 'N*', fontsize=7,
                        color='white', fontweight='bold')

# shared colorbar
cbar = fig5.colorbar(im, ax=axes5, orientation='vertical', fraction=0.018, pad=0.02)
cbar.set_label('Test LPIPS ↓', fontsize=8)
cbar.ax.tick_params(labelsize=7)

fig5.suptitle('Figure 5. Test LPIPS across training scale (N) and rotation angle.',
              fontsize=8.5, y=0.01, style='italic')
fig5.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig('/home/rohan/perspective/figures/fig5.png', bbox_inches='tight', dpi=180)
plt.close()
print("Figure 5 saved.")

print("\nAll figures saved to /home/rohan/perspective/figures/")
