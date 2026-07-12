# -*- coding: utf-8 -*-
"""Refine DEM by snapping contour-line components to 10m multiples of the prior surface."""
import cv2, numpy as np, os, json
from scipy.interpolate import RBFInterpolator

W = os.path.dirname(os.path.abspath(__file__))
topo = cv2.imread(os.path.join(W, "topo.jpg"))
h, w = topo.shape[:2]
prior = np.load(os.path.join(W, "dem.npy"))
meta = json.load(open(os.path.join(W, "dem_meta.json")))
GW, GH = meta["gw"], meta["gh"]
prior_full = cv2.resize(prior, (w, h), interpolation=cv2.INTER_CUBIC)

mask = cv2.imread(os.path.join(W, "contour_mask.png"), 0)
mask = (mask > 0).astype(np.uint8)
n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
print("components:", n)

snapped_pts = []
kept = skipped = 0
for i in range(1, n):
    area = stats[i, cv2.CC_STAT_AREA]
    if area < 30:
        continue
    ys, xs = np.where(labels == i)
    pv = prior_full[ys, xs]
    rng = pv.max() - pv.min()
    med = np.median(pv)
    snap = round(med / 10.0) * 10.0
    if rng > 22 or abs(med - snap) > 6:
        skipped += 1
        continue
    kept += 1
    # subsample points along component
    step = max(1, len(xs) // max(6, area // 40))
    for j in range(0, len(xs), step):
        snapped_pts.append((float(xs[j]), float(ys[j]), snap))
print(f"kept {kept} comps, skipped {skipped}, snapped pts {len(snapped_pts)}")

# anchors that must survive (wadi floor, summit)
ANCH = [(462, 356, 167), (231, 112, 92), (510, 104, 108),
        (10, 130, 80), (640, 150, 112), (100, 122, 84)]
pts = np.array([(p[0], p[1]) for p in snapped_pts] + [(a[0], a[1]) for a in ANCH], float)
vals = np.array([p[2] for p in snapped_pts] + [a[2] for a in ANCH], float)
print("total pts:", len(pts))

# subsample to keep RBF tractable
if len(pts) > 2600:
    idx = np.random.RandomState(7).choice(len(pts) - len(ANCH), 2600 - len(ANCH), replace=False)
    idx = np.concatenate([idx, np.arange(len(pts) - len(ANCH), len(pts))])
    pts, vals = pts[idx], vals[idx]

rbf = RBFInterpolator(pts, vals, kernel="thin_plate_spline", smoothing=1.0, neighbors=60)
gx, gy = np.meshgrid(np.linspace(0, w - 1, GW), np.linspace(0, h - 1, GH))
grid = rbf(np.stack([gx.ravel(), gy.ravel()], axis=1)).reshape(GH, GW)
# gentle smoothing to kill RBF ripples
grid = cv2.GaussianBlur(grid.astype(np.float32), (5, 5), 1.2)
print("refined dem min/max:", grid.min(), grid.max())
np.save(os.path.join(W, "dem2.npy"), grid)

# hillshade check (stronger)
km_px = meta["km_px"]
px_m = 1000.0 / km_px * (w / GW)
gzy, gzx = np.gradient(grid)
sx, sy = gzx / px_m, gzy / px_m
az, alt = np.radians(315), np.radians(45)
lx, ly, lz = np.sin(az)*np.cos(alt), -np.cos(az)*np.cos(alt), np.sin(alt)
norm = np.sqrt(sx**2 + sy**2 + 1)
shade = np.clip((-sx*lx - sy*ly + lz)/norm, 0, 1)
hs = cv2.resize((shade*255).astype(np.uint8), (w, h))
blend = cv2.addWeighted(topo, 0.35, cv2.cvtColor(hs, cv2.COLOR_GRAY2BGR), 0.65, 0)
cv2.imwrite(os.path.join(W, "dem2_check.png"), blend)

# also render pseudo-color elevation
gn = ((grid - grid.min())/(grid.max()-grid.min())*255).astype(np.uint8)
gc = cv2.applyColorMap(cv2.resize(gn, (w, h)), cv2.COLORMAP_TURBO)
cv2.imwrite(os.path.join(W, "dem2_color.png"), cv2.addWeighted(gc, 0.6, topo, 0.4, 0))
print("done")
