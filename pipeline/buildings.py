# -*- coding: utf-8 -*-
"""Extract building cells + shadow-based relative heights from the mosaic.
Output: buildings.png (grayscale height map, 0.25m units) in mosaic frame (1/4 res)."""
import cv2, numpy as np, os, json

W = os.path.dirname(os.path.abspath(__file__))
mos = cv2.imread(os.path.join(W, "mosaic_full.jpg"))
meta = json.load(open(os.path.join(W, "mosaic_meta.json")))
H, Wd = mos.shape[:2]
PX_M = 0.377  # meters per mosaic pixel

hsv = cv2.cvtColor(mos, cv2.COLOR_BGR2HSV)
Hh, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
gray = cv2.cvtColor(mos, cv2.COLOR_BGR2GRAY)

valid = (gray > 36)  # exclude empty canvas (filled with 30,30,30)
# vegetation: green hue
veg = ((Hh > 35) & (Hh < 90) & (S > 60)).astype(np.uint8)
# shadow: dark
vth = np.percentile(V[valid], 15)
shadow = ((V < vth) & valid).astype(np.uint8)
# bright rooftops / built: bright, low-ish saturation, not veg
bth = np.percentile(V[valid], 35)
built = ((V > bth) & (S < 95) & (veg == 0) & valid).astype(np.uint8)

# remove thin elongated streets/paths: morphological opening with disk
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
built_o = cv2.morphologyEx(built, cv2.MORPH_OPEN, k)
built_o = cv2.morphologyEx(built_o, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
# exclude bare ground / wide bright roads: low local texture (roofs have edges)
tex = cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_32F)
tex = cv2.GaussianBlur(np.abs(tex), (0, 0), 8)
tth = np.percentile(tex[valid], 45)
built_o = (built_o > 0) & (tex > tth)
built_o = built_o.astype(np.uint8)

# ---- shadow direction: shift building mask in 8 dirs, max overlap with shadow
dirs = {(1,0):"E",(1,1):"SE",(0,1):"S",(-1,1):"SW",(-1,0):"W",(-1,-1):"NW",(0,-1):"N",(1,-1):"NE"}
best_dir, best_ov = None, -1
for (dx, dy), nm in dirs.items():
    Mshift = np.float32([[1, 0, dx*6], [0, 1, dy*6]])
    sh = cv2.warpAffine(built_o, Mshift, (Wd, H))
    ov = float((sh & shadow & (built_o == 0)).sum())
    if ov > best_ov:
        best_ov, best_dir, best_name = ov, (dx, dy), nm
print("shadow direction:", best_name, best_dir)

dx, dy = best_dir
# ---- per-cell aggregation + shadow length
CELL = 12  # px = 4.5m
gh, gw = H // CELL, Wd // CELL
cover = cv2.resize(built_o.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
shadow_f = shadow.astype(np.float32)

# shadow run length: for each pixel on building border in shadow dir, count run
# approximation: cumulative shadow sum along direction using shifted adds
run = np.zeros((H, Wd), np.float32)
cur = shadow_f.copy()
MAXRUN = 34  # ~13m shadow max
for i in range(1, MAXRUN + 1):
    Mshift = np.float32([[1, 0, -dx*i], [0, 1, -dy*i]])
    cur = cv2.warpAffine(shadow_f, Mshift, (Wd, H)) * (cur > 0)
    run += (cur > 0).astype(np.float32)
# run[p] = consecutive shadow pixels starting at p going along dir
edge = ((built_o > 0) & (cv2.warpAffine(built_o, np.float32([[1,0,-dx],[0,1,-dy]]), (Wd, H)) == 0)).astype(np.float32)
# shadow may start a few px past the (morphologically eroded) mask edge: search a window
shadow_at_edge = np.zeros((H, Wd), np.float32)
for j in range(1, 9):
    rj = cv2.warpAffine(run, np.float32([[1, 0, -dx*j], [0, 1, -dy*j]]), (Wd, H))
    shadow_at_edge = np.maximum(shadow_at_edge, rj * edge)

cellrun = np.zeros((gh, gw), np.float32)
cnt = np.zeros((gh, gw), np.float32)
ys, xs = np.where(shadow_at_edge > 1)
for y, x in zip(ys, xs):
    cy, cx = min(y // CELL, gh - 1), min(x // CELL, gw - 1)
    cellrun[cy, cx] += shadow_at_edge[y, x]
    cnt[cy, cx] += 1
mean_run = np.where(cnt > 0, cellrun / np.maximum(cnt, 1), 0)

# smooth shadow-run field over cells (evidence is sparse in dense fabric)
run_s = cv2.GaussianBlur(mean_run, (0, 0), 2.5)
cnt_s = cv2.GaussianBlur((cnt > 0).astype(np.float32), (0, 0), 2.5)
run_field = np.where(cnt_s > 0.02, run_s / np.maximum(cnt_s, 0.02), 0)

builtcell = cover > 0.45
evid = run_field[builtcell & (run_field > 0)]
med = np.median(evid) if len(evid) else 4.0
print("cells built:", int(builtcell.sum()), "median shadow run(px):", round(float(med), 2))
# base 3m + shadow-modulated bonus; median evidence ~ 2 floors (~5.5m)
heights = np.zeros((gh, gw), np.float32)
bonus = np.clip(run_field / max(med, 1e-3), 0, 2.2) * 2.5
heights[builtcell] = np.clip(3.0 + bonus[builtcell], 3.0, 8.5)

out = np.clip(heights / 0.25, 0, 255).astype(np.uint8)
cv2.imwrite(os.path.join(W, "buildings.png"), out)
json.dump(dict(cell_px=CELL, gw=gw, gh=gh, unit=0.25, px_m=PX_M,
               shadow_dir=best_name),
          open(os.path.join(W, "buildings_meta.json"), "w"))

# preview
prev = mos.copy()
hm = cv2.resize(out, (Wd, H), interpolation=cv2.INTER_NEAREST)
overlay = cv2.applyColorMap((hm * 4), cv2.COLORMAP_JET)
prev = np.where(hm[..., None] > 0, (0.55*prev + 0.45*overlay).astype(np.uint8), prev)
cv2.imwrite(os.path.join(W, "buildings_prev.jpg"), cv2.resize(prev, (Wd//3, H//3)))
print("done")
