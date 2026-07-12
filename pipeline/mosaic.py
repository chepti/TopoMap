# -*- coding: utf-8 -*-
"""Build a full-resolution mosaic of all 6 partial aerials.
All partials share the same GSD (verified rel_scale=1.000), so we solve
translation-only placements from pairwise SIFT matches, anchored at part1.
Then compute the mosaic->overview similarity transform from template-match registrations.
"""
import cv2, numpy as np, os, json, itertools

W = os.path.dirname(os.path.abspath(__file__))
reg = json.load(open(os.path.join(W, "reg.json")))

NAMES = ["part1", "part2", "part3", "part4", "part5", "part6"]

def load_cropped(name):
    img = cv2.imread(os.path.join(W, name + ".jpg"))
    return img[:-30, :, :]

imgs = {n: load_cropped(n) for n in NAMES}
sift = cv2.SIFT_create(6000)
bf = cv2.BFMatcher()
feats = {}
for n in NAMES:
    g = cv2.cvtColor(imgs[n], cv2.COLOR_BGR2GRAY)
    feats[n] = sift.detectAndCompute(g, None)

# pairwise translations: t[a][b] = position of b's origin in a's coords
edges = []
for a, b in itertools.combinations(NAMES, 2):
    ka, da = feats[a]; kb, db = feats[b]
    matches = bf.knnMatch(db, da, k=2)
    good = [m for m, n_ in matches if m.distance < 0.75 * n_.distance]
    if len(good) < 15:
        continue
    src = np.float32([kb[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([ka[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, inl = cv2.estimateAffinePartial2D(src, dst, cv2.RANSAC, ransacReprojThreshold=3)
    if M is None or inl.sum() < 15:
        continue
    scale = np.hypot(M[0, 0], M[0, 1])
    if abs(scale - 1) > 0.01:
        continue
    # translation of b origin in a coords (force pure translation using inliers)
    src_i = src[inl.ravel() == 1].reshape(-1, 2)
    dst_i = dst[inl.ravel() == 1].reshape(-1, 2)
    t = (dst_i - src_i).mean(axis=0)
    edges.append((a, b, float(t[0]), float(t[1]), int(inl.sum())))
    print(f"{a}<-{b}: inliers={int(inl.sum())} t=({t[0]:.1f},{t[1]:.1f}) scale={scale:.4f}")

# least squares: positions p[n], p[part1]=(0,0); constraint p[b]-p[a]=t
idx = {n: i for i, n in enumerate(NAMES)}
A = []; bx = []; by = []; wts = []
A.append([1 if i == 0 else 0 for i in range(6)]); bx.append(0); by.append(0); wts.append(1000)
for a, b, tx, ty, ninl in edges:
    row = [0]*6; row[idx[b]] = 1; row[idx[a]] = -1
    A.append(row); bx.append(tx); by.append(ty); wts.append(min(ninl, 200))
# weak absolute constraints from template matching (relative to part1), for parts
# without SIFT overlap (isolated strips)
S_TM = 0.12655  # native->overview scale
for n in NAMES[1:]:
    r = reg[n]
    if n == "part3":
        continue
    row = [0]*6; row[idx[n]] = 1; row[idx["part1"]] = -1
    A.append(row)
    bx.append((r["x"] - reg["part1"]["x"]) / S_TM)
    by.append((r["y"] - reg["part1"]["y"]) / S_TM)
    wts.append(3)
A = np.array(A, float); wts = np.sqrt(np.array(wts, float))[:, None]
Aw = A * wts
px = np.linalg.lstsq(Aw, np.array(bx) * wts.ravel(), rcond=None)[0]
py = np.linalg.lstsq(Aw, np.array(by) * wts.ravel(), rcond=None)[0]
pos = {n: (float(px[i]), float(py[i])) for n, i in idx.items()}
print("positions:", {k: (round(v[0],1), round(v[1],1)) for k, v in pos.items()})

# canvas bounds
minx = min(pos[n][0] for n in NAMES); miny = min(pos[n][1] for n in NAMES)
maxx = max(pos[n][0] + imgs[n].shape[1] for n in NAMES)
maxy = max(pos[n][1] + imgs[n].shape[0] for n in NAMES)
cw, ch = int(np.ceil(maxx - minx)), int(np.ceil(maxy - miny))
print("canvas:", cw, "x", ch)

# feathered composite
acc = np.zeros((ch, cw, 3), np.float64)
wacc = np.zeros((ch, cw), np.float64)
for n in NAMES:
    img = imgs[n].astype(np.float64)
    h, w = img.shape[:2]
    # feather weight: distance to border
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.minimum.reduce([xx + 1, w - xx, yy + 1, h - yy]).astype(np.float64)
    wt = np.minimum(d / 60.0, 1.0)
    x0 = int(round(pos[n][0] - minx)); y0 = int(round(pos[n][1] - miny))
    acc[y0:y0+h, x0:x0+w] += img * wt[..., None]
    wacc[y0:y0+h, x0:x0+w] += wt
mos = (acc / np.maximum(wacc, 1e-9)[..., None]).astype(np.uint8)
empty = wacc < 1e-9
mos[empty] = (30, 30, 30)
cv2.imwrite(os.path.join(W, "mosaic_full.jpg"), mos, [cv2.IMWRITE_JPEG_QUALITY, 92])

# mosaic -> overview similarity: use template positions (average over parts)
# overview_xy = s * (mosaic_xy) + off  where mosaic_xy of part n origin = pos[n]-min
scales = []; offs = []
for n in NAMES:
    r = reg[n]
    if n == "part3":
        r = dict(x=188.3, y=121.3, w=imgs[n].shape[1]*0.1265)
    s = r["w"] / imgs[n].shape[1]
    scales.append(s)
s = float(np.mean(scales))
for n in NAMES:
    r = reg[n]
    if n == "part3":
        r = dict(x=188.3, y=121.3)
    mx = pos[n][0] - minx; my = pos[n][1] - miny
    offs.append((r["x"] - s * mx, r["y"] - s * my))
off = np.array(offs).mean(axis=0)
print("mosaic->overview: scale=%.5f offset=(%.1f, %.1f)" % (s, off[0], off[1]))
print("per-part offset spread:", np.array(offs).std(axis=0))

meta = dict(positions={n: [pos[n][0]-minx, pos[n][1]-miny] for n in NAMES},
            canvas=[cw, ch], to_overview=dict(scale=s, ox=float(off[0]), oy=float(off[1])))
json.dump(meta, open(os.path.join(W, "mosaic_meta.json"), "w"), indent=1)

# small preview
prev = cv2.resize(mos, (cw//4, ch//4))
cv2.imwrite(os.path.join(W, "mosaic_prev.jpg"), prev)
print("done")
