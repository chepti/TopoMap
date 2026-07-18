# -*- coding: utf-8 -*-
"""Unified pipeline: folder of map images -> 3D-viewer dataset.

Usage:
  python pipeline/run_all.py --input <folder> --site <id> --name "<display name>"
         [--config pipeline/<id>.config.json] [--overview <filename>]

Auto-classifies images (topo map / roads map / aerials), registers partial
aerials onto the overview aerial, builds a mosaic, extracts a DEM from the
topo map (needs elevation anchors from the config), extracts buildings
(footprint rectangles + shadow heights) and vegetation (trees/shrubs),
and writes webapp/data/<site>/ + updates webapp/data/sites.json.

Config JSON (per site):
{
  "anchors": [[topo_px_x, topo_px_y, elevation_m], ...],   // required for real DEM
  "roads_transform": [scale, ox, oy],                       // roads->overview px (optional)
  "topo_to_overview": [scale, ox, oy],                      // manual override (optional)
  "align_pairs": [[topo_x, topo_y, ov_x, ov_y], ...],       // >=2 matching points (optional)
  "topo": "file.jpg", "roads": "file.jpg",                  // explicit roles (optional)
  "overview": "file.jpg", "skip": ["file.jpg", ...]
}
"""
import argparse, glob, itertools, json, os, sys
import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator

# ---------------------------------------------------------------- helpers

def imread(path):
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit("cannot read " + path)
    return img

def grad_feat(g):
    g = cv2.GaussianBlur(g, (3, 3), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return cv2.magnitude(gx, gy)

# ---------------------------------------------------------------- 1) classify

def classify(paths):
    """Return topo, roads (may be None), aerials (list).
    Drawn maps are mostly FLAT (large uniform regions); aerial photos are
    textured everywhere. Among maps: topo is colorful, roads map is near-white."""
    kinds = {}
    for p in paths:
        img = imread(p)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
        grad = cv2.magnitude(gx, gy)
        flat = float((grad < 12).mean())
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        S, V = hsv[..., 1], hsv[..., 2]
        colored = float((S > 45).mean())
        white = float(((S < 40) & (V > 200)).mean())
        kinds[p] = dict(flat=flat, colored=colored, white=white)
    maps = [p for p in paths if kinds[p]["flat"] > 0.15]
    aerials = [p for p in paths if p not in maps]
    topo = max(maps, key=lambda p: kinds[p]["colored"], default=None)
    roads_cands = [p for p in maps if p != topo]
    roads = max(roads_cands, key=lambda p: kinds[p]["white"], default=None)
    for p in paths:
        tag = "topo" if p == topo else "roads" if p == roads else "aerial"
        k = kinds[p]
        print(f"  {os.path.basename(p)}: {tag}  (flat={k['flat']:.2f} colored={k['colored']:.2f} white={k['white']:.2f})")
    return topo, roads, aerials

# ---------------------------------------------------------------- 2) registration

def crop_watermark(img):
    return img[:-30, :, :] if img.shape[0] > 200 else img

def register_partials(overview, partials):
    """Template-match each partial into overview. Returns {i: (score, scale, x, y)}."""
    oh, ow = overview.shape[:2]
    UP = 3
    big = cv2.resize(overview, (ow * UP, oh * UP), interpolation=cv2.INTER_CUBIC)
    big_feat = grad_feat(cv2.cvtColor(big, cv2.COLOR_BGR2GRAY))
    out = {}
    for i, img in enumerate(partials):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        best = None
        for scale in np.arange(0.06, 0.60, 0.01):
            sw, sh = int(img.shape[1] * scale), int(img.shape[0] * scale)
            if sw < 40 or sh < 40 or sw >= big.shape[1] or sh >= big.shape[0]:
                continue
            tf = grad_feat(cv2.resize(g, (sw, sh), interpolation=cv2.INTER_AREA))
            _, mx, _, ml = cv2.minMaxLoc(cv2.matchTemplate(big_feat, tf, cv2.TM_CCOEFF_NORMED))
            if best is None or mx > best[0]:
                best = (float(mx), scale / UP, ml[0] / UP, ml[1] / UP)
        out[i] = best
        print(f"  partial {i}: score={best[0]:.2f} scale={best[1]:.4f} at ({best[2]:.0f},{best[3]:.0f})")
    return out

def build_mosaic(partials, tm):
    """SIFT graph among partials (translation-only, same GSD) + weak absolute
    template constraints. Returns mosaic image + to_overview transform."""
    n = len(partials)
    sift = cv2.SIFT_create(6000)
    bf = cv2.BFMatcher()
    feats = []
    for img in partials:
        feats.append(sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None))
    edges = []
    for a, b in itertools.combinations(range(n), 2):
        ka, da = feats[a]; kb, db = feats[b]
        if da is None or db is None:
            continue
        good = [m for m, n2 in bf.knnMatch(db, da, k=2) if m.distance < 0.75 * n2.distance]
        if len(good) < 15:
            continue
        src = np.float32([kb[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([ka[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, inl = cv2.estimateAffinePartial2D(src, dst, cv2.RANSAC, ransacReprojThreshold=3)
        if M is None or inl.sum() < 15 or abs(np.hypot(M[0, 0], M[0, 1]) - 1) > 0.015:
            continue
        s = src[inl.ravel() == 1].reshape(-1, 2); d = dst[inl.ravel() == 1].reshape(-1, 2)
        t = (d - s).mean(axis=0)
        edges.append((a, b, float(t[0]), float(t[1]), int(inl.sum())))
        print(f"  sift {a}<-{b}: inliers={int(inl.sum())}")
    # median template scale = native->overview
    S_TM = float(np.median([tm[i][1] for i in range(n)]))
    A = [[1 if j == 0 else 0 for j in range(n)]]; bx = [0.0]; by = [0.0]; w = [1000.0]
    for a, b, tx, ty, ninl in edges:
        row = [0] * n; row[b] = 1; row[a] = -1
        A.append(row); bx.append(tx); by.append(ty); w.append(min(ninl, 200))
    for i in range(1, n):
        row = [0] * n; row[i] = 1; row[0] = -1
        A.append(row)
        bx.append((tm[i][2] - tm[0][2]) / S_TM)
        by.append((tm[i][3] - tm[0][3]) / S_TM)
        w.append(3.0)
    A = np.array(A, float); w = np.sqrt(np.array(w))[:, None]
    px = np.linalg.lstsq(A * w, np.array(bx) * w.ravel(), rcond=None)[0]
    py = np.linalg.lstsq(A * w, np.array(by) * w.ravel(), rcond=None)[0]
    minx = min(px[i] for i in range(n)); miny = min(py[i] for i in range(n))
    maxx = max(px[i] + partials[i].shape[1] for i in range(n))
    maxy = max(py[i] + partials[i].shape[0] for i in range(n))
    cw, ch = int(np.ceil(maxx - minx)), int(np.ceil(maxy - miny))
    acc = np.zeros((ch, cw, 3), np.float64); wacc = np.zeros((ch, cw), np.float64)
    for i, img in enumerate(partials):
        h2, w2 = img.shape[:2]
        yy, xx = np.mgrid[0:h2, 0:w2]
        d = np.minimum.reduce([xx + 1, w2 - xx, yy + 1, h2 - yy]).astype(np.float64)
        wt = np.minimum(d / 60.0, 1.0)
        x0 = int(round(px[i] - minx)); y0 = int(round(py[i] - miny))
        acc[y0:y0 + h2, x0:x0 + w2] += img.astype(np.float64) * wt[..., None]
        wacc[y0:y0 + h2, x0:x0 + w2] += wt
    mos = (acc / np.maximum(wacc, 1e-9)[..., None]).astype(np.uint8)
    mos[wacc < 1e-9] = 0
    offs = [(tm[i][2] - S_TM * (px[i] - minx), tm[i][3] - S_TM * (py[i] - miny)) for i in range(n)]
    off = np.array(offs).mean(axis=0)
    print(f"  mosaic {cw}x{ch}, ->overview scale={S_TM:.5f} off=({off[0]:.1f},{off[1]:.1f}) spread={np.array(offs).std(axis=0)}")
    return mos, (S_TM, float(off[0]), float(off[1]))

# ---------------------------------------------------------------- 3) topo -> DEM

def detect_grid(topo):
    g = cv2.cvtColor(topo, cv2.COLOR_BGR2GRAY)
    dark = (g < 100).astype(np.uint8)
    def lines(frac):
        idx = np.where(frac > 0.5)[0]
        out = []
        for x in idx:
            if out and x - out[-1][-1] <= 3: out[-1].append(x)
            else: out.append([x])
        return [int(np.mean(gr)) for gr in out]
    sp = np.diff(lines(dark.mean(axis=0))).tolist() + np.diff(lines(dark.mean(axis=1))).tolist()
    return float(np.median(sp)) if sp else None

def contour_mask_clean(topo):
    """Brown contour lines only.
    Color: light sienna (V 135-205) excludes the dark building squares (V<135)
    and the pink urban fill (V>205). Shape: keep long, sparse, THIN strokes
    (stroke width ~ 2*area/perimeter)."""
    hsv = cv2.cvtColor(topo, cv2.COLOR_BGR2HSV)
    # line cores are saturated sienna (S>=100); building squares are duller (S~85-94, V<135)
    m = cv2.inRange(hsv, (5, 100, 120), (25, 230, 205)) | cv2.inRange(hsv, (5, 55, 150), (25, 230, 205))
    roads_m = cv2.dilate(cv2.inRange(hsv, (10, 80, 150), (30, 255, 255)), np.ones((3, 3), np.uint8))
    m = cv2.bitwise_and(m, cv2.bitwise_not(roads_m))
    # bridge small gaps (dashes, label crossings) so lines form long components
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        ext = a / max(bw * bh, 1)
        long_enough = (a >= 30 and max(bw, bh) >= 20 and ext < 0.42) or \
                      (a >= 12 and max(bw, bh) >= 9 and ext < 0.5)
        if not long_enough:
            continue
        comp = (labels[stats[i, cv2.CC_STAT_TOP]:stats[i, cv2.CC_STAT_TOP] + bh,
                       stats[i, cv2.CC_STAT_LEFT]:stats[i, cv2.CC_STAT_LEFT] + bw] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perim = sum(cv2.arcLength(c, True) for c in cnts)
        if perim > 0 and 2.0 * a / perim < 2.8:   # thin stroke = drawn line
            keep[i] = True
    return np.where(keep[labels], 255, 0).astype(np.uint8)

def build_dem(topo, anchors, km_px):
    h, w = topo.shape[:2]
    cmask = contour_mask_clean(topo)
    pts = np.array([(a[0], a[1]) for a in anchors], float)
    vals = np.array([a[2] for a in anchors], float)
    rbf = RBFInterpolator(pts, vals, kernel="thin_plate_spline", smoothing=2.0)
    GW, GH = 322, int(round(322 * h / w))
    gx, gy = np.meshgrid(np.linspace(0, w - 1, GW), np.linspace(0, h - 1, GH))
    prior = rbf(np.stack([gx.ravel(), gy.ravel()], axis=1)).reshape(GH, GW)
    prior_full = cv2.resize(prior.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    # snap contour components to 10m multiples of the prior
    n, labels, stats, _ = cv2.connectedComponentsWithStats((cmask > 0).astype(np.uint8), 8)
    sp = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 30:
            continue
        ys, xs = np.where(labels == i)
        pv = prior_full[ys, xs]
        med = np.median(pv); snap = round(med / 10.0) * 10.0
        if pv.max() - pv.min() > 22 or abs(med - snap) > 6:
            continue
        step = max(1, len(xs) // max(6, stats[i, cv2.CC_STAT_AREA] // 40))
        for j in range(0, len(xs), step):
            sp.append((float(xs[j]), float(ys[j]), snap))
    print(f"  dem: {len(sp)} snapped contour pts")
    allp = np.array([(p[0], p[1]) for p in sp] + [(a[0], a[1]) for a in anchors], float)
    allv = np.array([p[2] for p in sp] + [a[2] for a in anchors], float)
    if len(allp) > 2600:
        rs = np.random.RandomState(7)
        idx = rs.choice(len(sp), 2600 - len(anchors), replace=False)
        allp = np.concatenate([allp[idx], allp[len(sp):]]); allv = np.concatenate([allv[idx], allv[len(sp):]])
    rbf2 = RBFInterpolator(allp, allv, kernel="thin_plate_spline", smoothing=1.0, neighbors=60)
    grid = rbf2(np.stack([gx.ravel(), gy.ravel()], axis=1)).reshape(GH, GW)
    grid = cv2.GaussianBlur(grid.astype(np.float32), (5, 5), 1.2)
    print(f"  dem range: {grid.min():.0f}..{grid.max():.0f} m")
    return grid, cmask

def find_blue_markers(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.morphologyEx(cv2.inRange(hsv, (100, 120, 100), (135, 255, 255)),
                         cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        x, y, w2, h2 = cv2.boundingRect(c)
        if a > 100 and 0.6 < w2 / max(h2, 1) < 1.7 and 14 < w2 < 40:
            out.append((x + w2 / 2, y + h2 / 2))
    return out

def _sim_affine(s, ox, oy):
    """scale+translate (no rotation) -> 2x3 affine [[s,0,ox],[0,s,oy]]."""
    return np.array([[s, 0.0, ox], [0.0, s, oy]], float)

def topo_to_overview_transform(topo, overview, manual, pairs=None, affine=None):
    """Return a 2x3 affine A mapping topo pixels -> overview pixels
    (ov = A @ [tx, ty, 1]). Supports rotation via the studio overlay (affine)."""
    if affine and len(affine) == 6:
        a = np.array(affine, float).reshape(2, 3)
        sc = float(np.hypot(a[0, 0], a[1, 0]))
        rot = float(np.degrees(np.arctan2(a[1, 0], a[0, 0])))
        print(f"  topo->overview (studio overlay): scale={sc:.4f} rot={rot:.1f}deg off=({a[0,2]:.1f},{a[1,2]:.1f})")
        return a
    if manual:
        return _sim_affine(*manual)
    if pairs and len(pairs) >= 2:
        # least-squares full similarity (scale+rotation+translation) from
        # user-clicked matching points [[tx,ty,ox,oy],...]
        src = np.array([[p[0], p[1]] for p in pairs], np.float32).reshape(-1, 1, 2)
        dst = np.array([[p[2], p[3]] for p in pairs], np.float32).reshape(-1, 1, 2)
        M, _ = cv2.estimateAffinePartial2D(src, dst)
        if M is not None:
            sc = float(np.hypot(M[0, 0], M[1, 0]))
            print(f"  topo->overview (from {len(pairs)} pairs): scale={sc:.4f}")
            return M.astype(float)
    mt = find_blue_markers(topo); mo = find_blue_markers(overview)
    best = None
    for (a1, a2) in itertools.permutations(mt, 2):
        for (b1, b2) in itertools.permutations(mo, 2):
            va = np.array(a2) - np.array(a1); vb = np.array(b2) - np.array(b1)
            la, lb = np.linalg.norm(va), np.linalg.norm(vb)
            if la < 30 or lb < 30:
                continue
            ang = abs(np.degrees(np.arctan2(va[1], va[0]) - np.arctan2(vb[1], vb[0])))
            s = lb / la
            if ang < 4 and 0.5 < s < 2.0:
                err = abs(la * s - lb)
                if best is None or err < best[0]:
                    off = np.array(b1) - s * np.array(a1)
                    best = (err, s, float(off[0]), float(off[1]))
    if best is None:
        raise SystemExit("no matching blue markers topo<->overview; use the studio overlay or add topo_affine/topo_to_overview to config")
    print(f"  topo->overview: scale={best[1]:.4f} off=({best[2]:.1f},{best[3]:.1f})")
    return _sim_affine(best[1], best[2], best[3])

def inpaint_blue(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.dilate(cv2.inRange(hsv, (100, 120, 100), (135, 255, 255)), np.ones((5, 5), np.uint8))
    return cv2.inpaint(img, m, 5, cv2.INPAINT_TELEA)

# ---------------------------------------------------------------- 4) buildings + vegetation

def shadow_dir_and_masks(mos):
    hsv = cv2.cvtColor(mos, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    gray = cv2.cvtColor(mos, cv2.COLOR_BGR2GRAY)
    valid = gray > 36
    # vegetation: excess-green index catches dark olive trees too
    b, g2, r = mos[..., 0].astype(np.int16), mos[..., 1].astype(np.int16), mos[..., 2].astype(np.int16)
    exg = 2 * g2 - r - b
    veg = ((exg > 18) | ((H > 35) & (H < 90) & (S > 60))) & valid
    # "shadow" run also counts visible facades: the imagery is slightly oblique,
    # so building walls show as dark/gray strips next to the roof — their length
    # is proportional to the number of floors
    vth = np.percentile(V[valid], 22)
    facade = (S < 70) & (V < np.percentile(V[valid], 40)) & ~veg
    shadow = (((V < vth) | facade) & valid)
    bth = np.percentile(V[valid], 32)
    built = ((V > bth) & (S < 100) & ~veg & valid).astype(np.uint8)
    # dense built fabric: fill roof interiors/gaps first, then trim thin
    # roads/paths — keep the urban block solid (watershed splits it later)
    bo = cv2.morphologyEx(built, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    bo = cv2.morphologyEx(bo, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    # shadow direction
    sh_u8 = shadow.astype(np.uint8)
    best = (None, -1)
    for dx, dy in [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]:
        M = np.float32([[1, 0, dx * 6], [0, 1, dy * 6]])
        s = cv2.warpAffine(bo, M, (bo.shape[1], bo.shape[0]))
        ov = float((s & sh_u8 & (bo == 0)).sum())
        if ov > best[1]:
            best = ((dx, dy), ov)
    print("  shadow dir:", best[0])
    return bo, veg.astype(np.uint8), sh_u8, best[0]

def shadow_run_field(shadow, ddir, maxrun=34):
    dx, dy = ddir
    Wd, H = shadow.shape[1], shadow.shape[0]
    sf = shadow.astype(np.float32)
    run = np.zeros_like(sf); cur = sf.copy()
    for i in range(1, maxrun + 1):
        M = np.float32([[1, 0, -dx * i], [0, 1, -dy * i]])
        cur = cv2.warpAffine(sf, M, (Wd, H)) * (cur > 0)
        run += (cur > 0)
    return run

def _separate_buildings(bo, mos):
    """Split the built mask into individual buildings via distance-transform
    watershed (respects roof-to-roof edges in the imagery)."""
    from scipy import ndimage
    dist = cv2.distanceTransform(bo, cv2.DIST_L2, 5)
    dist = cv2.GaussianBlur(dist, (0, 0), 2.0)
    # seeds = local maxima of the distance transform (one per building core).
    # size 15 -> ~one seed per building block, not per roof fragment
    peak = (ndimage.maximum_filter(dist, size=15) == dist) & (dist > 3.0)
    seeds, nseeds = ndimage.label(peak)
    markers = np.zeros(bo.shape, np.int32)
    markers[seeds > 0] = seeds[seeds > 0] + 1          # foreground seeds: 2..N+1
    markers[cv2.dilate(bo, np.ones((3, 3), np.uint8), iterations=2) == 0] = 1  # background = 1
    cv2.watershed(mos, markers)                          # floods along image edges
    markers[bo == 0] = 0                                 # keep only built pixels
    return markers, nseeds

def extract_buildings(mos, px_m, to_ov):
    bo, veg, shadow, ddir = shadow_dir_and_masks(mos)
    run = shadow_run_field(shadow, ddir)
    dx, dy = ddir
    Wd, H = mos.shape[1], mos.shape[0]
    labels, nseeds = _separate_buildings(bo, mos)
    ids = [i for i in np.unique(labels) if i >= 2]

    # --- pass 1: geometry + shadow evidence per building ---
    builds = []       # (poly_px Nx2, centroid, med_run)
    runs_all = []
    for i in ids:
        mask = (labels == i).astype(np.uint8)
        area = int(mask.sum())
        if area < 55:                                    # < ~7.5 m^2
            continue
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        # simplify to the real roof shape (hexagon / trapezoid / L / rectangle+bump)
        poly = cv2.approxPolyDP(c, 0.018 * peri, True).reshape(-1, 2).astype(np.float32)
        if len(poly) < 3:
            continue
        if len(poly) > 10:                               # noisy blob -> convex hull
            poly = cv2.convexHull(c).reshape(-1, 2).astype(np.float32)
        # drop slivers
        if cv2.contourArea(poly.astype(np.int32)) * px_m * px_m < 6:
            continue
        cx, cy = poly[:, 0].mean(), poly[:, 1].mean()
        # shadow/facade run just past the down-sun edge
        rvals = []
        for (x, y) in poly:
            for j in range(1, 10):
                nx, ny = int(x + dx * j), int(y + dy * j)
                if 0 <= nx < Wd and 0 <= ny < H and run[ny, nx] > 1:
                    rvals.append(run[ny, nx]); break
        med_run = float(np.median(rvals)) if rvals else 0.0
        builds.append((poly, (cx, cy), med_run))
        if med_run > 0:
            runs_all.append(med_run)

    global_med = float(np.median(runs_all)) if runs_all else 4.0
    print(f"  buildings: {len(builds)} footprints, median facade/shadow run {global_med:.1f}px")

    FLOOR_M = 3.1
    s, ox, oy = to_ov
    out = []
    for poly, (cx, cy), med_run in builds:
        if med_run > 0:
            floors = int(np.clip(round(med_run / global_med * 1.7), 1, 3))
        else:
            # no shadow evidence: 1-3 floors, deterministic from position
            floors = 1 + (int(cx * 7.0 + cy * 13.0)) % 3
        # per-building jitter so neighbours read as distinct volumes
        jitter = (((int(cx * 131 + cy * 57)) % 9) - 4) * 0.25    # +-1.0 m
        hgt = float(max(2.6, floors * FLOOR_M + 0.3 + jitter))
        pts_ov = [[round(float(px) * s + ox, 1), round(float(py) * s + oy, 1)] for px, py in poly]
        out.append({"poly": pts_ov, "h": round(hgt, 1)})
    print(f"  buildings out: {len(out)} prisms")
    return out, veg, shadow, ddir, run

def extract_vegetation(mos, veg, run, ddir, px_m, to_ov, max_items=3500):
    dx, dy = ddir
    Wd, H = mos.shape[1], mos.shape[0]
    v = cv2.morphologyEx(veg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(v, 8)
    items = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 25:
            continue
        cx, cy = cents[i]
        r_m = float(np.sqrt(area / np.pi) * px_m)
        # shadow next to it -> tall (tree)
        rv = 0.0
        for j in range(1, 8):
            nx, ny = int(cx + dx * j), int(cy + dy * j)
            if 0 <= nx < Wd and 0 <= ny < H and run[ny, nx] > 1:
                rv = float(run[ny, nx]); break
        is_tree = r_m > 1.5 and (rv > 2 or r_m > 2.6)
        if is_tree:
            hgt = float(np.clip(3.5 + rv * px_m * 1.2, 3.5, 9.0))
            items.append((cx, cy, min(r_m, 5.0), hgt, 1, area))
        else:
            items.append((cx, cy, min(r_m, 2.2), float(np.clip(0.8 + r_m * 0.4, 0.8, 2.0)), 0, area))
    items.sort(key=lambda t: -t[5])
    items = items[:max_items]
    s, ox, oy = to_ov
    out = [[round(c[0] * s + ox, 2), round(c[1] * s + oy, 2), round(c[2], 1),
            round(c[3], 1), c[4]] for c in items]
    ntree = sum(1 for c in out if c[4] == 1)
    print(f"  vegetation: {ntree} trees, {len(out) - ntree} shrubs")
    return out

# ---------------------------------------------------------------- 5) compose

def compose(outdir, overview, mosaic, to_ov, topo, t2o, roads, r2o, cmask,
            dem, km_px, bld, vegj, name):
    os.makedirs(outdir, exist_ok=True)
    OH, OW = overview.shape[:2]
    A = np.asarray(t2o, float)                       # topo px -> overview px (2x3 affine)
    t2o_scale = float(np.hypot(A[0, 0], A[1, 0]))    # similarity scale
    m_per_ovpx = (1000.0 / km_px) / t2o_scale
    TEXW = 4096; SF = TEXW / OW; TEXH = int(round(OH * SF))
    base = cv2.resize(overview, (TEXW, TEXH), interpolation=cv2.INTER_CUBIC)
    a = to_ov[0] * SF
    Mt = np.float32([[a, 0, to_ov[1] * SF], [0, a, to_ov[2] * SF]])
    warp = cv2.warpAffine(mosaic, Mt, (TEXW, TEXH), flags=cv2.INTER_LINEAR)
    mval = cv2.erode((cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY) > 20).astype(np.float32), np.ones((5, 5), np.uint8))
    wm = cv2.GaussianBlur(cv2.warpAffine(mval, Mt, (TEXW, TEXH)), (0, 0), 6)
    aerial = (base * (1 - wm[..., None]) + warp * wm[..., None]).astype(np.uint8)
    cv2.imencode(".jpg", aerial, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tofile(os.path.join(outdir, "texture_aerial.jpg"))

    T2W = 2048; SF2 = T2W / OW; T2H = int(round(OH * SF2))
    # topo -> texture frame = (overview px * SF2); M_tex = SF2 * A
    Mt = (SF2 * A).astype(np.float32)
    topo_t = cv2.warpAffine(topo, Mt, (T2W, T2H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imencode(".jpg", topo_t, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(os.path.join(outdir, "texture_topo.jpg"))
    cont_t = cv2.warpAffine(cmask, Mt, (T2W, T2H), flags=cv2.INTER_LINEAR)
    cv2.imencode(".png", cont_t)[1].tofile(os.path.join(outdir, "contours.png"))

    has_roads = False
    if roads is not None and r2o is not None:
        # warp roads to the full overview frame; coverage alpha marks where the
        # roads image actually lands (works with rotation, unlike a bbox rect)
        Mr = (SF2 * np.asarray(r2o, float)).astype(np.float32)
        roads_t = cv2.warpAffine(roads, Mr, (T2W, T2H), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
        cover = cv2.warpAffine(np.full(roads.shape[:2], 255, np.uint8), Mr, (T2W, T2H),
                               flags=cv2.INTER_NEAREST, borderValue=0)
        rgba = cv2.cvtColor(roads_t, cv2.COLOR_BGR2BGRA)
        rgba[..., 3] = cover
        cv2.imencode(".png", rgba)[1].tofile(os.path.join(outdir, "texture_roads.png"))
        has_roads = True

    th, tw = topo.shape[:2]
    GW, GH = 262, 196
    ovx, ovy = np.meshgrid(np.linspace(0, OW - 1, GW), np.linspace(0, OH - 1, GH))
    # invert A (topo->ov) to map overview grid points back to topo px
    A3 = np.vstack([A, [0, 0, 1]])
    Ainv = np.linalg.inv(A3)
    tx = Ainv[0, 0] * ovx + Ainv[0, 1] * ovy + Ainv[0, 2]
    ty = Ainv[1, 0] * ovx + Ainv[1, 1] * ovy + Ainv[1, 2]
    mx = (tx / tw * (dem.shape[1] - 1)).astype(np.float32)
    my = (ty / th * (dem.shape[0] - 1)).astype(np.float32)
    hg = cv2.remap(dem.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    json.dump(dict(gw=GW, gh=GH, z=[round(float(v), 1) for v in hg.ravel()]),
              open(os.path.join(outdir, "heights.json"), "w"))
    json.dump(dict(polys=bld), open(os.path.join(outdir, "buildings.json"), "w"))
    json.dump(dict(items=vegj), open(os.path.join(outdir, "vegetation.json"), "w"))
    json.dump(dict(ow=OW, oh=OH, m_per_px=m_per_ovpx, world_w=OW * m_per_ovpx,
                   world_h=OH * m_per_ovpx, has_roads=has_roads, place=name,
                   zmin=float(hg.min()), zmax=float(hg.max())),
              open(os.path.join(outdir, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("  wrote", outdir)

# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--site", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--config")
    ap.add_argument("--overview", help="filename of the overview aerial (default: smallest aerial)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "webapp", "data"))
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8")) if args.config else {}
    paths = [p for p in glob.glob(os.path.join(args.input, "*"))
             if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png", ".webp")]
    skip = cfg.get("skip", [])
    paths = [p for p in paths if os.path.basename(p) not in skip]
    print("images:", len(paths))
    print("[1/6] classify")
    topo_p, roads_p, aerial_ps = classify(paths)
    # explicit role overrides from the config (set by the admin studio)
    by_name = {os.path.basename(p): p for p in paths}
    if cfg.get("topo") in by_name:
        topo_p = by_name[cfg["topo"]]
    if cfg.get("roads") in by_name:
        roads_p = by_name[cfg["roads"]]
    elif cfg.get("roads") == "":
        roads_p = None
    if cfg.get("aerials"):
        aerial_ps = [by_name[n] for n in cfg["aerials"] if n in by_name]
    aerial_ps = [p for p in aerial_ps if p not in (topo_p, roads_p)]
    if topo_p is None or not aerial_ps:
        raise SystemExit("need a topo map and at least one aerial")
    ov_name = args.overview or cfg.get("overview")
    if ov_name and ov_name in by_name:
        ov_p = by_name[ov_name]
    else:
        ov_p = min(aerial_ps, key=lambda p: imread(p).shape[0] * imread(p).shape[1])
    part_ps = [p for p in aerial_ps if p != ov_p]
    print("  overview:", os.path.basename(ov_p))

    overview = inpaint_blue(imread(ov_p))
    topo = imread(topo_p)
    roads = imread(roads_p) if roads_p else None
    partials = [inpaint_blue(crop_watermark(imread(p))) for p in part_ps]

    print("[2/6] register partials")
    tm = register_partials(overview, partials)
    print("[3/6] mosaic")
    if partials:
        mosaic, to_ov = build_mosaic(partials, tm)
    else:
        mosaic, to_ov = overview.copy(), (1.0, 0.0, 0.0)

    print("[4/6] topo/DEM")
    km_px = detect_grid(topo)
    if km_px is None:
        km_px = cfg.get("km_px", 335.0)
        print("  grid not detected, using", km_px)
    else:
        print(f"  grid: {km_px:.0f} px/km ({1000/km_px:.2f} m/px)")
    anchors = cfg.get("anchors")
    if not anchors:
        print("  WARNING: no anchors in config -> flat DEM")
        dem = np.zeros((196, 262), np.float32)
        cmask = contour_mask_clean(topo)
    else:
        dem, cmask = build_dem(topo, anchors, km_px)
    t2o = topo_to_overview_transform(topo, imread(ov_p), cfg.get("topo_to_overview"),
                                     cfg.get("align_pairs"), cfg.get("topo_affine"))
    t2o_scale = float(np.hypot(t2o[0, 0], t2o[1, 0]))

    print("[5/6] buildings + vegetation")
    px_m = (1000.0 / km_px) / t2o_scale * to_ov[0]  # meters per mosaic px
    print(f"  mosaic GSD: {px_m:.3f} m/px")
    bld, veg, shadow, ddir, run = extract_buildings(mosaic, px_m, to_ov)
    vegj = extract_vegetation(mosaic, veg, run, ddir, px_m, to_ov)

    # roads layer transform (studio overlay affine, or legacy scale/translate)
    r2o = None
    if roads is not None:
        if cfg.get("roads_affine") and len(cfg["roads_affine"]) == 6:
            r2o = np.array(cfg["roads_affine"], float).reshape(2, 3)
        elif cfg.get("roads_transform"):
            r2o = _sim_affine(*cfg["roads_transform"])

    print("[6/6] compose")
    outdir = os.path.join(args.out, args.site)
    compose(outdir, overview, mosaic, to_ov, inpaint_blue(topo), t2o, roads,
            r2o, cmask, dem, km_px, bld, vegj, args.name)

    sites_p = os.path.join(args.out, "sites.json")
    sites = json.load(open(sites_p, encoding="utf-8")) if os.path.exists(sites_p) else []
    sites = [s for s in sites if s["id"] != args.site] + [dict(id=args.site, name=args.name)]
    with open(sites_p, "w", encoding="utf-8") as f:
        json.dump(sites, f, ensure_ascii=False, indent=1)
    print("done. sites:", [s["id"] for s in sites])

if __name__ == "__main__":
    main()
