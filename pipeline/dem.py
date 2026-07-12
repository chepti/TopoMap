# -*- coding: utf-8 -*-
"""Build DEM from the topographic map.
1) Detect the 1km black grid -> meters per pixel.
2) Extract brown contour-line mask (for overlay + validation).
3) Interpolate elevation surface from control points read off the map
   (contour labels 120/180/200, spot height 167, wadi profile) via RBF.
4) Render hillshade preview for visual validation.
"""
import cv2, numpy as np, os, json
from scipy.interpolate import RBFInterpolator

W = os.path.dirname(os.path.abspath(__file__))
topo = cv2.imread(os.path.join(W, "topo.jpg"))
h, w = topo.shape[:2]

# ---- 1) grid detection: long straight dark lines
g = cv2.cvtColor(topo, cv2.COLOR_BGR2GRAY)
dark = (g < 100).astype(np.uint8)
col_frac = dark.mean(axis=0)   # fraction of dark pixels per column
row_frac = dark.mean(axis=1)
cols = [int(x) for x in np.where(col_frac > 0.5)[0]]
rows = [int(y) for y in np.where(row_frac > 0.5)[0]]
def collapse(v):
    out = []
    for x in v:
        if out and x - out[-1][-1] <= 3: out[-1].append(x)
        else: out.append([x])
    return [int(np.mean(gr)) for gr in out]
gcols, grows = collapse(cols), collapse(rows)
print("grid cols:", gcols, "rows:", grows)
spacings = np.diff(gcols).tolist() + np.diff(grows).tolist()
km_px = float(np.median(spacings)) if spacings else None
print("km spacing px:", spacings, "-> median", km_px)

# ---- 2) brown contour mask
hsv = cv2.cvtColor(topo, cv2.COLOR_BGR2HSV)
# brown contour lines: hue ~5-25, moderate sat, low-mid value
m1 = cv2.inRange(hsv, (5, 60, 60), (25, 220, 200))
# exclude thick orange roads (high sat + high value) and pink urban fill
roads_m = cv2.inRange(hsv, (10, 80, 150), (30, 255, 255))
roads_m = cv2.dilate(roads_m, np.ones((3, 3), np.uint8))
contour_mask = cv2.bitwise_and(m1, cv2.bitwise_not(roads_m))
cv2.imwrite(os.path.join(W, "contour_mask.png"), contour_mask)
print("contour pixels:", int(contour_mask.sum() / 255))

# ---- 3) control points (x, y, elevation[m]) in topo pixel coords
# read off the map: labels 200/180/120 (N slope), spot height 167 (S hill),
# wadi profile descending westward, urban slopes.
CP = [
    # wadi / road line (valley floor), east->west descent
    (640, 150, 112), (600, 152, 110), (560, 160, 108), (520, 150, 106),
    (480, 158, 104), (440, 130, 102), (400, 120, 100), (360, 115, 98),
    (320, 112, 96), (280, 110, 94), (231, 112, 92), (190, 115, 89),
    (150, 118, 86), (100, 122, 84), (50, 126, 82), (10, 130, 80),
    # north slope (rises fast): 120 contour, then 180, 200 labels
    (487, 123, 120), (560, 120, 122), (620, 110, 126), (430, 100, 118),
    (545, 72, 180), (560, 40, 195), (545, 25, 200), (620, 20, 205),
    (490, 40, 185), (430, 30, 185), (370, 25, 175), (300, 30, 165),
    (240, 40, 150), (180, 50, 140), (120, 55, 130), (60, 60, 122), (10, 60, 115),
    # mid north slope
    (480, 80, 160), (420, 70, 155), (360, 70, 145), (300, 75, 132),
    (240, 80, 118), (180, 85, 105), (120, 90, 96),
    # south bank / camp area, rising south from wadi
    (60, 180, 92), (120, 180, 98), (180, 175, 104), (240, 170, 108),
    (300, 165, 112), (360, 170, 116), (420, 185, 122), (480, 190, 126),
    (560, 200, 130), (630, 210, 132),
    (60, 240, 100), (120, 240, 106), (180, 235, 112), (240, 230, 118),
    (300, 230, 124), (360, 235, 130), (420, 245, 138), (480, 255, 146),
    (540, 265, 150), (620, 280, 148),
    # hill 167 (summit ring) and flanks
    (462, 356, 167), (430, 340, 160), (490, 370, 160), (430, 380, 158),
    (500, 330, 152), (380, 350, 148), (540, 360, 150), (462, 300, 140),
    # south / southwest
    (60, 300, 102), (120, 300, 108), (180, 300, 112), (240, 300, 118),
    (300, 310, 124), (350, 320, 132),
    (30, 360, 98), (100, 370, 102), (170, 380, 106), (240, 380, 112),
    (300, 400, 116), (380, 420, 130), (460, 425, 145), (560, 420, 148), (630, 420, 150),
    # west edge (Tulkarm urban, gentle)
    (10, 200, 88), (10, 260, 94), (10, 320, 96), (10, 400, 95),
]
pts = np.array([(p[0], p[1]) for p in CP], float)
vals = np.array([p[2] for p in CP], float)

rbf = RBFInterpolator(pts, vals, kernel="thin_plate_spline", smoothing=2.0)
GW, GH = 322, 217  # half-res grid
gx, gy = np.meshgrid(np.linspace(0, w - 1, GW), np.linspace(0, h - 1, GH))
grid = rbf(np.stack([gx.ravel(), gy.ravel()], axis=1)).reshape(GH, GW)
print("dem min/max:", grid.min(), grid.max())

np.save(os.path.join(W, "dem.npy"), grid)
json.dump(dict(km_px=km_px, w=w, h=h, gw=GW, gh=GH,
               zmin=float(grid.min()), zmax=float(grid.max())),
          open(os.path.join(W, "dem_meta.json"), "w"))

# ---- 4) hillshade preview
gzy, gzx = np.gradient(grid)
px_m = 1000.0 / km_px * (w / GW) if km_px else 10  # meters per grid cell
slope_x, slope_y = gzx / px_m, gzy / px_m
az, alt = np.radians(315), np.radians(45)
lx, ly, lz = np.sin(az)*np.cos(alt), -np.cos(az)*np.cos(alt), np.sin(alt)
norm = np.sqrt(slope_x**2 + slope_y**2 + 1)
shade = np.clip((-slope_x*lx - slope_y*ly + lz) / norm, 0, 1)
hs = (shade * 255).astype(np.uint8)
hs = cv2.resize(hs, (w, h))
# blend hillshade with topo + contour mask for validation
blend = cv2.addWeighted(topo, 0.45, cv2.cvtColor(hs, cv2.COLOR_GRAY2BGR), 0.55, 0)
cv2.imwrite(os.path.join(W, "dem_check.png"), blend)
print("meters/pixel:", 1000.0/km_px if km_px else "?")
print("done")
