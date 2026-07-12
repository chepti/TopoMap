# -*- coding: utf-8 -*-
"""Compose final web-app assets in the overview frame.
Frame: overview extent (524x392 ov px, 2.9424 m/px -> 1542m x 1153m).
Outputs (to ./out): texture_aerial.jpg (4096w), texture_topo.jpg (2048w),
texture_roads.jpg (2048w), contours.png (2048w), heights.json, buildings.json, meta.json
"""
import cv2, numpy as np, os, json

W = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(W, "out")
os.makedirs(OUT, exist_ok=True)

def inpaint_blue(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (100, 120, 100), (135, 255, 255))
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    return cv2.inpaint(img, m, 5, cv2.INPAINT_TELEA)

overview = inpaint_blue(cv2.imread(os.path.join(W, "overview.jpg")))
mosaic = inpaint_blue(cv2.imread(os.path.join(W, "mosaic_full.jpg")))
topo = cv2.imread(os.path.join(W, "topo.jpg"))
roads = cv2.imread(os.path.join(W, "roads.jpg"))
cmask = cv2.imread(os.path.join(W, "contour_mask.png"), 0)
bpng = cv2.imread(os.path.join(W, "buildings.png"), 0)
bmeta = json.load(open(os.path.join(W, "buildings_meta.json")))
mmeta = json.load(open(os.path.join(W, "mosaic_meta.json")))
dem = np.load(os.path.join(W, "dem2.npy"))
dmeta = json.load(open(os.path.join(W, "dem_meta.json")))

OW, OH = 524, 392
M_PER_OVPX = (1000.0 / dmeta["km_px"]) / 1.013   # topo px 2.98m; ov = 1.013*topo
print("m per ov px:", M_PER_OVPX)

# transforms to overview frame
S_MOS = mmeta["to_overview"]["scale"]; MOX = mmeta["to_overview"]["ox"]; MOY = mmeta["to_overview"]["oy"]
S_TOPO, TOX, TOY = 1.013, 34.4, -22.6
S_ROAD, ROX, ROY = 0.55, 149.0, 76.6

TEXW = 4096
SF = TEXW / OW           # texture px per ov px
TEXH = int(round(OH * SF))
print("aerial texture:", TEXW, "x", TEXH)

# ---- aerial texture: overview upscaled + mosaic pasted with feather
base = cv2.resize(overview, (TEXW, TEXH), interpolation=cv2.INTER_CUBIC)
# mosaic -> texture affine: tex = (mos_px * S_MOS + (MOX,MOY)) * SF
a = S_MOS * SF
Mt = np.float32([[a, 0, MOX * SF], [0, a, MOY * SF]])
warp = cv2.warpAffine(mosaic, Mt, (TEXW, TEXH), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
mgray = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
mvalid = (mgray > 36).astype(np.float32)
mvalid = cv2.erode(mvalid, np.ones((5, 5), np.uint8))
wmask = cv2.warpAffine(mvalid, Mt, (TEXW, TEXH), flags=cv2.INTER_LINEAR)
wmask = cv2.GaussianBlur(wmask, (0, 0), 6)
aerial = (base * (1 - wmask[..., None]) + warp * wmask[..., None]).astype(np.uint8)
cv2.imwrite(os.path.join(OUT, "texture_aerial.jpg"), aerial, [cv2.IMWRITE_JPEG_QUALITY, 82])

# ---- topo texture (2048w)
T2W = 2048; SF2 = T2W / OW; T2H = int(round(OH * SF2))
a = S_TOPO * SF2
Mt = np.float32([[a, 0, TOX * SF2], [0, a, TOY * SF2]])
topo_t = cv2.warpAffine(topo, Mt, (T2W, T2H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
cv2.imwrite(os.path.join(OUT, "texture_topo.jpg"), topo_t, [cv2.IMWRITE_JPEG_QUALITY, 85])

# ---- roads texture
a = S_ROAD * SF2
Mt = np.float32([[a, 0, ROX * SF2], [0, a, ROY * SF2]])
roads_t = cv2.warpAffine(roads, Mt, (T2W, T2H), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
cv2.imwrite(os.path.join(OUT, "texture_roads.jpg"), roads_t, [cv2.IMWRITE_JPEG_QUALITY, 85])
# roads valid rect in UV (for shader masking)
rx0, ry0 = ROX / OW, ROY / OH
rx1, ry1 = (ROX + roads.shape[1] * S_ROAD) / OW, (ROY + roads.shape[0] * S_ROAD) / OH
roads_rect = [max(0, rx0), max(0, ry0), min(1, rx1), min(1, ry1)]

# ---- contours overlay (from topo frame)
a = S_TOPO * SF2
Mt = np.float32([[a, 0, TOX * SF2], [0, a, TOY * SF2]])
cont_t = cv2.warpAffine(cmask, Mt, (T2W, T2H), flags=cv2.INTER_LINEAR)
cv2.imwrite(os.path.join(OUT, "contours.png"), cont_t)

# ---- heights: resample DEM (topo-frame grid) into ov frame 262x196
GW, GH = 262, 196
ovx, ovy = np.meshgrid(np.linspace(0, OW - 1, GW), np.linspace(0, OH - 1, GH))
tx = (ovx - TOX) / S_TOPO
ty = (ovy - TOY) / S_TOPO
# dem grid is GHxGW over topo extent (dmeta w,h)
mx = (tx / dmeta["w"] * (dmeta["gw"] - 1)).astype(np.float32)
my = (ty / dmeta["h"] * (dmeta["gh"] - 1)).astype(np.float32)
hgrid = cv2.remap(dem.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
print("heights min/max:", float(hgrid.min()), float(hgrid.max()))
json.dump(dict(gw=GW, gh=GH,
               z=[round(float(v), 1) for v in hgrid.ravel()]),
          open(os.path.join(OUT, "heights.json"), "w"))

# ---- buildings: cells (mosaic frame) -> ov coords
CELL = bmeta["cell_px"]; UNIT = bmeta["unit"]
cells = []
ys, xs = np.where(bpng > 0)
for cy, cx in zip(ys, xs):
    h = float(bpng[cy, cx]) * UNIT
    mpx = (cx + 0.5) * CELL; mpy = (cy + 0.5) * CELL
    ox = mpx * S_MOS + MOX; oy = mpy * S_MOS + MOY
    if 0 <= ox < OW and 0 <= oy < OH:
        cells.append([round(ox, 1), round(oy, 1), round(h, 1)])
print("building cells:", len(cells))
json.dump(dict(size_m=CELL * 0.377, cells=cells), open(os.path.join(OUT, "buildings.json"), "w"))

json.dump(dict(ow=OW, oh=OH, m_per_px=M_PER_OVPX,
               world_w=OW * M_PER_OVPX, world_h=OH * M_PER_OVPX,
               roads_rect=roads_rect,
               place="נור א-שמס — טול כרם", zmin=float(hgrid.min()), zmax=float(hgrid.max())),
          open(os.path.join(OUT, "meta.json"), "w"), ensure_ascii=False)
print("done")
for f in os.listdir(OUT):
    print(f, os.path.getsize(os.path.join(OUT, f)))
