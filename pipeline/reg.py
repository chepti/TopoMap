# -*- coding: utf-8 -*-
"""Register partial aerial photos onto the overview aerial via multi-scale template matching."""
import cv2, numpy as np, json, os

W = os.path.dirname(os.path.abspath(__file__))

def load(name):
    img = cv2.imread(os.path.join(W, name))
    if img is None:
        raise SystemExit("missing " + name)
    return img

overview = load("overview.jpg")
oh, ow = overview.shape[:2]

# Work on an upscaled overview for sub-pixel-ish precision
UP = 3
big = cv2.resize(overview, (ow*UP, oh*UP), interpolation=cv2.INTER_CUBIC)
big_gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

def prep(g):
    # normalize illumination differences: use gradient magnitude
    g = cv2.GaussianBlur(g, (3,3), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return cv2.magnitude(gx, gy)

big_feat = prep(big_gray)

results = {}
for name in ["part1","part2","part3","part4","part5","part6"]:
    img = load(name + ".jpg")
    # crop google watermark / attribution bar
    img = img[:-30, :, :]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    best = None
    for scale in np.arange(0.06, 0.60, 0.01):
        tw, th = int(img.shape[1]*scale*UP/UP), int(img.shape[0]*scale)
        # scale relative to the UPscaled overview
        sw = int(img.shape[1]*scale)
        sh = int(img.shape[0]*scale)
        if sw < 40 or sh < 40 or sw >= big.shape[1] or sh >= big.shape[0]:
            continue
        templ = cv2.resize(g, (sw, sh), interpolation=cv2.INTER_AREA)
        tf = prep(templ)
        res = cv2.matchTemplate(big_feat, tf, cv2.TM_CCOEFF_NORMED)
        _, mx, _, ml = cv2.minMaxLoc(res)
        if best is None or mx > best[0]:
            best = (float(mx), float(scale), int(ml[0]), int(ml[1]), sw, sh)
    score, scale, x, y, sw, sh = best
    # position in original overview coordinates
    results[name] = dict(score=round(score,3), scale_on_big=scale,
                         x=x/UP, y=y/UP, w=sw/UP, h=sh/UP)
    print(name, results[name])

with open(os.path.join(W, "reg.json"), "w") as f:
    json.dump(results, f, indent=1)

# preview: draw rectangles on overview
prev = overview.copy()
colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255),(255,0,255),(255,255,0)]
for i, (name, r) in enumerate(results.items()):
    p1 = (int(r["x"]), int(r["y"])); p2 = (int(r["x"]+r["w"]), int(r["y"]+r["h"]))
    cv2.rectangle(prev, p1, p2, colors[i], 2)
    cv2.putText(prev, name, (p1[0]+3, p1[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)
cv2.imwrite(os.path.join(W, "reg_preview.png"), prev)
print("done")
