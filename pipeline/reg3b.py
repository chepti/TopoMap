# -*- coding: utf-8 -*-
"""Place part3 by SIFT feature matching against already-placed neighbor partials.
Also verify pairwise relative scale (all partials should share the same GSD)."""
import cv2, numpy as np, os, json

W = os.path.dirname(os.path.abspath(__file__))
reg = json.load(open(os.path.join(W, "reg.json")))

def load_cropped(name):
    img = cv2.imread(os.path.join(W, name + ".jpg"))
    return img[:-30, :, :]

sift = cv2.SIFT_create(4000)
bf = cv2.BFMatcher()

p3 = load_cropped("part3")
g3 = cv2.cvtColor(p3, cv2.COLOR_BGR2GRAY)
k3, d3 = sift.detectAndCompute(g3, None)
print("part3 kp:", len(k3))

for name in ["part1", "part2", "part6", "part4", "part5"]:
    img = load_cropped(name)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k, d = sift.detectAndCompute(g, None)
    matches = bf.knnMatch(d3, d, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        print(name, "matches:", len(good), "- skip")
        continue
    src = np.float32([k3[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, inl = cv2.estimateAffinePartial2D(src, dst, cv2.RANSAC, ransacReprojThreshold=3)
    if M is None:
        print(name, "no model"); continue
    ninl = int(inl.sum())
    scale = float(np.hypot(M[0, 0], M[0, 1]))
    rot = float(np.degrees(np.arctan2(M[0, 1], M[0, 0])))
    # part3 origin (0,0) mapped into neighbor coords
    origin = M @ np.array([0.0, 0.0, 1.0])
    r = reg[name]
    # neighbor native -> overview: overview_xy = r.xy + native_xy * (r.w / native_w)
    nat_w = img.shape[1]
    k_ov = r["w"] / nat_w
    ov_x = r["x"] + origin[0] * k_ov
    ov_y = r["y"] + origin[1] * k_ov
    print(f"{name}: good={len(good)} inliers={ninl} rel_scale={scale:.4f} rot={rot:.2f}deg "
          f"-> part3 top-left in overview: ({ov_x:.1f}, {ov_y:.1f}), part3 scale_on_ov={k_ov*scale:.4f}")
