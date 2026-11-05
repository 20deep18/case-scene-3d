"""把点云画成三视图散点（不带高斯盘），用来判断 VGGT 的几何是否成形。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def load_ply(path: Path):
    raw = path.read_bytes()
    end = raw.find(b"end_header")
    body = raw.find(b"\n", end) + 1
    head = raw[:end].decode("ascii", "replace")
    n = int([l for l in head.splitlines() if l.startswith("element vertex")][0].split()[-1])
    # ⚠️ 属性类型要照原样映射：颜色是 uchar（1 字节），一律当 f4 会让 itemsize 对不上，
    #    报 "buffer is smaller than requested size"（实测踩过）。
    props = [(l.split()[1], l.split()[-1]) for l in head.splitlines() if l.startswith("property")]
    dt = np.dtype([(nm, "<f4" if tname in ("float", "float32", "double") else "u1")
                   for tname, nm in props])
    buf = np.frombuffer(raw, dtype=dt, count=n, offset=body)
    return np.stack([buf["x"], buf["y"], buf["z"]], axis=1).astype(np.float64)

path = Path(sys.argv[1])
out = Path(sys.argv[2])
pts = load_ply(path)
pts = pts[np.isfinite(pts).all(axis=1)]
if len(pts) > 40000:
    sel = np.random.default_rng(0).choice(len(pts), 40000, replace=False)
    pts = pts[sel]
span = np.ptp(pts, axis=0)
print(f"{len(pts)} 点，跨度 {np.round(span, 2)}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")
pairs = [(0, 2, "X–Z（正视）"), (1, 2, "Y–Z（侧视）"), (0, 1, "X–Y（俯视）")]
for ax, (i, j, title) in zip(axes, pairs):
    ax.scatter(pts[:, i], pts[:, j], s=0.4, c="#0e6b8a", alpha=0.5, linewidths=0)
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=7)
fig.suptitle(f"VGGT 点云三视图 · {path.name} · {len(pts)} 点", fontsize=12)
fig.tight_layout()
fig.savefig(out, dpi=110)
print(f"已写出 {out}")
