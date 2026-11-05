"""把家具盒子画到点云散点上（两视图），判断盒子有没有套住真实物体。"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d.art3d import Line3DCollection

sys.path.insert(0, "tools")


def load_splat_xyz(path: Path, limit: int = 80000):
    raw = path.read_bytes()
    n = len(raw) // 32
    f = np.frombuffer(raw[: n * 32], dtype="<f4").reshape(n, 8)
    pos = f[:, 0:3].astype(np.float64)
    if len(pos) > limit:
        idx = np.random.default_rng(0).choice(len(pos), limit, replace=False)
        pos = pos[idx]
    return pos


splat = Path(sys.argv[1])
furn = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
out = Path(sys.argv[3])

pts = load_splat_xyz(splat)
fr = furn["frame"]
o = np.asarray(fr["origin"], float)
u = np.asarray(fr["axisU"], float)
v = np.asarray(fr["axisV"], float)
w = np.asarray(fr["up"], float)
axes = np.stack([u, v, w], axis=1)

# 把点和盒子都转到"帧坐标"里画
local = (pts - o) @ axes
print(f"点 {len(pts)}  盒 {len(furn['objects'])}  帧坐标跨度 {np.round(np.ptp(local, axis=0), 2)}")

fig = plt.figure(figsize=(16, 7), facecolor="white")

# 左：俯视（u-v），右：正视（u-w）
for k, (i, j, title) in enumerate([(0, 1, "俯视 (u–v)"), (0, 2, "正视 (u–up)")]):
    ax = fig.add_subplot(1, 2, k + 1)
    ax.scatter(local[:, i], local[:, j], s=0.3, c="#9fb6c4", alpha=0.35, linewidths=0)
    for n, obj in enumerate(furn["objects"], 1):
        c = np.asarray(obj["center"], float) - o
        c = axes.T @ c
        s = np.asarray(obj["size"], float)
        lo = c - s / 2
        hi = c + s / 2
        # 在该视图上画矩形
        xs = [lo[i], hi[i], hi[i], lo[i], lo[i]]
        ys = [lo[j], lo[j], hi[j], hi[j], lo[j]]
        ax.plot(xs, ys, color="#d1622b", linewidth=1.4)
        ax.text(lo[i], hi[j], str(n), color="#d1622b", fontsize=9)
    ax.set_title(title + " · 橙框=家具盒", fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=7)

fig.suptitle(f"盒子 vs 点云 · {splat.name}", fontsize=12)
fig.tight_layout()
fig.savefig(out, dpi=110)
print("已写出", out)
