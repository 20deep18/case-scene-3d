#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""探针：柜子每个面在渲染画面里**最终可见多少像素**（水平面 vs 竖直面）。

为什么量这个
------------
层检测在 4 种拍摄几何的真实重建上都是 0 层，而"三块水平面各自 ±2% 高度内的点数占比"
只有 2.7~4.1%（均匀分布本来就该是 ~4%）—— 也就是说水平面上**基本没有点**。
要判断这是"SfM 不行"还是"水平面在画面里根本没几个像素"，就得量面积。

做法：与生成器同一套渲染（画家算法 + 背面剔除），但额外维护一个**索引缓冲**，
记录每个像素最后被哪个面覆盖 —— 这正是"该面最终可见的像素数"。
按面类型汇总后就能看出：水平面占了多少像素、每个面的像素密度够不够出特征点。

用法:
    pyruntime312\\python.exe tools\\probe_object_surface_area.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT / "tools",):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from make_object_dataset import build_cabinet, fit_radius  # noqa: E402
from make_textured_dataset import look_at, make_texture, quad_mask, warp_quad  # noqa: E402

# 面名 → 该面是不是水平面（水平面 = 法向沿重力方向，正是"层"要的东西）
HORIZONTAL = {"top", "bottom", "shelf"}


def base_name(panel_name: str) -> str:
    return panel_name.split("·")[0]


def main() -> int:
    ap_views = 36
    W, H = 960, 720
    fx = fy = 0.5 * W / np.tan(np.radians(35))
    Wm, Dm, Hm = 1.2, 0.6, 1.9
    panels, truth = build_cabinet(Wm, Dm, Hm)
    tex = {n: make_texture(256, 17000 + i) for i, (n, _, _) in enumerate(panels)}
    target = np.array([0.0, 0.0, Hm / 2])
    corners = np.array([[sx * Wm / 2, sy * Dm / 2, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (0, Hm)])

    # 两种典型机位：平视（elev=0）与俯仰（elev=0.8 + bias 0.25）
    for tag, elev_amp, elev_bias in (("平视", 0.0, 0.0), ("俯仰0.8+抬高0.25", 0.8, 0.25)):
        counts: dict[str, int] = {}
        total_px = 0
        for i in range(ap_views):
            ang = 2 * np.pi * i / ap_views
            z = target[2] + elev_amp * np.sin(2 * ang) + elev_bias
            r = fit_radius(ang, target, corners, fx, fy, W, H, z, margin=0.88, r_hi=24.0)
            if r is None:
                continue
            eye = np.array([r * np.cos(ang), r * np.sin(ang), z])
            R, t = look_at(eye, target, np.array([0.0, 0.0, 1.0]))
            canvas = np.zeros((H, W, 3), np.uint8)
            idx = np.full((H, W), -1, np.int16)          # 索引缓冲：最后覆盖该像素的面
            drawable = []
            for pi, (name, verts, normal) in enumerate(panels):
                c = np.mean(np.array(verts, float), axis=0)
                if np.dot(np.array(normal, float), eye - c) <= 0:
                    continue
                pr, ok = [], True
                for v in verts:
                    xc = R @ np.array(v, float) + t
                    if xc[2] <= 1e-6:
                        ok = False
                        break
                    pr.append((fx * xc[0] / xc[2] + W / 2, fy * xc[1] / xc[2] + H / 2))
                if not ok:
                    continue
                drawable.append((float(np.linalg.norm(eye - c)), pi, name, pr))
            drawable.sort(reverse=True)
            for _, pi, name, pr in drawable:
                xs = [p[0] for p in pr]
                ys = [p[1] for p in pr]
                bbox = (
                    max(0, int(np.floor(min(xs)))),
                    max(0, int(np.floor(min(ys)))),
                    min(W, int(np.ceil(max(xs))) + 1),
                    min(H, int(np.ceil(max(ys))) + 1),
                )
                if bbox[2] - bbox[0] < 2 or bbox[3] - bbox[1] < 2:
                    continue
                warp_quad(canvas, tex[name], pr, bbox)
                m = quad_mask(bbox, pr)
                sub = idx[bbox[1] : bbox[3], bbox[0] : bbox[2]]
                sub[m] = pi                              # 近处的面覆盖远处的面
            for pi, (name, _, _) in enumerate(panels):
                n = int((idx == pi).sum())
                counts[base_name(name)] = counts.get(base_name(name), 0) + n
            total_px += int((idx >= 0).sum())

        hor = sum(v for k, v in counts.items() if k in HORIZONTAL)
        ver = sum(v for k, v in counts.items() if k not in HORIZONTAL)
        print(f"\n[{tag}] 36 个视角合计：柜子占 {total_px:,} 像素"
              f"（每视角平均 {total_px // max(ap_views, 1):,}）")
        print(f"  水平面（顶/底/搁板）合计 {hor:,} 像素 = 柜子像素的 {100 * hor / max(total_px, 1):.1f}%")
        print(f"  竖直面（背/侧/门框）合计 {ver:,} 像素 = {100 * ver / max(total_px, 1):.1f}%")
        print("  逐面明细（像素占比）：")
        for k in sorted(counts, key=lambda x: -counts[x]):
            mark = "  ← 水平面" if k in HORIZONTAL else ""
            print(f"    {k:<10} {counts[k]:>10,}  {100 * counts[k] / max(total_px, 1):5.1f}%{mark}")

    print("\n对照：稀疏重建（1600×1200、32768 特征）里三块水平面 ±2% 高度内的点数占比 ~3~4%（= 均匀分布）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
