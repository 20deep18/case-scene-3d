#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""评测台：把「盒子」与**真值**比，输出 L1/L2 精度指标。

为什么需要它（专家 P0）：
  没有真值，"改进了多少"只能靠眼睛；而评审要的是**可复现的数字**。
  所以这里固定三组指标，每次改动都跑同一个脚本，数字直接可比。

指标：
  · 中心误差（米）      —— 盒子中心与真值中心的距离
  · 尺寸相对误差（%）    —— 长/宽/高逐项（按大小排序后配对，避免轴顺序影响）
  · 尺寸 IoU            —— 体积交并比（能反映"位置+大小都对"）
  · 召回/误检           —— 真值里几件被找到、多了几个假盒子

真值格式（JSON）：
  {
    "scene": "宿舍",
    "scaleToMeters": 0.0925,        // 重建单位 → 米（用已知尺寸标定出来的）
    "room": {"size": [2.0, 0.8, 2.8]},         // 可选：房间盒真值（米）
    "items": [
      {"name": "门", "size": [0.8, 0.8, 2.0], "tolerance": 0.15},
      {"name": "上下铺", "size": [2.0, 0.8, 1.8], "tolerance": 0.2}
    ]
  }

用法:
  # L1：合成场景（盒子 json + 真值 json）
  <python> tools/benchmark_boxes.py --boxes output/furniture/demo-room-furniture.json \
      --truth output/furniture/demo-room-truth.json --mode synthetic

  # L2：真实场景（结构/家具输出 + 米制真值 + 标定系数）
  <python> tools/benchmark_boxes.py --boxes output/dorm/v1/furniture.json \
      --structure output/dorm/v1/structure.json \
      --truth docs/truth/dorm.json --mode real
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def box_size_from_corners(corners: list) -> np.ndarray:
    c = np.asarray(corners, float)
    return c.max(axis=0) - c.min(axis=0)


def vol(size: np.ndarray) -> float:
    return float(np.prod(np.maximum(size, 1e-9)))


def iou_axis_aligned(center_a, size_a, center_b, size_b) -> float:
    """两个轴对齐盒的 3D IoU（忽略朝向差异，按同名轴配对）。"""
    a_lo, a_hi = np.asarray(center_a) - np.asarray(size_a) / 2, np.asarray(center_a) + np.asarray(size_a) / 2
    b_lo, b_hi = np.asarray(center_b) - np.asarray(size_b) / 2, np.asarray(center_b) + np.asarray(size_b) / 2
    inter = np.maximum(0.0, np.minimum(a_hi, b_hi) - np.maximum(a_lo, b_lo))
    vi = float(np.prod(inter))
    return vi / max(vol(np.asarray(size_a)) + vol(np.asarray(size_b)) - vi, 1e-12)


def sorted_rel_error(size_a, size_b) -> np.ndarray:
    a, b = np.sort(np.asarray(size_a, float)), np.sort(np.asarray(size_b, float))
    return np.abs(a - b) / np.maximum(b, 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", required=True, help="furniture.json（含 objects[]）")
    ap.add_argument("--truth", required=True, help="真值 json")
    ap.add_argument("--structure", default=None, help="structure.json（可选，用来评估房间盒）")
    ap.add_argument("--mode", choices=["synthetic", "real"], default="real")
    ap.add_argument("--scale", type=float, default=None, help="重建单位→米；不给就用真值里的 scaleToMeters")
    args = ap.parse_args()

    boxes = load(args.boxes)
    truth = load(args.truth)
    scale = args.scale or float(truth.get("scaleToMeters", 1.0) or 1.0)

    objs = [o for o in (boxes.get("objects") or []) if o.get("size")]
    print(f"场景：{truth.get('scene', Path(args.truth).stem)}（{args.mode}）  标定 {scale:.4f} m/单位")
    _items = truth.get("items") or truth.get("furniture") or []
    print(f"检出盒子 {len(objs)} 个；真值 {len(_items)} 件\n")

    # ---- 逐件匹配（按体积最接近的贪心匹配）------------------------------
    # 两种真值写法都认：合成房用 furniture[]，真实场景用 items[]
    items = truth.get("items") or truth.get("furniture") or []
    used = set()
    rows = []
    for it in items:
        t_size = np.asarray(it["size"], float)          # 真值已是米
        best, best_score = None, 1e9
        for k, o in enumerate(objs):
            if k in used:
                continue
            size_m = np.asarray(o["size"], float) * scale
            score = float(np.sum(sorted_rel_error(size_m, t_size)))
            if score < best_score:
                best, best_score = k, score
        if best is None:
            rows.append({"name": it["name"], "found": False})
            continue
        used.add(best)
        o = objs[best]
        size_m = np.asarray(o["size"], float) * scale
        center_m = np.asarray(o["center"], float) * scale
        rel = sorted_rel_error(size_m, t_size)
        rows.append({
            "name": it["name"], "found": True,
            "truth": t_size.tolist(), "measured": size_m.tolist(),
            "relErr": (rel * 100).tolist(), "relErrMax": float(rel.max() * 100),
            "iou": iou_axis_aligned([0, 0, 0], size_m, [0, 0, 0], t_size),
            "tolerance": it.get("tolerance"),
            "points": o.get("points"),
        })

    # ---- 输出 --------------------------------------------------------
    hdr = f"{'真值':<12}{'真值尺寸(m)':<22}{'测得(m)':<22}{'逐项误差%':<18}{'IoU':>6}{'判定':>6}"
    print(hdr)
    print("-" * 96)
    n_ok = 0
    for r in rows:
        if not r["found"]:
            print(f"{r['name']:<12}{'—':<22}{'未检出':<22}{'—':<18}{'—':>6}{'✗':>6}")
            continue
        tol = r.get("tolerance") or 0.2
        ok = r["relErrMax"] <= tol * 100
        n_ok += int(ok)
        t = " ".join(f"{v:5.2f}" for v in r["truth"])
        m = " ".join(f"{v:5.2f}" for v in r["measured"])
        e = " ".join(f"{v:5.1f}" for v in r["relErr"])
        print(f"{r['name']:<12}{t:<22}{m:<22}{e:<18}{r['iou']:6.2f}{('✓' if ok else '✗'):>6}")

    print()
    found = sum(1 for r in rows if r["found"])
    print(f"召回：{found}/{len(items)}")
    print(f"达标（误差 ≤ 容差）：{n_ok}/{found if found else 1}")
    print(f"多检出的盒子：{len(objs) - len(used)} 个")

    # ---- 房间盒（可选）----------------------------------------------
    if args.structure:
        st = load(args.structure)
        size_m = np.asarray(st.get("room", {}).get("size") or [], float) * scale
        if size_m.size == 3:
            print(f"\n房间盒：{np.round(size_m, 2)} m"
                  f"（estimated={st.get('estimated')}, 墙 {st.get('roomLikeWalls')} 面）")
            room_truth = truth.get("room")
            room_size = None
            if isinstance(room_truth, dict):
                room_size = room_truth.get("size")
            elif isinstance(room_truth, (list, tuple)):
                room_size = room_truth
            if room_size:
                t = np.asarray(room_size, float)
                rel = sorted_rel_error(size_m, t)
                print(f"  真值 {np.round(t, 2)} m   逐项误差 {np.round(rel * 100, 1)}%")

    # ---- 结论（给评审看的口径）---------------------------------------
    if args.mode == "synthetic":
        print("\nL1（相对精度）：中心误差 <4 cm、尺寸 IoU >0.8 视为达标")
    else:
        print("\nL2（绝对精度）：报告 90% 分位误差，且必须注明采集规范与已标定比例尺")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
