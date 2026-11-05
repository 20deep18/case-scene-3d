#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""盒子去重：跨类别 3D NMS + 类别层次消解（专家 Q5 建议）。

为什么需要：同一片区域会被检出 bed / bunk bed / sofa 等多个类别，三个盒子摞在一起，
视觉上"重合体积很大"，也让人看不出到底识别成了什么。

做法（按专家建议）：
1. **类别层次消解**：`bunk bed` 抑制同区域的 `bed`（上下铺是床的一种，不该同时报两个）
2. **跨类别 3D IoU NMS**：按置信度降序，IoU > 阈值就抑制（不分类别）
3. 保留策略：IoU 相同看置信度；置信度相近时保留体积大的（专家建议）

用法：
  <python> tools/nms_boxes.py --boxes output/dorm/detected.json --out output/dorm/detected-nms.json \
      --iou 0.45
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# 类别层次：key 抑制 value（同区域）
HIERARCHY = {
    "bunk bed": ["bed"],
    "bed": ["sofa", "mattress"],
}


def box_volume(o: dict) -> float:
    return float(np.prod(np.maximum(np.asarray(o["size"], float), 1e-9)))


def iou3d(a: dict, b: dict) -> float:
    ca, sa = np.asarray(a["center"], float), np.asarray(a["size"], float)
    cb, sb = np.asarray(b["center"], float), np.asarray(b["size"], float)
    lo_a, hi_a = ca - sa / 2, ca + sa / 2
    lo_b, hi_b = cb - sb / 2, cb + sb / 2
    inter = np.maximum(0.0, np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b))
    vi = float(np.prod(inter))
    return vi / max(box_volume(a) + box_volume(b) - vi, 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iou", type=float, default=0.45, help="跨类别 3D IoU NMS 阈值（专家建议 0.45）")
    ap.add_argument("--iou-hierarchy", type=float, default=0.20,
                    help="类别层次消解的 IoU 阈值。要比 --iou 低：\n"
                         "实测 bed 与 sofa 在 3D 上只有 0.25~0.35 IoU（同一片区域被识成两类），\n"
                         "而 door 与 cabinet 的 0.28 是**相邻的两个真实物体**，用 0.45 的几何阈值才不会误合")
    ap.add_argument("--keep-votes", type=int, default=0, help="票数低于该值的先丢掉（0=不过滤）")
    args = ap.parse_args()

    data = json.loads(Path(args.boxes).read_text(encoding="utf-8"))
    objs = list(data.get("objects") or [])
    n0 = len(objs)
    if args.keep_votes:
        objs = [o for o in objs if int(o.get("votes") or 0) >= args.keep_votes]
    # 置信度降序；置信度相近（差 <0.02）时体积大的优先
    objs.sort(key=lambda o: (-round(float(o.get("score") or 0), 2), -box_volume(o)))
    print(f"输入 {n0} 个盒子" + (f"，按票数过滤后 {len(objs)} 个" if args.keep_votes else ""))

    kept: list[dict] = []
    dropped: list[tuple[str, str, float]] = []
    for o in objs:
        label = str(o.get("label") or "object")
        if any(label in HIERARCHY.get(k.get("label", ""), []) and iou3d(k, o) > args.iou_hierarchy
               for k in kept):
            sup = next(k["label"] for k in kept
                       if label in HIERARCHY.get(k["label"], []) and iou3d(k, o) > args.iou_hierarchy)
            dropped.append((label, f"被 {sup} 层次抑制（IoU>{args.iou_hierarchy}）", 0.0))
            continue
        hit = None
        for k in kept:
            v = iou3d(k, o)
            if v > args.iou:
                hit = (k, v)
                break
        if hit is not None:
            dropped.append((label, f"与最高分盒 IoU {hit[1]:.2f} > {args.iou}", hit[1]))
            continue
        kept.append(o)

    print(f"保留 {len(kept)} 个：")
    for o in kept:
        size = [round(float(v), 2) for v in o["size"]]
        print(f"  {str(o.get('label')):<10} {size}  置信 {o.get('score')}  票 {o.get('votes')}")
    if dropped:
        print(f"抑制 {len(dropped)} 个：")
        for label, why, _v in dropped[:8]:
            print(f"  {label:<10} {why}")

    data["objects"] = kept
    data["note"] = (data.get("note", "") +
                    f" | 已做跨类别 3D NMS（IoU>{args.iou}）+ 类别层次消解")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
