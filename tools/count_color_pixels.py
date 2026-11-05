#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统计图片里「接近某个颜色」的像素数 —— 用于**像素级**验证叠加层真的画出来了。

为什么需要它：本项目没有 Three.js 渲染器，叠加层走 DOM/SVG。
「往场景里加了对象」这种断言根本证明不了用户能在屏幕上看到东西
（我踩过：测试 12/12 全过、截图里一条线都没有）。
所以验收必须是「画面里真的有这些颜色的像素」。

用法:
    python tools/count_color_pixels.py --image a.png --color "#4dd0e1" --tol 40
    python tools/count_color_pixels.py --image a.png --color "#4dd0e1,#ff8a80" --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def hex_to_rgb(text: str) -> tuple[int, int, int]:
    t = text.strip().lstrip("#")
    if len(t) == 3:
        t = "".join(c * 2 for c in t)
    return int(t[0:2], 16), int(t[2:4], 16), int(t[4:6], 16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--color", required=True, help="十六进制颜色，逗号分隔可多个")
    ap.add_argument("--tol", type=int, default=40, help="每通道容差（默认 40）")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出")
    args = ap.parse_args()

    try:
        from PIL import Image
    except ImportError:
        print("[错误] 需要 Pillow", file=sys.stderr)
        return 3

    img = Image.open(args.image).convert("RGB")
    px = img.load()
    w, h = img.size
    targets = [hex_to_rgb(c) for c in args.color.split(",") if c.strip()]
    counts = [0] * len(targets)
    tol = args.tol
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            for i, (tr, tg, tb) in enumerate(targets):
                if abs(r - tr) <= tol and abs(g - tg) <= tol and abs(b - tb) <= tol:
                    counts[i] += 1
    total = w * h
    out = {
        "image": str(Path(args.image).name),
        "size": [w, h],
        "totalPixels": total,
        "tol": tol,
        "counts": [
            {"color": args.color.split(",")[i].strip(), "pixels": counts[i],
             "ratio": round(counts[i] / max(1, total), 6)}
            for i in range(len(targets))
        ],
        "sum": sum(counts),
    }
    if args.json:
        print(json.dumps(out, ensure_ascii=False))
    else:
        print(f"{out['image']}  {w}×{h}")
        for c in out["counts"]:
            print(f"  {c['color']}: {c['pixels']} 像素（{c['ratio'] * 100:.4f}%）")
        print(f"  合计 {out['sum']} 像素")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
