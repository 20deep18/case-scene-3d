#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断「模型是一朵白云」：白色/超大高斯到底占多少，是不是它们在遮蔽有颜色的部分。

.splat 布局（32 字节/高斯，与 app/web/js/splat-loader.js 一致）：
    0..11   位置 3×float32
    12..23  尺度 3×float32（线性，米）
    24..27  颜色 R,G,B,A（4×uint8）
    28..31  旋转四元数（4×uint8）

要回答的问题：
  1) 是不是「白色」高斯占多数？
  2) 它们是不是**又大又淡**（大 = 糊、淡 = 叠起来变白雾）？
  3) 大高斯和白高斯是不是同一批（→ 遮蔽有颜色的细节）？
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

GAUSS_BYTES = 32


def load_splat(path: Path):
    raw = path.read_bytes()
    n = len(raw) // GAUSS_BYTES
    a = np.frombuffer(raw[: n * GAUSS_BYTES], dtype="<f4").reshape(n, 8)
    pos = a[:, 0:3].astype(np.float64)
    scale = a[:, 3:6].astype(np.float64)
    b = np.frombuffer(raw[: n * GAUSS_BYTES], dtype=np.uint8).reshape(n, GAUSS_BYTES)
    rgba = b[:, 24:28].astype(np.float64)
    return pos, scale, rgba


def label_for(path: Path, all_paths: list[Path]) -> str:
    """给表格用的短标签。

    ⚠️ 2026-10-06 修：原来直接用 `path.name`，于是**同名文件会打印成两行一样的名字** ——
    做背景分离 A/B 时我拿两个 `point_cloud.splat` 对比，表格里
    "point_cloud.splat / point_cloud.splat" 根本分不清哪行是哪个，
    只能靠顺序去猜（而顺序一错，结论就反了）。
    现在：只有当 basename 重复时才加上父目录名做区分，且长度自适应。
    """
    base = path.name
    if sum(1 for p in all_paths if p.name == base) <= 1:
        return base
    return f"{path.parent.name}/{base}"


def _fit(text: str, width: int) -> str:
    """截断到 width（按显示宽度粗略算：CJK 字符占 2 列），并补齐。"""
    def w(s: str) -> int:
        return sum(2 if ord(c) > 0x2E80 else 1 for c in s)

    out = text
    while out and w(out) > width:
        out = out[:-1]
    return out + " " * max(0, width - w(out))


def describe(path: Path) -> dict:
    pos, scale, rgba = load_splat(path)
    n = len(pos)
    smax = scale.max(axis=1)
    alpha = rgba[:, 3] / 255.0
    lum = rgba[:, :3].mean(axis=1)
    white = (rgba[:, :3].min(axis=1) > 200)          # 三通道都高 = 近白
    big = smax > 0.5
    big_white = big & white
    # 「雾」的判据：又大又偏白又偏淡 —— 这种高斯叠起来就是白纱
    foggy = big & white & (alpha < 0.6)
    # 明显有色（饱和度足够高）：这才是能让人认出物体的部分
    mx = rgba[:, :3].max(axis=1)
    mn = rgba[:, :3].min(axis=1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0)
    colorful = (sat > 0.25) & (alpha > 0.3)
    return {
        "name": path.name,
        "n": n,
        "scale_p50": float(np.percentile(smax, 50)),
        "scale_p95": float(np.percentile(smax, 95)),
        "scale_max": float(smax.max()),
        "big_pct": 100.0 * float(big.mean()),
        "white_pct": 100.0 * float(white.mean()),
        "bigWhite_pct": 100.0 * float(big_white.mean()),
        "foggy_pct": 100.0 * float(foggy.mean()),
        "colorful_pct": 100.0 * float(colorful.mean()),
        "alpha_p50": float(np.percentile(alpha, 50)),
        "lum_p50": float(np.percentile(lum, 50)),
        # 大高斯 vs 小高斯的「白度」对比 —— 若大高斯明显更白，就说明它们就是那朵云
        "lum_big": float(lum[big].mean()) if big.any() else float("nan"),
        "lum_small": float(lum[~big].mean()) if (~big).any() else float("nan"),
        "alpha_big": float(alpha[big].mean()) if big.any() else float("nan"),
        "alpha_small": float(alpha[~big].mean()) if (~big).any() else float("nan"),
    }


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        paths = sorted(Path("data/models").glob("*.splat"), key=lambda p: -p.stat().st_mtime)
    rows = []
    files: list[Path] = []
    for p in paths:
        if p.is_file():
            try:
                rows.append(describe(p))
                files.append(p)
            except Exception as exc:  # noqa: BLE001
                print(f"跳过 {p.name}: {exc}")
    if not rows:
        print("没有可分析的 .splat")
        return 1

    labels = [label_for(p, files) for p in files]
    name_w = max(28, max(len(l) for l in labels) + 2)

    print(f"{_fit('模型', name_w)}{'高斯':>7}{'尺度P50':>9}{'尺度P95':>9}{'>0.5m':>8}{'近白':>7}{'大+白':>7}{'雾':>7}{'有色':>7}")
    print("-" * (name_w + 61))
    for r, lab in zip(rows, labels):
        print(
            f"{_fit(lab, name_w)}{r['n']:>7}{r['scale_p50']:>9.3f}{r['scale_p95']:>9.3f}"
            f"{r['big_pct']:>7.0f}%{r['white_pct']:>6.0f}%{r['bigWhite_pct']:>6.0f}%"
            f"{r['foggy_pct']:>6.0f}%{r['colorful_pct']:>6.0f}%"
        )
    print("\n大高斯 vs 小高斯（若大高斯明显更白更淡，那朵云就是它们）：")
    print(f"{_fit('模型', name_w)}{'大·亮度':>10}{'小·亮度':>10}{'大·alpha':>10}{'小·alpha':>10}")
    print("-" * (name_w + 40))
    for r, lab in zip(rows, labels):
        print(
            f"{_fit(lab, name_w)}{r['lum_big']:>10.1f}{r['lum_small']:>10.1f}"
            f"{r['alpha_big']:>10.3f}{r['alpha_small']:>10.3f}"
        )
    print("\n（亮度 0~255；alpha 0~1）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
