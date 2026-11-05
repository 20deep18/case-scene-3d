#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断：模型为什么「看不出房间」——量化离群浮尘（floaters）。

方法：读 point_cloud.ply 的高斯中心，比较
  · min/max 包围盒（现在 to_splat.py 用的就是这个 → 决定前端取景半径）
  · 1%/99% 分位包围盒（稳健包围盒）
并统计「落在稳健包围盒之外」的高斯占比与它们离开核心的距离。

用法:
    python tools/diag_floaters.py --ply <point_cloud.ply> [--ply ...]
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

PLY_TYPES = {
    "float": ("f", 4), "float32": ("f", 4),
    "double": ("d", 8), "float64": ("d", 8),
    "int": ("i", 4), "int32": ("i", 4),
    "uint": ("I", 4), "uint32": ("I", 4),
    "short": ("h", 2), "int16": ("h", 2),
    "ushort": ("H", 2), "uint16": ("H", 2),
    "char": ("b", 1), "int8": ("b", 1),
    "uchar": ("B", 1), "uint8": ("B", 1),
    "long": ("q", 8), "int64": ("q", 8),
    "ulong": ("Q", 8), "uint64": ("Q", 8),
}


def read_ply_positions(path: Path) -> tuple[np.ndarray, list[str], str]:
    """读 PLY 的顶点坐标（只关心 x/y/z，二进制小端）。"""
    with path.open("rb") as fh:
        header_lines: list[str] = []
        while True:
            line = fh.readline()
            if not line:
                raise ValueError("PLY 头部没有 end_header")
            text = line.decode("ascii", "replace").strip()
            header_lines.append(text)
            if text == "end_header":
                break
        fmt = "ascii"
        count = 0
        props: list[tuple[str, str]] = []
        in_vertex = False
        for text in header_lines:
            if text.startswith("format"):
                fmt = text.split()[1]
            elif text.startswith("element vertex"):
                count = int(text.split()[2])
                in_vertex = True
            elif text.startswith("element "):
                in_vertex = False
            elif text.startswith("property") and in_vertex:
                parts = text.split()
                props.append((parts[-1], parts[1]))

        head = header_lines[:0]
        del head
        names = [p[0] for p in props]
        if fmt == "ascii":
            data = np.loadtxt(fh, max_rows=count)
            idx = [names.index(n) for n in ("x", "y", "z")]
            return np.asarray(data)[:, idx], names, fmt

        struct_fmt = "<" + "".join(PLY_TYPES[p[1]][0] for p in props)
        size = struct.calcsize(struct_fmt)
        raw = fh.read(size * count)
        arr = np.frombuffer(raw, dtype=np.dtype([(f"f{i}", PLY_TYPES[p[1]][0]) for i, p in enumerate(props)]))
        xyz = np.stack([arr[f"f{names.index('x')}"], arr[f"f{names.index('y')}"], arr[f"f{names.index('z')}"]], axis=1)
        return xyz.astype(np.float64), names, fmt


def report(path: Path) -> dict:
    xyz, names, fmt = read_ply_positions(path)
    n = len(xyz)
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    p01, p99 = np.percentile(xyz, 1, axis=0), np.percentile(xyz, 99, axis=0)
    p05, p95 = np.percentile(xyz, 5, axis=0), np.percentile(xyz, 95, axis=0)

    # 稳健中心 = 中位数；稳健半径 = 95% 分位到中心的距离
    center = np.median(xyz, axis=0)
    dist = np.linalg.norm(xyz - center, axis=1)
    r95 = float(np.percentile(dist, 95))
    r99 = float(np.percentile(dist, 99))
    rmax = float(dist.max())

    outside = dist > max(r95 * 2.0, 1e-6)
    far = dist > max(r99 * 3.0, 1e-6)

    size_minmax = hi - lo
    size_p01p99 = p99 - p01
    size_p05p95 = p95 - p05

    print(f"\n=== {path.name} ===")
    print(f"  格式 {fmt}，属性 {len(names)} 个，高斯数 {n}")
    print(f"  min/max 包围盒尺寸   : {size_minmax[0]:8.2f} × {size_minmax[1]:8.2f} × {size_minmax[2]:8.2f} m"
          f"   半径(半对角) {float(np.linalg.norm(size_minmax) / 2):8.2f} m")
    print(f"  1%/99% 分位包围盒尺寸: {size_p01p99[0]:8.2f} × {size_p01p99[1]:8.2f} × {size_p01p99[2]:8.2f} m"
          f"   半径 {float(np.linalg.norm(size_p01p99) / 2):8.2f} m")
    print(f"  5%/95% 分位包围盒尺寸: {size_p05p95[0]:8.2f} × {size_p05p95[1]:8.2f} × {size_p05p95[2]:8.2f} m"
          f"   半径 {float(np.linalg.norm(size_p05p95) / 2):8.2f} m")
    print(f"  到中位中心的距离     : P50 {np.median(dist):.2f}  P95 {r95:.2f}  P99 {r99:.2f}  max {rmax:.2f} m")
    print(f"  离群（>2×P95）       : {int(outside.sum()):6d} 个（{outside.mean() * 100:5.2f}%）")
    print(f"  极远（>3×P99）       : {int(far.sum()):6d} 个（{far.mean() * 100:5.2f}%）")
    ratio = rmax / r95 if r95 > 0 else float("inf")
    print(f"  max/P95 比值         : {ratio:6.1f}×   {'← 明显有浮尘，取景会被撑爆' if ratio > 5 else '正常'}")
    return {
        "name": path.name, "count": n,
        "sizeMinMax": size_minmax.tolist(), "sizeP01P99": size_p01p99.tolist(),
        "r95": r95, "r99": r99, "rmax": rmax,
        "outsideRatio": float(outside.mean()), "ratio": ratio,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", nargs="+", required=True)
    args = ap.parse_args()
    for p in args.ply:
        path = Path(p)
        if not path.exists():
            print(f"[跳过] 不存在 {path}")
            continue
        report(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
