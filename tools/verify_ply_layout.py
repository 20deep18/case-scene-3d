#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
严格校验 3DGS PLY 的字节布局：
  1) 精确定位 end_header 的字节偏移，用它推算数据起点
  2) 按结构 dtype 直接解析，而不是手工 reshape
  3) 逐字段检查分布是否「各自不同」（若各字段统计量雷同，说明读偏了）

用法: python tools/verify_ply_layout.py <a.ply> [b.ply]
"""
import sys
from pathlib import Path

import numpy as np


def parse(path):
    data = Path(path).read_bytes()
    # 关键：用字节查找，而不是 readline，避免行尾/编码差异导致偏移错位
    marker = b"end_header"
    i = data.find(marker)
    if i < 0:
        raise SystemExit(f"{path}: 找不到 end_header")
    # 跳过 end_header 之后的行尾符（\n 或 \r\n）
    j = i + len(marker)
    if data[j : j + 2] == b"\r\n":
        j += 2
    elif data[j : j + 1] in (b"\n", b"\r"):
        j += 1
    header = data[:i].decode("ascii", "replace")
    body = data[j:]
    print(f"--- {path} ---")
    print(f"end_header 字节偏移 {i}   数据起点 {j}   数据字节 {len(body):,}")

    lines = [ln.strip() for ln in header.splitlines() if ln.strip()]
    props = []
    n = None
    for ln in lines:
        if ln.startswith("element vertex"):
            n = int(ln.split()[-1])
            in_v = True
        elif ln.startswith("element"):
            in_v = False
        elif ln.startswith("property") and in_v:
            parts = ln.split()
            props.append((parts[-1], parts[1]))
    print(f"顶点 {n:,}  属性 {len(props)}")
    dt = np.dtype([(name, "<f4") for name, _ in props])
    print(f"单条记录 {dt.itemsize} 字节   期望总字节 {n*dt.itemsize:,}")
    ok = len(body) >= n * dt.itemsize
    print(f"字节数吻合: {ok}")
    if not ok:
        print("  ✘ 数据不完整，文件可能被截断")
    arr = np.frombuffer(body[: n * dt.itemsize], dtype=dt)

    # 逐字段统计：正常数据里，位置/尺度/颜色/旋转的分布应当**明显不同**
    print()
    print("=== 各字段统计（若都雷同 → 解析错位）===")
    for name in props[:1] + [p for p in props if p[0] in ("x", "y", "z", "scale_0", "opacity", "f_dc_0", "rot_0")]:
        v = arr[name[0]]
        print(
            f"  {name[0]:10s} min {v.min():9.4f}  P1 {np.percentile(v,1):9.4f}  "
            f"中位 {np.median(v):9.4f}  P99 {np.percentile(v,99):9.4f}  max {v.max():9.4f}"
        )
    print()
    # 位置字段的实际跨度
    for c in ("x", "y", "z"):
        v = arr[c]
        print(f"  {c}: 跨度 {v.max()-v.min():8.3f} m   有限值 {np.isfinite(v).all()}")
    return arr, props


for p in sys.argv[1:]:
    parse(p)
    print()
