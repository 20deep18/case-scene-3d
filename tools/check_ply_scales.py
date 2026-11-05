#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""直接检查 PLY 里 scale_* 字段的原始值与统计，确认尺度约定是否一致。"""
import struct
import sys
from pathlib import Path

import numpy as np


def read_ply_raw(path):
    with open(path, "rb") as f:
        header = b""
        while b"end_header" not in header:
            header += f.readline()
        lines = header.decode("ascii", "replace").splitlines()
        n = 0
        props = []
        in_vertex = False
        for ln in lines:
            if ln.startswith("element vertex"):
                n = int(ln.split()[-1])
                in_vertex = True
            elif ln.startswith("element"):
                in_vertex = False
            elif ln.startswith("property") and in_vertex:
                props.append(ln.split()[-1])
        raw = f.read()
    stride = len(props) * 4
    print(f"声明顶点 {n:,}  属性 {len(props)}  记录长度 {stride} 字节")
    print(f"文件剩余字节 {len(raw):,}  期望 {n*stride:,}  吻合: {len(raw) >= n*stride}")
    return props, np.frombuffer(raw[: n * stride], dtype=np.float32).reshape(n, len(props))


props, a = read_ply_raw(sys.argv[1])
idx = {p: i for i, p in enumerate(props)}

print()
print("=== scale_* 原始值（PLY 里实际存的数字）===")
for i in range(3):
    v = a[:, idx[f"scale_{i}"]]
    print(
        f"  scale_{i}: min {v.min():9.5f}  P1 {np.percentile(v,1):9.5f}  中位 {np.median(v):9.5f}  "
        f"P99 {np.percentile(v,99):9.5f}  max {v.max():9.5f}  负值数 {(v<0).sum()}"
    )
print()
print("=== 若按「线性尺度」解释 ===")
lin = a[:, [idx["scale_0"], idx["scale_1"], idx["scale_2"]]]
print(f"  中位最大尺度 {np.median(lin.max(1)):.5f} m   >0.1m: {(lin.max(1)>0.1).mean()*100:.1f}%")
print()
print("=== 若按「log 尺度」解释（再取 exp）===")
from numpy import exp

le = exp(lin)
print(f"  中位最大尺度 {np.median(le.max(1)):.5f} m   >0.1m: {(le.max(1)>0.1).mean()*100:.1f}%")
print()
print("=== 其它字段抽样是否合理 ===")
for name in ("x", "opacity", "f_dc_0", "rot_0"):
    if name in idx:
        v = a[:, idx[name]]
        print(f"  {name:9s} min {v.min():9.4f}  中位 {np.median(v):9.4f}  max {v.max():9.4f}")
