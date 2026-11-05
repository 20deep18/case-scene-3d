#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bug A 前置确认：--max-scale 到底作用在哪个空间？

文档里那张表说得很对，同一个 0.5 有三种含义：
    if scale_raw > 0.5            → 砍 exp(0.5)=1.65 m 以上（温和）
    if exp(scale_raw) > 0.5       → 砍真实尺度 > 0.5 m（最狠，铺墙铺地全死）
    if exp(scale_raw)*spatial > 0.5 → 与训练器口径对齐（正确）

这里把 PLY 里的 scale_0/1/2 读出来，直接打印分布与各阈值下的丢弃比例，
用数据回答「当前走的是哪一条」，而不是靠猜。
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

PLY = Path(sys.argv[1] if len(sys.argv) > 1 else
           "data/projects/case-20261003/recon/recon-muvf7pbo-k9nz1/train/point_cloud.ply")


def read_header(path: Path):
    with path.open("rb") as fh:
        raw = b""
        while b"end_header" not in raw:
            chunk = fh.read(1)
            if not chunk:
                raise SystemExit("PLY 头没有结束标记")
            raw += chunk
        body = fh.read()
    lines = raw.decode("ascii", "replace").splitlines()
    fmt = None
    props: list[str] = []
    count = 0
    cur = None
    elements: list[tuple[str, int]] = []
    for ln in lines:
        p = ln.split()
        if not p:
            continue
        if p[0] == "format":
            fmt = p[1]
        elif p[0] == "element":
            cur = p[1]
            elements.append((p[1], int(p[2])))
            if cur == "vertex":
                count = int(p[2])
        elif p[0] == "property":
            # ⚠️ 必须只收集 **vertex 元素** 的属性：PLY 头里可能还有 face 等元素，
            #    把它们的属性也算进来会让 dtype 步长错位，读出满屏 inf/NaN
            #    （第一版就踩了这个坑，导致 P50 = nan）。
            if cur == "vertex":
                props.append(p[-1])
    return fmt, props, count, body, elements


fmt, props, count, body, elements = read_header(PLY)
print(f"文件: {PLY.name}")
print(f"格式: {fmt}")
print(f"元素: {elements}")
print(f"vertex 属性数: {len(props)}")
print(f"属性: {','.join(props)}")

if fmt != "binary_little_endian":
    raise SystemExit("本探针只处理 binary_little_endian")

dtype = np.dtype([(p, "<f4") for p in props])
arr = np.frombuffer(body[: count * dtype.itemsize], dtype=dtype, count=count)
need = ["scale_0", "scale_1", "scale_2"]
missing = [k for k in need if k not in props]
if missing:
    raise SystemExit(f"缺少属性 {missing}")

raw = np.stack([arr[k].astype(np.float64) for k in need], axis=1)
# ⚠️ 关键更正：train_gsplat.save_ply 写的是 torch.exp(scales_log)，
#    所以 PLY 里的 scale_* **已经是线性尺度（米）**，不能再 exp 一次。
#    第一版这里多 exp 了一次，把「线性 > 6.9 m」误报成 50.9%，结论方向也带偏了。
s = raw
smax = s.max(axis=1)

print(f"\nPLY 的 scale_*（**线性尺度，米**；save_ply 已 exp 过）分位：")
for q in (50, 90, 95, 99, 99.5, 99.9, 100):
    print(f"  P{q:<5} = {np.percentile(smax, q):.4f} m")
print(f"  含 NaN 的行: {int(np.isnan(smax).sum())} / {count}")

# 训练器的钳位上限：_clamp_scales 用 log(prune_scale * spatial_scale) 夹 log 参数，
# 等价于线性上限 = prune_scale * spatial_scale。
TRAIN_CAP = 0.1 * 11.185  # prune_scale=0.1, spatial_scale=11.185（取自该任务日志）
print(f"\n训练器钳位上限（0.1 × spatial_scale）= {TRAIN_CAP:.4f} m")
print(f"导出脚本 --max-scale                  = 0.5 m   ← 两者差 {TRAIN_CAP/0.5:.2f} 倍")

print(f"\n各阈值下的丢弃比例：")
for label, thr in [
    ("线性 > 0.5（当前 Web 默认）", 0.5),
    ("线性 > 1.0", 1.0),
    (f"线性 > {TRAIN_CAP:.4f}（与训练器同源）", TRAIN_CAP),
    ("线性 > 1.5", 1.5),
    ("线性 > 999（不限制）", 999.0),
]:
    drop = float((smax > thr).mean())
    print(f"  {label:<40} 丢弃 {drop*100:5.1f}%  保留 {int(round((1-drop)*count)):>6}")


# 训练器的剪枝口径：0.1 * spatial_scale。spatial_scale ≈ 相机包围球半径×1.1
print("\n提示：训练器剪枝用 0.1 * spatial_scale，所以导出上限应当与它同源，")
print("      而不是硬编码 0.5 —— 这正是文档第四节的结论。")
