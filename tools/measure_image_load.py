#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测量训练实际加载的图像张量（用真实 loader + torch）。"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import train_gsplat  # noqa: E402

imgs_dir = Path("data/scene-big/images")
files = sorted(p for p in imgs_dir.iterdir() if p.is_file() and p.suffix.lower() in train_gsplat.IMAGE_SUFFIXES)
print(f"目录: {imgs_dir}")
print(f"找到 {len(files)} 张图")
print()
print("=== 用真实 loader 读前 6 张 ===")
for p in files[:6]:
    w, h, data = train_gsplat.load_image_rgb(p)
    t = train_gsplat.image_to_tensor(p, 3, torch, torch.device("cpu"))
    a = t.numpy()  # (3,H,W)
    print(
        f"  {p.name:20s} 原始 {w}x{h}  张量 {tuple(a.shape)}  "
        f"RGB 通道均值 {np.round(a.reshape(3, -1).mean(1), 3)}  整体均值 {a.mean():.3f}"
    )

print()
print("=== 对照：直接解 PNG（独立实现）看真实亮度 ===")
import struct
import zlib


def read_png_rgb(path):
    d = Path(path).read_bytes()
    pos = 8
    idat = bytearray()
    w = h = ct = None
    while pos < len(d):
        (ln,) = struct.unpack(">I", d[pos : pos + 4])
        tag = d[pos + 4 : pos + 8]
        chunk = d[pos + 8 : pos + 8 + ln]
        pos += 12 + ln
        if tag == b"IHDR":
            w, h, bd, ct, _, _, il = struct.unpack(">IIBBBBB", chunk)
        elif tag == b"IDAT":
            idat += chunk
        elif tag == b"IEND":
            break
    raw = zlib.decompress(bytes(idat))
    ch = {0: 1, 2: 3, 4: 2, 6: 4}[ct]
    stride = w * ch
    out = np.empty((h, stride), np.uint8)
    prev = np.zeros(stride, np.uint8)
    p = 0
    for y in range(h):
        ft = raw[p]; p += 1
        line = np.frombuffer(raw[p : p + stride], np.uint8).copy(); p += stride
        if ft == 1:
            for i in range(ch, stride):
                line[i] = (int(line[i]) + int(line[i - ch])) & 0xFF
        elif ft == 2:
            line = (line.astype(np.int16) + prev.astype(np.int16)).astype(np.uint8)
        elif ft == 3:
            for i in range(stride):
                left = int(line[i - ch]) if i >= ch else 0
                line[i] = (int(line[i]) + ((left + int(prev[i])) >> 1)) & 0xFF
        elif ft == 4:
            for i in range(stride):
                a = int(line[i - ch]) if i >= ch else 0
                b = int(prev[i])
                c = int(prev[i - ch]) if i >= ch else 0
                pp = a + b - c
                pa, pb, pc = abs(pp - a), abs(pp - b), abs(pp - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (int(line[i]) + pr) & 0xFF
        out[y] = line
        prev = line
    return out.reshape(h, w, ch)[..., :3]


for p in files[:3]:
    a = read_png_rgb(p)
    print(f"  {p.name:20s} shape {a.shape}  RGB 均值 {np.round(a.reshape(-1, a.shape[2]).mean(0), 1)}  整体 {a.mean():.1f}")
