#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
极简 PNG 读写（只依赖 numpy + zlib），用于在没有 Pillow 的环境里查看/拼接图片。

支持：8 位、非隔行、灰度 / RGB / RGBA / 调色板之外的常见形式。
用途：把抽帧结果拼成一张对比图，快速判断素材可用性。

用法：
    pyruntime312\\python.exe tools/pngsheet.py --glob "data/scene-real/images/*.png" ^
        --out output/verify/sheet.png --cols 4 --tile 200
"""

import argparse
import glob
import os
import struct
import zlib

import numpy as np


def read_png(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("不是 PNG：" + path)
    pos = 8
    width = height = bit_depth = color_type = None
    idat = bytearray()
    palette = None
    trns = None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        ctype = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if ctype == b"IHDR":
            width, height, bit_depth, color_type, comp, filt, interlace = struct.unpack(">IIBBBBB", chunk)
            if bit_depth != 8:
                raise ValueError(f"只支持 8 位，实际 {bit_depth}")
            if interlace != 0:
                raise ValueError("不支持隔行 PNG")
        elif ctype == b"PLTE":
            palette = np.frombuffer(chunk, dtype=np.uint8).reshape(-1, 3)
        elif ctype == b"tRNS":
            trns = np.frombuffer(chunk, dtype=np.uint8)
        elif ctype == b"IDAT":
            idat += chunk
        elif ctype == b"IEND":
            break
    raw = zlib.decompress(bytes(idat))
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    stride = width * channels
    out = np.empty((height, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.uint8)
    p = 0
    for y in range(height):
        ft = raw[p]
        p += 1
        line = np.frombuffer(raw[p : p + stride], dtype=np.uint8).copy()
        p += stride
        if ft == 1:  # Sub
            for i in range(channels, stride):
                line[i] = (int(line[i]) + int(line[i - channels])) & 0xFF
        elif ft == 2:  # Up
            line = (line.astype(np.int16) + prev.astype(np.int16)).astype(np.uint8)
        elif ft == 3:  # Average
            for i in range(stride):
                left = int(line[i - channels]) if i >= channels else 0
                line[i] = (int(line[i]) + ((left + int(prev[i])) >> 1)) & 0xFF
        elif ft == 4:  # Paeth
            for i in range(stride):
                a = int(line[i - channels]) if i >= channels else 0
                b = int(prev[i])
                c = int(prev[i - channels]) if i >= channels else 0
                pp = a + b - c
                pa, pb, pc = abs(pp - a), abs(pp - b), abs(pp - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (int(line[i]) + pr) & 0xFF
        out[y] = line
        prev = line
    img = out.reshape(height, width, channels)
    if color_type == 3 and palette is not None:
        img = palette[img[:, :, 0]]
    elif channels == 1:
        img = np.repeat(img, 3, axis=2)
    elif channels == 2:
        img = np.repeat(img[:, :, :1], 3, axis=2)
    elif channels == 4:
        img = img[:, :, :3]
    return img


def write_png(path, arr):
    h, w, _ = arr.shape

    def chunk(tag, payload):
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))
    blob = b"\x89PNG\r\n\x1a\n"
    blob += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    blob += chunk(b"IDAT", zlib.compress(raw, 6))
    blob += chunk(b"IEND", b"")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", default="output/verify/sheet.png")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--tile", type=int, default=200)
    ap.add_argument("--max", type=int, default=24)
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))[: args.max]
    if not files:
        print("没有匹配到文件：" + args.glob)
        return 1
    tiles = []
    for f in files:
        img = read_png(f)
        h = img.shape[0]
        step = max(1, int(np.ceil(h / args.tile)))
        small = img[::step, ::step]
        tiles.append((os.path.basename(f), small))
        print(f"{os.path.basename(f):20s} {img.shape[1]}x{img.shape[0]}  亮度 {img.mean():6.1f}")
    cols = max(1, args.cols)
    rows = []
    for i in range(0, len(tiles), cols):
        chunk = tiles[i : i + cols]
        hh = min(t.shape[0] for _, t in chunk)
        ww = min(t.shape[1] for _, t in chunk)
        cells = [t[:hh, :ww] for _, t in chunk]
        # ⚠️ 最后一行常常不足 cols 个（实测 8 张图配 --cols 3 必崩）：
        #    这一行会比别的行**窄**，于是 `np.concatenate(rows, axis=0)` 直接抛
        #    "all the input array dimensions except for the concatenation axis
        #     must match exactly"。补黑块补齐即可 —— 黑块也直观地表示"这里没有图了"。
        while len(cells) < cols:
            cells.append(np.zeros((hh, ww, 3), dtype=cells[0].dtype))
        rows.append(np.concatenate(cells, axis=1))
    sheet = np.concatenate(rows, axis=0)
    write_png(args.out, sheet)
    print(f"\n已写出 {args.out}（{sheet.shape[1]}x{sheet.shape[0]}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
