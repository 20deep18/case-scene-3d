#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""粗定位 JS 文件的括号不平衡位置（足够用来找语法错误）。"""
import re
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "app/server/index.js")
text = path.read_text(encoding="utf-8")


def strip_line(line: str) -> str:
    """去掉字符串与注释（粗糙但够用）。"""
    out = []
    i = 0
    n = len(line)
    quote = None
    while i < n:
        ch = line[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "\"'`":
            quote = ch
            i += 1
            continue
        if ch == "/" and i + 1 < n and line[i + 1] == "/":
            break
        out.append(ch)
        i += 1
    return "".join(out)


depth = 0
start_line = None
for idx, raw in enumerate(text.splitlines(), start=1):
    line = strip_line(raw)
    depth += line.count("{") - line.count("}")
    depth += line.count("(") - line.count(")")
    if depth < 0 and start_line is None:
        start_line = idx
        print(f"第一次出现负深度：第 {idx} 行  {raw.strip()}")
    if depth == 0:
        print(f"在第 {idx} 行回到深度 0: {raw.strip()[:70]}")

print(f"\n最终深度 = {depth}")
