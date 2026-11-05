#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""打印 JS/TS 文件里每个函数块的开始/结束行，用于快速定位括号不平衡。"""
import sys
from pathlib import Path


def strip_line(line: str) -> str:
    out = []
    i = 0
    n = len(line)
    quote = None
    while i < n:
        ch = line[i]
        if quote is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ('"', "'", "`"):
            quote = ch
            i += 1
            continue
        if ch == "/" and i + 1 < n and line[i + 1] == "/":
            break
        out.append(ch)
        i += 1
    return "".join(out)


path = Path(sys.argv[1])
lines = path.read_text(encoding="utf-8").splitlines()
targets = set()
if len(sys.argv) > 2:
    for tok in sys.argv[2].split(","):
        targets.add(int(tok))

depth = 0
for n, raw in enumerate(lines, start=1):
    s = strip_line(raw)
    before = depth
    depth += s.count("{") - s.count("}")
    interesting = (
        n in targets
        or (before == 0 and depth == 1)
        or (before == 1 and depth == 0)
        or depth < 0
    )
    if interesting:
        print(f"{n:5d}  depth {before:3d} -> {depth:3d}   {raw.strip()[:76]}")
print(f"\n最终深度 = {depth}（非 0 说明大括号不平衡）")
