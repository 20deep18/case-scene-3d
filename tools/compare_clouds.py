#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把多个点云（.splat / .ply）跑同一条家具分割流水线，并排比较。

用途：路线 A（COLMAP CUDA 稠密 MVS）与路线 C（VGGT/DUSt3R）产出的 PLY，
要和现有的 3DGS `.splat` 比一比 —— 到底哪个能嵌出家具级的盒子。
不比较就没有判断依据，所以这个脚本是"决策脚本"，不是演示脚本。

用法:
    <python> tools/compare_clouds.py \
        --cloud 合成房=output/furniture/demo-room.splat:output/furniture/demo-room.structure.json \
        --cloud 厕所splat=output/bathroom/v2/model.splat:output/bathroom/v2/structure.json \
        --cloud 厕所ply=output/bathroom/v1/train-hd/point_cloud.ply:output/bathroom/v1/structure.json

格式: 名字=点云路径[:结构文件路径]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / "pyruntime312" / "python.exe") if (ROOT / "pyruntime312" / "python.exe").exists() else sys.executable


def run_one(cloud: Path, structure: Path | None) -> dict:
    out = Path(tempfile.gettempdir()) / f"cmp-{cloud.stem}-{abs(hash(str(cloud))) % 9999}.json"
    cmd = [PY, "-u", str(ROOT / "tools" / "furniture_lines.py"), "--splat", str(cloud), "--out", str(out)]
    if structure and structure.exists():
        cmd += ["--structure", str(structure)]
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=3600)
    log = (p.stdout or "") + (p.stderr or "")

    def grab(pattern: str, default=0, cast=int):
        m = re.search(pattern, log)
        return cast(m.group(1)) if m else default

    res = {
        "name": cloud.name,
        "kind": cloud.suffix.lower().lstrip("."),
        "total": grab(r"高斯基数: (\d+)"),
        "kept": grab(r"过滤后: (\d+)"),
        "shell": grab(r"剥壳.*?(\d+) / (\d+) 点留下"),          # 第一个捕获组
        "plane_left": grab(r"平面剥离：去掉 \d+ 点，剩下 (\d+) 点"),
        "clusters": grab(r"聚类: (\d+) 个候选簇"),
        "boxes": grab(r"找到 (\d+) 个家具级盒子"),
    }
    # 找到的盒子尺寸（用于算"占场景跨度多少"）
    boxes = re.findall(r"尺寸 \[([\d.]+), ([\d.]+), ([\d.]+)\]", log)
    res["box_sizes"] = [[float(x) for x in b] for b in boxes[:6]]
    if out.exists():
        try:
            data = json.loads(out.read_text(encoding="utf-8"))
            res["span"] = (data.get("params") or {}).get("span") or 0
        except Exception:
            res["span"] = 0
        out.unlink(missing_ok=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", action="append", required=True,
                    help="名字=点云路径[:结构文件路径]；可重复")
    args = ap.parse_args()

    rows = []
    for spec in args.cloud:
        name, _, rest = spec.partition("=")
        cloud_s, _, st_s = rest.partition(":")
        cloud = (ROOT / cloud_s) if not Path(cloud_s).is_absolute() else Path(cloud_s)
        st = None
        if st_s:
            st = (ROOT / st_s) if not Path(st_s).is_absolute() else Path(st_s)
        if not cloud.exists():
            print(f"✗ 找不到 {cloud}")
            rows.append({"name": name, "kind": "-", "total": 0, "kept": 0, "shell": 0,
                         "plane_left": 0, "clusters": 0, "boxes": 0, "box_sizes": [], "span": 0})
            continue
        r = run_one(cloud, st)
        r["name"] = name
        rows.append(r)

    print()
    hdr = f"{'点云':<14}{'格式':<7}{'总点':>8}{'过滤后':>9}{'剥壳后':>9}{'剥平面后':>10}{'簇':>5}{'盒':>5}{'最大盒/跨度':>12}"
    print(hdr)
    print("-" * len(hdr.encode("gbk", errors="replace")))
    for r in rows:
        span = r.get("span") or 0
        biggest = max((max(b) for b in r["box_sizes"]), default=0.0)
        ratio = f"{biggest / span:.2f}" if span else "-"
        print(f"{r['name'][:13]:<14}{r['kind']:<7}{r['total']:>8}{r['kept']:>9}"
              f"{r['shell']:>9}{r['plane_left']:>10}{r['clusters']:>5}{r['boxes']:>5}{ratio:>12}")

    print("\n判定标准（看最后一列 最大盒/跨度）：")
    print("  < 0.15  → 像家具（相对场景很小的一件东西）")
    print("  0.15~0.3 → 存疑（可能是大件家具，也可能是墙段）")
    print("  > 0.3   → 仍是房间级（说明表面没分成一个个物体）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
