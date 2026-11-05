#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把工作区里**全部有源模型**的素材跑一遍结构提取，列出三态判定与证据。

用途：判据的阈值是**实测定的**，那就必须能一键复现这张表。
任何人（包括下一个接手的 AI）都能用它核对"为什么这份素材被判成 object/uncertain"。

用法:
    pyruntime312\\python.exe tools/diag_scene_mode_table.py [--json 输出路径]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT / "tools", ROOT / "pipeline"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import scene_structure as ss  # noqa: E402


def candidate_models() -> list[tuple[str, Path]]:
    """(标签, 稀疏模型目录)。标签尽量带上人可读的素材名。"""
    out: list[tuple[str, Path]] = []
    named = {
        "room-real": ROOT / "data" / "scene-big" / "colmap-ex" / "sparse" / "0",
    }
    for k, v in named.items():
        if (v / "images.bin").exists():
            out.append((k, v))
    # 案件里的重建任务（用 inputs.json 里的 name 当标签）
    for d in sorted(glob.glob(str(ROOT / "data" / "projects" / "*" / "recon" / "*"))):
        p = Path(d)
        inp = p / "inputs.json"
        sp = p / "colmap" / "sparse" / "0"
        if not (sp / "images.bin").exists() or not inp.exists():
            continue
        try:
            nm = json.loads(inp.read_text(encoding="utf-8")).get("name") or p.name
        except Exception:  # noqa: BLE001
            nm = p.name
        out.append((f"{nm}[{p.name[-6:]}]", sp))
    # 阈值实验的输出（真跑过的真实素材）
    for d in sorted(glob.glob(str(ROOT / "output" / "threshold*" / "run-*" / "colmap" / "sparse" / "0"))):
        p = Path(d)
        if (p / "images.bin").exists():
            out.append((f"{p.parts[-4]}/{p.parts[-3]}", p))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="把整张表写成 json")
    args = ap.parse_args()

    rows = []
    print("=" * 118)
    print(f"{'标签':<26} {'判定':<10} {'包围度':>7} {'(洗点后)':>8} {'墙':>3} {'相机高自洽':>9} {'sane':>5} {'层数':>4}  拒绝理由")
    print("=" * 118)
    for tag, d in candidate_models():
        t0 = time.time()
        try:
            data = ss.extract_structure(d)
        except Exception as exc:  # noqa: BLE001
            print(f"{tag:<26} [失败] {type(exc).__name__}: {exc}")
            continue
        ev = data.get("sceneModeEvidence") or {}
        sc = ev.get("surroundCoverage") or {}
        ob = data.get("object") or {}
        reasons = " / ".join((ev.get("rejectReasons") or [])[:2])
        rows.append({
            "tag": tag,
            "dir": str(d),
            "sceneMode": data.get("sceneMode"),
            "surround": sc.get("median"),
            "surroundAfterClean": sc.get("medianAfterOutlierRemoval"),
            "roomLikeWalls": ev.get("roomLikeWalls"),
            "camHeightSane": (ev.get("cameraHeight") or {}).get("sane"),
            "sane": data.get("sane"),
            "layers": len(ob.get("layers") or []),
            "reasons": ev.get("rejectReasons") or [],
            "rayMinSpreadRel": (ev.get("rayConvergence") or {}).get("minSpreadRel"),
            "elapsedSec": round(time.time() - t0, 2),
        })
        print(f"{tag:<26} {str(data.get('sceneMode')):<10} "
              f"{str(sc.get('median')):>7} {str(sc.get('medianAfterOutlierRemoval')):>8} "
              f"{str(ev.get('roomLikeWalls')):>3} {str((ev.get('cameraHeight') or {}).get('sane')):>9} "
              f"{str(data.get('sane')):>5} {len(ob.get('layers') or []):>4}  {reasons[:70]}")
    print("=" * 118)
    n_room = sum(1 for r in rows if r["sceneMode"] == "room")
    n_obj = sum(1 for r in rows if r["sceneMode"] == "object")
    n_unc = sum(1 for r in rows if r["sceneMode"] == "uncertain")
    print(f"合计 {len(rows)} 份素材：room {n_room} / object {n_obj} / uncertain {n_unc}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写出 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
