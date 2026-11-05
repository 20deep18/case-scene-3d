#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一条龙：视频 → 帧 → COLMAP → .splat → 结构 → 家具盒 → 家具级线稿。

为什么需要这条链：家具级线稿用的是 `.splat`（每个高斯坐在表面上，点足够密），
而不是 COLMAP 稀疏点云（只有特征点，家具表面几乎没有点 —— 实测过）。

用法:
    <python> tools/pipeline_furniture.py --video "厕所测试1 (1).mp4" --out output/bathroom/v1
可选项:
    --fps 3  --long-side 1600  --blur 70  --iterations 7000  --downscale 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / "pyruntime312" / "python.exe") if (ROOT / "pyruntime312" / "python.exe").exists() else sys.executable


def run(step: str, cmd: list[str] | str, *, timeout: int = 3600) -> tuple[bool, str]:
    t0 = time.time()
    # ⚠️ 传**字符串**时走 shell=True：让 cmd.exe 自己解析 `&&` 与引号。
    #    之前用 ["cmd","/c", "cd /d \"...\" && ..."]，Python 会给整串再包一层引号
    #    （MS C 运行时的转义规则 ≠ cmd 的），cmd 直接报"文件名语法不正确"（实测踩过）。
    use_shell = isinstance(cmd, str)
    print(f"\n===== {step} =====", flush=True)
    print("  " + (cmd if use_shell else " ".join(cmd)), flush=True)
    try:
        p = subprocess.run(cmd, shell=use_shell, cwd=str(ROOT), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  ✗ 超时（>{timeout}s）", flush=True)
        return False, "timeout"
    out = (p.stdout or "") + (p.stderr or "")
    tail = [ln for ln in out.splitlines() if ln.strip()][-12:]
    print("\n".join("  " + ln[:200] for ln in tail), flush=True)
    print(f"  → exit={p.returncode}  用时 {time.time() - t0:.1f}s", flush=True)
    return p.returncode == 0, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--long-side", type=int, default=1600)
    ap.add_argument("--blur", type=float, default=70.0)
    ap.add_argument("--iterations", type=int, default=7000)
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--grad-threshold", default="1e-4",
                    help="致密化梯度阈值（低纹理场景调低才有足够高斯：默认 1e-4）")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = out / "frames"
    colmap_ws = out / "colmap"
    train_dir = out / "train"
    splat = out / "model.splat"
    structure = out / "structure.json"
    furniture = out / "furniture.json"
    lineart = out / "lineart"
    log = {"video": args.video, "steps": []}

    # 1) 抽帧
    ok, _ = run("1/6 抽帧", [PY, "pipeline/extract_frames.py", "--video", args.video,
                             "--out", str(frames), "--fps", str(args.fps),
                             "--long-side", str(args.long_side), "--blur-threshold", str(args.blur)])
    log["steps"].append({"step": "frames", "ok": ok})
    if not ok:
        (out / "log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1

    # 2) COLMAP
    ok, _ = run("2/6 COLMAP 位姿", [PY, "pipeline/colmap_recon.py", "--images", str(frames),
                                    "--workspace", str(colmap_ws), "--matcher", "exhaustive",
                                    "--camera-model", "OPENCV"], timeout=5400)
    log["steps"].append({"step": "colmap", "ok": ok})
    if not ok:
        (out / "log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        return 2

    # 3) 训练 -> ply
    if not args.skip_train:
        # ⚠️ 训练必须走 tools\msvc-env.bat：它把 JIT 编译缓存指到工作区
        #    （默认的 %LOCALAPPDATA%\torch_extensions 在被拒的目录里 → PermissionError），
        #    并把 VS 的 cl.exe / ninja / CUDA 放进 PATH（缺一个 gsplat 就编不过）。
        inner = (f'cd /d "{ROOT}" && call tools\\msvc-env.bat && "{PY}" -u pipeline\\train_gsplat.py '
                 f'--data "{colmap_ws}" --out "{train_dir}" --images "{frames}" '
                 f'--iterations {args.iterations} --downscale {args.downscale} --sh-degree 0 '
                 f'--densify-grad-threshold {args.grad_threshold}')
        ok, _ = run("3/6 训练 .splat（gsplat）", inner, timeout=7200)
        log["steps"].append({"step": "train", "ok": ok})
        if not ok:
            (out / "log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
            return 3
    ply = train_dir / "point_cloud.ply"
    if not ply.exists():
        print(f"✗ 没有找到 {ply}")
        return 3

    # 4) ply -> splat
    ok, _ = run("4/6 转 .splat", [PY, "pipeline/to_splat.py", "--ply", str(ply), "--out", str(splat)])
    log["steps"].append({"step": "to_splat", "ok": ok})
    if not ok:
        (out / "log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        return 4

    # 5) 结构（提供曼哈顿系 + 房间盒）
    ok, _ = run("5/6 结构提取", [PY, "tools/scene_structure.py", "--model", str(colmap_ws / "sparse/0"),
                                 "--out", str(structure)])
    log["steps"].append({"step": "structure", "ok": ok})

    # 6) 家具盒 + 线稿
    if ok:
        ok2, _ = run("6/6a 家具盒", [PY, "tools/furniture_lines.py", "--splat", str(splat),
                                     "--structure", str(structure), "--out", str(furniture)])
        log["steps"].append({"step": "furniture", "ok": ok2})
        if ok2:
            ok3, _ = run("6/6b 线稿", [PY, "tools/render_lineart.py", "--structure", str(structure),
                                       "--objects", str(furniture), "--out", str(lineart),
                                       "--views", "2", "--elevation", "30"], timeout=1200)
            log["steps"].append({"step": "lineart", "ok": ok3})

    (out / "log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 汇总 ===")
    for s in log["steps"]:
        print(f"  {s['step']:10s} {'✅' if s['ok'] else '❌'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
