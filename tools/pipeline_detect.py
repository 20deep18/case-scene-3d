#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一条命令：视频 → VGGT（点云/位姿/深度）→ 2D 检测 → 3D 家具盒 → 结构 → 线稿。

为什么单独一条链：VGGT 路线（小房间/弱纹理更稳）与 COLMAP+3DGS 路线
（需要 GPU 训练、对白瓷砖会成雾）是**两套后端**，各自的坑不一样，不该混在一个
脚本里靠开关切换 —— 混在一起的结果是"改一处坏一处"，实测已经踩过。

用法:
    # 最常用：取视频 8~32 秒、每秒 1 帧、24 帧
    <python> tools/pipeline_detect.py --video "宿舍测试 (1).mp4" --out output/dorm/auto \
        --start 8 --duration 24 --frames 24

    # 只要盒子不要线稿
    ... --no-lineart
    # 顺带训一个 3DGS .splat（需要 GPU，约 3~5 分钟）
    ... --train-splat

产出（全部在 --out 下）：
    frames/          抽帧（518 宽，方图补边前的原图）
    scene.ply        点云
    scene.poses.json 位姿与内参
    scene.depth.npz  稠密深度（检测反投影用）
    detected.json    3D 家具盒（带 label/score）
    structure.json   点云推出的房间盒（供线稿/界面）
    lineart/         线稿（房间盒 + 检测盒）
    model.splat      可选（--train-splat）
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
FFMPEG = ROOT / "pydeps312" / "imageio_ffmpeg" / "binaries" / "ffmpeg-win-x86_64-v7.1.exe"


def run(step: str, cmd: list[str] | str, *, timeout: int = 3600) -> tuple[bool, str]:
    t0 = time.time()
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
    tail = [ln for ln in out.splitlines() if ln.strip()][-10:]
    print("\n".join("  " + ln[:200] for ln in tail), flush=True)
    print(f"  → exit={p.returncode}  用时 {time.time() - t0:.1f}s", flush=True)
    return p.returncode == 0, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=float, default=0.0, help="从第几秒开始取（默认 0）")
    ap.add_argument("--duration", type=float, default=24.0, help="取多少秒（默认 24）")
    ap.add_argument("--frames", type=int, default=24, help="VGGT 用多少帧（显存随帧数增长）")
    ap.add_argument("--fps", type=float, default=1.0, help="抽帧帧率（1 秒 1 帧最稳）")
    ap.add_argument("--conf", type=float, default=4.2, help="VGGT 深度置信阈值（按 P90 取）")
    ap.add_argument("--threshold", type=float, default=0.3, help="检测置信度阈值")
    ap.add_argument("--prompts", default=None, help="检测提示词（默认床/上下铺/门/柜子…）")
    ap.add_argument("--train-splat", action="store_true", help="顺带训 3DGS（需要 GPU）")
    ap.add_argument("--no-lineart", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    frames = out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    ply = out / "scene.ply"
    poses = out / "scene.poses.json"
    depth = out / "scene.depth.npz"
    det = out / "detected.json"
    st = out / "structure.json"
    steps: list[tuple[str, bool]] = []

    # 1) 抽帧（518 宽，1 秒 1 帧；VGGT 最吃"连续小基线"）
    cmd = [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-ss", str(args.start),
           "-t", str(args.duration), "-i", str(ROOT / args.video),
           "-vf", f"fps={args.fps},scale=518:-2", "-q:v", "2", str(frames / "f_%03d.jpg")]
    ok, _ = run("1/5 抽帧", cmd, timeout=900)
    n_frames = len(list(frames.glob("*.jpg")))
    print(f"  抽到 {n_frames} 帧", flush=True)
    steps.append(("抽帧", ok and n_frames > 0))
    if not ok or n_frames == 0:
        return 1

    # 2) VGGT：点云 + 位姿 + 深度
    ok, _ = run("2/5 VGGT（点云/位姿/深度）",
                [PY, "-u", "tools/vggt_cloud.py", "--frames-dir", str(frames),
                 "--out", str(ply), "--frames", str(min(args.frames, n_frames)),
                 "--conf", str(args.conf), "--save-depth"], timeout=3600)
    steps.append(("VGGT", ok))
    if not ok or not poses.exists():
        print("✗ VGGT 失败，后续步骤跳过")
        return 2

    # 3) 2D 检测 + 深度反投影 → 3D 盒
    det_cmd = [PY, "-u", "tools/detect_furniture.py", "--frames", str(frames),
               "--depth", str(depth), "--poses", str(poses), "--out", str(det),
               "--threshold", str(args.threshold)]
    if args.prompts:
        det_cmd += ["--prompts", args.prompts]
    ok, _ = run("3/5 检测 + 反投影 → 3D 盒", det_cmd, timeout=3600)
    steps.append(("检测", ok))
    if not ok:
        return 3
    n_boxes = len((json.loads(det.read_text(encoding="utf-8")).get("objects") or []))
    print(f"  盒子 {n_boxes} 个")

    # 4) 结构（用点云推出房间盒；VGGT 没有 COLMAP 模型，走 cloud_to_structure）
    # 房间盒：用**墙面拟合**（RANSAC 竖直平面 + 曼哈顿聚类 + 半空间求交），
    # 比"点云分位数"稳；墙面只覆盖少数方向时会如实标 estimated=True
    ok, _ = run("4/5 结构（墙面拟合）",
                [PY, "-u", "tools/cloud_walls.py", "--cloud", str(ply),
                 "--poses", str(poses), "--out", str(st)], timeout=1800)
    steps.append(("结构", ok))

    # 5) 线稿
    if not args.no_lineart and ok:
        ok, _ = run("5/5 线稿",
                    [PY, "-u", "tools/render_lineart.py", "--structure", str(st),
                     "--objects", str(det), "--out", str(out / "lineart"),
                     "--views", "2", "--elevation", "30", "--no-trajectory"], timeout=1800)
        steps.append(("线稿", ok))

    # 可选：训 3DGS（要 GPU；先建 COLMAP 模型再训）
    if args.train_splat:
        ws = out / "colmap-ws"
        ok, _ = run("6a 建 COLMAP 模型（VGGT 位姿）",
                    [PY, "-u", "tools/vggt_poses_to_colmap.py", "--frames", str(frames),
                     "--poses", str(poses), "--out", str(ws),
                     "--frames-limit", str(min(args.frames, n_frames)), "--init-ply", str(ply)],
                    timeout=1800)
        if ok:
            # 文本模型 → 二进制（train_gsplat 只认 bin）
            gen = (
                "import json,sys;from pathlib import Path;import numpy as np;"
                "sys.path.insert(0,'tools');"
                "from colmap_bin_writer import write_cameras,write_images,write_points3d;"
                "from vggt_poses_to_colmap import rotmat_to_qvec;"
                f"ws=Path(r'{ws}');sp=ws/'sparse'/'0';"
                f"pj=json.loads(Path(r'{poses}').read_text(encoding='utf-8'));"
                "E=np.asarray(pj['extrinsic'],float);K=np.asarray(pj['intrinsic'],float);"
                "n=min(len(E),len(pj['frames']));W=H=1024;"
                "f=float(np.median(np.concatenate([K[:,0,0],K[:,1,1]])))*(W/518.0);"
                f"raw=Path(r'{ply}').read_bytes();end=raw.find(b'end_header');body=raw.find(b'\\n',end)+1;"
                "head=raw[:end].decode('ascii','replace');"
                "m=int([l for l in head.splitlines() if l.startswith('element vertex')][0].split()[-1]);"
                "names=[l.split()[-1] for l in head.splitlines() if l.startswith('property')];"
                "dt=np.dtype([(nm,'<f4') for nm in names]);"
                "buf=np.frombuffer(raw,dtype=dt,count=m,offset=body);"
                "P=np.stack([buf['x'],buf['y'],buf['z']],axis=1).astype(np.float64);"
                "P=P[np.isfinite(P).all(axis=1)];"
                "P=P[np.random.default_rng(0).choice(len(P),min(30000,len(P)),replace=False)];"
                "qv=np.stack([rotmat_to_qvec(E[i,:3,:3]) for i in range(n)]);"
                "write_cameras(sp/'cameras.bin',w=W,h=H,fx=f,fy=f,cx=W/2,cy=H/2);"
                "write_images(sp/'images.bin',names=[Path(x).name for x in pj['frames']][:n],"
                "qvecs=qv,tvecs=E[:n,:3,3],n_points_per_image=[len(P)]+[0]*(n-1));"
                "write_points3d(sp/'points3D.bin',P,np.full((len(P),3),200,np.uint8),track_image_id=1);"
                "print('模型已写')"
            )
            okbin, _ = run("6b 写二进制 COLMAP 模型", [PY, "-c", gen], timeout=600)
            if okbin:
                inner = (f'cd /d "{ROOT}" && call tools\\msvc-env.bat && "{PY}" -u pipeline\\train_gsplat.py '
                         f'--data "{ws}" --out "{ws / "train"}" --images "{frames}" '
                         f'--iterations 12000 --downscale 1 --sh-degree 0 --densify-grad-threshold 1e-4')
                okt, _ = run("6c 训练 3DGS", inner, timeout=7200)
                if okt:
                    oks, _ = run("6d 导出 .splat",
                                 [PY, "-u", "pipeline/to_splat.py", "--ply", str(ws / "train/point_cloud.ply"),
                                  "--out", str(out / "model.splat")], timeout=1800)
                    steps.append(("3DGS", oks))
                else:
                    steps.append(("3DGS", False))

    print("\n=== 汇总 ===")
    for name, ok in steps:
        print(f"  {name:<8} {'✅' if ok else '❌'}")
    print(f"\n产出目录：{out}")
    if det.exists():
        d = json.loads(det.read_text(encoding="utf-8"))
        print(f"检测盒 {len(d.get('objects') or [])} 个：")
        for o in (d.get("objects") or [])[:6]:
            print(f"  {o.get('label','?'):<10} {[round(v, 2) for v in o.get('size', [])]}  "
                  f"置信 {o.get('score')}  票 {o.get('votes')}")
    return 0 if all(ok for _, ok in steps) else 1


if __name__ == "__main__":
    raise SystemExit(main())
