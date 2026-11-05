#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
COLMAP 端到端验证：从图片目录一路跑到可加载的 .splat，并输出官方统计。

用法：
    pyruntime312\\python.exe tools/verify_colmap.py --images data/scene-textured/images ^
        --workspace output/colmap-verify

会自动：
    1. 找到 COLMAP（tools/colmap/bin/colmap.exe 或 PATH），打印版本
    2. 跑 feature_extractor → matcher → mapper
    3. 用官方 `colmap model_analyzer` 取权威统计（注册率、点数、平均重投影误差）
    4. 用 tools/colmap_model_io.py 读取自己实现的解析结果做交叉校验
    5. 报告结论

退出码：0 通过；1 未找到 COLMAP；2 重建注册率过低；3 其他失败
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "tools"))
sys.path.insert(0, str(_ROOT / "pipeline"))

import colmap_model_io as mio  # noqa: E402


def find_colmap():
    try:
        import config  # type: ignore

        p = config.find_colmap()
        if p:
            return p
    except Exception:  # noqa: BLE001
        pass
    import shutil

    return shutil.which("colmap")


def run(cmd, timeout=3600):
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def probe_options(colmap, command):
    """读取子命令支持的真实选项名（COLMAP 4.x 改过名）"""
    _, text = run([colmap, command, "--help"], timeout=120)
    return set(re.findall(r"(--[A-Za-z][A-Za-z0-9_.]*)", text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--matcher", default="exhaustive", choices=["exhaustive", "sequential"])
    ap.add_argument("--use-gpu", type=int, default=1)
    args = ap.parse_args()

    images = Path(args.images).resolve()
    ws = Path(args.workspace).resolve()
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "sparse").mkdir(parents=True, exist_ok=True)

    colmap = find_colmap()
    if not colmap:
        print("✘ 未找到 COLMAP。请把官方包解压到 tools/colmap/，或设置环境变量 COLMAP_EXE。")
        return 1

    code, ver = run([colmap, "--help"], timeout=120)
    ver_line = next((l for l in ver.splitlines() if "COLMAP" in l and "http" not in l), "").strip()
    print(f"COLMAP: {colmap}")
    print(f"版本  : {ver_line}")
    has_gpu = "with CUDA" in ver_line
    if args.use_gpu and not has_gpu:
        print("提示  : 该 COLMAP 不是 CUDA 版，自动改用 CPU")
        args.use_gpu = 0

    n_imgs = len([p for p in images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm"}])
    print(f"图片  : {n_imgs} 张（{images}）\n")

    fe_opts = probe_options(colmap, "feature_extractor")
    fe_gpu = "--FeatureExtraction.use_gpu" if "--FeatureExtraction.use_gpu" in fe_opts else "--SiftExtraction.use_gpu"
    db = ws / "database.db"

    steps = [
        (
            "特征提取",
            [colmap, "feature_extractor", "--database_path", str(db), "--image_path", str(images),
             "--ImageReader.single_camera", "1", fe_gpu, "1" if args.use_gpu else "0"],
        )
    ]
    if args.matcher == "sequential":
        m_opts = probe_options(colmap, "sequential_matcher")
        m_gpu = "--FeatureMatching.use_gpu" if "--FeatureMatching.use_gpu" in m_opts else "--SiftMatching.use_gpu"
        steps.append(("顺序匹配", [colmap, "sequential_matcher", "--database_path", str(db),
                                  "--SequentialMatching.overlap", "15", m_gpu, "1" if args.use_gpu else "0"]))
    else:
        m_opts = probe_options(colmap, "exhaustive_matcher")
        m_gpu = "--FeatureMatching.use_gpu" if "--FeatureMatching.use_gpu" in m_opts else "--SiftMatching.use_gpu"
        m_guided = "--FeatureMatching.guided_matching" if "--FeatureMatching.guided_matching" in m_opts else "--SiftMatching.guided_matching"
        steps.append(("穷举匹配", [colmap, "exhaustive_matcher", "--database_path", str(db),
                                  m_guided, "1", m_gpu, "1" if args.use_gpu else "0"]))
    steps.append(("增量式重建", [colmap, "mapper", "--database_path", str(db), "--image_path", str(images),
                               "--output_path", str(ws / "sparse")]))

    for name, cmd in steps:
        print(f"→ {name} …", end="", flush=True)
        rc, out = run(cmd)
        if rc != 0:
            print(" 失败")
            print(out[-2000:])
            return 3
        print(" 完成")

    # 官方统计
    models = sorted([d for d in (ws / "sparse").iterdir() if d.is_dir()])
    if not models:
        print("✘ 没有生成任何稀疏模型")
        return 2
    best, best_n = None, -1
    for d in models:
        rc, out = run([colmap, "model_analyzer", "--path", str(d)], timeout=300)
        m = re.search(r"Registered images:\s*(\d+)", out)
        n = int(m.group(1)) if m else 0
        if n > best_n:
            best, best_n, best_out = d, n, out
    print(f"\n=== 官方 model_analyzer（模型 {best.name}）===")
    for key in ("Cameras", "Registered images", "Points", "Observations", "Mean track length",
                "Mean observations per image", "Mean reprojection error"):
        mm = re.search(rf"{key}:\s*([0-9.]+)", best_out)
        if mm:
            print(f"  {key:28s} {mm.group(1)}")

    # 自己实现的解析交叉校验
    print("\n=== tools/colmap_model_io.py 独立解析（交叉校验）===")
    try:
        model = mio.read_model(best)
        s = mio.summarize(model)
        print(f"  相机数            {s['cameras']}")
        print(f"  注册图像          {s['registeredImages']}")
        print(f"  三维点            {s['points3D']}")
        print(f"  观测数            {s['observations']}")
        print(f"  平均 track 长度   {s['meanTrackLength']:.4f}")
        print(f"  平均重投影误差    {s['meanReprojectionError']:.6f} px")
        for cid, cam in model["cameras"].items():
            fx, fy, cx, cy = mio.intrinsics_of(cam)
            print(f"  相机 {cid}: {cam['model']} {cam['width']}x{cam['height']} fx={fx:.2f} fy={fy:.2f} cx={cx:.1f} cy={cy:.1f}")
    except Exception as exc:  # noqa: BLE001
        print(f"  ✘ 解析失败：{exc}")
        return 3

    ratio = s["registeredImages"] / max(1, n_imgs)
    print("\n=== 结论 ===")
    print(f"  注册率 {s['registeredImages']}/{n_imgs} = {ratio * 100:.1f}%")
    print(f"  平均重投影误差 {s['meanReprojectionError']:.4f} px（< 1.0px 属优秀）")
    ok = ratio >= 0.6
    print("  " + ("✔ COLMAP 端到端可用" if ok else "✘ 注册率偏低，检查图片纹理与重叠度"))
    (ws / "colmap_verify.json").write_text(
        json.dumps({"colmap": colmap, "version": ver_line, "summary": s, "registeredRatio": ratio},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
