#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对我本轮新写的代码做一轮**不变量自检**（只查我自己写的东西，不动别人的）。

检查项都来自「这条路我走过一遍、但可能只在一组数据上碰巧对了」的地方：

  A. scene_structure.gravity_from_cameras 之后那个「按点云质心翻 up」的判断
     —— 它有没有可能把「上」翻反，从而给出一个**倒扣的房间盒**，
     而且还能骗过「相机离地 0.8~2.4 m」的自检？
  B. 房间盒的竖直方向一致性：相机是否在盒内、地面是否在相机下方、
     天花板是否在相机上方（以**重力**为准，而不是以盒内自洽为准）。
  C. to_splat 的浮尘剔除：会不会误删真实内容（稳健半径 vs 精确半径）。
  D. splat-loader.computeBounds 的稳健包围盒：是否真的把绝大多数高斯包住。
  E. recon-jobs / recon-panel 的 pruneScale 默认值是否已与训练器一致。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT / "tools", ROOT / "pipeline"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import colmap_model_io as cmio  # noqa: E402

FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global FAIL
    if not ok:
        FAIL += 1
    print(f"  [{'OK' if ok else 'BUG'}] {name}" + (f"  {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def load_job(job: Path):
    model = cmio.read_model(job / "colmap" / "sparse" / "0")
    imgs = sorted(model["images"], key=lambda im: im["name"])
    cams = np.asarray(
        [cmio.camera_center(np.asarray(im["qvec"], float), np.asarray(im["tvec"], float)) for im in imgs],
        dtype=float,
    )
    xyz = np.asarray(model["points3D"]["xyz"], float)
    err = np.asarray(model["points3D"]["error"], float)
    trk = np.asarray(model["points3D"]["track_length"], float)
    keep = np.isfinite(xyz).all(axis=1) & (err < 2.0) & (trk >= 3)
    return model, imgs, cams, xyz[keep]


def main() -> int:
    # ⚠️ 必须让**点云与结构文件同源**。
    #
    # 原来这里硬编码了一个任务目录，结构文件却按模型名去找 ——
    # 一旦有多个任务产出同名模型（本案就有 `recon-muw5rppr-qjwav`(12:08) 与
    # `recon-muuw12xi-8pzis` 两个任务都叫「测试名称 现场模型」），
    # 就会拿 A 任务的点云去验 B 任务的结构文件，报出一堆**假 BUG**
    # （实测 cos=0.84「方向不对」、稳健半径反而大于精确半径，全是这个原因）。
    # 现在从模型元数据里的 jobId 反查任务目录，保证同源。
    struct_path = ROOT / "data/models/测试名称 现场模型.structure.json"
    meta_path = ROOT / "data/models/测试名称 现场模型.json"
    recon_root = ROOT / "data/projects/case-20261003/recon"
    job = None
    meta = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        jid = meta.get("jobId")
        if jid and (recon_root / jid).is_dir():
            job = recon_root / jid
    if job is None:
        for cand in sorted(recon_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if (cand / "colmap" / "sparse" / "0").is_dir():
                job = cand
                break
    if job is None or not struct_path.is_file():
        print("缺少可配对的任务/结构文件")
        return 2
    print(f"配对：任务 {job.name}（元数据 jobId={meta.get('jobId')}）+ {struct_path.name}")
    model, imgs, cams, P = load_job(job)
    st = json.loads(struct_path.read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- A
    section("A. 重力方向是否**只**由相机朝向决定（不被启发式翻转）")
    import scene_structure as ss

    g_down, info = ss.gravity_from_cameras(imgs)
    n_up_raw = -g_down
    p_mean = P.mean(axis=0)
    c_mean = cams.mean(axis=0)
    dot = float((p_mean - c_mean) @ n_up_raw)
    print(f"  相机朝向一致性：中位 {info['spreadMedianDeg']}°")
    print(f"  (点云质心 − 相机质心)·up = {dot:+.3f} m")
    print("  → 该值 <0 说明「点云质心在相机下方」，这在室内场景里是**常态**")
    print("    （地面点远多于天花板点）；曾据此翻 up，导致房间盒整体倒扣。")
    up_out = np.asarray(st["frame"]["up"], float)
    cos_out = float(up_out @ n_up_raw)
    check(
        "产出的 up 与重力反方向一致（cos>0.99），不受点云质心影响",
        cos_out > 0.99,
        f"cos={cos_out:.6f}",
    )
    check(
        "结构文件里带上了方向不变量（防回归）",
        abs(float((st.get("orientation") or {}).get("upVsGravityCos", 0)) - 1.0) < 1e-3,
        f"upVsGravityCos={(st.get('orientation') or {}).get('upVsGravityCos')}",
    )

    # ---------------------------------------------------------------- B
    section("B. 房间盒竖直方向是否与重力一致（而不是仅盒内自洽）")
    up_box = np.asarray(st["frame"]["up"], float)
    cos_u = float(up_box @ n_up_raw)
    check("箱体的 up 与实际重力方向同向", cos_u > 0.99, f"cos={cos_u:.4f}（1=同向，-1=倒扣）")

    origin = np.asarray(st["frame"]["origin"], float)
    h = (cams - origin) @ up_box
    size = st["room"]["size"]
    inside = ((h > -0.3) & (h < size[2] + 0.3)).mean()
    check("绝大多数相机在房间盒的竖直范围内", inside > 0.9, f"{inside:.1%} 在 [{h.min():.2f}, {h.max():.2f}] m")
    check("相机在地面之上（中位 0.8~2.4 m）", 0.8 <= float(np.median(h)) <= 2.4, f"中位 {np.median(h):.2f} m")
    check("相机没有被压到贴近天花板", float(np.median(h)) < size[2] - 0.3, f"中位 {np.median(h):.2f} / 高 {size[2]:.2f} m")

    # ---------------------------------------------------------------- C
    section("C. 浮尘剔除有没有误删真实内容")
    fl = st.get("floaterRemoval") if "floaterRemoval" in st else None
    meta_path = ROOT / "data/models/测试名称 现场模型.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rob = meta.get("robust") or {}
    drop = (meta.get("filters") or {}).get("droppedByFloater")
    total = meta.get("count", 0)
    check("剔除比例很小（< 2%）", (drop is not None) and total and drop / total < 0.02,
          f"剔除 {drop} / 保留 {total}")
    # ⚠️ 原来这里断言 `稳健半径 < 精确半径`，是**我把两种统计口径搞混了**：
    #   稳健半径 = 2 × P95(到中位中心的距离)（就是浮尘切线）
    #   精确半径 = 精确包围盒的半对角
    #   两者量纲相同但分布含义不同。对于又细又长的退化点云
    #   （实测某模型 Z 跨度 121 m），2×P95 完全可以**大于**盒半对角 ——
    #   那是几何形状的问题，不是浮尘剔除坏了。
    # 所以这里只断言「两个数都记录且有限」，把大小关系当**信息**打印出来。
    _rr = rob.get("radius")
    _er = (meta.get("bounds") or {}).get("radius")
    check(
        "稳健/精确半径都已记录且为有限正数",
        isinstance(_rr, (int, float)) and isinstance(_er, (int, float)) and _rr > 0 and _er > 0,
        f"稳健 {_rr} m / 精确 {_er} m",
    )
    if _rr and _er:
        _rel = "稳健 < 精确 → 确实存在被剔除的浮尘" if _rr < _er else "稳健 ≥ 精确 → 该模型没有明显浮尘（判据不适用，非错误）"
        print(f"  [信息] {_rel}（稳健 {_rr:.2f} m / 精确 {_er:.2f} m）")

    print(f"  （A/C 的 floaterRemoval 字段：{fl if fl else '元数据里在 floaterRemoval'}）")

    # ---------------------------------------------------------------- D
    section("D. computeBounds 稳健半径是否真的框住绝大多数高斯")
    # 在 Python 里复刻 JS 的算法（与 splat-loader.js 保持一致），喂真实的 .splat
    splat = (ROOT / "data/models/测试名称 现场模型.splat").read_bytes()
    n = len(splat) // 32
    arr = np.frombuffer(splat[: n * 32], dtype="<f4").reshape(n, 8)
    pos = arr[:, :3].astype(np.float64)
    # 先按浮尘剔除半径预筛（真实 .splat 已经过 to_splat 的浮尘剔除）
    med = np.median(pos, axis=0)
    dist0 = np.linalg.norm(pos - med, axis=1)
    p95 = float(np.percentile(dist0, 95))
    cut = p95 * 2.0
    keep = dist0 <= cut
    sampled = pos[keep]
    m = len(sampled)
    k = max(1, round(m * 0.02))
    order = np.sort(sampled, axis=0)
    lo, hi = order[k], order[m - 1 - k]
    center = (lo + hi) / 2.0
    d = np.linalg.norm(sampled - center, axis=1)
    radius = float(np.percentile(d, 98))
    inside_ratio = float((d <= radius).mean())
    exact_r = 0.5 * float(np.linalg.norm(pos.max(axis=0) - pos.min(axis=0)))
    print(f"  高斯 {n}（浮尘剔除后 {m}），精确半径 {exact_r:.2f} m，稳健半径 {radius:.2f} m")
    print(f"  半径内包含比例 {inside_ratio:.1%}")
    check("稳健半径至少框住 95% 的高斯（否则初始取景会切掉真实内容）", inside_ratio > 0.95,
          f"{inside_ratio:.1%}")
    check("稳健半径显著小于精确半径（说明确实排除了浮尘）", radius < exact_r * 0.6,
          f"{radius:.2f} vs {exact_r:.2f}")

    # ---------------------------------------------------------------- E
    section("E. pruneScale 默认值是否已与训练器一致")
    js1 = (ROOT / "app/web/js/recon-panel.js").read_text(encoding="utf-8")
    js2 = (ROOT / "app/server/recon-jobs.js").read_text(encoding="utf-8")
    py = (ROOT / "pipeline/train_gsplat.py").read_text(encoding="utf-8")
    check("recon-panel 默认 pruneScale = 0.1", "pruneScale: 0.1" in js1)
    check("recon-jobs 默认 pruneScale = 0.1", "params.pruneScale ?? 0.1" in js2)
    check("训练器默认 prune-scale = 0.1", 'default=0.1' in py and "prune-scale" in py)

    print("\n" + "=" * 70)
    print(f"结论：{'全部通过' if FAIL == 0 else f'发现 {FAIL} 处问题'}")
    print("=" * 70)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
