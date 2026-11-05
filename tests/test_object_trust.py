#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回归：物体包围盒的**可信度判据**，以及两个被它量出来的真缺陷。

为什么需要合成真值
-----------------
真实素材里我手上**只有坏样本**（柜子、run-32mb 都是"一次贴面拍"的退化素材），
没有一份可信的物体重建 —— 那就没法定阈值、也没法判断"算对了没有"。
所以这里用合成真值：`structure_object.build_object_model()` 是「点云 + 相机位姿」
的纯函数，直接喂进去即可（不需要跑 COLMAP），而柜子的尺寸是我自己定的、**已知**。

它守住的四件事（都是实测踩出来的）：

  1. **竖直范围不能按"相机高度带"裁剪**（真缺陷，已修）
     人拍柜子时相机在眼睛高度，柜子却从地面到 2 m —— 相机高度带根本不覆盖物体。
     修复前：1.2×0.6×**1.9** m 的柜子被算成高 **0.58 m**（砍掉 70%）。
  2. **包围盒只能取"相机附近"的点**（真缺陷，已修）
     合成C：小柜 + 15% 撒在 8~14 m 外的背景点。用全部点算盒子得到
     18.93×19.07×19.06 m（完全没用）；限制到相机群 2 倍半径内就恢复成那个柜子。
  3. **顶面/底面要能被发现**（层的边界问题，已修）
     柜子的顶/底水平面正好落在点云极值上，而落在直方图第 0/最后 bin 的峰
     永远不被"局部极大"发现 → 三层只报 1 层。外扩 3 个 bin 后 3 层都在。
  4. **不可信时必须说得出理由**，而不是给一个"看起来还行"的盒子。

用法: pyruntime312\\python.exe tests/test_object_trust.py
退出码 0 通过 / 1 失败
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

from probe_object_trust import cabinet_points, metrics, orbit_cameras  # noqa: E402
from structure_object import (  # noqa: E402
    OBJECT_NEAR_RADIUS_FACTOR,
    TRUST_MAX_EDGE_OVER_CAM_R,
    evaluate_object_trust,
)

passed = 0
failed = 0
TOL = 0.05          # 盒子尺寸容差 5 cm
UP = np.array([0.0, 0.0, 1.0])


def check(cond, msg):
    global passed, failed
    if cond:
        passed += 1
        print(f"  [OK]   {msg}")
    else:
        failed += 1
        print(f"  [FAIL] {msg}")


def synth(size, radius, *, bg_frac=0.0, seed=7, cam_seed=11):
    """合成一次"绕物体一圈"的拍摄：返回 (P, C, F, 真值尺寸)。"""
    P = cabinet_points(size, seed=seed)
    if bg_frac > 0:
        rng = np.random.default_rng(seed + 1)
        n_bg = int(len(P) * bg_frac / (1 - bg_frac))
        r = rng.uniform(8, 14, n_bg)
        d = rng.normal(size=(n_bg, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        P = np.concatenate([P, d * r[:, None]], axis=0)
    C, F = orbit_cameras((0, 0, size[2] / 2), radius, seed=cam_seed)
    return P, C, F


def robust_truth(P, C, n_up) -> np.ndarray:
    """真值的**稳健**包围盒（与生产口径一致：锚点=相机质心，1/99 分位）。

    为什么不拿"名义尺寸"比：盒子的定义就是 P1/P99 稳健包围盒，**本来就会比极值略小**。
    第一版测试拿名义尺寸比，于是 2.2 m 的柜子算出 2.15 m 就被判失败 ——
    那 5 cm 是 P99 裁掉的尾巴，不是缺陷。要比就跟"同样口径的真值"比。
    """
    import geom_common as gc  # noqa: PLC0415

    u, v = gc.orthonormal_basis(n_up)
    axes = np.stack([u, v, n_up], axis=1)
    anchor = C.mean(axis=0)
    proj = (P - anchor) @ axes
    lo, hi = np.percentile(proj, 1, axis=0), np.percentile(proj, 99, axis=0)
    return hi - lo


def main() -> int:
    global passed, failed
    print("=" * 74)
    print("物体包围盒与可信度判据（合成真值 + 真实素材对照）")
    print("=" * 74)

    cases = [
        ("合成A 1.2×0.6×1.9 绕1.9m", (1.2, 0.6, 1.9), 1.9, 0.0),
        ("合成B 2.5×1.0×2.2 绕4.0m", (2.5, 1.0, 2.2), 4.0, 0.0),
    ]
    for tag, size, radius, bg in cases:
        P, C, F = synth(size, radius, bg_frac=bg)
        r = metrics(P, C, F, tag=tag, n_up=UP)
        print(f"\n[{tag}]")
        got = r.get("size") or []
        check(bool(got), f"算出了物体盒：{got}")
        if got:
            truth = robust_truth(P, C, UP)
            err = [abs(g - t) for g, t in zip(got, truth)]
            check(max(err) < 0.01,
                  f"三边与**同口径真值** {[round(float(t), 2) for t in truth]} 一致"
                  f"（最大误差 {max(err) * 100:.2f} cm）")
            # 与名义尺寸的偏差要小 —— 这条才能真正抓住"高度被砍掉 70%"那种缺陷
            nom = [abs(g - t) for g, t in zip(got, size)]
            check(max(nom) < 0.12,
                  f"与名义尺寸 {size} 的偏差 < 12 cm（最大 {max(nom) * 100:.1f} cm）")
            check(abs(got[2] - size[2]) < 0.12,
                  f"**高度没有被裁剪**：{got[2]:.2f} m ≈ 名义 {size[2]:.2f} m（修复前是 0.58 m）")
            check(len(r["layerProminence"]) == 3,
                  f"三块水平面（顶/中/底）都报出来了：{len(r['layerProminence'])} 层"
                  f"（修复前只报 1 层）")
            check(r["longestEdgeOverCamR"] < TRUST_MAX_EDGE_OVER_CAM_R,
                  f"可信度第一项通过：最长边/相机半径 = {r['longestEdgeOverCamR']}")
            check(abs(r["centerOffsetOverCamR"]) < 1.0,
                  f"可信度第二项通过：盒心偏移/相机半径 = {r['centerOffsetOverCamR']}")

    print("\n[合成C 小柜 + 15% 背景野点（8~14 m 外）]")
    P, C, F = synth((1.2, 0.6, 1.9), 1.9, bg_frac=0.15)
    r = metrics(P, C, F, tag="合成C", n_up=UP)
    got = r.get("size") or []
    print(f"       用了 {r['nPoints']} 个点（其中背景点约占 15%）")
    check(bool(got) and max(abs(g - t) for g, t in zip(got, (1.2, 0.6, 1.9))) < TOL,
          f"背景点**没有**把盒子撑坏：{got}（用全部点会算成 ~19×19×19 m）")
    check(r["longestEdgeOverCamR"] < TRUST_MAX_EDGE_OVER_CAM_R,
          f"最长边/相机半径 = {r['longestEdgeOverCamR']}（< {TRUST_MAX_EDGE_OVER_CAM_R}）")

    # ---- 真实素材：必须判为不可信，且理由要说得出 -------------------------
    print("\n[真实 柜子现场（一次贴面拍的退化素材）]")
    real_dir = ROOT / "data/projects/case-20261003/recon/recon-muwcu0a4-889pb/colmap/sparse/0"
    if not (real_dir / "images.bin").exists():
        print("       ⚠ 找不到该素材，跳过（不是失败；这条依赖案件数据）")
    else:
        from probe_object_trust import real_case

        Pr, Cr, Fr, n_up_real = real_case(real_dir)
        rr = metrics(Pr, Cr, Fr, tag="真实柜子", n_up=n_up_real)
        mom = np.median(np.linalg.norm(Pr - Pr.mean(axis=0), axis=1)) if len(Pr) else 0.0
        size_r = np.asarray(rr.get("size") or [0, 0, 0], float)
        center_r = None
        # 直接调判据函数，拿到完整理由（metrics 里只带了比值）
        anchor = Cr.mean(axis=0)
        _, _, _ = None, None, None
        from structure_object import build_object_model  # noqa: PLC0415
        import geom_common as gc  # noqa: PLC0415

        u, v = gc.orthonormal_basis(n_up_real)
        obj, layers, _ev = build_object_model(
            Pr, Cr, u=u, v=v, n_up=n_up_real, floor_pt=anchor,
            scale=float(np.percentile(np.linalg.norm(Pr - np.median(Pr, axis=0), axis=1), 95)),
            mode="object", layers=[], layers_ev={},
        )
        trust = (obj or {}).get("trust") or {}
        print(f"       盒子={[round(x, 2) for x in (obj or {}).get('size') or []]} m，"
              f"相机群半径={trust.get('metrics', {}).get('camRadius')} m")
        check(trust.get("ok") is False, "判为**不可信**（不会画到界面上）")
        check(len(trust.get("reasons") or []) >= 2,
              f"给出了 {len(trust.get('reasons') or [])} 条理由，例如：{(trust.get('reasons') or ['—'])[0][:48]}…")
        check(trust.get("metrics", {}).get("fracPtsWithin2CamR", 1) < 0.01,
              f"相机群 2 倍半径内的点占比 = {trust.get('metrics', {}).get('fracPtsWithin2CamR')}"
              f"（相机在 3.5 m 内绕，点却在十几米外）")
        check(trust.get("metrics", {}).get("pointsUsedForBox", -1) == 0,
              f"用于算盒子的「相机附近点」数量 = {trust.get('metrics', {}).get('pointsUsedForBox')}（一个都没有）")

    # ---- 判据函数本身的边界行为 -----------------------------------------
    print("\n[判据函数的边界行为]")
    none_val = evaluate_object_trust(np.array([1.0, 1.0, 1.0]), np.zeros(3), np.zeros(3),
                                     np.zeros((0, 3)), np.zeros((0, 3)))
    check(none_val["ok"] is False and any("相机太少" in x for x in none_val["reasons"]),
          "没有相机时判为不可信并说明原因（不抛异常）")
    check(OBJECT_NEAR_RADIUS_FACTOR == 2.0,
          f"「相机附近」半径倍数 = {OBJECT_NEAR_RADIUS_FACTOR}（合成真值定出来的）")

    # ---- 真实数据上的判定表（正/负样本都来自真实文件）---------------------
    # 这一节的价值在于：**负样本是真实存在的退化重建**，不是我想象出来的。
    # 其中三个"不确定"模型（248~571 点 / 16 台相机）曾经把三条比值判据全部骗过
    # （比值 0.36~0.58 / 0.05~0.16 / 1.00），是端到端测试之后补的点数门限挡住它们的。
    print("\n[真实结构文件的判定表（该不该画物体盒）]")
    expect = {
        "room-real": False,          # 房间，且物体盒 3.17× 相机半径
        "柜子现场-致密": False,       # 相机附近 0 个点
        "柜子现场": False,
        "测试名称 现场模型": False,    # 248 点 / 16 台 —— 比值全过、密度不够
        "测试名称 现场模型1": False,   # 571 点 / 16 台
        "豆包测试现场模型": False,     # 250 点 / 16 台
    }
    for f in sorted((ROOT / "data" / "models").glob("*.structure.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        ob = d.get("object") or {}
        if not ob:
            continue
        key = f.name.replace(".structure.json", "")
        want = expect.get(key)
        if want is None:
            print(f"       （未在期望表里，跳过）{f.name}")
            continue
        got = (ob.get("trust") or {}).get("ok")
        m = (ob.get("trust") or {}).get("metrics") or {}
        check(got is want,
              f"{key}：trust.ok={got}（期望 {want}）点={m.get('pointsForBox')} "
              f"点/相机={m.get('pointsPerCamera')} 长/半径={m.get('longestEdgeOverCamR')}")

    # ---- 端到端正样本：合成柜子 → 真 COLMAP ------------------------------
    # 这是本轮补的那个缺口：**可信 → 画出来**这条路径要在真实重建产物上验证。
    # 真值柜子 1.2×0.6×1.9 m；单目 SfM 没有公制尺度，所以比**形状比值**。
    print("\n[端到端：合成柜子 → 真 COLMAP 重建（正向路径）]")
    truth_ratio = np.sort(np.array([1.2, 0.6, 1.9])) / 1.9   # [0.316, 0.632, 1.0]
    e2e = sorted((ROOT / "output" / "object-e2e").glob("*.structure.json"))
    if not e2e:
        print("       ⚠ 没有端到端产物（先跑 tools/make_object_dataset.py + pipeline/colmap_recon.py）")
    # 期望表：名字 → (是否应当 trust.ok, 形状比值容差；None = **已知偏**)
    #
    # ⚠️ `cabinet-aim`（相机"探头往柜子里看"）是**已知反例**，必须写在这里：
    #    它 trust.ok=True，但形状偏得厉害（换算成米 1.22×1.54×1.90，真值 1.2×0.6×1.9）。
    #    这说明 **trust 通过 ≠ 形状一定对** —— trust 只回答"这个盒子像不像
    #    「相机围着一个小物体拍」的盒子"，不回答"每一边准不准"。
    #    所以这里用 None 表示"已知偏"，测试反过来**断言它确实偏**：
    #    既不会假通过，也不会被当成偶发失败而被忽略。
    e2e_expect = {
        "cabinet-plain": (True, 0.06),
        "cabinet-sweep": (True, 0.06),
        "cabinet-high": (True, 0.06),
        "cabinet-aim": (True, None),
        # 高处俯视（相机在柜顶之上往下看）：盒子仍然准（2.6%），但**层依然是 0** ——
        # 这一版是用来验证"把水平面在画面里占比从 4% 提到 21% 也救不了层检测"的。
        "cabinet-topdown": (True, 0.06),
    }
    for f in e2e:
        key = f.name.replace(".structure.json", "")
        if key not in e2e_expect:
            print(f"       （未在期望表里，跳过）{f.name}")
            continue
        want_ok, tol = e2e_expect[key]
        d = json.loads(f.read_text(encoding="utf-8"))
        ob = d.get("object") or {}
        tr = ob.get("trust") or {}
        if not ob:
            check(False, f"{key}：没有物体记录（场景判定不确定时也应当算出来）")
            continue
        size = np.asarray(ob.get("size") or [0, 0, 0], float)
        ratio = np.sort(size) / max(size.max(), 1e-9)
        err = float(np.abs(ratio - truth_ratio).max())
        got_m = [round(float(x) / (size.max() / 1.9), 2) for x in np.sort(size)]
        check(tr.get("ok") is want_ok, f"{key}：真实重建上 trust.ok={tr.get('ok')}（期望 {want_ok}）")
        if tol is None:
            check(err > 0.15,
                  f"{key}：**已知形状偏**（最大偏差 {err * 100:.0f}%，换算 {got_m} m，"
                  f"真值 [0.6, 1.2, 1.9]）—— trust 通过 ≠ 形状对，见期望表注释")
        else:
            check(err < tol,
                  f"{key}：盒子形状比值 {[round(float(x), 3) for x in ratio]} 对上真值 "
                  f"{[round(float(x), 3) for x in truth_ratio]}（最大偏差 {err * 100:.1f}%，"
                  f"换算 {got_m} m，真值 [0.6, 1.2, 1.9]）")
        # 层：**实测在稀疏点云上找不出层**（掠射角的面点太少）。
        # 这条不是在断言"层是对的"，而是钉住"找不到层时必须留下为什么"的证据。
        lev = ob.get("layersEvidence") or {}
        if not (ob.get("layers") or []):
            check(bool(lev.get("rawPeakCount") is not None or lev.get("reason")),
                  f"{key}：没找到层，但留下了证据"
                  f"（原始峰 {lev.get('rawPeakCount')}，被突出度拒 {lev.get('rejectedByProminence')}）")

    print("\n" + "=" * 74)
    print(f"结果：{passed} 通过 / {failed} 失败")
    print("=" * 74)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
