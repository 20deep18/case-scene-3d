#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""素材质量闸门：判一段素材"能不能出几何"，并说清为什么。

为什么需要（专家 P0）：
  白瓷砖那次是"跑完 25 分钟才发现是雾"。闸门的作用是**在花时间之前**给出判断：
  注册率、PSNR、雾比例、结构证据——每一条都有阈值和出处，不是拍脑袋。

用法（跑完管线后）：
  <python> tools/quality_gate.py --dir output/dorm/v1
  <python> tools/quality_gate.py --dir output/bathroom/v2 --strict

输出：GOOD / MARGINAL / UNUSABLE + 逐项证据 + 建议动作。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402


def read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def fog_stats(splat: Path) -> dict | None:
    """粗算雾指标：大尺度且近白且低不透明度的高斯占比（阈值与 analyze_model_fog 一致）。"""
    if not splat.exists():
        return None
    raw = splat.read_bytes()
    n = len(raw) // 32
    if n == 0:
        return None
    f = np.frombuffer(raw[: n * 32], dtype="<f4").reshape(n, 8)
    scale = f[:, 3:6].max(axis=1)
    b = np.frombuffer(raw[: n * 32], dtype=np.uint8).reshape(n, 32)
    rgb = b[:, 24:27].astype(float)
    alpha = b[:, 27].astype(float) / 255.0
    bright = rgb.mean(axis=1)
    near_white = (bright > 200) & (rgb.max(axis=1) - rgb.min(axis=1) < 18)
    fog = near_white & (scale > np.percentile(scale, 70)) & (alpha < 0.35)
    return {
        "gaussians": int(n),
        "nearWhiteRatio": round(float(near_white.mean()), 4),
        "fogRatio": round(float(fog.mean()), 4),
        "scaleP50": round(float(np.percentile(scale, 50)), 4),
        "alphaP50": round(float(np.percentile(alpha, 50)), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="管线输出目录（含 colmap/、train/、structure.json 等）")
    ap.add_argument("--strict", action="store_true", help="严格模式：任一项不达标即判 UNUSABLE")
    args = ap.parse_args()

    d = Path(args.dir)
    print(f"素材质量闸门：{d}\n")

    checks: list[dict] = []

    def add(name, ok, detail, suggest=""):
        checks.append({"name": name, "ok": ok, "detail": detail, "suggest": suggest})

    # ---- 1) COLMAP 注册率 ------------------------------------------
    summ = read_json(d / "colmap" / "colmap_summary.json")
    reg = None
    if summ:
        reg = summ.get("registeredRatio")
        if reg is None:
            n_reg = summ.get("registered") or summ.get("numRegistered")
            n_tot = summ.get("images") or summ.get("numImages")
            reg = (n_reg / n_tot) if (n_reg and n_tot) else None
    if reg is not None:
        add("注册率（位姿是否成功）", reg >= 0.6,
            f"{reg:.1%}（阈值 ≥60%）",
            "低于 60% 通常是：走太快/甩镜头/白墙无纹理。建议慢走、白墙前放参照物")
    else:
        add("注册率（位姿是否成功）", False, "没有 colmap_summary.json（重建没跑或失败）",
            "先确认 COLMAP 跑通")

    # ---- 2) 画质 PSNR（训练报告里）-----------------------------------
    import re

    psnr = None
    cands = [d / "train" / "train.log", d / "train.log", Path(f"{d}-pipeline.log")]
    cands += sorted(d.glob("*.log")) + sorted(d.glob("train/*.log"))
    for cand in dict.fromkeys(cands):
        if cand.exists():
            txt = cand.read_text(encoding="utf-8", errors="replace")
            m = re.findall(r"平均 PSNR ([\d.]+) dB", txt)
            if m:
                psnr = float(m[-1])
                break
    if psnr is not None:
        add("训练画质 PSNR", psnr >= 22.0, f"{psnr:.2f} dB（阈值 ≥22 dB）",
            "低于 22 dB 说明模型只是'糊上去'，几何不可信（白瓷砖实测 18.8~21.2）")

    # ---- 3) 雾比例 -------------------------------------------------
    splats = sorted(d.glob("*.splat"))
    fog = fog_stats(splats[0]) if splats else None
    if fog:
        # 三条一起看，任何一条越界都算"雾风险"（实测白瓷砖：近白 33%、alpha P50 0.031）：
        #   · 近白占比 > 20%       —— 大片纯白面没有纹理
        #   · alpha P50 < 0.08     —— 高斯普遍半透明 = 糊出来的
        #   · 雾占比 > 15%         —— 大尺度+近白+低透明度
        fog_risk = (fog["nearWhiteRatio"] > 0.20) or (fog["alphaP50"] < 0.08) or (fog["fogRatio"] > 0.15)
        add("雾风险（近白/半透明/大尺度）", not fog_risk,
            f"近白 {fog['nearWhiteRatio']:.1%}（阈值 ≤20%），alpha P50 {fog['alphaP50']}（阈值 ≥0.08），"
            f"雾占比 {fog['fogRatio']:.1%}；尺度 P50 {fog['scaleP50']}",
            "雾多 = 表面没收敛：换素材或在现场加纹理参照物（毛巾/纸箱/瓶子）最有效")

    # ---- 4) 结构证据 -----------------------------------------------
    st = read_json(d / "structure.json")
    if st:
        walls = st.get("roomLikeWalls")
        sane = st.get("sane")
        est = st.get("estimated")
        add("房间结构自检", bool(sane), f"sane={sane}，可信竖墙 {walls} 面，estimated={est}",
            "0~1 面墙 → 走单墙补全（已支持，会标'推测'）；0 面墙且未推定 → 结构不可用")

    # ---- 5) 家具盒 ------------------------------------------------
    furn = read_json(d / "furniture.json")
    if furn:
        n = len(furn.get("objects") or [])
        add("家具盒数量", n >= 2, f"{n} 个（阈值 ≥2）",
            "0~1 个通常说明表面没分出物体（雾/太稀）")

    # ---- 汇总 ------------------------------------------------------
    print(f"{'检查项':<30}{'结果':<6}说明")
    print("-" * 100)
    for c in checks:
        print(f"{c['name']:<30}{'✅' if c['ok'] else '❌':<6}{c['detail']}")
    bad = [c for c in checks if not c["ok"]]
    verdict = "GOOD" if not bad else ("UNUSABLE" if (args.strict or len(bad) >= 2) else "MARGINAL")
    print()
    print(f"结论：{verdict}（{len(checks) - len(bad)}/{len(checks)} 项通过）")
    if bad:
        print("需要改进的地方：")
        for c in bad:
            if c["suggest"]:
                print(f"  · {c['name']}：{c['suggest']}")
    return 0 if verdict == "GOOD" else 1


if __name__ == "__main__":
    raise SystemExit(main())
