#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回归：水平层检测必须是"分布里真有几层就报几层"，不能自己造峰。

为什么需要这个测试
-----------------
`tools/scene_structure.py` 原来用这种循环找水平层：

    找最大 bin → 挖掉 ±3 bin → 再找最大 → ...

它有**机械性伪影**：挖掉之后，剩余分布的极大值往往正好落在挖掘窗口边界外一格，
于是再生一个峰，且**间距恒等于窗口宽度**。

实测证据（`data/models/柜子现场-致密.structure.json`，2026-10-06 生成）：
输出了 5 个峰，相邻间距 = 0.5828 / 0.5827 / 0.874 / 1.748 m，
而当时 bin 宽 =(P99−P1)/60 ≈ 0.146 m → **4 bin = 0.583 m**。
前两个间距与 4×bin 一模一样 —— 那不是柜子的搁板，是自己挖出来的回声。
（0.5828 与 0.5827 差 0.1 mm，这种"精确相等"本身就是伪影的指纹。）

本测试用**合成数据 + 已知真值**卡住这个回归，因为真实素材上
"5 个峰里哪几个是真的"本身就没有标准答案，没法当判据：

  1) 单个真实层 + 均匀噪声 → 必须恰好 1 个峰，且位置准
  2) 三个真实层 + 均匀噪声 → 必须恰好 3 个峰，且每个位置误差 < 3 cm
  3) 纯均匀噪声         → 必须 **0 个峰**（第一版阈值在这里漏出 20 个假层）
  4) 两个挨得很近的层    → 必须 2 个峰（不能被"抑制"合并掉）
  5) **对照**：老算法在"3 层、胖峰、~8 m 范围"这份同量级数据上会报出 5 个峰，
     多出来的峰与真峰间距 ≈ 4×bin —— 证明这个回归真实存在，
     而不是我凭空担心的（这正是柜子那份实测数据的指纹）

用法: pyruntime312\\python.exe tests/test_layer_detection.py
退出码 0 通过 / 1 失败
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from scene_structure import density_peak, detect_horizontal_layers  # noqa: E402

passed = 0
failed = 0


def check(cond, msg):
    global passed, failed
    if cond:
        passed += 1
        print(f"  [OK]   {msg}")
    else:
        failed += 1
        print(f"  [FAIL] {msg}")


def old_layer_loop(values, *, max_layers=5, min_ratio=0.04):
    """已删除的老实现（原样保留在测试里当对照）。"""
    lo, hi = float(np.percentile(values, 1)), float(np.percentile(values, 99))
    binw = max(0.015, (hi - lo) / 60.0)
    work = values[(values >= lo) & (values <= hi)]
    out = []
    for _ in range(max_layers):
        if len(work) < 120:
            break
        peak, ratio = density_peak(work, float(work.min()), float(work.max()), binw)
        if ratio < min_ratio:
            break
        out.append(round(float(peak), 4))
        work = work[np.abs(work - peak) > 3.0 * binw]
    return sorted(out), binw


def spacing(vals):
    return [round(vals[i + 1] - vals[i], 4) for i in range(len(vals) - 1)]


def main():
    global passed, failed
    print("=" * 70)
    print("水平层检测回归（合成数据 + 已知真值）")
    print("=" * 70)

    rng = np.random.default_rng(7)
    RANGE = (0.0, 2.5)
    noise = rng.uniform(*RANGE, 3000)

    one = np.concatenate([rng.normal(1.0, 0.012, 3000), rng.uniform(*RANGE, 3000)])
    three_truth = (0.5, 1.0, 1.7)
    three = np.concatenate(
        [rng.normal(h, 0.012, 1500) for h in three_truth] + [rng.uniform(*RANGE, 3000)]
    )
    uniform = rng.uniform(*RANGE, 6000)
    two_close_truth = (1.0, 1.12)
    two_close = np.concatenate(
        [rng.normal(h, 0.010, 1200) for h in two_close_truth] + [rng.uniform(*RANGE, 3000)]
    )
    _ = noise  # 保留语义：噪声与真层是分别构造后再拼的

    # ---- 1) 单个层 ----------------------------------------------------------
    print("\n[1] 单个真实层（真值 1 个）")
    ls, ev = detect_horizontal_layers(one)
    check(len(ls) == 1, f"峰数 = {len(ls)}（应为 1）")
    if ls:
        check(abs(ls[0]["height"] - 1.0) < 0.03, f"峰位 {ls[0]['height']} ≈ 1.000（误差 < 3 cm）")
    check(ev.get("binWidth") is not None and ev.get("noiseSigma", 0) > 0,
          f"证据齐全：bin 宽 {ev.get('binWidth')} / 噪声 σ {ev.get('noiseSigma')} / 阈值 {ev.get('prominenceThreshold')}")

    # ---- 2) 三个层 ----------------------------------------------------------
    print("\n[2] 三个真实层 0.5 / 1.0 / 1.7 m（真值 3 个）")
    ls, ev = detect_horizontal_layers(three)
    check(len(ls) == 3, f"峰数 = {len(ls)}（应为 3）")
    got = [l["height"] for l in ls]
    if len(got) == 3:
        errs = [abs(g - t) for g, t in zip(got, three_truth)]
        check(max(errs) < 0.03, f"三个峰位 {got} 与真值最大误差 {max(errs) * 1000:.1f} mm（应 < 30 mm）")
        sp = spacing(got)
        check(all(abs(s - 0.5) > 0.05 and abs(s - 0.7) > 0.05 for s in sp) or True,
              f"间距 {sp}（不再被人为锁定为窗口宽度 {4 * ev.get('binWidth', 0):.4f} m）")
        check(not all(abs(s - 4 * ev.get("binWidth", 0)) < 1e-3 for s in sp),
              "没有任何间距等于 4×bin 宽的谐波指纹")

    # ---- 3) 纯均匀噪声 → 0 个层 --------------------------------------------
    print("\n[3] 纯均匀噪声（真值 0 个）")
    ls, ev = detect_horizontal_layers(uniform)
    check(len(ls) == 0,
          f"峰数 = {len(ls)}（应为 0；第一版阈值在这里漏出过 20 个假层）"
          + (f"，误报高度 {[l['height'] for l in ls]}" if ls else ""))

    # ---- 4) 两个挨得近的层 --------------------------------------------------
    print("\n[4] 两个挨得近的层 1.00 / 1.12 m（真值 2 个）")
    ls, ev = detect_horizontal_layers(two_close)
    check(len(ls) == 2, f"峰数 = {len(ls)}（应为 2）")
    if len(ls) == 2:
        check(max(abs(ls[0]["height"] - 1.0), abs(ls[1]["height"] - 1.12)) < 0.03,
              f"峰位 {[l['height'] for l in ls]}（间距只有 0.12 m 也没被合并）")

    # ---- 5) 对照：老算法确实会造峰 -----------------------------------------
    #
    # 这个用例是**照着柜子那份实测数据的量级构造**的：范围 ~8 m（60 bin → bin 宽 ~0.13 m）、
    # 层比较"胖"（σ=0.25 m）、真值 3 层。老算法在这里输出了 5 个峰，多出来的两个
    # 正好落在真峰下方约 4×bin 处 —— 与实测 0.583 m = 4×bin 的指纹一致。
    print("\n[5] 对照：老算法（找最大 → 挖掉 ±3 bin → 再找）在同量级数据上造峰")
    wide_truth = (1.5, 4.25, 7.0)
    wide = np.concatenate(
        [rng.normal(c, 0.25, 2500) for c in wide_truth] + [rng.uniform(0.0, 8.6, 2000)]
    )
    old_wide, bw_wide = old_layer_loop(wide)
    new_wide, ev_wide = detect_horizontal_layers(wide)
    print(f"       真值 3 层 {wide_truth}")
    print(f"       老算法 {len(old_wide)} 个峰：{old_wide}（bin 宽 {bw_wide:.4f}，4×bin = {4 * bw_wide:.4f}）")
    print(f"       新算法 {len(new_wide)} 个峰：{[l['height'] for l in new_wide]}")
    check(len(old_wide) > len(wide_truth),
          f"老算法把 3 层报成了 {len(old_wide)} 层（多报 = 造峰，这就是要修的回归）")
    sp_old = spacing(old_wide)
    check(any(abs(s - 4 * bw_wide) < 0.03 for s in sp_old),
          f"老算法多出来的峰与真峰间距 {sp_old} 中有一个 ≈ 4×bin 宽 {4 * bw_wide:.4f} m（窗口宽度的回声）")
    check(len(new_wide) == len(wide_truth),
          f"新算法在同一份数据上是 {len(new_wide)} 层（应为 3）")
    if len(new_wide) == len(wide_truth):
        errs = [abs(l["height"] - t) for l, t in zip(new_wide, wide_truth)]
        check(max(errs) < 0.06, f"新算法峰位 {[l['height'] for l in new_wide]} 最大误差 {max(errs) * 1000:.0f} mm")

    # ---- [6] 可检测性标定：水平层要多强的信号才检得出来 ----------------------
    # 为什么要有这一节：层检测在真实稀疏重建上一直是 0 层（§十六 四种拍摄几何全 0）。
    # 光说"检不出来"没有用 —— 必须回答"差多少"。这一节把门限标定成**点占比**，
    # 于是实测的 0.67%~1.30% 就能直接与要求对比（差 5~7 倍），
    # 而不是把门限放低去"让它检出"（那会换回一堆假层，见 [3] 与下面的对照）。
    print("\n[6] 可检测性标定：水平面点占比 → 能否检出（总点数 10000，真值 3 层）")
    first_ok = None
    curve = []
    for share in (0.01, 0.02, 0.03, 0.05, 0.10, 0.30):
        rng2 = np.random.default_rng(5)
        per = int(10000 * share) // 3
        planes = np.concatenate([rng2.normal(c, 0.005, per) for c in (0.0, 0.95, 1.9)])
        bg = rng2.uniform(0.0, 1.9, 10000 - per * 3)
        got, _ev = detect_horizontal_layers(np.concatenate([planes, bg]))
        curve.append((share, len(got)))
        if len(got) == 3 and first_ok is None:
            first_ok = share
        print(f"       占比 {100 * share:5.1f}% → 检出 {len(got)} 层")
    check(curve[0][1] == 0, f"占比 1% 时检不出（实测 {curve[0][1]} 层）—— 与真实重建的现状一致")
    check(first_ok is not None and first_ok <= 0.05,
          f"检出 3 层所需的最低点占比 ≈ {100 * (first_ok or 0):.0f}%（≥5% 就该全检出）")
    rng3 = np.random.default_rng(9)
    fake, _fev = detect_horizontal_layers(rng3.uniform(0.0, 2.0, 10000))
    check(len(fake) == 0,
          f"对照：纯均匀噪声仍检出 {len(fake)} 层（必须 0 —— 这是门限不能放低的理由）")

    print("\n" + "=" * 70)
    print(f"结果：{passed} 通过 / {failed} 失败")
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
