#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""探针：水平层**要多强的信号才能被检测出来**（可检测性标定）。

为什么做这个
------------
层检测在真实稀疏重建上一直是 0 层。我量到的现状是：
  · 三块水平面只占柜子像素的 1.8%~5.5%（面积探针）；
  · 它们在点云里的占比 ≈ 均匀分布（±0.5% 高度带内 0.7%~1.3%，均匀期望 1.0%）。
于是问题变成：**"这么弱的信号"与"检测门限"差多少？**
这个脚本给出标定曲线：**水平面点占比 → 能不能检出**。

造数据的方式（与生产完全同一条路径：`detect_horizontal_layers`）：
  竖直背景（均匀分布，模拟柜子的侧板/背板）+ 三块水平面（底/搁板/顶），
  平面点占 `share`，总点数与真实柜子同量级（10000）。
  平面有 1 cm 厚度（真实重建里点不会严格在一个数学平面上）。

用法:
    pyruntime312\\python.exe tools\\probe_layer_detectability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from geom_common import detect_horizontal_layers  # noqa: E402


def synth(n_total: int, share: float, *, height: float = 1.9, thickness: float = 0.01, seed: int = 5):
    """造"竖直背景 + 三块水平面"，平面点占 share。返回竖直坐标数组。"""
    rng = np.random.default_rng(seed)
    n_plane = int(n_total * share)
    per = n_plane // 3
    zs = []
    for lvl in (0.0, height / 2, height):
        zs.append(rng.normal(lvl, thickness / 2, per))
    bg = rng.uniform(0, height, n_total - per * 3)      # 竖直面：高度上均匀分布
    return np.concatenate([np.concatenate(zs), bg])


def main() -> int:
    n_total = 10000
    print("水平层可检测性标定（同一条检测路径：geom_common.detect_horizontal_layers）")
    print("总点数 %d，真值 3 块水平面（底/搁板/顶）" % n_total)
    print("-" * 86)
    print(f"{'水平面点占比':>12}{'检出层数':>10}{'原始峰':>8}{'被突出度拒':>12}{'最高突出度':>12}   结论")
    first_ok = None
    for share in (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.30, 0.45):
        z = synth(n_total, share)
        layers, ev = detect_horizontal_layers(z)
        prom = [l["prominence"] for l in layers]
        ok = len(layers) == 3
        if ok and first_ok is None:
            first_ok = share
        print(f"{100 * share:>11.1f}%{len(layers):>10}{ev.get('rawPeakCount', 0):>8}"
              f"{ev.get('rejectedByProminence', 0):>12}{(max(prom) if prom else 0):>12.2f}"
              f"   {'✅ 三层都在' if ok else ('⚠ 只检出 %d 层' % len(layers) if layers else '❌ 一个都没检出')}")
    print("-" * 86)
    if first_ok is not None:
        print(f"→ **检出三块水平面所需的最低点占比 ≈ {100 * first_ok:.1f}%**")
    # 已知的均匀噪声行为（防止我把门限放低换来一堆假层）
    rng = np.random.default_rng(9)
    fake, fake_ev = detect_horizontal_layers(rng.uniform(0, 2.0, 10000))
    print(f"→ 对照：纯均匀噪声检出 {len(fake)} 层（应该 0 层；这是门限不能被放低的理由）")
    print("\n实测对照（真实稀疏重建，见 §十六/§十八）：三块水平面 ±0.5% 高度带内占比")
    print("   0.67%~1.30%（而均匀分布本来就给 1.00%）—— 也就是**连均匀水平都没有**。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
