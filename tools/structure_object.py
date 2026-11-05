#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""物体算法：以**相机轨迹为锚**的包围盒 + 水平层（柜子这类贴拍物体）。

为什么必须与房间算法分开（交接文档 §一 的步骤 ②）
--------------------------------------------------
两者的几何假设根本不同，混在一起时"末尾一个 if 换锚点"必然出错：

| | 房间 | 物体（柜子） |
| --- | --- | --- |
| 结构信息在哪 | **竖直**平面（墙） | **水平**面（顶面/搁板/底面） |
| 地面 | 确定存在（相机下方） | 未必存在 —— 实测这份柜子素材有 **30 台相机（23%）落到拟合地面以下** |
| 相机构型 | 在空间**内部**走动 | 在物体**外面**绕着看 |
| 锚点 | 地面（密度峰） | 相机轨迹质心 + 相机高度的稳健区间 |

所以物体这一套**不假设地面存在**：它只用「相机大致以固定距离、固定高度绕着物体转」
这条在环绕拍摄里非常稳的性质。这样"相机落在物体范围之外"在**定义上**就不可能发生。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT / "tools", ROOT / "pipeline"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from geom_common import detect_horizontal_layers  # noqa: E402

_SGN = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]

# ---------------------------------------------------------------------------
# 「物体盒可不可信」的判据阈值（**实测得来，不要凭感觉改**）
# ---------------------------------------------------------------------------
# 依据：tools/probe_object_trust.py（合成真值 + 真实素材对照）。全部是**尺度无关**的比值，
# 因为 COLMAP 的尺度是任意的。实测：
#
#   样本                                最长边/R  盒心偏移/R  ≤2R点占比  跨度/R
#   合成A 小柜 1.2×0.6×1.9 绕 1.9 m      0.63      0.01       1.00     0.78   ← 可信
#   合成B 大柜 2.5×1.0×2.2 绕 4.0 m      0.62      0.01       1.00     0.52   ← 可信
#   合成C 小柜 + 15% 背景野点             4.37      0.35       0.85     7.12   ← 盒子被背景撑坏
#   真实 柜子（一次贴面拍）                4.73      3.88       0.00     3.62   ← 几何已退化
#
# 三条一起用（缺一不可）：
#   1. 最长边/R ≤ 2.5   —— 物体不可能比"相机绕着它的半径"大出这么多，
#                          超了说明盒子被背景点撑大（C 与真实柜子都栽在这条）
#   2. 盒心偏移/R ≤ 1.0 —— 相机是**围着物体看**的，所以盒心必须靠近相机质心。
#                          真实柜子这条是 3.88：盒子中心离相机群 15 m 远
#   3. ≤2R 点占比 ≥ 0.6 —— 点云要**真的在相机附近**。真实柜子这条是 **0.00**：
#                          没有任何一个点落在相机群 2 倍半径之内
TRUST_MAX_EDGE_OVER_CAM_R = 2.5
TRUST_MAX_CENTER_OFFSET_OVER_CAM_R = 1.0
TRUST_MIN_FRAC_PTS_WITHIN_2CAM_R = 0.6

# 「点云本身够不够密」——**2026-10-06 端到端测试补的一条**。
# 起因：那三条比值判据会被"点极少 + 相机散得很开"的退化重建**轻松骗过**。
# 实测（data/models 里三个不确定模型）：
#     点数 248 / 571 / 250，相机 16 台，相机群半径 32.4 / 11.6 / 32.1
#     → 长/半径 0.58 / 0.36 / 0.51，偏移/半径 0.06 / 0.16 / 0.05，≤2R 占比 1.00
#     三条全过，可那点云根本不足以定义一个物体的形状。
# 对照（应当通过的真实重建，合成柜子 → 真 COLMAP，4 种拍摄几何）：
#     点数 8035~11127，36 台 → 223~309 点/相机
# 门限取 1500 点 / 50 点每相机，距离两侧都有 3~5 倍余量。
TRUST_MIN_POINTS_FOR_BOX = 1500
TRUST_MIN_POINTS_PER_CAMERA = 50

# 「物体 = 相机附近那批点」的半径倍数。
# 为什么要有这个限制（合成真值量出来的）：
#   合成C（小柜 + 15% 撒在 8~14 m 外的背景点）用**全部点**算盒子得到
#   18.93 × 19.07 × 19.06 m —— 完全没用；而把范围限制到相机群 2 倍半径内，
#   剩下的正是那个 1.2×0.6×1.9 m 的柜子。真实柜子素材在这条限制下剩 **0 个点**，
#   于是"物体盒不可信"这件事有了一个非常干脆的表达。
# 取 2 倍：相机绕半径 R 拍时，物体在锚点 ±(半对角线) 之内，实测 A/B 只有 0.5~0.7 R，
# 而背景点在 8~14 m（≫2R），两者分得很开。
OBJECT_NEAR_RADIUS_FACTOR = 2.0


def evaluate_object_trust(
    size: np.ndarray, center: np.ndarray, anchor: np.ndarray,
    cam_xyz: np.ndarray, P: np.ndarray,
) -> dict:
    """算出物体盒的**可信度证据**，决定前端该不该画它。

    为什么必须有这道闸：实测那份柜子素材算出来的盒子是 4.42 × 16.77 × 8.60 m
    （柜子实际约 1×0.5×2 m）。那不是"画得不准"，而是**几何已经退化** ——
    点云退化成货架棱边、且绝大部分点在十几米外的背景上。
    把它画到界面上，就是重演用户已经提过意见的"假房间盒"。
    所以这里的立场与房间盒一致：**宁可什么都不画，也不画一个假的**。
    """
    cam_r = float(np.median(np.linalg.norm(cam_xyz - anchor, axis=1))) if len(cam_xyz) else 0.0
    r = max(cam_r, 1e-9)
    size = np.asarray(size, float)
    d_pts = np.linalg.norm(P - anchor, axis=1) if len(P) else np.zeros(0)
    n_pts = int(len(P))
    n_cam = int(len(cam_xyz))
    metrics = {
        "camRadius": round(cam_r, 4),
        "longestEdgeMeters": round(float(size.max()), 4),
        "longestEdgeOverCamR": round(float(size.max() / r), 3),
        "centerOffsetMeters": round(float(np.linalg.norm(np.asarray(center, float) - anchor)), 4),
        "centerOffsetOverCamR": round(float(np.linalg.norm(np.asarray(center, float) - anchor) / r), 3),
        "fracPtsWithin2CamR": round(float((d_pts <= 2 * cam_r).mean()), 3) if len(d_pts) else 0.0,
        "pointsForBox": n_pts,
        "cameras": n_cam,
        "pointsPerCamera": round(n_pts / max(n_cam, 1), 1),
    }
    reasons: list[str] = []
    if n_cam < 8:
        reasons.append(f"相机太少（{n_cam} 台），无法判断是不是「绕着物体拍」")
    if metrics["longestEdgeOverCamR"] > TRUST_MAX_EDGE_OVER_CAM_R:
        reasons.append(
            f"物体盒最长边 {metrics['longestEdgeMeters']:.2f} m 是相机环绕半径 "
            f"{cam_r:.2f} m 的 {metrics['longestEdgeOverCamR']:.1f} 倍"
            f"（＞{TRUST_MAX_EDGE_OVER_CAM_R}）—— 盒子被背景点撑大了"
        )
    if metrics["centerOffsetOverCamR"] > TRUST_MAX_CENTER_OFFSET_OVER_CAM_R:
        reasons.append(
            f"盒心离相机群中心 {metrics['centerOffsetMeters']:.2f} m（= {metrics['centerOffsetOverCamR']:.1f} × 半径，"
            f"＞{TRUST_MAX_CENTER_OFFSET_OVER_CAM_R}）—— 相机不是围着这个盒子在拍"
        )
    if metrics["fracPtsWithin2CamR"] < TRUST_MIN_FRAC_PTS_WITHIN_2CAM_R:
        reasons.append(
            f"只有 {metrics['fracPtsWithin2CamR']:.0%} 的点落在相机群 2 倍半径内"
            f"（＜{TRUST_MIN_FRAC_PTS_WITHIN_2CAM_R:.0%}）—— 重建出来的点大多在远处的背景上"
        )
    # 点数密度：三条比值判据能被"点极少、相机散很开"的退化重建骗过（实测 248 点/16 台）
    if n_pts < TRUST_MIN_POINTS_FOR_BOX:
        reasons.append(
            f"相机附近只有 {n_pts} 个点（＜{TRUST_MIN_POINTS_FOR_BOX}）——"
            f"这么少的点定义不出物体形状，盒子是噪声拟合出来的"
        )
    if metrics["pointsPerCamera"] < TRUST_MIN_POINTS_PER_CAMERA:
        reasons.append(
            f"每台相机平均只有 {metrics['pointsPerCamera']} 个点"
            f"（＜{TRUST_MIN_POINTS_PER_CAMERA}）—— 重建太稀疏"
        )
    # 相机太少时无论比值多好都不算可信（比值本身不可靠）
    ok = not reasons
    return {
        "ok": bool(ok),
        "metrics": metrics,
        "thresholds": {
            "maxLongestEdgeOverCamR": TRUST_MAX_EDGE_OVER_CAM_R,
            "maxCenterOffsetOverCamR": TRUST_MAX_CENTER_OFFSET_OVER_CAM_R,
            "minFracPtsWithin2CamR": TRUST_MIN_FRAC_PTS_WITHIN_2CAM_R,
            "minPointsForBox": TRUST_MIN_POINTS_FOR_BOX,
            "minPointsPerCamera": TRUST_MIN_POINTS_PER_CAMERA,
            "source": "tools/probe_object_trust.py（合成真值 A/B 可信 0.62~0.63；"
                      "C 与真实柜子 4.4~4.7 判不可信）；点数门限由端到端合成柜子"
                      "（真 COLMAP，8035~11127 点 / 36 台）对照三个退化模型（248~571 点 / 16 台）定出",
        },
        "reasons": reasons,
        "note": "不可信时前端不应画物体盒与水平层（宁可什么都不画，也不要画假的）",
    }


def build_object_model(
    P: np.ndarray,
    cam_xyz: np.ndarray,
    *,
    u: np.ndarray,
    v: np.ndarray,
    n_up: np.ndarray,
    floor_pt: np.ndarray,
    scale: float,
    mode: str,
    layers: list[dict],
    layers_ev: dict,
) -> tuple[dict | None, list[dict], dict]:
    """返回 (物体记录, 水平层, 层证据)。

    物体包围盒用**重力+墙对齐**的坐标轴（比 PCA 主轴更符合人的直觉：
    宽/深/高就是人看柜子的三个方向），范围取 1/99 分位抗野点。

    `mode == "object"` 时锚在相机轨迹（环绕拍摄）；其余情况退回地面锚点。
    """
    if len(P) < 100:
        return None, layers, layers_ev
    axes = np.stack([u, v, n_up], axis=1)          # 3×3，列是基向量

    anchor = floor_pt
    # ⚠️ 2026-10-06 端到端之后改：**`uncertain` 也要按物体处理**。
    #    原来这里只认 `mode == "object"`，于是被判成 uncertain 的模型（而实测
    #    已知尺寸的柜子**全部**被判成 uncertain）会：
    #      · 锚点退回"地面密度峰"（物体拍摄里那个"地面"本来就是错的）；
    #      · 近点参考也跟着错（`d_anchor` 用的是地面峰，不是相机群中心）；
    #      · **水平层不重算**（层还是在房间坐标系里算的）；
    #      · 而 trust 判据的"盒心偏移"文档写的是**相机质心**，实际量的却是地面峰。
    #    既然前端已经按"trust 通过就画物体盒"来工作，物体分支就必须一视同仁：
    #    凡是**不是房间**（object / uncertain）都用相机轨迹锚 + 相机质心做参考。
    _object_style = mode != "room"
    if _object_style and len(cam_xyz) >= 8:
        anchor = cam_xyz.mean(axis=0)
        obj_anchor = {
            "method": "camera-orbit",
            "center": [round(float(x), 4) for x in anchor],
            "note": "物体/不确定模式：以相机轨迹质心为锚（环绕拍摄下比地面密度峰稳得多）",
        }
    else:
        obj_anchor = {"method": "floor-density-peak", "center": [round(float(x), 4) for x in anchor]}

    # ---- 包围盒 = **相机附近那批点**的稳健包围盒（1/99 分位）----
    #
    # ⚠️ 2026-10-06 两处都改过，都是被合成真值量出来的：
    #
    # (a) **去掉"按相机高度带裁剪竖直范围"**。原来物体模式会把竖直范围裁到相机高度的
    #     P2/P98 再外扩，理由是"这样相机落在盒外是定义上成立的"。但人拍柜子时相机在
    #     眼睛高度（约 1.5 m），柜子却是从地面到 2 m —— 相机高度带**根本不覆盖物体**。
    #     合成真值当场暴露：1.2×0.6×**1.9** m 的柜子、相机绕 1.9 m 一圈（高度抖动 ±0.2 m），
    #     算出来的盒子高度只有 **0.58 m**（砍掉 70%）。
    #
    # (b) **只用"相机附近"的点**。相机是围着物体看的，所以物体必然在锚点附近；
    #     而 8~14 m 外的背景点只会把盒子撑坏。合成C（小柜 + 15% 背景点）用全部点算出
    #     18.93×19.07×19.06 m，限制到 2×相机半径内就恢复成那个 1.2×0.6×1.9 m 的柜子。
    #     真实柜子素材在这条限制下剩 **0 个点** —— "几何已退化"变得不可辩驳。
    cam_r = float(np.median(np.linalg.norm(cam_xyz - anchor, axis=1))) if len(cam_xyz) else 0.0
    near_r = OBJECT_NEAR_RADIUS_FACTOR * max(cam_r, 1e-9)
    d_anchor = np.linalg.norm(P - anchor, axis=1) if len(P) else np.zeros(0)
    near = d_anchor <= near_r if len(P) else np.zeros(0, dtype=bool)
    P_near = P[near] if len(P) else P
    near_frac = float(near.mean()) if len(P) else 0.0

    if len(P_near) >= 50:
        proj = (P_near - anchor) @ axes
    else:
        # 相机附近几乎没有点：盒子无从谈起。仍给一个"空盒"，
        # 由 trust 判据说清原因（不要在这里抛异常 —— 结构提取是增强步骤）。
        proj = (P - anchor) @ axes if len(P) else np.zeros((0, 3))
    if len(proj) >= 50:
        o_lo, o_hi = np.percentile(proj, 1, axis=0), np.percentile(proj, 99, axis=0)
    else:
        o_lo, o_hi = np.zeros(3), np.zeros(3)
    o_size = o_hi - o_lo
    o_center = anchor + axes @ ((o_lo + o_hi) / 2.0)

    # 物体模式**重算水平层**：房间那一套用的是相对 `floor_pt` 的高度，
    # 而那个"地面"在物体模式里本来就不可靠。改用物体锚定的竖直坐标（proj[:,2]）。
    #
    # ⚠️ 无论找没找到层都要保留 `new_ev`：它是"**为什么**没找到层"的唯一证据
    #    （bin 宽 / 原始峰数 / 被突出度拒掉几个 / reason）。
    #    第一版写成 `if new_layers:` 才赋值，于是 0 层时把证据丢成 `{}` ——
    #    端到端测试里我因此看不出"是数据没峰，还是阈值太严"。
    if _object_style and len(proj) >= 200:
        new_layers, new_ev = detect_horizontal_layers(proj[:, 2])
        layers_ev = new_ev
        if new_layers:
            layers = new_layers
    o_corners = [o_center + axes @ (np.array(s, dtype=float) * o_size / 2.0) for s in _SGN]
    # 参考点必须是**相机质心**（判据文档写的就是这个）；对 room 模式锚点可能是地面峰，
    # 那时如果直接拿 anchor 当参考，"盒心偏移"量错对象 —— trust 会给出无意义的结论。
    trust = evaluate_object_trust(o_size, o_center, cam_xyz.mean(axis=0), cam_xyz, P_near)
    trust["metrics"]["nearRadiusMeters"] = round(float(near_r), 4)
    trust["metrics"]["nearRadiusFactor"] = OBJECT_NEAR_RADIUS_FACTOR
    trust["metrics"]["fracAllPtsUsedForBox"] = round(near_frac, 3)
    trust["metrics"]["pointsUsedForBox"] = int(len(P_near))
    obj = {
        "mode": "object",
        "center": [round(float(x), 4) for x in o_center],
        "size": [round(float(x), 4) for x in o_size],       # 宽 / 深 / 高（对齐墙面与重力）
        "corners": [[round(float(x), 4) for x in c] for c in o_corners],
        "axes": {
            "u": [round(float(x), 6) for x in u],
            "v": [round(float(x), 6) for x in v],
            "up": [round(float(x), 6) for x in n_up],
        },
        "layers": layers,          # 水平层 = 柜子的顶面/搁板/底面（按高度排序）
        "layersEvidence": layers_ev,   # 层检测的判据证据（bin 宽 / 原始峰数 / 突出度阈值）
        "anchor": obj_anchor,      # 锚点来源（相机轨迹 or 地面密度峰）
        # 可信度：ok=false 时前端**不应**画这个盒子（实测那份柜子的盒子是 16.7 m）
        "trust": trust,
        "note": "重力+墙对齐的稳健包围盒；size 依次为宽/深/高。layers 是水平面（两层柜=2 个峰）。"
                "trust.ok=false 表示几何已退化，不要画",
    }
    return obj, layers, layers_ev
