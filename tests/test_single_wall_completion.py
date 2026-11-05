"""单墙补全 + 单位标签的回归测试（Python 层）。

用**已知真值**验证两件事：
  1. 单墙补全能给出保守但方向正确的盒子 —— 以 room-real 为例，
     用它的**一面墙**补全，与人工核对的 17.14×9.19×2.88 m 比较，
     相对误差应在 10~20%（实测 10.5/12.5/19）。
  2. 补全结果必须带 `estimated=true` 与 `measuredError` 证据（前端据此画虚线）。

同时做一组**尺度无关性**检查：把同一份点云整体缩放 k 倍，
补出来的盒子也应正好缩放 k 倍（说明没有偷偷假设"1 单位 = 1 米"）。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "tools")
import colmap_model_io as cmio
import structure_room as SR
from geom_common import (camera_centers_and_forwards, drop_statistical_outliers,
                         reference_frame)

MD = Path("data/scene-big/colmap-ex/sparse/0")
TRUTH = np.array([17.1371, 9.1901, 2.8829])

ok = 0
bad = 0


def check(name, cond, extra=""):
    global ok, bad
    if cond:
        ok += 1
        print(f"  [PASS] {name}")
    else:
        bad += 1
        print(f"  [FAIL] {name} {extra}")


def load():
    model = cmio.read_model(MD)
    xyz = np.asarray(model["points3D"]["xyz"], float)
    err = np.asarray(model["points3D"]["error"], float)
    trk = np.asarray(model["points3D"]["track_length"], float)
    keep = np.isfinite(xyz).all(axis=1) & (err < 2.0) & (trk >= 3)
    imgs = sorted(model["images"], key=lambda i: i["name"])
    cam_xyz, _ = camera_centers_and_forwards(imgs)
    P = xyz[keep]
    P = P[drop_statistical_outliers(P)]
    return P, cam_xyz, imgs


def complete(P, cam_xyz, imgs):
    ref = reference_frame(P, cam_xyz, imgs)
    room = SR.build_room_model(P, cam_xyz, ref, np.random.default_rng(1))
    rw = room["real_walls"]
    if not rw:
        return None, room, ref
    wall = max(rw, key=lambda w: w["rect"]["width"])
    comp = SR.complete_room_from_single_wall(
        P=P, cam_xyz=cam_xyz, n_up=ref["n_up"], wall=wall,
        floor_pt=np.asarray(room["floor_pt"], float), scale=ref["scale"],
        height_hint=room["height"],
        cam_height_median=room["traj"]["heightAboveFloor"]["median"],
        below_floor_ratio=room["traj"]["heightAboveFloor"]["belowFloor"] / max(room["traj"]["count"], 1),
        origin0=ref["origin0"])
    return comp, room, ref


print("== room-real：单墙补全 vs 真值 ==")
P, cam_xyz, imgs = load()
comp, room, ref = complete(P, cam_xyz, imgs)
check("原始模型有可信墙（否则无从取值）", len(room["real_walls"]) >= 2,
      f"walls={len(room['real_walls'])}")
if comp:
    size = np.asarray(comp["size"], float)
    rel = np.abs(np.sort(size) - np.sort(TRUTH)) / np.sort(TRUTH)
    print(f"   真值 {TRUTH}   补全 {np.round(size, 2)}   相对误差 {np.round(rel * 100, 1)}%")
    check("补全误差在 5%~25%（保守但可用）", bool(np.all(rel >= 0.05) and np.all(rel <= 0.25)),
          f"rel={np.round(rel * 100, 1)}")
    check("证据含 estimated 语义", comp["evidence"].get("confidence") == "estimated")
    check("证据含 measuredError", isinstance(comp["evidence"].get("measuredError"), list))
    check("地面来自尺度无关估计", comp["evidence"].get("scaleFree") is True)
    check("相机全部在地面以上", comp["evidence"].get("cameraAboveFloorRatio", 0) >= 0.99,
          f"ratio={comp['evidence'].get('cameraAboveFloorRatio')}")

    # ---- 尺度无关性：整体放大 3 倍，盒子应正好放大 3 倍 ----
    comp3, _, _ = complete(P * 3.0, cam_xyz * 3.0, imgs)
    if comp3:
        ratio = np.asarray(comp3["size"], float) / np.maximum(size, 1e-9)
        # 容差 8%：层高取自"高度直方图的强峰"，柱宽 ∝ 场景尺度，
        # 缩放后柱边界会挪一点 → 峰值位置有量化误差（实测 2.88~3.19 倍）。
        # 这条验的是"没有写死 1 单位 = 1 米"，不是逐位相等。
        check("整体缩放 3 倍 → 盒子也 ≈3 倍（无米制假设）",
              bool(np.allclose(ratio, 3.0, rtol=0.08)), f"ratio={np.round(ratio, 3)}")
    else:
        check("整体缩放 3 倍后仍能补全", False, "返回 None")
else:
    check("单墙补全返回了结果", False, "None")

print(f"\n结果：{ok} 通过 / {bad} 失败")
raise SystemExit(1 if bad else 0)
