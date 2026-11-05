#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回归：场景算法与物体算法**必须保持分开**（交接文档 §一 的步骤 ②）。

为什么需要这个测试
-----------------
`tools/scene_structure.py` 原来是一个 1600 多行的单文件，把三件事混在一起：
纯几何原语、房间算法、物体算法。混着的时候，"物体模式"只是末尾一个
`if scene_mode == "object"` 换锚点 —— 而两者的几何假设根本不同
（房间靠竖直平面、物体靠水平面；房间有确定地面、物体没有）。

拆开之后最容易发生的退化是**悄悄长回去**：下一个改动图省事，
又把新逻辑写进 `scene_structure.py`，几个月后又变成一个单文件。
所以这里用几条**结构断言**把它钉住：

  1. 三个模块都能独立导入，且 `scene_structure` 仍然再导出旧名字
     （外部探针/自检/测试是靠 `import scene_structure as ss` 用的，
      再导出断了等于把别人的脚本一次性全弄坏）
  2. `scene_structure.py` 是**调度层**：行数必须明显小于拆分前（1600+）
  3. 房间/物体各自的算法函数存在，且**不在** `scene_structure` 里定义
     （即 `build_room_model.__module__ == "structure_room"`）
  4. `geom_common` 不许出现 `align_frame_to_walls` 那种"靠模块级全局传参"的隐式依赖
     （它已经改成显式参数；这条防止有人再加回来）
  5. 房间盒的 8 角点硬校验仍然在（2026-10-06 那个真回归的护栏）

用法: pyruntime312\\python.exe tests/test_module_split.py
退出码 0 通过 / 1 失败
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import geom_common  # noqa: E402
import scene_structure as ss  # noqa: E402
import structure_object  # noqa: E402
import structure_room  # noqa: E402

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


def main():
    global passed, failed
    print("=" * 72)
    print("模块拆分回归（场景算法 / 物体算法 / 通用几何）")
    print("=" * 72)

    # ---- 1) 再导出（向后兼容）------------------------------------------------
    print("\n[1] scene_structure 仍然再导出旧名字（外部脚本靠它）")
    legacy = ["density_peak", "detect_horizontal_layers", "gravity_from_cameras",
              "drop_statistical_outliers", "align_frame_to_walls", "ransac_plane",
              "robust_frame", "orthonormal_basis", "segment_rooms", "plan_outline",
              "frustum_coverage", "extract_structure", "main"]
    missing = [n for n in legacy if not hasattr(ss, n)]
    check(not missing, f"全部可访问（缺失：{missing or '无'}）")
    # 而且必须是**同一个对象**，不能是复制粘贴出来的一份
    check(ss.density_peak is geom_common.density_peak,
          "再导出的 density_peak 与 geom_common 里是同一个对象（不是复制品）")
    check(ss.detect_horizontal_layers is geom_common.detect_horizontal_layers,
          "再导出的 detect_horizontal_layers 是同一个对象")
    check(ss.segment_rooms is structure_room.segment_rooms,
          "再导出的 segment_rooms 来自 structure_room")

    # ---- 2) 调度层不能重新长成单文件 ----------------------------------------
    print("\n[2] scene_structure.py 是调度层，不是单文件")
    n_lines = len((ROOT / "tools" / "scene_structure.py").read_text(encoding="utf-8").splitlines())
    print(f"       当前 {n_lines} 行（拆分前 1600+）")
    check(n_lines < 700, f"行数 {n_lines} < 700（超过说明又有人把算法写回调度层了）")

    # ---- 3) 两套算法各在自己的模块里 ----------------------------------------
    print("\n[3] 房间/物体算法各归其位")
    check(hasattr(structure_room, "build_room_model"), "structure_room.build_room_model 存在")
    check(hasattr(structure_object, "build_object_model"), "structure_object.build_object_model 存在")
    check(ss.build_room_model.__module__ == "structure_room",
          f"build_room_model 定义在 structure_room（实际 {ss.build_room_model.__module__}）")
    check(ss.build_object_model.__module__ == "structure_object",
          f"build_object_model 定义在 structure_object（实际 {ss.build_object_model.__module__}）")
    # 物体算法**不许**依赖房间算法（否则又耦合回去了）
    obj_src = (ROOT / "tools" / "structure_object.py").read_text(encoding="utf-8")
    check("import structure_room" not in obj_src and "from structure_room" not in obj_src,
          "structure_object 不导入 structure_room（物体算法不依赖房间算法）")
    # 通用几何**不许**包含分类策略（阈值属于策略，放调度层）
    # 注意判据是"有没有**定义**"，而不是"有没有提到" —— 注释里引用常量名是正常的。
    geom_src = (ROOT / "tools" / "geom_common.py").read_text(encoding="utf-8")
    check("SURROUND_ROOM_MIN =" not in geom_src and "SURROUND_OBJECT_MAX =" not in geom_src,
          "geom_common 里没有分类阈值的定义（量 vs 判分离：阈值在调度层）")
    check("SURROUND_ROOM_MIN = 0.38" in (ROOT / "tools" / "scene_structure.py").read_text(encoding="utf-8"),
          "分类阈值定义在 scene_structure（调度层）里")

    # ---- 4) 不许再用模块级全局传参 ------------------------------------------
    print("\n[4] align_frame_to_walls 的地面法向必须是显式参数")
    sig = inspect.signature(geom_common.align_frame_to_walls)
    check("n_up" in sig.parameters, f"签名含 n_up：{sig}")
    check(not hasattr(geom_common, "n_up_global"),
          "geom_common 里不再有 n_up_global（隐式全局依赖已移除）")
    for name in ("scene_structure", "structure_room"):
        src = (ROOT / "tools" / f"{name}.py").read_text(encoding="utf-8")
        check("n_up_global" not in src, f"{name}.py 里没有 n_up_global")

    # ---- 5) 旧回归的护栏还在 -------------------------------------------------
    print("\n[5] 房间盒 8 角点的硬校验仍在（2026-10-06 那个真回归）")
    room_src = (ROOT / "tools" / "structure_room.py").read_text(encoding="utf-8")
    check("len(room_corners) != 8" in room_src,
          "structure_room 里有「角点数必须为 8」的硬校验")
    check("room_size" in room_src and "room_origin" in room_src,
          "房间尺寸/原点由房间算法算好后交出（上层没机会拿墙的尺寸拼房间尺寸）")

    # ---- 6) 探针不许自己再实现一遍几何 --------------------------------------
    # 为什么值得钉住：探针是"先量再改"的依据，它一旦与线上实现漂移，
    # 量出来的数就不再代表线上行为，而且**不会报错**。
    # 交接文档里「-Z 是观察方向」那个错误就是两份实现不一致导致的。
    print("\n[6] 探针必须复用 geom_common，不许各自再实现一遍")
    for name in ("probe_scene_mode_features.py", "probe_ray_convergence.py"):
        src = (ROOT / "tools" / name).read_text(encoding="utf-8")
        check("import geom_common as gc" in src, f"{name} 导入了 geom_common")
    sc = (ROOT / "tools" / "probe_scene_mode_features.py").read_text(encoding="utf-8")
    check("camera_surround_coverage(" in sc or "gc.camera_surround_coverage" in sc,
          "包围度直接调 geom_common.camera_surround_coverage（同一函数、同一参数）")
    check("1 + 5 ** 0.5" not in sc and "1+5**0.5" not in sc,
          "probe_scene_mode_features 里不再有自己那份 Fibonacci 球面方向")
    rc = (ROOT / "tools" / "probe_ray_convergence.py").read_text(encoding="utf-8")
    check("quat_to_R" not in rc,
          "probe_ray_convergence 里不再自己把 qvec 换成前向（改用生产实现）")

    print("\n" + "=" * 72)
    print(f"结果：{passed} 通过 / {failed} 失败")
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
