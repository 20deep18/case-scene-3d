#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 COLMAP 稀疏点云 + 相机位姿里提取**现场的结构骨架**（调度层）。

为什么需要它
------------
3D 高斯泼溅无论重建得多好，呈现出来的都是一团「照片云」：它能让你看见**材质**，
但看不见**结构** —— 哪边是地面、墙在哪里、房间多大、人是怎么走的。
用户的原话是「没有呈现出房间的整体布局和空间线条」，这就是缺的东西。

拆分说明（交接文档 §一 的步骤 ②，2026-10-06 完成）
---------------------------------------------------
本文件现在只做**调度与装配**，三块内容各归其位：

    tools/geom_common.py      纯几何/统计原语（平面拟合、重力、密度峰、包围度、离群剔除）
    tools/structure_room.py   房间算法（地面/天花板/竖墙/房间盒/单间/平面轮廓/轨迹）
    tools/structure_object.py 物体算法（以相机轨迹为锚的包围盒 + 水平层）
    tools/scene_structure.py  本文件：读模型 → 洗点 → 三态分类 → 调两套算法 → 装配 JSON

**分类夹在房间算法与物体算法之间**，而且判据本身**不依赖**房间算法是否成功
（只看相机位姿与点云包围度）—— 这正是交接文档指出的那个坑：
「用下游拟合失败的结果去决定上游分类」。

判据的实测依据见 `docs/场景与物体-判据实测与修复.md`：
文档提议的「视线汇聚度」在真实柜子素材上判反（柜子 1.0000 vs 房间 0.9652），
故改用「方向包围度」+ 三态保守判定（room / object / uncertain）+ `--scene-mode` 人工覆盖。

用法
----
    python tools/scene_structure.py --model <稀疏模型目录> --out scene.structure.json
    python tools/scene_structure.py --model <...> --out <...> --scene-mode room   # 人工改判
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT / "tools", ROOT / "pipeline"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import colmap_model_io as cmio  # noqa: E402

from geom_common import (  # noqa: E402
    camera_centers_and_forwards,
    camera_ray_convergence,
    camera_surround_coverage,
    drop_statistical_outliers,
    frustum_coverage,
    near_camera_protect_mask,
    reference_frame,
)
from structure_object import build_object_model  # noqa: E402
from structure_room import build_room_model, complete_room_from_single_wall  # noqa: E402

# ---------------------------------------------------------------------------
# 场景分类阈值（**策略，实测得来，不要凭感觉改**）
# ---------------------------------------------------------------------------
# 「方向包围度」= 相机周围 256 个方向上，有多少比例能找到重建点（见
# geom_common.camera_surround_coverage 的说明）。实测（tools/probe_scene_mode_features.py）：
#     注册良好的房间：0.523（room-real 178 相机）/ 0.523（run-100mb）/ 0.406（run-64mb）
#     柜子贴拍      ：0.234（130 相机）
#     ⚠️ 同一段房间视频用 sequential 匹配（只注册 61/175）：0.227
# 灰区（0.28~0.38）故意留宽：宁可报"不确定"，也不要猜。
#
# ⚠️ 阈值放在**调度层**而不是 geom_common：那里只做测量、不做判断。
SURROUND_ROOM_MIN = 0.38
SURROUND_OBJECT_MAX = 0.28

# ---------------------------------------------------------------------------
# 向后兼容的再导出
# ---------------------------------------------------------------------------
# 拆分之前这些东西都在本模块里，外部（探针/自检/测试）是靠
# `import scene_structure as ss; ss.xxx` 或 `from scene_structure import xxx` 用的。
# 拆完仍然从本模块再导出一次，避免为了搬家去改一堆调用点（也更不容易漏）。
# ⚠️ 新代码请直接从 geom_common / structure_room / structure_object 导入。
from geom_common import (  # noqa: E402,F401
    _local_maxima,
    _prominence,
    align_frame_to_walls,
    density_peak,
    detect_horizontal_layers,
    fit_plane_lsq,
    gravity_from_cameras,
    orthonormal_basis,
    ransac_plane,
    robust_frame,
)
from structure_room import plan_outline, segment_rooms  # noqa: E402,F401

__all__ = [
    "extract_structure",
    "main",
    "density_peak",
    "detect_horizontal_layers",
    "gravity_from_cameras",
    "drop_statistical_outliers",
    "camera_ray_convergence",
    "camera_surround_coverage",
    "camera_centers_and_forwards",
    "reference_frame",
    "frustum_coverage",
    "align_frame_to_walls",
    "build_room_model",
    "build_object_model",
]


def _empty_structure(model_dir: Path, reason: str, images: list[dict] | None = None) -> dict:
    """点云为空/过少时的**降级输出**：不崩，但明确说"不可信"。

    为什么要有这条路径：COLMAP 偶尔会产出**空模型**（实测 `output/threshold/run-4mb`
    只有 8 字节的 points3D.bin）。原来这里直接
    `np.isfinite(xyz).all(axis=1)` → `AxisError: axis 1 is out of bounds for array of
    dimension 1`（空模型读出来是 1 维数组），整个结构提取**崩掉**，
    于是结构文件根本不生成，前端什么都没有、也没有任何解释。

    三态判定里本来就有 `uncertain`，空点云正是它该出现的地方 ——
    宁可输出一份"明说不可信"的结构文件，也不要抛异常让上游猜。
    相机轨迹仍然给出：它只依赖位姿，不依赖点云，是这种情况下唯一还有用的信息。
    """
    cam_xyz, _ = camera_centers_and_forwards(images or [])
    cams = [[round(float(x), 5) for x in c] for c in cam_xyz]
    return {
        "version": 1,
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": str(model_dir),
        "generator": "tools/scene_structure.py",
        "sane": False,
        "degraded": True,
        "stats": {"pointsTotal": 0, "pointsUsed": 0, "pointsKeptByError": 0,
                  "scaleP95": 0.0, "epsMeters": 0.0, "elapsedSec": 0.0},
        "gravity": {"source": "unavailable", "frames": 0},
        "orientation": {"upVsGravityCos": None, "ok": False, "note": "点云为空，未拟合"},
        "frame": {"up": [0, 0, 1], "origin": [0, 0, 0], "axisU": [1, 0, 0], "axisV": [0, 1, 0]},
        "ground": {"normal": [0, 0, 1], "point": [0, 0, 0], "inliers": 0, "rms": None,
                   "ratio": 0.0, "method": "未拟合（点云为空）", "eyeHeightMeters": None},
        "ceiling": {"found": False, "height": 0.0, "method": "未拟合（点云为空）",
                    "cappedByCameraHeight": False, "cameraP95AboveFloor": None},
        "walls": [],
        "sceneMode": "uncertain",
        "sceneModeSource": "auto",
        "sceneModeEvidence": {
            "mode": "uncertain",
            "rejectReasons": [reason],
            "degraded": True,
            "note": "点云为空/过少，无法做任何几何判定；前端不应画房间盒或物体盒",
        },
        "object": None,
        "roomLikeWalls": 0,
        "frameAlignment": {"alignedToWalls": False, "rotationDeg": 0.0,
                           "note": "未做（点云为空）"},
        "rooms": [],
        "roomsQuality": {"usable": False, "note": "点云为空"},
        "planOutline": {"segmentsWorld": []},
        "room": {"size": [0.0, 0.0, 0.0], "floorY": 0.0, "ceilingY": 0.0, "corners": [],
                 "axisU": [1, 0, 0], "axisV": [0, 1, 0], "origin": [0, 0, 0]},
        "trajectory": {"count": len(cams), "points": cams, "lengthMeters": 0.0,
                       "netDisplacement": 0.0, "plan": [], "heightAboveFloor": None},
        "coverage": {},
    }


def classify_scene_mode(
    cam_xyz: np.ndarray,
    cam_fwd: np.ndarray,
    P_before_clean: np.ndarray,
    P_cleaned: np.ndarray,
    *,
    room_like_walls: int,
    cam_ok: bool,
    cam_h_rel: np.ndarray,
) -> tuple[str, str, bool, bool, list[str], dict]:
    """三态判定：room / object / uncertain，并给出**全量证据**。

    判据只用相机位姿与点云的**方向包围度**，不用任何拟合结果 ——
    这是交接文档的核心诉求（原判据拿"地面是否拟合成功"当分类依据，
    等于用下游失败的结果决定上游分类）。

    实测（tools/probe_scene_mode_features.py，2026-10-06）：
        注册良好的房间：0.523 / 0.523 / 0.406（三份素材）
        柜子贴拍      ：0.234
        ⚠️ 同一段房间视频用 sequential 匹配（只注册 61/175）：0.227
    最后一行说明这个量**同时反映重建质量**，所以它只作佐证：
    包围度低却"相机高度自洽得像房间"时**不猜**，报 uncertain。

    返回 (mode, mode_source, room_strong, object_strong, reasons, evidence)
    """
    raymode = camera_ray_convergence(cam_xyz, cam_fwd)
    # 证据用**洗点前**的点集：`drop_statistical_outliers` 是相对密度判据，
    # 会系统性删掉离相机最近的表面点（房间的"被包围"恰恰由那批近点体现）。
    # 实测 room-real：洗点前 0.5234 / 洗点后 0.2344 —— 证据会被洗没。
    surround = camera_surround_coverage(cam_xyz, P_before_clean)
    surround_after = camera_surround_coverage(cam_xyz, P_cleaned)
    if surround.get("ok") and surround_after.get("ok"):
        surround["medianAfterOutlierRemoval"] = surround_after.get("median")
        surround["note"] = (
            surround.get("note", "")
            + f"（另附洗点后的值 {surround_after.get('median')}：SOR 会删掉离相机最近的表面点，"
              "故判定统一用洗点前的值）"
        )
    cov_med = surround.get("median") if surround.get("ok") else None

    reasons: list[str] = []
    room_strong = False
    object_strong = False
    if cov_med is None:
        reasons.append(f"方向包围度算不出来（{surround.get('reason', '点数/相机不足')}）")
    if room_like_walls < 2:
        reasons.append(f"可信竖墙只有 {room_like_walls} 面（房间判据要 ≥2 面）")
    if not cam_ok:
        reasons.append(
            f"相机高度不自洽：中位 {float(np.median(cam_h_rel)):.2f} m，"
            f"落在拟合地面以下 {int((cam_h_rel < -0.2).sum())} 台 —— 说明地面/重力估计不可信"
        )
    if cov_med is not None:
        if cov_med >= SURROUND_ROOM_MIN:
            if room_like_walls >= 2 and cam_ok:
                room_strong = True
            else:
                reasons.append(
                    f"方向包围度 {cov_med:.3f} 像房间（≥{SURROUND_ROOM_MIN}），"
                    "但墙/相机高度这两项不支持"
                )
        elif cov_med <= SURROUND_OBJECT_MAX:
            if not cam_ok:
                object_strong = True
            else:
                # 关键的一道防线：**姿态上像个自洽的房间，点云却不包围相机** ——
                # 这更像是重建质量差（实测 sequential 匹配的房间就是 0.227），
                # 而不是"这是个物体"。此时不猜，报不确定。
                reasons.append(
                    f"方向包围度 {cov_med:.3f} 偏低（≤{SURROUND_OBJECT_MAX}）本像物体，"
                    "但相机高度自洽得像房间 → 更可能是重建质量差，故不判定"
                )
        else:
            reasons.append(
                f"方向包围度 {cov_med:.3f} 落在灰区"
                f"（{SURROUND_OBJECT_MAX}~{SURROUND_ROOM_MIN}），既不像房间也不像贴拍物体"
            )

    mode = "room" if room_strong else ("object" if object_strong else "uncertain")
    evidence = {
        "mode": mode,
        "modeSource": "auto",
        "autoDecision": mode,
        "thresholds": {
            "surroundRoomMin": SURROUND_ROOM_MIN,
            "surroundObjectMax": SURROUND_OBJECT_MAX,
            "thresholdSource": "实测：注册良好的房间 0.406~0.523；柜子贴拍 0.234；"
                               "重建差的房间（sequential 61/175）0.227",
        },
        "surroundCoverage": surround,
        "rayConvergence": raymode,
        "roomLikeWalls": room_like_walls,
        "cameraHeight": {
            "medianAboveFittedFloor": round(float(np.median(cam_h_rel)), 3) if len(cam_h_rel) else None,
            "belowFittedFloor": int((cam_h_rel < -0.2).sum()) if len(cam_h_rel) else None,
            "sane": cam_ok,
        },
        "rejectReasons": reasons,
        "note": "三态判定：证据强才敢断言；其余为 uncertain，前端不应画房间盒/地面网格",
    }
    return mode, "auto", room_strong, object_strong, reasons, evidence


def extract_structure(model_dir: Path, *, seed: int = 20261005, force_mode: str | None = None,
                      protect_near_camera: bool = False, min_track: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    model = cmio.read_model(model_dir)
    pts = model["points3D"]
    xyz = np.asarray(pts["xyz"], dtype=float)
    # 空模型/点太少：降级输出而不是崩（见 _empty_structure 的说明）
    _n_pts = int(xyz.reshape(-1, 3).shape[0]) if xyz.size else 0
    if xyz.ndim != 2 or _n_pts < 50:
        return _empty_structure(
            model_dir,
            f"稀疏点云不足以做结构提取（只有 {_n_pts} 个点）"
            + ("：COLMAP 没能重建出这个现场" if _n_pts == 0 else "：几何证据太少"),
            model.get("images"),
        )
    err = np.asarray(pts["error"], dtype=float)
    trk = np.asarray(pts["track_length"], dtype=float)

    # 只用「重投影误差小、track 够长」的点参与结构拟合：
    # 这些点才是可信的几何证据；野点会把平面法向带歪。
    # min_track 默认 3（COLMAP 正常三角化）；VGGT/位姿直给的模型没有真实 track，
    # 传 1 才能用（否则一个点都过不了，结构直接判空）。
    keep = np.isfinite(xyz).all(axis=1) & (err < 2.0) & (trk >= min_track)
    P = xyz[keep]
    # ⚠️ 留一份**洗点之前**的点集给"相机是否被点云包围"这个证据用（见 classify_scene_mode）。
    #    drop_statistical_outliers 是**相对密度**判据，会系统性地删掉
    #    **离相机最近的那批表面点** —— 因为贴着相机的表面在三维里天生更稀疏。
    #    实测：room-real 洗点前最近点中位 1.065 m / 方向包围度 0.5234；
    #          洗点后 3.782 m / 0.2344（证据被洗没了，与柜子的 0.234 无法区分）。
    #    洗点对"拟合平面/包围盒"是必要的（它治好了 229 m → 25 m 的跨度问题），
    #    故两者分开取用，并把两个数值都写进证据里，便于复核。
    P_before_clean = P

    # ⚠️ 必须按文件名排序再当时间序列用：COLMAP 的 image_id 顺序不等于拍摄顺序
    #    （视频抽帧的帧名是 frame_0001…，字典序即时间序）。
    # 相机位姿要在**洗点之前**算好：保护掩码需要它（见下）。
    images_sorted = sorted(model["images"], key=lambda im: im["name"])
    cam_xyz, cam_fwd = camera_centers_and_forwards(images_sorted)

    # ---- 洗点：统计式离群点剔除必须在**任何拟合之前** ----------------------
    # 实测某份重建：全跨度 229 m，稳健(1/99)跨度只有 25 m（差 9 倍）。
    # 这批极端野点会把包围盒/取景撑大，还会让 RANSAC 拟合出「假平面」，
    # 于是输出「墙宽 11~16 m 而房间只有 6.8 m」「相机跑到地面以下」这类自相矛盾的结果。
    #
    # ⚠️ 但 SOR 是**相对密度**判据，会把「贴着相机的表面」也误杀（它们天生稀疏）：
    #    实测 room-real 删掉的 4417 点里有 1828 个是"离相机 ≤ 中位"的，
    #    它们到最近保留点只有 0.362 m —— 是真实表面，而且视差最大、三角化最准。
    #
    #    `--protect-near-camera` 可以保护这批点。但**默认不开**，因为**实测收益不明显**：
    #      roomreal 43748 → 44949 点，房间 17.14×9.19×2.88 → 17.11×9.26×2.61，墙 5 → 4
    #      guizi    12931 → 13638 点，房间/判定基本不变
    #      r64      28925 → 29383 点，几乎不变
    #    也就是说"删掉近表面会明显拖坏拟合"这个担心**被实测否掉了**（见 docs §四）。
    #    没有证据就不动默认值 —— 开关留着供对照。
    _protect = None
    if protect_near_camera:
        _protect = near_camera_protect_mask(P, cam_xyz, cam_fwd)
    _out_ok = drop_statistical_outliers(P, protect=_protect)
    _removed = int((~_out_ok).sum())
    _protected = int((_protect & ~_out_ok).sum()) if _protect is not None else 0
    P = P[_out_ok]
    if _removed:
        print(f"  离群点剔除：去掉 {_removed} / {len(_out_ok)} 个稀疏点（{_removed/max(len(_out_ok),1):.1%}）"
              + (f"；其中 {_protected} 个因「贴近相机且在相机前方」被保护性保留" if _protected else ""))

    t0 = time.time()

    # ---- 公共参考系（尺度 / 重力「上」 / 地面轴 / 高度分布）
    ref = reference_frame(P, cam_xyz, images_sorted)
    scale = ref["scale"]
    eps = ref["eps"]
    n_up = ref["n_up"]
    u = ref["u"]
    v = ref["v"]
    hh = None  # 由房间算法给出（相对地面）
    grav_info = ref["grav_info"]
    up_vs_gravity = ref["up_vs_gravity"]

    # ---- 房间算法（无论最终判成什么都要跑：分类需要它给出的墙/相机高度证据）
    room = build_room_model(P, cam_xyz, ref, rng)
    floor_pt = room["floor_pt"]
    floor_source = room["floor_source"]
    ceil_source = room["ceil_source"]
    eye = room["eye"]
    height = room["height"]
    height_capped = room["height_capped"]
    cam_hi_rel = room["cam_hi_rel"]
    wall_lines = room["wall_lines"]
    wall_theta_deg = room["wall_theta_deg"]
    room_corners = room["room_corners"]
    u = room["u"]
    v = room["v"]
    uu = room["uu"]
    vv = room["vv"]
    use = room["use"]
    hh = room["hh"]
    traj = room["traj"]
    rooms = room["rooms"]
    outline = room["outline"]
    wall_theta_deg = room["wall_theta_deg"]
    real_walls = room["real_walls"]
    cam_ok = room["cam_ok"]
    cam_h_rel = room["cam_h_rel"]

    # ---- 场景分类：房间 / 物体 / **不确定**
    scene_mode, scene_mode_source, room_strong, object_strong, reasons, scene_mode_evidence = classify_scene_mode(
        cam_xyz, cam_fwd, P_before_clean, P,
        room_like_walls=len(real_walls), cam_ok=cam_ok, cam_h_rel=cam_h_rel,
    )
    # ---- 人工覆盖（用户可覆盖）--------------------------------------------
    # 几何本身**分不开**这两种素材：柜子那段视频与 run-32mb 那段"以旋转为主"的
    # 房间视频，在相机位姿统计上属于同一类拍摄方式（贴着看、转动为主、平移很少），
    # 区别只在镜头前是柜子还是屋子 —— 那是**语义**，几何量拿不到。
    # 所以判不准时不该硬猜，而应该：① 报不确定并说明；② 允许拍摄的人直接指定。
    if force_mode in ("room", "object"):
        if force_mode != scene_mode:
            reasons.insert(0, f"人工指定为 {force_mode}（自动判定原本是 {scene_mode}）")
        scene_mode = force_mode
        scene_mode_source = "manual"
    scene_mode_evidence["mode"] = scene_mode
    scene_mode_evidence["modeSource"] = scene_mode_source

    # ---- 物体算法
    # ⚠️ 2026-10-06 端到端测试后改：**不确定时也要算物体几何**（原来直接跳过）。
    #
    # 原来的理由写的是"一个来路不明的盒子画到界面上比不画更误导"——那个顾虑是对的，
    # 但用错了手段：物体盒**自带可信判据**（`object.trust`：盒长/相机半径、
    # 盒心偏移、相机附近点占比三条），它才是"这个盒子可不可信"的直接答案。
    # 用"场景模式不确定"去一刀切，会把**几何完全可信**的物体现场也一起毙掉。
    #
    # 实测（tools/make_object_dataset.py 生成的已知尺寸柜子 → 真 COLMAP）：
    #   4 种拍摄几何（平视 / 俯仰0.55 / 俯仰0.8+抬高 / 探头看柜内）重建出来全都是
    #   `trust.ok=True`（比值 0.68 / 0.07~0.11 / 1.00），但场景判定一律给 `uncertain`
    #   （方向包围度 0.11~0.14 像物体，可"相机高度自洽得像房间"这条把它压回不确定）——
    #   于是**物体记录被整条抑制**，前端永远画不出那个其实正确的盒子。
    #
    # 现在：几何照算、trust 照评、证据照写进 json。**画不画由 trust 决定**（前端门控），
    # 而不是由那个更模糊的场景模式判定决定。
    obj = None
    layers = room["layers"]
    layers_ev = room["layers_ev"]
    obj, layers, layers_ev = build_object_model(
        P, cam_xyz,
        u=u, v=v, n_up=n_up, floor_pt=floor_pt, scale=scale,
        mode=scene_mode, layers=layers, layers_ev=layers_ev,
    )
    if scene_mode == "uncertain" and obj is not None:
        # 场景模式仍报 uncertain（那是"房间/物体"的判断，证据确实不够）；
        # 但把物体几何的可信度单独交代清楚，供前端与人工复核。
        obj["note"] = (
            "场景模式判定为不确定（房间/物体的证据都不足），但物体几何照算并附可信度："
            + ("trust.ok=true —— 几何自检通过，可作为「物体」呈现" if (obj.get("trust") or {}).get("ok")
               else "trust.ok=false —— 几何不可信，不要画")
        )

    camera_height = traj["heightAboveFloor"]
    # 方向自检：箱体「上」必须是重力反方向。
    # 这条是补上的 —— 之前只校验「相机离地 1.5 m」，而那个条件对**倒扣的盒子**
    # 同样成立，所以一个上下颠倒的模型能一路骗过自检（实测踩过）。
    orientation_ok = bool(up_vs_gravity > 0.99)
    sane = bool(
        orientation_ok
        and camera_height
        and 0.6 <= camera_height["median"] <= 2.6
        and camera_height["belowFloor"] <= max(2, 0.05 * traj["count"])
    )

    # ---- 单墙曼哈顿补全（专家方案 a）----------------------------------------
    # 背景：白瓷砖卫生间只找到 **1 面**可信竖墙，自检因此 sane=false，房间盒不画，
    #       下游的家具盒也失去参考系。专家建议的兜底是：
    #       "地面平面 + 相机高度 + 曼哈顿正交先验"给出**保守但方向正确**的盒，
    #       并明确标成「推测」——比"什么都不给"有用得多，也比"硬拟合一个歪盒子"诚实。
    estimated = False
    estimation = None
    if (not sane) and len(real_walls) == 1:
        comp = complete_room_from_single_wall(
            P=P, cam_xyz=cam_xyz, n_up=n_up,
            wall=real_walls[0], floor_pt=floor_pt, scale=scale,
            height_hint=height,
            cam_height_median=(camera_height or {}).get("median"),
            below_floor_ratio=(
                float(camera_height["belowFloor"]) / max(int(traj.get("count", 0)), 1)
                if camera_height else 0.0
            ),
            eye_prior=1.5,
            origin0=ref["origin0"],
        )
        if comp is not None:
            estimated = True
            estimation = comp["evidence"]
            sane = True                       # 允许画，但 front-end 必须显示「推测」
            room_corners = comp["corners"]
            room["room_size"] = comp["size"]
            room["room_origin"] = comp["origin"]
            u, v = comp["u"], comp["v"]
            floor_pt = np.asarray(comp["floor_pt"], dtype=float)
            height = comp["size"][2]
            if scene_mode != "room":
                scene_mode = "room"
                scene_mode_source = "auto"
                scene_mode_evidence["note"] = (
                    "判定依据：只有 1 面可信竖墙 → 单墙曼哈顿补全（estimated）"
                )
            print(f"  单墙补全：房间 {comp['size'][0]:.2f}×{comp['size'][1]:.2f}×{comp['size'][2]:.2f}"
                  f"（地面={comp['evidence']['floorSource']}，推测）")

    # ---- 覆盖度：哪些方向其实从来没被拍到过（决定模型会在哪里塌）
    # 只用可信点（重投影误差小、track 长）来数，避免野点把覆盖度虚高
    coverage = frustum_coverage(model, P, [im["name"] for im in images_sorted])

    return {
        "version": 1,
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": str(model_dir),
        "generator": "tools/scene_structure.py",
        "sane": sane,
        "stats": {
            "pointsTotal": int(len(xyz)),
            "pointsUsed": int(len(P)),
            "pointsKeptByError": int(keep.sum()),
            "pointsProtectedNearCamera": int(_protected),
            "protectNearCamera": bool(protect_near_camera),
            "scaleP95": round(scale, 4),
            "epsMeters": round(eps, 5),
            "elapsedSec": round(time.time() - t0, 3),
        },
        "gravity": grav_info,
        "orientation": {
            "upVsGravityCos": round(up_vs_gravity, 6),
            "ok": orientation_ok,
            "note": "箱体「上」与重力反方向的夹角余弦；必须 > 0.99，否则整个房间盒是倒扣的",
        },
        "frame": {
            "up": [round(float(x), 6) for x in n_up],
            "origin": [round(float(x), 5) for x in floor_pt],
            "axisU": [round(float(x), 6) for x in u],
            "axisV": [round(float(x), 6) for x in v],
        },
        "ground": {
            "normal": [round(float(x), 6) for x in n_up],
            "point": [round(float(x), 5) for x in floor_pt],
            "inliers": int(((hh > -0.06 * scale) & (hh < 0.06 * scale)).sum()),
            "rms": None,
            "ratio": round(float(((hh > -0.06 * scale) & (hh < 0.06 * scale)).mean()), 4),
            "method": f"{floor_source} + 相机离地 {eye:.2f} m 校验",
            "eyeHeightMeters": round(float(eye), 3),
        },
        "ceiling": {
            "found": True,
            "height": round(float(height), 4),
            "method": ceil_source,
            "cappedByCameraHeight": height_capped,
            "cameraP95AboveFloor": round(float(cam_hi_rel), 3),
        },
        "walls": wall_lines,
        # A：场景分类与物体包围盒。
        # "room"     → 房间模型可用（房间盒/地面网格/墙面矩形都可信）
        # "object"   → 这是**物体**（柜子等），别拿房间模型去套
        # "uncertain"→ 证据不足以断言；前端**什么都不画**（含房间盒），并说明原因
        "sceneMode": scene_mode,
        # 房间盒是否为「推测」（只有 1 面可信墙 + 曼哈顿正交补全出来的）。
        # 前端必须据此标注，不能和"已确认"混为一谈。
        "estimated": estimated,
        "estimation": estimation,
        # 判定来源：auto = 证据自动判的；manual = 人工 --scene-mode 指定的
        "sceneModeSource": scene_mode_source,
        # 分类证据（三态判定的全部原始数值 + 拒绝理由）。
        # 交接文档的教训："不要只写一个布尔结论" —— 事后复盘要能看到为什么这么判。
        "sceneModeEvidence": scene_mode_evidence,
        "object": obj,
        "roomLikeWalls": len(real_walls),
        "frameAlignment": {
            "alignedToWalls": True,
            "rotationDeg": round(float(wall_theta_deg), 2),
            "note": "地面坐标系已旋转到与主墙面平行；这样量出的房间尺寸才不是歪的包围盒",
        },
        # 单间列表（按墙线切分）：供俯视平面示意图画房间轮廓
        #
        # ⚠️ 目前**只是候选区域，不是可靠的单间分割**。实测：
        #   用户那个案件 → 1 个 3.6×9.2 m；room-real → 1 个 3.9×10.3 m（另一格 13.4×10.3 被判为「没被切开」已剔除）。
        #   原因是我的「按墙线切网格」只在墙法向与坐标轴夹角很小时才产生切割线，
        #   而检测到的 8 面墙里真正属于曼哈顿两族的不多 → 切出来的格子过大。
        #   要做对需要：先把墙聚成两族正交平面、剔除斜墙与镜面残影，再做平面图分解。
        #   在做到那一步之前，前端**不要**画它。
        "rooms": rooms,
        "roomsQuality": {
            "usable": len(rooms) >= 2,
            "note": "候选区域（按墙线切网格），尚未做正交平面图分解；不足以作为单间分割使用",
        },
        # 地面平面轮廓：俯视态势下的「现场平面示意图」
        "planOutline": outline,
        "room": {
            "size": [round(float(x), 4) for x in room["room_size"]],
            "floorY": 0.0,
            "ceilingY": round(float(height), 4),
            "corners": room_corners,
            "axisU": [round(float(x), 6) for x in u],
            "axisV": [round(float(x), 6) for x in v],
            "origin": room["room_origin"],
        },
        "trajectory": traj,
        "coverage": coverage,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="从 COLMAP 稀疏模型提取现场结构骨架（地面/墙/房间盒/相机轨迹）")
    ap.add_argument("--model", required=True, help="COLMAP 稀疏模型目录（含 cameras.bin/images.bin/points3D.bin）")
    ap.add_argument("--out", required=True, help="输出 structure.json 路径")
    ap.add_argument("--seed", type=int, default=20261005, help="RANSAC 随机种子（固定以便复现）")
    ap.add_argument(
        "--scene-mode", choices=["auto", "room", "object"], default="auto",
        help="人工指定场景模式（默认 auto=按证据判定）。几何上分不开「贴着物体拍」"
             "与「以旋转为主拍的房间」，此时由拍摄的人指定最可靠",
    )
    ap.add_argument(
        "--min-track", type=int, default=3,
        help="参与拟合的最小 track 长度（COLMAP 正常三角化默认 3；"
             "由 VGGT 位姿直接生成的模型没有真实 track，用 1）",
    )
    ap.add_argument(
        "--protect-near-camera", dest="protect_near_camera", action="store_true", default=False,
        help="保护「贴近相机且在相机前方」的点，避免被相对密度判据误杀"
             "（默认关：实测对拟合的收益不明显，见 docs §四；留作对照开关）",
    )
    args = ap.parse_args()

    model_dir = Path(args.model)
    if not model_dir.is_dir():
        print(f"[错误] 模型目录不存在：{model_dir}")
        return 2
    data = extract_structure(
        model_dir, seed=args.seed,
        force_mode=None if args.scene_mode == "auto" else args.scene_mode,
        protect_near_camera=bool(args.protect_near_camera),
        # ⚠️ 必须显式传：默认 3 会让"由 VGGT 位姿生成、没有真实 track"的模型
        #    过滤后剩 0 个点 → 下游 percentile 对空数组崩溃（实测踩过一次，
        #    现象是 scene_structure 直接 IndexError 退出，structure.json 根本没写出）。
        min_track=int(args.min_track),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # `default=` 兜住 numpy 标量/数组：结构里只要漏出一个 ndarray，
    # 整个 json.dumps 就会抛 TypeError，**前面几分钟的拟合全部白跑**。
    # 与其逐个排查哪一处漏了，不如在唯一的写出点统一转换。
    def _jsonable(o):
        if hasattr(o, "tolist"):
            return o.tolist()
        if hasattr(o, "item"):
            return o.item()
        return str(o)

    out.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=_jsonable), encoding="utf-8"
    )

    room = data["room"]
    traj = data["trajectory"]
    print("=" * 68)
    print("现场结构提取完成")
    print(f"  参与拟合的稀疏点 : {data['stats']['pointsUsed']} / {data['stats']['pointsTotal']}")
    print(f"  重力来源         : {data['gravity']['source']}，"
          f"相机朝向离散度 中位 {data['gravity'].get('spreadMedianDeg')}° / "
          f"P90 {data['gravity'].get('spreadP90Deg')}°")
    print(f"  地面高度分位内点 : {data['ground']['inliers']}（{data['ground']['ratio']:.1%}）")
    print(f"  主墙面数量       : {len(data['walls'])}")
    print(f"  房间尺寸(宽×深×高): {room['size'][0]:.2f} × {room['size'][1]:.2f} × {room['size'][2]:.2f} m")
    ch = traj.get("heightAboveFloor")
    if ch:
        print(f"  相机离地高度     : 中位 {ch['median']:.2f} m（P05 {ch['p05']:.2f} ~ P95 {ch['p95']:.2f}），"
              f"落到地面以下 {ch['belowFloor']} 个")
    print(f"  相机轨迹         : {traj['count']} 个位姿，总长 {traj['lengthMeters']:.2f} m"
          f"（稳健 {traj.get('lengthRobustMeters', 0):.2f} m，异常步 {traj.get('outlierSteps', 0)} 个），"
          f"净位移 {traj['netDisplacement']:.2f} m")
    cov = data.get("coverage") or {}
    if cov.get("frames"):
        print(f"  方向覆盖度       : 视锥内稀疏点 中位 {cov['medianPointsInView']}，"
              f"最少 {cov['minPointsInView']}；"
              f"弱帧 {cov['weakFrames']}/{cov['frames']}（{cov['weakRatio']:.1%}），最差帧/中位 {cov['worstToMedianRatio']:.1%}"
              f"→ {cov['verdict']}")
    print(f"  自检             : {'✅ 通过（相机高度落在手持拍摄的合理区间）' if data['sane'] else '❌ 不通过 —— 上方向或地面位置可疑，结果不要直接用'}")
    ev = data.get("sceneModeEvidence") or {}
    sc = (ev.get("surroundCoverage") or {})
    mode_cn = {"room": "房间", "object": "物体（贴拍）", "uncertain": "⚠ 不确定"}.get(
        data.get("sceneMode"), data.get("sceneMode")
    )
    print(f"  场景判定         : {mode_cn}   （方向包围度 "
          f"{sc.get('median') if sc.get('ok') else '算不出'}；"
          f"可信竖墙 {ev.get('roomLikeWalls')} 面；相机高度自洽 {ev.get('cameraHeight', {}).get('sane')}）")
    for _r in (ev.get("rejectReasons") or [])[:4]:
        print(f"                     ↳ {_r}")
    if data.get("sceneMode") == "uncertain":
        # ⚠️ 这句话 2026-10-06 改过。原来写的是"不确定时不输出房间盒/物体盒"，
        #    而端到端测试证明那会**把几何完全可信的物体现场一起毙掉** ——
        #    现在物体几何照算、照带可信度，画不画由 `object.trust` 决定。
        _ob = data.get("object") or {}
        _tr = _ob.get("trust") or {}
        if _ob:
            print(f"                     ↳ 不画房间盒（场景判定不确定）；物体几何已算出，"
                  f"可信度 trust.ok={_tr.get('ok')}"
                  + ("（几何自检通过，可作为「物体」呈现）" if _tr.get("ok")
                     else "（不可信，不画物体盒）"))
        else:
            print("                     ↳ 不画房间盒；也没算出物体几何（点太少）")
    print(f"  输出             : {out}")
    return 0 if data["sane"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
