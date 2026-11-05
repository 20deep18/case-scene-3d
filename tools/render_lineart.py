#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""线稿渲染：把结构骨架（房间盒/墙/地面网格/物体盒/相机轨迹）画成**二维直线**，
并且做**解析遮挡**（后面的线被前面的面挡住就不画）。

为什么要单独写这个渲染器（而不是在浏览器里画）
----------------------------------------------
用户要的是"整个空间和家具的三维线条"，而且要能一眼读出空间含义。
浏览器里现在的覆盖层是**全画不遮挡**的 —— 线一多就成蜘蛛网，读不出空间。
遮挡在这里用**几何解析**算（射线与平面/矩形求交），比屏幕空间 z-buffer 稳：
  · 不依赖泼溅渲染器有没有深度缓冲（实测它 `depth: false`，没有）；
  · 每条线被谁挡住可以查证（可解释）；
  · 遮挡关系只由面决定，不随渲染噪声抖。

"娃娃屋"切法：房间盒的那些**朝向相机的面**（近侧的墙）会被剔掉，
这正是手绘户型图/线稿的惯例 —— 否则从外面只能看到一个盒子轮廓。

用法:
    pyruntime312\\python.exe tools\\render_lineart.py --structure <structure.json> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# 线稿配色：白底 + 深色细线（越近越深，远线淡一些）
BG = (255, 255, 255)
COL_STRUCT = (28, 32, 38)      # 房间盒/墙：主结构线
COL_GRID = (170, 180, 190)     # 地面网格：淡
COL_PLAN = (120, 130, 145)     # 平面轮廓
COL_OBJ = (200, 90, 20)        # 物体盒：橙（与界面一致）
COL_TRAJ = (232, 140, 140)     # 相机轨迹：淡红（手持轨迹很乱，画浓会抢视线）


def look_at(eye, target, up):
    """世界→相机（与 make_textured_dataset 同一套约定：相机 x 右 / y 下 / z 前）。"""
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    up = np.asarray(up, float)
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-12)          # 前
    r = np.cross(f, up)
    r = r / (np.linalg.norm(r) + 1e-12)          # 右
    d = np.cross(f, r)                            # 下
    R = np.stack([r, d, f], axis=0)              # 世界→相机（行是相机基）
    t = -R @ eye
    return R, t


def quad_normal(quad):
    """矩形（4 点，按顺序）的单位法向。"""
    a, b, c = (np.asarray(p, float) for p in quad[:3])
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n)
    return n / ln if ln > 1e-12 else np.zeros(3)


def ray_quad_hit(orig, dirv, quad, *, eps=1e-6):
    """射线与四边形求交，返回 (命中?, 距离)。四边形按两个三角形处理。"""
    v0, v1, v2, v3 = (np.asarray(p, float) for p in quad)
    best = np.inf
    for (a, b, c) in ((v0, v1, v2), (v0, v2, v3)):
        e1, e2 = b - a, c - a
        pv = np.cross(dirv, e2)
        det = float(np.dot(e1, pv))
        if abs(det) < 1e-12:
            continue
        inv = 1.0 / det
        tv = orig - a
        u = float(np.dot(tv, pv)) * inv
        if u < -eps or u > 1 + eps:
            continue
        qv = np.cross(tv, e1)
        v = float(np.dot(dirv, qv)) * inv
        if v < -eps or u + v > 1 + eps:
            continue
        dist = float(np.dot(e2, qv)) * inv
        if dist > eps:
            best = min(best, dist)
    return (best < np.inf), best


def room_box_faces(room):
    """房间盒的 6 个面（每面 4 点）+ 各自的**朝外**法向。"""
    c = room.get("corners") or []
    if len(c) != 8 or not room.get("axisU") or not room.get("axisV"):
        return []
    u = np.asarray(room["axisU"], float)
    v = np.asarray(room["axisV"], float)
    up = np.cross(u, v)
    faces = []
    # corners 顺序：(-u,-v,-up) (u,-v,-up) (u,v,-up) (-u,v,-up) 然后同样四个抬到 +up
    idx = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    for a, b, cc, d in idx:
        faces.append([c[a], c[b], c[cc], c[d]])
    return faces


def clip_rect_to_box(quad, room):
    """把实测墙矩形裁到房间盒范围内（在 (u,v,up) 局部系里夹到 ±size/2）。

    为什么需要：墙是**拟合**出来的矩形，尺寸通常比"理想房间盒"略大/略偏，
    直接画会伸出盒外 —— 看起来像画错，读图的人会怀疑整套几何（实测第一版就这样）。
    """
    if not (room.get("corners") and room.get("axisU") and room.get("axisV") and room.get("size")):
        return quad
    u = np.asarray(room["axisU"], float)
    v = np.asarray(room["axisV"], float)
    up = np.cross(u, v)
    size = np.asarray(room["size"], float)
    o = np.mean(np.array(room["corners"], float), axis=0)
    axes = np.stack([u, v, up], axis=1)
    out = []
    for p in quad:
        loc = axes.T @ (np.asarray(p, float) - o)
        loc = np.clip(loc, -size / 2.0, size / 2.0)
        out.append((o + axes @ loc).tolist())
    return out


def rect_from_wall(w):
    """墙面矩形 → 4 点（结构文件里叫 rect.corners）。"""
    r = (w or {}).get("rect") or {}
    c = r.get("corners")
    if isinstance(c, list) and len(c) == 4:
        return [list(map(float, p)) for p in c]
    return None


def collect(structure: dict, *, keep_ceiling: bool, cut_near: bool, cam_eye, world_up,
            grid_step: float = 1.0, with_plan_outline: bool = False,
            with_trajectory: bool = True, furniture: dict | None = None):
    """把结构文件拆成 (线段列表, 遮挡面列表)。线段带颜色与所属类别。"""
    lines: list[tuple[np.ndarray, np.ndarray, tuple, str]] = []
    room = structure.get("room") or {}
    c = room.get("corners") or []
    if len(c) == 8:
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        for a, b in edges:
            lines.append((np.asarray(c[a], float), np.asarray(c[b], float), COL_STRUCT, "房间盒"))
    # 墙：既是遮挡面，也画出矩形轮廓。
    # ⚠️ 只画**背对相机**的那些墙（远侧）：近侧墙在娃娃屋视图里应当被"切掉"，
    #    否则它们自己的轮廓会横在画面中间，把空间关系压平（第一版就是这样）。
    surfaces = []
    for f in room_box_faces(room):
        surfaces.append(("box", f))
    eye0 = np.asarray(cam_eye, float)
    for w in (structure.get("walls") or []):
        q = rect_from_wall(w)
        if q is None or (w.get("kind") or "wall") != "wall":
            continue
        n = quad_normal(q)
        ctr = np.mean(np.array(q, float), axis=0)
        if cut_near and float(n @ (eye0 - ctr)) > 0:
            continue
        # ⚠️ 实测墙矩形常常比理想房间盒大一点，画出来会"伸出盒外"像画错了（实测）。
        #    这里把它在盒内裁一下：在 (u,v,up) 局部系里把平面内范围夹到 ±size/2。
        q = clip_rect_to_box(q, room)
        surfaces.append(("wall", q))
        for i in range(4):
            lines.append((np.asarray(q[i], float), np.asarray(q[(i + 1) % 4], float),
                          COL_STRUCT, "墙"))
    # 地面：盒子内的**规则网格**（每 --grid-step 米一条）。
    # ⚠️ 原来用 planOutline 的 segmentsWorld，那是**栅格化**出来的占比轮廓，
    #    画出来是一圈锯齿（第一版实测），读起来像噪声而不是地面。
    if room.get("corners") and room.get("axisU") and room.get("axisV") and grid_step > 0:
        cc = np.array(room["corners"], float)
        u = np.asarray(room["axisU"], float)
        v = np.asarray(room["axisV"], float)
        up = np.cross(u, v)
        o = np.mean(cc, axis=0)
        size = room.get("size") or [0, 0, 0]
        nu = int(size[0] / grid_step)
        nv = int(size[1] / grid_step)
        lo = -0.5 * np.array([size[0], size[1]], float)
        for i in range(1, max(nu, 0)):
            a = o + u * (lo[0] + i * grid_step) + v * lo[1] - up * (size[2] / 2)
            b = o + u * (lo[0] + i * grid_step) + v * -lo[1] - up * (size[2] / 2)
            lines.append((a, b, COL_GRID, "地面网格"))
        for j in range(1, max(nv, 0)):
            a = o + u * lo[0] + v * (lo[1] + j * grid_step) - up * (size[2] / 2)
            b = o + u * -lo[0] + v * (lo[1] + j * grid_step) - up * (size[2] / 2)
            lines.append((a, b, COL_GRID, "地面网格"))
    if with_plan_outline:
        for seg in ((structure.get("planOutline") or {}).get("segmentsWorld") or []):
            if isinstance(seg, (list, tuple)) and len(seg) == 2:
                lines.append((np.asarray(seg[0], float), np.asarray(seg[1], float), COL_PLAN, "平面轮廓"))
    # 物体盒（只有可信时才有意义；结构文件里带 trust）
    ob = structure.get("object") or {}
    if ob.get("corners") and (ob.get("trust") or {}).get("ok"):
        oc = ob["corners"]
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                     (0, 4), (1, 5), (2, 6), (3, 7)]:
            lines.append((np.asarray(oc[a], float), np.asarray(oc[b], float), COL_OBJ, "物体盒"))
        idx = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        for a, b, cc, d in idx:
            surfaces.append(("obj", [oc[a], oc[b], oc[cc], oc[d]]))
    # ---- 家具盒（来自 tools/furniture_lines.py 的输出）----------------------
    # 家具盒用橙色细线画；同时把盒面加进**遮挡面**里，这样后面的线会被家具挡住，
    # 读起来才像"屋里有东西"，而不是一堆透明线。
    if furniture:
        fr = furniture.get("frame") or {}
        fo = np.asarray(fr.get("origin") or [0, 0, 0], float)
        fu = np.asarray(fr.get("axisU") or [1, 0, 0], float)
        fv = np.asarray(fr.get("axisV") or [0, 1, 0], float)
        fw = np.asarray(fr.get("up") or [0, 0, 1], float)
        faxes = np.stack([fu, fv, fw], axis=1)
        signs = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                 (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
        for obj in (furniture.get("objects") or []):
            c = np.asarray(obj["center"], float)
            s = np.asarray(obj["size"], float)
            corners = [c + faxes @ (np.asarray(sg, float) * s / 2) for sg in signs]
            for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                         (0, 4), (1, 5), (2, 6), (3, 7)]:
                lines.append((corners[a], corners[b], COL_OBJ, "家具"))
            for a, b, cc, d in [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
                                (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]:
                surfaces.append(("furniture", [corners[a], corners[b], corners[cc], corners[d]]))

    # 相机轨迹（可选：手持轨迹很乱，干净版可以不画）
    pts = ((structure.get("trajectory") or {}).get("points") or []) if with_trajectory else []
    for i in range(len(pts) - 1):
        lines.append((np.asarray(pts[i], float), np.asarray(pts[i + 1], float), COL_TRAJ, "相机轨迹"))
    # ---- 娃娃屋切法：剔掉朝向相机的面（近侧的墙/顶），否则从外面只看得到一个盒子 ----
    kept = []
    eye = np.asarray(cam_eye, float)
    up_axis = None
    if room.get("axisU") and room.get("axisV"):
        up_axis = np.cross(np.asarray(room["axisU"], float), np.asarray(room["axisV"], float))
    for kind, q in surfaces:
        n = quad_normal(q)
        center = np.mean(np.array(q, float), axis=0)
        if kind == "box":
            if not keep_ceiling and up_axis is not None and abs(float(n @ up_axis)) > 0.9:
                # 顶/底面：默认保留地面、剔掉天花（否则俯视看不到里面）
                if float(n @ up_axis) > 0:
                    continue
            if cut_near and float(n @ (eye - center)) > 0:
                continue          # 朝相机的面 = 近侧 → 剔掉（娃娃屋）
        kept.append((kind, q))
    return lines, kept


def render(structure: dict, eye, target, *, world_up, width=1400, height=950, fov_deg=45.0,
           keep_ceiling=False, cut_near=True, samples_per_meter=22.0, grid_step=1.0,
           with_plan_outline=False, with_trajectory=True, furniture=None):
    """渲染一张线稿。返回 (numpy RGB 图, 统计信息)。"""
    from PIL import Image, ImageDraw

    # ⚠️ 相机上方向必须用**这个重建自己的"上"**（结构文件的 frame.up），不能硬编码 +z：
    #    COLMAP 的世界坐标系是任意的，实测 room-real 里"上"是 ≈ −y，
    #    硬编码 +z 会把整个房间转倒 90°（长边竖起来，实测截图）。
    R, t = look_at(eye, target, np.asarray(world_up, float))
    fx = fy = 0.5 * width / math.tan(math.radians(fov_deg) / 2)
    cx, cy = width / 2.0, height / 2.0

    lines, surfaces = collect(structure, keep_ceiling=keep_ceiling, cut_near=cut_near, cam_eye=eye,
                              world_up=np.asarray(world_up, float),
                              grid_step=grid_step, with_plan_outline=with_plan_outline,
                              with_trajectory=with_trajectory, furniture=furniture)
    surf_pts = [(np.asarray(q, float), quad_normal(q)) for _k, q in surfaces]

    img = Image.new("RGB", (width, height), BG)
    dr = ImageDraw.Draw(img)
    hidden = 0
    total = 0

    def project(p):
        xc = R @ np.asarray(p, float) + t
        if xc[2] <= 1e-6:
            return None
        return (fx * xc[0] / xc[2] + cx, fy * xc[1] / xc[2] + cy, xc[2])

    scale = float(np.linalg.norm(np.asarray(target, float) - np.asarray(eye, float)))
    for a, b, color, _cls in lines:
        seg_len = float(np.linalg.norm(b - a))
        n = max(2, int(min(400, seg_len * samples_per_meter)))
        ts = np.linspace(0.0, 1.0, n)
        pts = a[None, :] + ts[:, None] * (b - a)[None, :]
        vis = []
        for p in pts:
            total += 1
            dirv = np.asarray(eye, float) - p            # 从点看相机
            dist_cam = float(np.linalg.norm(dirv))
            dirv = dirv / (dist_cam + 1e-12)
            blocked = False
            for q, _n in surf_pts:
                ok, d = ray_quad_hit(p, dirv, q)
                if ok and d < dist_cam - 1e-4:         # 面在点与相机之间 → 被挡
                    blocked = True
                    break
            if blocked:
                hidden += 1
            vis.append(not blocked)
        # 把连续可见的样本连成折线画出来（近处线更粗）
        run_start = None
        for i, v in enumerate(vis + [False]):
            if v and run_start is None:
                run_start = i
            elif not v and run_start is not None:
                seg = pts[run_start:i]
                if len(seg) >= 1:
                    pp = [project(p) for p in seg]
                    pp = [q for q in pp if q is not None]
                    if len(pp) >= 2:
                        zc = float(np.mean([q[2] for q in pp]))
                        wpx = float(np.clip(scale / max(zc, 1e-6) * 1.1, 0.7, 2.4))
                        dr.line([(q[0], q[1]) for q in pp], fill=color, width=int(round(wpx)))
                run_start = None
    # 右下角写一行小字：房间尺寸 + 线/遮挡统计。
    # 线稿本身没有比例尺就"读不出来"，尺寸必须写在图上（用户要的是能量尺寸的东西）。
    try:
        from PIL import ImageDraw, ImageFont
        d2 = ImageDraw.Draw(img)
        # ⚠️ PIL 默认位图字体**没有中文字形**，中文会渲染成方块（实测）。
        #    这里显式加载系统雅黑；找不到就退回英文说明，绝不输出方块。
        font = None
        for cand in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
                     r"C:\Windows\Fonts\msyhbd.ttc"):
            try:
                font = ImageFont.truetype(cand, 15)
                break
            except Exception:
                continue
        size = (structure.get("room") or {}).get("size") or []
        if font is not None:
            cap = ""
            if len(size) == 3:
                cap = f"房间 {size[0]:.2f} × {size[1]:.2f} × {size[2]:.2f} m"
            cap += f"   线 {len(lines)} 条   遮挡面 {len(surfaces)} 个   被挡 {100 * hidden / max(total, 1):.0f}%"
        else:
            cap = (f"room {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} m" if len(size) == 3 else "")
            cap += f"   lines {len(lines)}   occluders {len(surfaces)}   hidden {100 * hidden / max(total, 1):.0f}%"
        d2.text((18, height - 26), cap, fill=(90, 98, 110), font=font)
    except Exception:
        pass
    return np.asarray(img), {"lines": len(lines), "surfaces": len(surfaces),
                             "samples": total, "hiddenSamples": hidden}


def main() -> int:
    ap = argparse.ArgumentParser(description="结构骨架 → 带解析遮挡的二维线稿")
    ap.add_argument("--structure", required=True)
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--views", type=int, default=4, help="环绕几个方位各一张")
    ap.add_argument("--elevation", type=float, default=32.0, help="俯视角（度）")
    ap.add_argument("--keep-ceiling", action="store_true", help="保留天花作为遮挡面（默认剔掉）")
    ap.add_argument("--no-cut-near", action="store_true", help="不剔近侧墙面（娃娃屋关闭）")
    ap.add_argument("--grid-step", type=float, default=1.0, help="地面网格间距（米）；0=不画")
    ap.add_argument("--plan-outline", action="store_true", help="额外画栅格化的平面轮廓（锯齿，默认关）")
    ap.add_argument("--no-trajectory", action="store_true", help="不画相机轨迹（干净版）")
    ap.add_argument("--objects", default=None, help="家具盒 JSON（tools/furniture_lines.py 的输出）")
    args = ap.parse_args()

    st = json.loads(Path(args.structure).read_text(encoding="utf-8"))
    furniture = None
    if args.objects and Path(args.objects).exists():
        furniture = json.loads(Path(args.objects).read_text(encoding="utf-8"))
        print(f"家具盒：{len(furniture.get('objects') or [])} 个（来自 {Path(args.objects).name}）")
    room = st.get("room") or {}
    if not room.get("corners"):
        print("结构文件里没有房间盒，无法出线稿")
        return 1
    c = np.array(room["corners"], float)
    # ⚠️ 目标点必须是**盒心**（8 个角点的均值）。
    #    第一版用了 room.origin —— 那是拟合参考系的原点，可能是盒子的**角点**，
    #    结果整个盒子被画到画面右上角、大半张纸是空的（实测截图）。
    center = c.mean(axis=0)
    diag = float(np.linalg.norm(c.max(axis=0) - c.min(axis=0)))
    # 世界坐标里的"上"：优先用结构文件记的 frame.up（就是重力反方向）；
    # 没有就退回 axisU×axisV。**绝不硬编码**（见 render 里的说明）。
    world_up = None
    fr = st.get("frame") or {}
    if isinstance(fr.get("up"), list) and len(fr["up"]) == 3:
        world_up = np.asarray(fr["up"], float)
    if world_up is None and room.get("axisU") and room.get("axisV"):
        world_up = np.cross(np.asarray(room["axisU"], float), np.asarray(room["axisV"], float))
    if world_up is None:
        world_up = np.array([0.0, 0.0, 1.0])
    world_up = world_up / (np.linalg.norm(world_up) + 1e-12)
    # 视角的方位角要在**水平面内**取（垂直于 world_up），不能再用世界 xy
    a1 = np.asarray(room.get("axisU") or [1.0, 0, 0], float)
    a1 = a1 - world_up * float(a1 @ world_up)
    a1 = a1 / (np.linalg.norm(a1) + 1e-12)
    a2 = np.cross(world_up, a1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    W, H = 1400, 950
    fov = 45.0
    fx = fy = 0.5 * W / math.tan(math.radians(fov) / 2)

    def fits(eye, margin=0.92):
        R, t = look_at(eye, center, world_up)
        for p in c:
            xc = R @ p + t
            if xc[2] <= 1e-6:
                return False
            u = fx * xc[0] / xc[2] + W / 2
            v = fy * xc[1] / xc[2] + H / 2
            if not ((1 - margin) * W / 2 <= u <= (1 + margin) * W / 2):
                return False
            if not ((1 - margin) * H / 2 <= v <= (1 + margin) * H / 2):
                return False
        return True

    made = []
    for i in range(args.views):
        az = 2 * math.pi * i / max(args.views, 1) + math.radians(35)
        el = math.radians(args.elevation)
        # 自动求刚好装下整个房间的距离（第一版固定 diag*0.95，实测装不下还偏）
        dist = None
        for d in np.linspace(diag * 0.6, diag * 3.0, 90):
            eye_try = center + d * (math.cos(el) * (math.cos(az) * a1 + math.sin(az) * a2)
                                    + math.sin(el) * world_up)
            if fits(eye_try):
                dist = float(d)
                break
        if dist is None:
            dist = diag * 1.6
        eye = center + dist * (math.cos(el) * (math.cos(az) * a1 + math.sin(az) * a2)
                               + math.sin(el) * world_up)
        img, stats = render(st, eye, center, world_up=world_up, width=W, height=H, fov_deg=fov,
                            keep_ceiling=args.keep_ceiling,
                            cut_near=not args.no_cut_near, grid_step=args.grid_step,
                            with_plan_outline=args.plan_outline,
                            with_trajectory=not args.no_trajectory,
                            furniture=furniture)
        from PIL import Image
        p = out / f"lineart_{i + 1}.png"
        Image.fromarray(img).save(p)
        made.append((p, stats))
        print(f"  {p.name}: 线 {stats['lines']} 条，遮挡面 {stats['surfaces']} 个，"
              f"采样 {stats['samples']}，被挡 {stats['hiddenSamples']} "
              f"({100 * stats['hiddenSamples'] / max(stats['samples'], 1):.0f}%)")
    print(f"\n已写出 {len(made)} 张线稿到 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
