"""pipeline/make_test_scene.py —— 生成不依赖任何第三方库的合成测试场景。

用途：在真实拍摄数据 / COLMAP / gsplat 到位之前，先把「Web 端能不能正确加载与显示」
这条链路跑通，并给后端提供演示轨迹与 PnP 测试数据。

生成内容：
  1. 一个程序化「房间」场景，输出为 .splat（默认 data/models/test-scene.splat）：
     地面、两面墙、一张桌子（长方体贴合面片）、桌上一个球与一个盒子、一个门框；
     每个面用大量各向异性高斯铺满（面内尺度贴合面片、法向尺度很薄），
     朝向由面法向构造四元数，颜色按物体区分并叠加光照渐变与少量噪点。
  2. 元数据 JSON（默认 data/models/test-scene.json）。
  3. 环绕相机 JSON（默认 data/models/test-scene-cameras.json）：8 个环绕相机位姿，
     坐标系为 COLMAP / 3DGS 世界系（y 轴向下，单位米）。
  4. 2D 观测 JSON（默认 data/models/test-scene-observations.json）：6 个已知三维点
     （桌角、门框角等）在 3 个相机中的投影像素坐标 + 相机内参，
     供自动化测试 OpenCV PnP / 重投影精度。

可选：--ply 同时输出一份标准 3DGS 二进制 PLY，便于用 to_splat.py 反向验证。

用法：
    python pipeline/make_test_scene.py --out data/models/test-scene.splat --count 60000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

_PIPELINE_DIR = Path(__file__).resolve().parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

import config  # noqa: E402
import to_splat  # noqa: E402  （复用 .splat 编码与 SH 常数，保证两个脚本完全一致）

# ---------------------------------------------------------------------------
# 场景尺寸（单位：米；y 轴向下，地面 y = 0，天花板 y = -2.8）
# ---------------------------------------------------------------------------

ROOM_X: float = 3.0  # 左右墙 ±3.0
ROOM_Z: float = 2.5  # 前后墙 ±2.5
CEILING_Y: float = -2.80
WALL_THICKNESS: float = 0.02  # 墙面高斯的厚度（法向很薄）

FLOOR_COLOR = (0.20, 0.20, 0.23)
WALL_COLOR = (0.80, 0.79, 0.76)
TABLE_COLOR = (0.46, 0.29, 0.16)
LEG_COLOR = (0.34, 0.21, 0.11)
BALL_COLOR = (0.78, 0.26, 0.18)
BOX_COLOR = (0.24, 0.36, 0.68)
FRAME_COLOR = (0.70, 0.63, 0.50)

TABLE_TOP_Y: float = -0.75  # 桌面上表面高度
TABLE_CENTER = (0.20, 0.0, 0.30)  # 桌子在 (x, z) 平面的中心
TABLE_HALF_X: float = 0.90
TABLE_HALF_Z: float = 0.50
TABLE_THICKNESS: float = 0.06

BALL_CENTER = (-0.35, TABLE_TOP_Y - 0.16, 0.30)
BALL_RADIUS: float = 0.16

BOX_MIN = (0.45, TABLE_TOP_Y - 0.30, 0.14)
BOX_MAX = (0.75, TABLE_TOP_Y, 0.44)

# 门框：贴在 z = -ROOM_Z 的后墙上
DOOR_CENTER_X: float = 1.90
DOOR_WIDTH: float = 1.00
DOOR_HEIGHT: float = 2.05
DOOR_POST: float = 0.09
DOOR_DEPTH: float = 0.13

# 相机
IMAGE_WIDTH: int = 1920
IMAGE_HEIGHT: int = 1080
FX: float = 1250.0
FY: float = 1250.0
CX: float = 960.0
CY: float = 540.0
WORLD_UP = (0.0, -1.0, 0.0)  # y 轴向下，因此「上」是 -y
ORBIT_RADIUS: float = 2.00
ORBIT_EYE_Y: float = -1.15
SCENE_TARGET = (0.20, -1.10, -0.20)
CAMERA_COUNT: int = 8

# ---------------------------------------------------------------------------
# 极简向量工具（不依赖 numpy）
# ---------------------------------------------------------------------------

Vec3 = tuple[float, float, float]


def v_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_mul(a: Vec3, k: float) -> Vec3:
    return (a[0] * k, a[1] * k, a[2] * k)


def v_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def v_len(a: Vec3) -> float:
    return math.sqrt(v_dot(a, a))


def v_norm(a: Vec3) -> Vec3:
    n = v_len(a)
    if n <= 1e-12:
        return (0.0, 0.0, 1.0)
    return (a[0] / n, a[1] / n, a[2] / n)


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else (high if value > high else value)


# ---------------------------------------------------------------------------
# 高斯与面片
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Gaussian:
    """一个高斯：位置 + 线性尺度 + 颜色 + 原始 opacity(logit) + 四元数 (w,x,y,z)。"""

    p: Vec3
    scale: Vec3
    color: tuple[float, float, float]
    opacity_raw: float
    quat: tuple[float, float, float, float]


@dataclass(slots=True)
class Face:
    """一块矩形面片：origin 为 u-v 原点，u_axis / v_axis 为面内两条边向量。"""

    name: str
    origin: Vec3
    u_axis: Vec3
    v_axis: Vec3
    normal: Vec3
    color: tuple[float, float, float]
    thin: float = WALL_THICKNESS

    @property
    def area(self) -> float:
        return v_len(self.u_axis) * v_len(self.v_axis)


def box_faces(mn: Vec3, mx: Vec3, color: tuple[float, float, float], name: str, *, include: str = "all") -> list[Face]:
    """由轴对齐长方体生成 6 个（或指定若干）面片。"""
    x0, y0, z0 = mn
    x1, y1, z1 = mx
    faces: list[Face] = []
    if include in ("all", "top"):
        # y 轴向下：y0（数值较小）是「上」表面，法向朝上 = -y
        faces.append(Face(f"{name}-顶面", (x0, y0, z0), (x1 - x0, 0, 0), (0, 0, z1 - z0), (0, -1, 0), color, 0.008))
    if include in ("all", "sides"):
        faces.append(Face(f"{name}-前面", (x0, y0, z1), (x1 - x0, 0, 0), (0, y1 - y0, 0), (0, 0, 1), color))
        faces.append(Face(f"{name}-后面", (x0, y0, z0), (x1 - x0, 0, 0), (0, y1 - y0, 0), (0, 0, -1), color))
        faces.append(Face(f"{name}-左面", (x0, y0, z0), (0, 0, z1 - z0), (0, y1 - y0, 0), (-1, 0, 0), color))
        faces.append(Face(f"{name}-右面", (x1, y0, z0), (0, 0, z1 - z0), (0, y1 - y0, 0), (1, 0, 0), color))
    if include == "all":
        faces.append(Face(f"{name}-底面", (x0, y1, z0), (x1 - x0, 0, 0), (0, 0, z1 - z0), (0, 1, 0), color, 0.008))
    return faces


def build_surfaces() -> list[Face]:
    """搭建房间的所有面片。"""
    faces: list[Face] = []

    # 地面（法向朝上 = -y）
    faces.append(Face("地面", (-ROOM_X, 0.0, -ROOM_Z), (2 * ROOM_X, 0, 0), (0, 0, 2 * ROOM_Z), (0, -1, 0), FLOOR_COLOR, 0.012))
    # 后墙 z = -ROOM_Z（法向朝房间内 = +z）
    faces.append(Face("后墙", (-ROOM_X, CEILING_Y, -ROOM_Z), (2 * ROOM_X, 0, 0), (0, -CEILING_Y, 0), (0, 0, 1), WALL_COLOR, 0.014))
    # 左墙 x = -ROOM_X（法向朝房间内 = +x）
    faces.append(Face("左墙", (-ROOM_X, CEILING_Y, -ROOM_Z), (0, 0, 2 * ROOM_Z), (0, -CEILING_Y, 0), (1, 0, 0), WALL_COLOR, 0.014))

    # 桌面（只有上表面 + 四周薄边，避免内部不可见的面浪费高斯）
    tx, tz = TABLE_CENTER[0], TABLE_CENTER[2]
    top_y = TABLE_TOP_Y - TABLE_THICKNESS
    faces.append(
        Face("桌面上表面", (tx - TABLE_HALF_X, top_y, tz - TABLE_HALF_Z), (2 * TABLE_HALF_X, 0, 0), (0, 0, 2 * TABLE_HALF_Z), (0, -1, 0), TABLE_COLOR, 0.010)
    )
    for dx in (-TABLE_HALF_X, TABLE_HALF_X):
        for dz in (-TABLE_HALF_Z, TABLE_HALF_Z):
            normal: Vec3 = (0.0, 0.0, 1.0 if dz > 0 else -1.0) if abs(dx) < TABLE_HALF_X else (1.0 if dx > 0 else -1.0, 0.0, 0.0)
            if abs(dx) == TABLE_HALF_X:
                faces.append(
                    Face(
                        f"桌面侧边-{ '右' if dx > 0 else '左' }",
                        (tx + dx, top_y, tz - TABLE_HALF_Z),
                        (0, 0, 2 * TABLE_HALF_Z),
                        (0, TABLE_THICKNESS, 0),
                        normal,
                        TABLE_COLOR,
                        0.008,
                    )
                )
            else:
                faces.append(
                    Face(
                        f"桌面侧边-{ '前' if dz > 0 else '后' }",
                        (tx - TABLE_HALF_X, top_y, tz + dz),
                        (2 * TABLE_HALF_X, 0, 0),
                        (0, TABLE_THICKNESS, 0),
                        normal,
                        TABLE_COLOR,
                        0.008,
                    )
                )

    # 四条桌腿（只画四个侧面）
    leg = 0.06
    for sx in (-1, 1):
        for sz in (-1, 1):
            cx = tx + sx * (TABLE_HALF_X - 0.10)
            cz = tz + sz * (TABLE_HALF_Z - 0.10)
            faces.extend(
                box_faces(
                    (cx - leg, top_y + TABLE_THICKNESS, cz - leg),
                    (cx + leg, 0.0, cz + leg),
                    LEG_COLOR,
                    f"桌腿{sx:+d}{sz:+d}",
                    include="sides",
                )
            )

    # 桌上的盒子
    faces.extend(box_faces(BOX_MIN, BOX_MAX, BOX_COLOR, "桌上盒子", include="all"))

    # 门框（两根立柱 + 一根横梁）
    wall_z = -ROOM_Z + 0.01
    left_x = DOOR_CENTER_X - DOOR_WIDTH / 2
    right_x = DOOR_CENTER_X + DOOR_WIDTH / 2
    faces.extend(
        box_faces(
            (left_x - DOOR_POST, -DOOR_HEIGHT, wall_z),
            (left_x, 0.0, wall_z + DOOR_DEPTH),
            FRAME_COLOR,
            "门框左柱",
            include="sides",
        )
    )
    faces.extend(
        box_faces(
            (right_x, -DOOR_HEIGHT, wall_z),
            (right_x + DOOR_POST, 0.0, wall_z + DOOR_DEPTH),
            FRAME_COLOR,
            "门框右柱",
            include="sides",
        )
    )
    faces.extend(
        box_faces(
            (left_x, -DOOR_HEIGHT - DOOR_POST, wall_z),
            (right_x + DOOR_POST, -DOOR_HEIGHT, wall_z + DOOR_DEPTH),
            FRAME_COLOR,
            "门框横梁",
            include="sides",
        )
    )
    return faces


# ---------------------------------------------------------------------------
# 姿态与着色
# ---------------------------------------------------------------------------


def frame_from_normal(normal: Vec3) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """由法向构造一组局部坐标轴，返回 (t1, t2, n)，n 为局部 z 轴（尺度最薄的方向）。"""
    n = v_norm(normal)
    helper: Vec3 = (0.0, 1.0, 0.0) if abs(v_dot(n, (0.0, 1.0, 0.0))) < 0.9 else (1.0, 0.0, 0.0)
    t1 = v_norm(v_cross(helper, n))
    t2 = v_norm(v_cross(n, t1))
    return t1, t2, n


def quat_from_axes(t1: Vec3, t2: Vec3, t3: Vec3) -> tuple[float, float, float, float]:
    """把列向量为 (t1, t2, t3) 的旋转矩阵转成四元数 (w, x, y, z)。"""
    m00, m10, m20 = t1
    m01, m11, m21 = t2
    m02, m12, m22 = t3
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return v_norm4((w, x, y, z))


def v_norm4(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if n <= 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


#: 点光源位置（贴近天花板中央，用于制造光照渐变）
LIGHT_POS: Vec3 = (0.4, CEILING_Y + 0.25, 0.6)


def shade(color: tuple[float, float, float], position: Vec3, normal: Vec3, rng: random.Random) -> tuple[float, float, float]:
    """给固定基色叠加点光源衰减、朝上增益与少量噪点。"""
    d = v_sub(position, LIGHT_POS)
    intensity = 0.58 + 0.42 / (1.0 + v_dot(d, d) / 3.2)
    intensity += 0.08 * max(0.0, -normal[1])  # 朝上的面亮一点
    noise = rng.uniform(-0.025, 0.025)
    return tuple(clamp(color[i] * intensity + noise, 0.02, 0.98) for i in range(3))  # type: ignore[return-value]


def opacity_logit(p: float) -> float:
    p = clamp(p, 1e-4, 1 - 1e-4)
    return math.log(p / (1.0 - p))


# ---------------------------------------------------------------------------
# 采样：把面片 / 球面铺满高斯
# ---------------------------------------------------------------------------


def sample_face(face: Face, count: int, rng: random.Random, jitter: float = 0.45) -> list[Gaussian]:
    """在一块矩形面片上按分层抖动采样 count 个各向异性高斯。"""
    if count <= 0:
        return []
    u_len = v_len(face.u_axis)
    v_len_ = v_len(face.v_axis)
    if u_len <= 0 or v_len_ <= 0:
        return []
    u_dir = v_mul(face.u_axis, 1.0 / u_len)
    v_dir = v_mul(face.v_axis, 1.0 / v_len_)
    nu = max(1, int(round(math.sqrt(count * u_len / v_len_))))
    nv = max(1, math.ceil(count / nu))
    spacing = math.sqrt(face.area / max(1, count))
    in_plane = spacing * 0.78
    thin = min(face.thin, spacing * 0.6)
    t1, t2, n = frame_from_normal(face.normal)
    quat = quat_from_axes(t1, t2, n)
    gaussians: list[Gaussian] = []
    for i in range(nu):
        for j in range(nv):
            if len(gaussians) >= count:
                break
            fu = (i + 0.5 + rng.uniform(-jitter, jitter)) / nu
            fv = (j + 0.5 + rng.uniform(-jitter, jitter)) / nv
            if not (0.0 <= fu <= 1.0 and 0.0 <= fv <= 1.0):
                fu = clamp(fu, 0.001, 0.999)
                fv = clamp(fv, 0.001, 0.999)
            base = v_add(face.origin, v_add(v_mul(u_dir, fu * u_len), v_mul(v_dir, fv * v_len_)))
            position = v_add(base, v_mul(n, rng.uniform(0.0, 0.004)))
            scale = (
                in_plane * rng.uniform(0.85, 1.15),
                in_plane * rng.uniform(0.85, 1.15),
                max(0.0025, thin * rng.uniform(0.8, 1.2)),
            )
            gaussians.append(
                Gaussian(position, scale, shade(face.color, position, n, rng), opacity_logit(rng.uniform(0.86, 0.95)), quat)
            )
        if len(gaussians) >= count:
            break
    return gaussians


def sample_sphere(center: Vec3, radius: float, count: int, color: tuple[float, float, float], rng: random.Random) -> list[Gaussian]:
    """在球面上用黄金螺旋均匀铺点（法向即径向）。"""
    gaussians: list[Gaussian] = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(count):
        y = 1.0 - 2.0 * (i + 0.5) / count
        r = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden * i
        n: Vec3 = (math.cos(theta) * r, y, math.sin(theta) * r)
        position = v_add(center, v_mul(n, radius + rng.uniform(-0.002, 0.002)))
        t1, t2, _ = frame_from_normal(n)
        patch = 2.0 * math.pi * radius / math.sqrt(max(1.0, count)) * 0.9
        scale = (patch * rng.uniform(0.85, 1.15), patch * rng.uniform(0.85, 1.15), max(0.003, radius * 0.08))
        gaussians.append(
            Gaussian(position, scale, shade(color, position, n, rng), opacity_logit(rng.uniform(0.86, 0.95)), quat_from_axes(t1, t2, n))
        )
    return gaussians


def allocate_counts(areas: Sequence[float], total: int) -> list[int]:
    """按面积比例分配高斯数量（最大余数法，保证总数精确等于 total）。"""
    positive = [a if a > 0 else 0.0 for a in areas]
    area_sum = sum(positive)
    if area_sum <= 0 or total <= 0:
        return [0] * len(areas)
    raw = [a / area_sum * total for a in positive]
    counts = [int(x) for x in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(areas)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in order[:remainder]:
        counts[i] += 1
    for i, a in enumerate(positive):
        if a > 0 and counts[i] == 0:
            counts[i] = 1
    return counts


def build_scene(total: int, seed: int) -> tuple[list[Gaussian], list[Face]]:
    """程序化生成整个房间的高斯集合。"""
    rng = random.Random(seed)
    faces = build_surfaces()
    sphere_area = 4.0 * math.pi * BALL_RADIUS * BALL_RADIUS * 0.55  # 球只有朝外一半可见
    areas = [f.area for f in faces] + [sphere_area]
    counts = allocate_counts(areas, total)
    gaussians: list[Gaussian] = []
    for face, n in zip(faces, counts):
        gaussians.extend(sample_face(face, n, rng))
    gaussians.extend(sample_sphere(BALL_CENTER, BALL_RADIUS, counts[-1], BALL_COLOR, rng))
    if len(gaussians) > total:  # 采样可能略微超出，按顺序截断保证精确
        gaussians = gaussians[:total]
    return gaussians, faces


# ---------------------------------------------------------------------------
# 相机与投影
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Camera:
    """一台针孔相机（世界系 y 向下）。"""

    cam_id: str
    position: Vec3
    target: Vec3
    rotation: tuple[Vec3, Vec3, Vec3]  # 世界 → 相机 的 3x3 旋转（行向量）
    translation: Vec3  # t = -R @ C

    def project(self, point: Vec3) -> tuple[float, float] | None:
        """把世界点投影到像素坐标；在相机后方返回 None。"""
        d = v_sub(point, self.position)
        x = v_dot(self.rotation[0], d)
        y = v_dot(self.rotation[1], d)
        z = v_dot(self.rotation[2], d)
        if z <= 1e-3:
            return None
        return (FX * x / z + CX, FY * y / z + CY)

    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON 结构（同时给出 position/target 与完整外参）。"""
        r = self.rotation
        t = self.translation
        c2w = [
            r[0][0], r[1][0], r[2][0], self.position[0],
            r[0][1], r[1][1], r[2][1], self.position[1],
            r[0][2], r[1][2], r[2][2], self.position[2],
            0.0, 0.0, 0.0, 1.0,
        ]
        w2c = [
            r[0][0], r[0][1], r[0][2], t[0],
            r[1][0], r[1][1], r[1][2], t[1],
            r[2][0], r[2][1], r[2][2], t[2],
            0.0, 0.0, 0.0, 1.0,
        ]
        return {
            "id": self.cam_id,
            "position": [round(v, 6) for v in self.position],
            "target": [round(v, 6) for v in self.target],
            "up": [0.0, -1.0, 0.0],
            "rotation": [round(v, 10) for row in r for v in row],
            "translation": [round(v, 10) for v in t],
            "c2w": [round(v, 10) for v in c2w],
            "w2c": [round(v, 10) for v in w2c],
        }


def look_at(position: Vec3, target: Vec3, world_up: Vec3 = WORLD_UP) -> tuple[tuple[Vec3, Vec3, Vec3], Vec3]:
    """构造 OpenCV 约定的外参：x 右、y 下、z 前（p_cam = R (p_world - C)）。"""
    forward = v_norm(v_sub(target, position))
    right = v_norm(v_cross(forward, world_up))
    down = v_norm(v_cross(forward, right))
    rotation = (right, down, forward)
    translation: Vec3 = (
        -v_dot(right, position),
        -v_dot(down, position),
        -v_dot(forward, position),
    )
    return rotation, translation


def build_cameras() -> list[Camera]:
    """生成 8 个环绕相机（在房间内部，视线朝向场景中心）。"""
    cameras: list[Camera] = []
    for i in range(CAMERA_COUNT):
        angle = 2.0 * math.pi * i / CAMERA_COUNT
        position: Vec3 = (
            SCENE_TARGET[0] + ORBIT_RADIUS * math.cos(angle),
            ORBIT_EYE_Y + 0.12 * math.sin(angle * 2.0),
            SCENE_TARGET[2] + ORBIT_RADIUS * math.sin(angle),
        )
        rotation, translation = look_at(position, SCENE_TARGET)
        cameras.append(Camera(f"cam{i}", position, SCENE_TARGET, rotation, translation))
    return cameras


#: 用于 PnP / 重投影校验的 6 个已知三维点（桌角、门框角、盒子角）
def known_points() -> list[dict[str, Any]]:
    tx, tz = TABLE_CENTER[0], TABLE_CENTER[2]
    left_x = DOOR_CENTER_X - DOOR_WIDTH / 2 - DOOR_POST
    right_x = DOOR_CENTER_X + DOOR_WIDTH / 2 + DOOR_POST
    wall_z = -ROOM_Z + 0.01 + DOOR_DEPTH
    return [
        {"id": "P1", "label": "桌面左前角", "xyz": (tx - TABLE_HALF_X, TABLE_TOP_Y, tz + TABLE_HALF_Z)},
        {"id": "P2", "label": "桌面右前角", "xyz": (tx + TABLE_HALF_X, TABLE_TOP_Y, tz + TABLE_HALF_Z)},
        {"id": "P3", "label": "桌面左后角", "xyz": (tx - TABLE_HALF_X, TABLE_TOP_Y, tz - TABLE_HALF_Z)},
        {"id": "P4", "label": "门框左上角", "xyz": (left_x, -DOOR_HEIGHT - DOOR_POST, wall_z)},
        {"id": "P5", "label": "门框右上角", "xyz": (right_x, -DOOR_HEIGHT - DOOR_POST, wall_z)},
        {"id": "P6", "label": "桌上盒子右下角", "xyz": (BOX_MAX[0], BOX_MAX[1], BOX_MAX[2])},
    ]


def build_observations(cameras: list[Camera], margin: float = 24.0, per_point: int = 3) -> dict[str, Any]:
    """为每个已知三维点挑选 3 个「看得见」的相机，输出投影像素坐标。"""
    points = known_points()
    used_cameras: dict[str, Camera] = {}
    observations: list[dict[str, Any]] = []
    visibility: dict[str, int] = {}
    for point in points:
        candidates: list[tuple[float, Camera, tuple[float, float]]] = []
        for cam in cameras:
            uv = cam.project(point["xyz"])
            if uv is None:
                continue
            if not (margin <= uv[0] <= IMAGE_WIDTH - margin and margin <= uv[1] <= IMAGE_HEIGHT - margin):
                continue
            border = min(uv[0], IMAGE_WIDTH - uv[0], uv[1], IMAGE_HEIGHT - uv[1])
            candidates.append((border, cam, uv))
        candidates.sort(key=lambda item: item[0], reverse=True)
        visibility[point["id"]] = len(candidates)
        for _border, cam, uv in candidates[:per_point]:
            used_cameras[cam.cam_id] = cam
            observations.append(
                {
                    "pointId": point["id"],
                    "label": point["label"],
                    "cameraId": cam.cam_id,
                    "uv": [round(uv[0], 6), round(uv[1], 6)],
                }
            )
    return {
        "note": "6 个已知三维点的投影像素坐标，用于校验 PnP / 重投影精度（像素原点在左上角）；"
                "每个点选取可见度最高的 3 个相机（按到图像边界的余量排序）",
        "coordinateSystem": "COLMAP / 3DGS 世界系，y 轴向下，单位米",
        "imageSize": {"width": IMAGE_WIDTH, "height": IMAGE_HEIGHT},
        "intrinsics": {"model": "PINHOLE", "fx": FX, "fy": FY, "cx": CX, "cy": CY},
        "points3D": [{"id": p["id"], "label": p["label"], "xyz": [round(v, 6) for v in p["xyz"]]} for p in points],
        "cameras": [used_cameras[key].to_dict() for key in sorted(used_cameras)],
        "observations": observations,
        "visibilityPerPoint": visibility,
    }


# ---------------------------------------------------------------------------
# 写出 .splat / PLY / JSON
# ---------------------------------------------------------------------------


def write_splat(gaussians: Iterable[Gaussian], out_path: Path) -> dict[str, Any]:
    """写 .splat，同时统计包围盒。"""
    min_b = [float("inf")] * 3
    max_b = [float("-inf")] * 3
    count = 0
    with to_splat.SplatWriter(out_path) as writer:
        for g in gaussians:
            rgba = (
                to_splat.to_u8_half_up(g.color[0] * 255.0),
                to_splat.to_u8_half_up(g.color[1] * 255.0),
                to_splat.to_u8_half_up(g.color[2] * 255.0),
                to_splat.to_u8_half_up(to_splat.sigmoid(g.opacity_raw) * 255.0),
            )
            writer.write_gaussian(g.p, g.scale, rgba, to_splat.encode_quaternion(*g.quat))
            count += 1
            for axis in range(3):
                if g.p[axis] < min_b[axis]:
                    min_b[axis] = g.p[axis]
                if g.p[axis] > max_b[axis]:
                    max_b[axis] = g.p[axis]
    size = [max_b[i] - min_b[i] for i in range(3)]
    return {
        "count": count,
        "min": [round(v, 6) for v in min_b],
        "max": [round(v, 6) for v in max_b],
        "center": [round((min_b[i] + max_b[i]) / 2.0, 6) for i in range(3)],
        "size": [round(v, 6) for v in size],
        "radius": round(0.5 * math.sqrt(sum(v * v for v in size)), 6),
    }


def write_ply(gaussians: Sequence[Gaussian], out_path: Path, sh_degree: int = 0) -> None:
    """写一份标准 3DGS 二进制小端 PLY（便于用 to_splat.py 反向验证）。"""
    rest_per_channel = (sh_degree + 1) ** 2 - 1 if sh_degree > 0 else 0
    props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    props += [f"f_rest_{i}" for i in range(rest_per_channel * 3)]
    props += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    header = "ply\nformat binary_little_endian 1.0\ncomment generated by pipeline/make_test_scene.py (synthetic room)\n"
    header += f"element vertex {len(gaussians)}\n"
    header += "".join(f"property float {name}\n" for name in props)
    header += "end_header\n"
    stride = len(props)
    record = struct.Struct("<" + "f" * stride)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as handle:
        handle.write(header.encode("ascii"))
        buffer = bytearray()
        for g in gaussians:
            dc = [(g.color[i] - 0.5) / to_splat.SH_C0 for i in range(3)]
            values = [g.p[0], g.p[1], g.p[2], 0.0, 0.0, 0.0, dc[0], dc[1], dc[2]]
            values += [0.0] * (rest_per_channel * 3)
            values += [
                g.opacity_raw,
                # 标准 3DGS PLY 里 scale_* 是**线性**尺度（已激活），不是 log 值。
                # 早期这里写的是 math.log(scale)，导致产物与标准查看器不兼容，
                # 也让 to_splat.py 必须 exp 一次才正确。现已统一为标准约定。
                g.scale[0],
                g.scale[1],
                g.scale[2],
                g.quat[0],
                g.quat[1],
                g.quat[2],
                g.quat[3],
            ]
            buffer.extend(record.pack(*values))
            if len(buffer) >= 1 << 20:
                handle.write(buffer)
                buffer.clear()
        if buffer:
            handle.write(buffer)


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make_test_scene.py",
        description="生成合成测试场景（房间 + 相机位姿 + PnP 观测），用于在真实数据到位前验证整条链路。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python pipeline/make_test_scene.py --out data/models/test-scene.splat --count 60000\n"
            "  python pipeline/make_test_scene.py --ply data/models/test-scene.ply --count 20000\n"
        ),
    )
    parser.add_argument("--out", default=str(config.DEFAULT_TEST_SPLAT), help="输出 .splat 路径（默认 data/models/test-scene.splat）")
    parser.add_argument("--json", default=str(config.DEFAULT_TEST_JSON), help="输出元数据 JSON 路径")
    parser.add_argument("--cameras", default=str(config.DEFAULT_TEST_OBSERVATIONS), help="输出 2D 观测 JSON 路径（含相机位姿与内参）")
    parser.add_argument("--camera-poses", default=str(config.DEFAULT_TEST_CAMERAS), help="输出环绕相机位姿 JSON 路径")
    parser.add_argument("--count", type=int, default=60000, help="高斯数量（默认 60000）")
    parser.add_argument("--seed", type=int, default=20240601, help="随机种子（默认 20240601）")
    parser.add_argument("--ply", default=None, help="可选：同时输出一份标准 3DGS PLY，便于反向验证 to_splat.py")
    parser.add_argument("--ply-sh-degree", type=int, default=0, choices=[0, 1, 2, 3], help="PLY 里写入的高阶球谐阶数（默认 0）")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    config.ensure_utf8_stdio()
    config.require_python()
    args = build_parser().parse_args(argv)
    if args.count <= 0:
        config.log_error("--count 必须为正整数。")
        return 2

    config.log_rule("生成合成测试场景")
    started = time.time()
    config.ensure_dirs()

    config.log_info(f"构建房间面片并采样 {args.count} 个高斯（随机种子 {args.seed}）…")
    gaussians, faces = build_scene(args.count, args.seed)
    config.log_info(f"面片数量：{len(faces)}；实际生成高斯：{len(gaussians)}")

    out_path = Path(args.out)
    stats = write_splat(gaussians, out_path)
    size_bytes = out_path.stat().st_size
    config.log_ok(
        f"已写出 {out_path}：{stats['count']} 个高斯，{config.human_bytes(size_bytes)}"
        f"（{size_bytes} 字节 = {stats['count']} × 32）"
    )
    config.log_info(f"包围盒 min={stats['min']} max={stats['max']} size={stats['size']} radius={stats['radius']}")

    cameras = build_cameras()
    poses_path = Path(args.camera_poses)
    poses_payload = {
        "note": "8 个环绕相机位姿，供后端生成演示轨迹；坐标系为 COLMAP / 3DGS 世界系（y 轴向下，单位米）",
        "coordinateSystem": "COLMAP / 3DGS",
        "worldUp": [0.0, -1.0, 0.0],
        "units": "meters",
        "imageSize": {"width": IMAGE_WIDTH, "height": IMAGE_HEIGHT},
        "intrinsics": {"model": "PINHOLE", "fx": FX, "fy": FY, "cx": CX, "cy": CY},
        "orbit": {"center": list(SCENE_TARGET), "radius": ORBIT_RADIUS, "eyeY": ORBIT_EYE_Y, "count": CAMERA_COUNT},
        "cameras": [cam.to_dict() for cam in cameras],
    }
    config.dump_json(poses_path, poses_payload)
    config.log_ok(f"已写出环绕相机位姿：{poses_path}（{len(cameras)} 个相机）")

    observations = build_observations(cameras)
    obs_path = Path(args.cameras)
    config.dump_json(obs_path, observations)
    config.log_ok(
        f"已写出 2D 观测数据：{obs_path}（{len(observations['observations'])} 条观测，"
        f"{len(observations['cameras'])} 个相机，{len(observations['points3D'])} 个三维点）"
    )
    for pid, visible in observations["visibilityPerPoint"].items():
        if visible < 3:
            config.log_warn(f"三维点 {pid} 只有 {visible} 个相机可见（期望 ≥3），观测文件仍会写出可用部分。")

    metadata: dict[str, Any] = {
        "format": "splat",
        "bytesPerGaussian": to_splat.GAUSSIAN_BYTES,
        "count": stats["count"],
        "shDegree": 0,
        "shRestCount": 0,
        "sourcePly": args.ply or None,
        "sourceGaussians": stats["count"],
        "outputBytes": size_bytes,
        "bounds": {
            "min": stats["min"],
            "max": stats["max"],
            "center": stats["center"],
            "size": stats["size"],
            "radius": stats["radius"],
        },
        "aabbMeters": {"min": stats["min"], "max": stats["max"], "size": stats["size"]},
        "createdAt": config.now_iso(),
        "units": "meters",
        "axes": {"system": "COLMAP 3DGS", "x": "right", "y": "down", "z": "forward"},
        "generator": "pipeline/make_test_scene.py",
        "synthetic": True,
        "scene": {
            "type": "synthetic-room",
            "description": "程序化合成的房间：地面 / 后墙 / 左墙 / 桌子 / 球 / 盒子 / 门框",
            "surfaceCount": len(faces),
            "seed": args.seed,
            "roomSizeMeters": {"width": 2 * ROOM_X, "height": -CEILING_Y, "depth": 2 * ROOM_Z},
            "tableTopY": TABLE_TOP_Y,
        },
        "camerasFile": poses_path.name,
        "observationsFile": obs_path.name,
        "filters": {"shDegree": 0, "minOpacity": 0.0, "maxScaleMeters": None, "decimate": 1.0},
        "elapsedSec": round(time.time() - started, 3),
    }
    config.dump_json(Path(args.json), metadata)
    config.log_ok(f"已写出元数据：{args.json}")

    if args.ply:
        ply_path = Path(args.ply)
        write_ply(gaussians, ply_path, args.ply_sh_degree)
        config.log_ok(f"已写出 PLY（球谐 {args.ply_sh_degree} 阶）：{ply_path}（{config.human_bytes(ply_path.stat().st_size)}）")

    config.log_info(f"总耗时 {config.format_duration(time.time() - started)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
