"""pipeline/colmap_recon.py —— 相机位姿重建（SfM）：优先 pycolmap，其次 COLMAP 命令行。

命令行：
    python pipeline/colmap_recon.py --images <图片目录> --workspace <输出目录>
        [--camera-model OPENCV] [--matcher auto|exhaustive|sequential] [--use-gpu 1|0] [--dry-run]

产物：
    <workspace>/database.db              特征与匹配数据库（COLMAP 格式）
    <workspace>/sparse/0/{cameras,images,points3D}.bin
    <workspace>/colmap_summary.json      注册图片数、平均重投影误差、稀疏点数量等

本模块同时提供了**自己实现的 COLMAP 二进制模型读写器**（read_model / write_model），
不依赖 pycolmap，供 train_gsplat.py 与自动化测试使用：
    cameras.bin : uint64 相机数，随后每台：int32 camera_id, int32 model_id,
                  uint64 width, uint64 height, double params[num_params]
    images.bin  : uint64 图片数，随后每张：int32 image_id, double qvec[4](w,x,y,z),
                  double tvec[3], int32 camera_id, 以 \\0 结尾的 name,
                  uint64 num_points2D, 随后 num_points2D × (double x, double y, uint64 point3D_id)
    points3D.bin: uint64 点数，随后每个：uint64 point3D_id, double xyz[3], uint8 rgb[3],
                  double error, uint64 track_length, 随后 track_length × (int32 image_id, int32 point2D_idx)

退出码：0 成功；2 注册图片过少（附补救建议）；3 输入目录不存在/为空；4 缺少 COLMAP 工具；5 重建失败。
"""

from __future__ import annotations

import argparse
import inspect
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

_PIPELINE_DIR = Path(__file__).resolve().parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

import config  # noqa: E402

# ---------------------------------------------------------------------------
# COLMAP 相机模型表
# ---------------------------------------------------------------------------

#: model_id → (名称, 参数个数)
CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}

CAMERA_MODEL_IDS: dict[str, int] = {name: mid for mid, (name, _n) in CAMERA_MODELS.items()}

#: 图像文件扩展名（COLMAP 原生支持 PPM/PGM，tools/make_textured_dataset.py 会用它）
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm"}

#: COLMAP 用 uint64 最大值表示「该二维点没有对应三维点」
INVALID_POINT3D_ID: int = 0xFFFFFFFFFFFFFFFF


class ColmapFormatError(Exception):
    """COLMAP 二进制模型解析错误。"""


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class ColmapCamera:
    """一台相机（内参）。"""

    camera_id: int
    model_id: int
    width: int
    height: int
    params: list[float]

    @property
    def model(self) -> str:
        return CAMERA_MODELS.get(self.model_id, (f"UNKNOWN({self.model_id})", 0))[0]

    def intrinsics(self) -> tuple[float, float, float, float]:
        """返回 (fx, fy, cx, cy)；不支持的模型按 PINHOLE 约定尽力推断。"""
        p = self.params
        model = self.model
        try:
            if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
                f, cx, cy = p[0], p[1], p[2]
                return float(f), float(f), float(cx), float(cy)
            if model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE"):
                return float(p[0]), float(p[1]), float(p[2]), float(p[3])
        except (IndexError, TypeError):
            pass
        raise ColmapFormatError(f"无法从相机模型 {model} 的参数 {p} 中取出内参。")


@dataclass
class ColmapImage:
    """一张注册成功的图片（外参 + 观测到的二维点）。"""

    image_id: int
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    camera_id: int
    name: str
    xys: list[tuple[float, float]] = field(default_factory=list)
    point3D_ids: list[int] = field(default_factory=list)

    def rotation_matrix(self) -> tuple[tuple[float, float, float], ...]:
        """qvec(w,x,y,z) → 世界到相机的 3×3 旋转矩阵 R。"""
        return qvec_to_rotmat(self.qvec)

    def camera_center(self) -> tuple[float, float, float]:
        """相机光心在世界坐标中的位置：C = -R^T t。"""
        r = self.rotation_matrix()
        t = self.tvec
        return (
            -(r[0][0] * t[0] + r[1][0] * t[1] + r[2][0] * t[2]),
            -(r[0][1] * t[0] + r[1][1] * t[1] + r[2][1] * t[2]),
            -(r[0][2] * t[0] + r[1][2] * t[1] + r[2][2] * t[2]),
        )


@dataclass
class ColmapPoint3D:
    """一个稀疏三维点。"""

    point3D_id: int
    xyz: tuple[float, float, float]
    rgb: tuple[int, int, int]
    error: float
    image_ids: list[int] = field(default_factory=list)
    point2D_idxs: list[int] = field(default_factory=list)


@dataclass
class ColmapModel:
    """一个完整的稀疏模型。"""

    cameras: dict[int, ColmapCamera]
    images: dict[int, ColmapImage]
    points3D: dict[int, ColmapPoint3D]
    model_dir: Path | None = None

    @property
    def num_registered(self) -> int:
        return len(self.images)

    @property
    def mean_reprojection_error(self) -> float:
        if not self.points3D:
            return 0.0
        return float(sum(p.error for p in self.points3D.values()) / len(self.points3D))

    @property
    def mean_track_length(self) -> float:
        if not self.points3D:
            return 0.0
        return float(sum(len(p.image_ids) for p in self.points3D.values()) / len(self.points3D))


def qvec_to_rotmat(qvec: Sequence[float]) -> tuple[tuple[float, float, float], ...]:
    """COLMAP 四元数 (w, x, y, z) → 旋转矩阵（行优先）。"""
    w, x, y, z = (float(v) for v in qvec)
    return (
        (1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y),
        (2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x),
        (2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y),
    )


def rotmat_to_qvec(r: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    """旋转矩阵 → COLMAP 四元数 (w, x, y, z)。"""
    m00, m01, m02 = r[0]
    m10, m11, m12 = r[1]
    m20, m21, m22 = r[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = (0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s)
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        q = ((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        q = ((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        q = ((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)
    norm = math.sqrt(sum(v * v for v in q))
    if norm <= 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return (q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm)


# ---------------------------------------------------------------------------
# COLMAP 二进制读取（自实现，不依赖 pycolmap）
# ---------------------------------------------------------------------------


def read_cameras_binary(path: Path | str) -> dict[int, ColmapCamera]:
    """读取 cameras.bin。"""
    cameras: dict[int, ColmapCamera] = {}
    with open(path, "rb") as handle:
        data = handle.read()
    if len(data) < 8:
        raise ColmapFormatError(f"cameras.bin 太短：{path}")
    (num_cameras,) = struct.unpack_from("<Q", data, 0)
    offset = 8
    for _ in range(num_cameras):
        if offset + 24 > len(data):
            raise ColmapFormatError("cameras.bin 数据被截断（相机头信息）。")
        camera_id, model_id, width, height = struct.unpack_from("<iiQQ", data, offset)
        offset += 24
        num_params = CAMERA_MODELS.get(model_id, (None, 0))[1]
        if num_params == 0:
            raise ColmapFormatError(f"未知的相机模型 id={model_id}（camera_id={camera_id}）。")
        if offset + 8 * num_params > len(data):
            raise ColmapFormatError("cameras.bin 数据被截断（相机参数）。")
        params = list(struct.unpack_from(f"<{num_params}d", data, offset))
        offset += 8 * num_params
        cameras[camera_id] = ColmapCamera(camera_id, model_id, int(width), int(height), params)
    return cameras


def read_images_binary(path: Path | str) -> dict[int, ColmapImage]:
    """读取 images.bin（name 按 COLMAP 约定以 \\0 结尾）。"""
    images: dict[int, ColmapImage] = {}
    with open(path, "rb") as handle:
        data = handle.read()
    if len(data) < 8:
        raise ColmapFormatError(f"images.bin 太短：{path}")
    (num_images,) = struct.unpack_from("<Q", data, 0)
    offset = 8
    for _ in range(num_images):
        if offset + 64 > len(data):
            raise ColmapFormatError("images.bin 数据被截断（图片头信息）。")
        image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = struct.unpack_from("<idddddddi", data, offset)
        offset += 64
        end = data.find(b"\x00", offset)
        if end < 0:
            raise ColmapFormatError("images.bin 中图片名缺少结束符。")
        name = data[offset:end].decode("utf-8", "replace")
        offset = end + 1
        if offset + 8 > len(data):
            raise ColmapFormatError("images.bin 数据被截断（二维点数量）。")
        (num_points2D,) = struct.unpack_from("<Q", data, offset)
        offset += 8
        need = 16 * num_points2D
        if offset + need > len(data):
            raise ColmapFormatError("images.bin 数据被截断（二维点坐标）。")
        xys: list[tuple[float, float]] = []
        ids: list[int] = []
        for _i in range(num_points2D):
            x, y, pid = struct.unpack_from("<ddQ", data, offset)
            offset += 24
            xys.append((x, y))
            ids.append(pid)
        images[image_id] = ColmapImage(image_id, (qw, qx, qy, qz), (tx, ty, tz), camera_id, name, xys, ids)
    return images


def read_points3d_binary(path: Path | str) -> dict[int, ColmapPoint3D]:
    """读取 points3D.bin。"""
    points: dict[int, ColmapPoint3D] = {}
    with open(path, "rb") as handle:
        data = handle.read()
    if len(data) < 8:
        raise ColmapFormatError(f"points3D.bin 太短：{path}")
    (num_points,) = struct.unpack_from("<Q", data, 0)
    offset = 8
    for _ in range(num_points):
        if offset + 43 > len(data):
            raise ColmapFormatError("points3D.bin 数据被截断（点信息）。")
        pid, x, y, z, r, g, b = struct.unpack_from("<QdddBBB", data, offset)
        offset += 8 + 24 + 3
        (error,) = struct.unpack_from("<d", data, offset)
        offset += 8
        (track_length,) = struct.unpack_from("<Q", data, offset)
        offset += 8
        if offset + 8 * track_length > len(data):
            raise ColmapFormatError("points3D.bin 数据被截断（track 信息）。")
        image_ids: list[int] = []
        point2D_idxs: list[int] = []
        for _t in range(track_length):
            image_id, point2D_idx = struct.unpack_from("<ii", data, offset)
            offset += 8
            image_ids.append(image_id)
            point2D_idxs.append(point2D_idx)
        points[pid] = ColmapPoint3D(pid, (x, y, z), (r, g, b), error, image_ids, point2D_idxs)
    return points


def read_model(model_dir: Path | str) -> ColmapModel:
    """读取一个 COLMAP 稀疏模型目录（sparse/0）。"""
    directory = Path(model_dir)
    if not directory.is_dir():
        raise ColmapFormatError(f"模型目录不存在：{directory}")
    cameras_bin = directory / "cameras.bin"
    images_bin = directory / "images.bin"
    points_bin = directory / "points3D.bin"
    for required in (cameras_bin, images_bin):
        if not required.is_file():
            raise ColmapFormatError(f"缺少文件：{required}")
    cameras = read_cameras_binary(cameras_bin)
    images = read_images_binary(images_bin)
    points = read_points3d_binary(points_bin) if points_bin.is_file() else {}
    return ColmapModel(cameras, images, points, directory)


# ---------------------------------------------------------------------------
# COLMAP 二进制写入（自实现，供测试做往返校验，也可用于修复/裁剪模型）
# ---------------------------------------------------------------------------


def write_cameras_binary(path: Path | str, cameras: dict[int, ColmapCamera]) -> None:
    """写出 cameras.bin。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as handle:
        handle.write(struct.pack("<Q", len(cameras)))
        for camera in cameras.values():
            num_params = CAMERA_MODELS[camera.model_id][1]
            if len(camera.params) != num_params:
                raise ColmapFormatError(
                    f"相机 {camera.camera_id} 的模型 {camera.model} 需要 {num_params} 个参数，实际 {len(camera.params)} 个。"
                )
            handle.write(struct.pack("<iiQQ", camera.camera_id, camera.model_id, camera.width, camera.height))
            handle.write(struct.pack(f"<{num_params}d", *camera.params))


def write_images_binary(path: Path | str, images: dict[int, ColmapImage]) -> None:
    """写出 images.bin。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as handle:
        handle.write(struct.pack("<Q", len(images)))
        for image in images.values():
            handle.write(
                struct.pack(
                    "<idddddddi",
                    image.image_id,
                    image.qvec[0],
                    image.qvec[1],
                    image.qvec[2],
                    image.qvec[3],
                    image.tvec[0],
                    image.tvec[1],
                    image.tvec[2],
                    image.camera_id,
                )
            )
            handle.write(image.name.encode("utf-8") + b"\x00")
            handle.write(struct.pack("<Q", len(image.xys)))
            for (x, y), pid in zip(image.xys, image.point3D_ids):
                # 负数视为「无效点」，统一写成 COLMAP 的 uint64 哨兵值
                handle.write(struct.pack("<ddQ", x, y, INVALID_POINT3D_ID if pid < 0 else pid))


def write_points3d_binary(path: Path | str, points: dict[int, ColmapPoint3D]) -> None:
    """写出 points3D.bin。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as handle:
        handle.write(struct.pack("<Q", len(points)))
        for point in points.values():
            handle.write(struct.pack("<QdddBBB", point.point3D_id, point.xyz[0], point.xyz[1], point.xyz[2], *point.rgb))
            handle.write(struct.pack("<d", point.error))
            handle.write(struct.pack("<Q", len(point.image_ids)))
            for image_id, idx in zip(point.image_ids, point.point2D_idxs):
                handle.write(struct.pack("<ii", image_id, idx))


def write_model(model_dir: Path | str, model: ColmapModel) -> None:
    """写出完整稀疏模型（cameras.bin + images.bin + points3D.bin）。"""
    directory = Path(model_dir)
    directory.mkdir(parents=True, exist_ok=True)
    write_cameras_binary(directory / "cameras.bin", model.cameras)
    write_images_binary(directory / "images.bin", model.images)
    write_points3d_binary(directory / "points3D.bin", model.points3D)


# ---------------------------------------------------------------------------
# 输入扫描
# ---------------------------------------------------------------------------


def link_dataset_images(images_dir: Path, workspace: Path) -> Path | None:
    """
    把图片目录接到 `<workspace>/images`，形成标准 COLMAP 数据集布局。

    优先做**目录联接/符号链接**（省空间、不复制），失败时退化为复制。
    若图片本来就在 `<workspace>/images`，直接返回。
    """
    target = workspace / "images"
    src = images_dir.resolve()
    if target.exists():
        try:
            if target.resolve() == src:
                return target
        except OSError:
            pass
        # 已存在但不是同一目录：不覆盖，交给下游
        return target

    # 1) Windows 目录联接（不需要管理员权限）
    if os.name == "nt":
        try:
            proc = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(target), str(src)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
            if proc.returncode == 0 and target.exists():
                config.log_info(f"已把图片目录联接为 {target}（不占额外空间）")
                return target
        except (OSError, subprocess.SubprocessError):
            pass

    # 2) 普通符号链接（Linux / 有权限的 Windows）
    try:
        target.symlink_to(src, target_is_directory=True)
        if target.exists():
            config.log_info(f"已把图片目录软链为 {target}")
            return target
    except (OSError, NotImplementedError):
        pass

    # 3) 兜底：真的复制
    try:
        shutil.copytree(src, target, dirs_exist_ok=True)
        config.log_info(f"已把图片复制到 {target}")
        return target
    except OSError as exc:
        config.log_warn(f"无法把图片接入工作区（{exc}）；下游训练脚本需要自行指定 --images")
        return None


def find_image_subdirs(images_dir: Path) -> list[tuple[Path, int]]:
    """找出图目录里**含图片的子目录**，返回 [(子目录, 图片数)]。

    为什么要查这个（2026-10-07 实测的真 bug）：
      COLMAP 的 `--image_path` 会**递归扫描子目录**，而抽帧工具的模糊过滤
      曾经把剔掉的模糊帧放在 `images/_rejected/` —— 结果那批帧照样被拿去重建：
      用户视频实测 227 张保留 + 45 张剔除，数据库里却是 **272 张**，其中 45 张
      来自 `_rejected/`，而重建断链的帧区间正好就是这些模糊帧。
      抽帧工具已经改成把剔除帧放到图目录**之外**，但历史数据、
      或人手往图目录里塞子目录，都会再踩同一个坑 —— 所以这里显式查出来并告警。
    """
    out: list[tuple[Path, int]] = []
    if not images_dir.is_dir():
        return out
    try:
        children = sorted(images_dir.iterdir())
    except OSError:
        return out
    for child in children:
        if not child.is_dir():
            continue
        n = sum(
            1
            for p in child.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if n:
            out.append((child, n))
    return out


def list_images(images_dir: Path) -> list[Path]:
    """列出目录里的图片文件（**只看本层**，不看子目录）。

    注意：这里"只看本层"是对的（我们不想把子目录里的图算进来），
    但**COLMAP 自己会递归** —— 所以调用方必须先用 find_image_subdirs 告警。
    """
    if not images_dir.is_dir():
        return []
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def pick_best_model(sparse_dir: Path) -> Path | None:
    """在 sparse/ 下挑选注册图片最多的模型目录（COLMAP 可能输出 0/1/2 …）。"""
    if not sparse_dir.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for child in sorted(sparse_dir.iterdir()):
        images_bin = child / "images.bin"
        if child.is_dir() and images_bin.is_file():
            try:
                (count,) = struct.unpack("<Q", images_bin.read_bytes()[:8])
            except (struct.error, OSError):
                count = 0
            candidates.append((int(count), child))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


# ---------------------------------------------------------------------------
# pycolmap 适配层（不同版本 API 差异较大，这里做尽力而为的调用）
# ---------------------------------------------------------------------------


def _accepted_params(fn: Callable[..., Any]) -> set[str] | None:
    """
    尽力找出目标函数接受的**关键字参数名**。

    pycolmap 是 pybind11 绑定的，`inspect.signature()` 对它直接抛
    `ValueError: no signature found for builtin`。好消息是 pybind11 会把签名写进
    `__doc__` 的第一行，形如：
        extract_features(database_path, image_path, image_names=..., camera_mode=..., ...) -> None
    所以这里退回解析 docstring。拿不到就返回 None（表示"未知，走试错"）。
    """
    try:
        params = inspect.signature(fn).parameters
        if params:
            return set(params)
    except (TypeError, ValueError):
        pass
    doc = getattr(fn, "__doc__", None) or ""
    head = doc.strip().split("\n", 1)[0]
    if "(" not in head or ")" not in head:
        return None
    inner = head[head.index("(") + 1 : head.rindex(")")]
    if not inner.strip():
        return set()
    names = set()
    for chunk in inner.split(","):
        token = chunk.strip().lstrip("*")
        if not token:
            continue
        name = token.split("=")[0].split(":")[0].strip()
        if name.isidentifier():
            names.add(name)
    return names or None


def _call_supported(fn: Callable[..., Any], **kwargs: Any) -> Any:
    """只把目标函数真正接受的参数传进去，兼容不同版本的 pycolmap。"""
    accepted = _accepted_params(fn)
    if accepted is None:
        # 完全无法预知签名 → 先按原样试，失败再逐个丢掉可疑参数
        try:
            return fn(**kwargs)
        except TypeError:
            reduced = dict(kwargs)
            for drop in ("sift_options", "extraction_options", "image_dir", "image_path", "use_gpu", "camera_model"):
                reduced.pop(drop, None)
                try:
                    return fn(**reduced)
                except TypeError:
                    continue
            raise
    # 同一个含义在不同 pycolmap 版本里可能叫不同名字（例如 image_dir → image_path）。
    # 直接按 accepted 过滤会把**两个别名同时**传进去，触发
    #   TypeError: extract_features(): incompatible function arguments
    # 所以这里对每个「别名组」只保留签名真正接受的那一个。
    alias_groups = [
        ("image_path", "image_dir"),
        ("extraction_options", "sift_options"),
        ("reader_options", "image_reader_options"),
        ("device", "use_gpu"),
    ]
    resolved = dict(kwargs)
    for group in alias_groups:
        present = [name for name in group if name in resolved]
        if len(present) <= 1:
            continue
        keep = next((name for name in group if name in accepted), None)
        if keep is None:
            # 一个都不被接受 → 全部丢掉，让调用方走默认值
            for name in present:
                resolved.pop(name, None)
            continue
        for name in present:
            if name != keep:
                resolved.pop(name, None)
    filtered = {k: v for k, v in resolved.items() if k in accepted}
    try:
        return fn(**filtered)
    except TypeError:
        # 仍不匹配（例如某版本要求位置参数）→ 逐步减少可选参数再试一次
        for drop in ("sift_options", "extraction_options", "use_gpu", "device"):
            filtered.pop(drop, None)
        return fn(**filtered)


def run_pycolmap(
    images_dir: Path,
    workspace: Path,
    *,
    camera_model: str,
    matcher: str,
    use_gpu: bool,
    dry_run: bool,
    mapper: str = "incremental",
) -> bool:
    """用 pycolmap 跑完整 SfM 流程；成功返回 True（失败时调用方回退到命令行）。"""
    pycolmap = config.try_import("pycolmap")
    if pycolmap is None:
        return False
    database = workspace / "database.db"
    sparse = workspace / "sparse"
    device = "cuda" if use_gpu else "cpu"

    if dry_run:
        config.log_info("[dry-run] 将调用 pycolmap，等价操作如下：")
        config.log_info(f"  pycolmap.extract_features(database_path='{database}', image_dir='{images_dir}', camera_model='{camera_model}', device='{device}')")
        if matcher == "sequential":
            config.log_info(f"  pycolmap.match_sequential(database_path='{database}', device='{device}')")
        else:
            config.log_info(f"  pycolmap.match_exhaustive(database_path='{database}', device='{device}')")
        config.log_info(f"  pycolmap.incremental_mapping(database_path='{database}', image_dir='{images_dir}', output_path='{sparse}')")
        return True

    workspace.mkdir(parents=True, exist_ok=True)
    sparse.mkdir(parents=True, exist_ok=True)
    try:
        config.log_info("使用 pycolmap 提取特征…")
        _call_supported(
            pycolmap.extract_features,
            database_path=str(database),
            image_dir=str(images_dir),
            image_path=str(images_dir),
            camera_model=camera_model,
            device=device,
            use_gpu=use_gpu,
            sift_options=None,
        )
        config.log_info(f"使用 pycolmap 进行{ '顺序' if matcher == 'sequential' else '穷举' }匹配…")
        if matcher == "sequential":
            _call_supported(pycolmap.match_sequential, database_path=str(database), device=device, use_gpu=use_gpu)
        else:
            _call_supported(pycolmap.match_exhaustive, database_path=str(database), device=device, use_gpu=use_gpu)
        # 建图方式：global = 全局建图（原 GLOMAP，COLMAP 3.12+ 已并入）。
        # 实测宿舍 211 帧：global_mapping 47 秒、注册 210 张；incremental 8.9 分钟、
        # 注册 198 张，且增量法出现**尺度爆炸**（点云长宽比 2.94 vs 1.79）。
        n_imgs = len([p for p in images_dir.iterdir() if p.is_file()]) if images_dir.exists() else 0
        use_global = mapper == "global" or (mapper == "auto" and n_imgs > 150)
        has_global = hasattr(pycolmap, "global_mapping")
        if use_global and has_global:
            config.log_info(f"使用 pycolmap.global_mapping（原 GLOMAP，{n_imgs} 张）…")
            maps = _call_supported(
                pycolmap.global_mapping,
                database_path=str(database),
                image_dir=str(images_dir),
                image_path=str(images_dir),
                output_path=str(sparse),
            )
        else:
            if use_global and not has_global:
                config.log_warn("当前 pycolmap 没有 global_mapping，退回增量式"
                                "（可升级 pycolmap，或直接用 colmap global_mapper 命令行）")
            config.log_info("使用 pycolmap 做增量式重建（incremental_mapping）…")
            maps = _call_supported(
                pycolmap.incremental_mapping,
                database_path=str(database),
                image_dir=str(images_dir),
                image_path=str(images_dir),
                output_path=str(sparse),
            )
        if isinstance(maps, dict):
            if not maps:
                config.log_warn("pycolmap 没有输出任何重建模型。")
                return False
            # 把注册图片最多的模型写到 sparse/0，保证下游路径一致
            best = max(maps.values(), key=lambda m: len(getattr(m, "images", {})))
            best_dir = sparse / "0"
            if best_dir.exists():
                shutil.rmtree(best_dir, ignore_errors=True)
            best_dir.mkdir(parents=True, exist_ok=True)
            if hasattr(best, "write"):
                try:
                    best.write(str(best_dir))
                except TypeError:
                    best.write(best_dir)
            config.log_ok(f"pycolmap 重建完成：{len(getattr(best, 'images', {}))} 张图片注册成功。")
    except Exception as exc:  # noqa: BLE001 - pycolmap 各版本异常类型不同
        config.log_warn(f"pycolmap 流程失败：{type(exc).__name__}: {exc}")
        config.log_warn("将回退到 COLMAP 命令行。若本机没有 colmap 命令，请按提示安装或改用 Python 包。")
        return False
    return True


# ---------------------------------------------------------------------------
# COLMAP 命令行流程
# ---------------------------------------------------------------------------


def _supported_options(colmap: str, command: str) -> set[str] | None:
    """
    探测某个 COLMAP 子命令支持哪些 `--Xxx.yyy` 选项。

    COLMAP 4.x 把若干选项改名了（例如 `--SiftExtraction.use_gpu`
    → `--FeatureExtraction.use_gpu`），写死任何一个版本的名字都会在另一个版本上直接报
    「unrecognised option」。这里先读一次 `--help`，拿到真实支持的选项集合，
    再从中挑可用的写法；探测失败（返回 None）时退化为「全部传上去」。
    """
    try:
        proc = subprocess.run(
            [colmap, command, "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = (proc.stdout or "") + (proc.stderr or "")
    found = set(re.findall(r"(--[A-Za-z][A-Za-z0-9_.]*)", text))
    return found or None


def _pick_options(supported: set[str] | None, candidates: Sequence[tuple[str, str]]) -> list[str]:
    """从候选 (选项名, 取值) 里挑出第一组被支持的写法；都不支持就返回第一组。"""
    if supported is None:
        return list(candidates[0])
    for name, value in candidates:
        if name in supported:
            return [name, value]
    return list(candidates[0])


def colmap_commands(
    colmap: str,
    images_dir: Path,
    workspace: Path,
    *,
    camera_model: str,
    matcher: str,
    use_gpu: bool,
    mask_dir: Path | None = None,
    mapper: str = "incremental",
) -> list[list[str]]:
    """构造 COLMAP 命令行三件套（feature_extractor / matcher / mapper）。

    选项名对版本敏感，这里用 `_supported_options` 探测后再选，
    同时兼容 COLMAP 3.x 与 4.x。

    `mask_dir`（可选）：背景分离的 mask 目录。约定与 COLMAP 一致 ——
    **mask 与图片同名（扩展名换 .png）、黑 = 忽略该区域**（见 tools/object_mask.py）。
    这是"物体算法专有"的步骤：房间拍摄没有"背景"可言，给房间加 mask 只会挖掉主体。
    """
    database = workspace / "database.db"
    sparse = workspace / "sparse"
    gpu_flag = "1" if use_gpu else "0"

    fe_supported = _supported_options(colmap, "feature_extractor")
    fe_gpu = _pick_options(
        fe_supported,
        [("--FeatureExtraction.use_gpu", gpu_flag), ("--SiftExtraction.use_gpu", gpu_flag)],
    )
    fe_mask: list[str] = []
    if mask_dir is not None:
        mask_dir = Path(mask_dir)
        if fe_supported is not None and "--ImageReader.mask_path" not in fe_supported:
            # ⚠️ 不支持时必须**明确说出来**并退化为"不用 mask"。
            #    静默传一个不被识别的选项会让 COLMAP 直接报 unrecognised option 退出，
            #    而用户只会看到"建模失败"，根本猜不到是 mask 引起的。
            config.log_warn(
                f"这个 COLMAP 版本不支持 --ImageReader.mask_path，"
                f"本次**忽略**背景分离 mask（{mask_dir}）。"
            )
        else:
            fe_mask = ["--ImageReader.mask_path", str(mask_dir)]
            n_masks = len(list(mask_dir.glob("*.png"))) if mask_dir.is_dir() else 0
            if fe_supported is None:
                config.log_warn("无法探测 COLMAP 选项（--help 解析失败），仍尝试使用 mask。")
            if n_masks == 0:
                config.log_warn(
                    f"mask 目录里没有 png：{mask_dir} —— COLMAP 会因缺少同名 mask 而行为异常，"
                    "请先跑 tools/object_mask.py。"
                )
            else:
                config.log_info(f"已启用背景分离：{n_masks} 个 mask（黑=忽略）来自 {mask_dir}")

    commands: list[list[str]] = [
        [
            colmap,
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(images_dir),
            "--ImageReader.camera_model",
            camera_model,
            "--ImageReader.single_camera",
            "1",
            *fe_mask,
            *fe_gpu,
        ]
    ]

    if matcher == "sequential":
        sm_supported = _supported_options(colmap, "sequential_matcher")
        sm_gpu = _pick_options(
            sm_supported,
            [("--FeatureMatching.use_gpu", gpu_flag), ("--SiftMatching.use_gpu", gpu_flag)],
        )
        commands.append(
            [
                colmap,
                "sequential_matcher",
                "--database_path",
                str(database),
                "--SequentialMatching.overlap",
                "15",
                "--SequentialMatching.loop_detection",
                "0",
                *sm_gpu,
            ]
        )
    else:
        em_supported = _supported_options(colmap, "exhaustive_matcher")
        em_gpu = _pick_options(
            em_supported,
            [("--FeatureMatching.use_gpu", gpu_flag), ("--SiftMatching.use_gpu", gpu_flag)],
        )
        em_guided = _pick_options(
            em_supported,
            [("--FeatureMatching.guided_matching", "1"), ("--SiftMatching.guided_matching", "1")],
        )
        commands.append(
            [
                colmap,
                "exhaustive_matcher",
                "--database_path",
                str(database),
                *em_guided,
                *em_gpu,
            ]
        )

    # mapper 的 --output_path 必须是一个已存在的目录，否则 COLMAP 直接报
    # "output_path is not a directory" 并退出（实测 COLMAP 4.2.1）
    sparse.mkdir(parents=True, exist_ok=True)
    # 建图方式：global = COLMAP 的全局建图（原 GLOMAP，3.12+ 已并入）。
    # 实测宿舍 211 帧：colmap global_mapper **47 秒 / 注册 210 张**；
    # 增量 mapper 8.9 分钟 / 198 张，且出现尺度爆炸（点云长宽比 2.94 vs 1.79）。
    # ⚠️ pycolmap 里也有 global_mapping，但它是 pybind 内建、签名探测不到，
    #    传参容易不匹配（实测会静默失败并回退 CLI）—— 所以统一走命令行。
    n_images = len([p for p in images_dir.iterdir() if p.is_file()]) if images_dir.exists() else 0
    mapper_cmd = "mapper"
    if mapper == "global" or (mapper == "auto" and n_images > 150):
        mapper_cmd = "global_mapper"
    config.log_info(f"建图命令：colmap {mapper_cmd}")
    mapper_supported = _supported_options(colmap, mapper_cmd)
    # ⚠️ global_mapper **不接受** --Mapper.ba_refine_principal_point（实测报
    #    "unrecognised option" 并直接返回码 1）。_supported_options 的探测对
    #    global_mapper 不可靠，这里直接不传任何 Mapper.* 选项。
    mapper_pp = [] if mapper_cmd == "global_mapper" else _pick_options(
        mapper_supported,
        [("--Mapper.ba_refine_principal_point", "0")],
    )
    commands.append(
        [
            colmap,
            mapper_cmd,
            "--database_path",
            str(database),
            "--image_path",
            str(images_dir),
            "--output_path",
            str(sparse),
            *mapper_pp,
        ]
    )
    return commands


def choose_matcher(matcher: str, image_names: Sequence[str], total: int) -> str:
    """auto 模式下根据文件名/数量挑选匹配策略：视频抽帧 → sequential，否则 exhaustive。"""
    if matcher != "auto":
        return matcher
    looks_like_video = total > 0 and sum(1 for name in image_names if name.lower().startswith("frame_")) >= max(1, total // 2)
    if looks_like_video or total > 150:
        return "sequential"
    return "exhaustive"


def run_colmap_cli(
    images_dir: Path,
    workspace: Path,
    *,
    colmap: str,
    camera_model: str,
    matcher: str,
    use_gpu: bool,
    dry_run: bool,
    mask_dir: Path | None = None,
    mapper: str = "incremental",
) -> bool:
    """执行 COLMAP 命令行流程。"""
    commands = colmap_commands(
        colmap, images_dir, workspace,
        camera_model=camera_model, matcher=matcher, use_gpu=use_gpu, mask_dir=mask_dir,
        mapper=mapper,
    )
    if dry_run:
        config.log_info("[dry-run] 将执行以下命令：")
        for command in commands:
            config.log_info("  " + " ".join(f'"{part}"' if " " in part else part for part in command))
        return True

    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "sparse").mkdir(parents=True, exist_ok=True)
    stages = ["特征提取", "特征匹配", "增量式重建"]
    for index, command in enumerate(commands):
        config.log_info(f"COLMAP {stages[index]}：{Path(command[0]).name} {command[1]} …")
        result = config.run_command(command, timeout=None, echo=False)
        if not result.ok:
            config.log_error(f"COLMAP {stages[index]}失败（返回码 {result.returncode}）。")
            if result.tail(12):
                config.log_error(result.tail(12))
            config.log_info("排查建议：")
            config.log_info("  1) 确认安装的是 CUDA 版且显卡驱动正常；不行就加 --use-gpu 0 走 CPU；")
            config.log_info("  2) 路径不要含中文/空格（COLMAP 对非 ASCII 路径支持不佳）；")
            config.log_info("  3) 图片数量太少时 mapper 可能返回 0，但这不是致命错误，脚本会继续检查结果。")
            if command[1] == "mapper":
                return False
            return False
        config.log_ok(f"COLMAP {stages[index]}完成（耗时 {config.format_duration(result.elapsed_sec)}）")
    return True


# ---------------------------------------------------------------------------
# 结果汇总
# ---------------------------------------------------------------------------


def summarize(model: ColmapModel, total_images: int, matcher: str, elapsed: float) -> dict[str, Any]:
    """生成 colmap_summary.json 的内容。"""
    camera = next(iter(model.cameras.values()), None)
    return {
        "registeredImages": model.num_registered,
        "totalImages": total_images,
        "registeredRatio": round(model.num_registered / total_images, 4) if total_images else 0.0,
        "meanReprojectionError": round(model.mean_reprojection_error, 4),
        "points3D": len(model.points3D),
        "meanTrackLength": round(model.mean_track_length, 3),
        "cameraModel": camera.model if camera else "UNKNOWN",
        "matcher": matcher,
        "elapsedSec": round(elapsed, 3),
        "cameras": [
            {
                "cameraId": cam.camera_id,
                "model": cam.model,
                "width": cam.width,
                "height": cam.height,
                "intrinsics": dict(zip(("fx", "fy", "cx", "cy"), cam.intrinsics())) if cam.model != "UNKNOWN" else None,
            }
            for cam in model.cameras.values()
        ],
        "createdAt": config.now_iso(),
        "generator": "pipeline/colmap_recon.py",
    }


def print_remediation(registered: int, total: int) -> None:
    """打印注册图片过少时的中文补救建议。"""
    config.log_error(f"注册成功的图片太少：{registered}/{total}（要求 ≥60% 且 ≥8 张）。")
    config.log_error("可执行的补救建议（按性价比排序）：")
    config.log_error("  1) 提高重叠率：抽帧更密（--fps 3~5），相邻帧画面位移不要超过画面宽度的 30%；")
    config.log_error("  2) 换相机模型：python pipeline/colmap_recon.py ... --camera-model OPENCV（手机视频畸变明显时必选）；")
    config.log_error("  3) 打开引导匹配：命令行版会自动加 --SiftMatching.guided_matching 1，pycolmap 版请升级到最新；")
    config.log_error("  4) 顺序匹配只在「一路单向拍下去」时才更快；")
    config.log_error("     实测（本机、175 帧手持视频）：--matcher sequential 只注册 57/175（32.6%），")
    config.log_error("     而 --matcher exhaustive 能注册 178/178（100%）。素材有回头/绕行时不要用 sequential。")
    config.log_error("  5) 先剔除模糊帧：python pipeline/extract_frames.py ... --blur-threshold 80；")
    config.log_error("  6) 纹理太少的场景（白墙、纯色地面、玻璃）无法定位，请贴临时标记或摆放有纹理的物体；")
    config.log_error("  7) GPU 相关报错可试 --use-gpu 0；反之 CPU 很慢时可装 CUDA 版 COLMAP 再开 --use-gpu 1；")
    config.log_error("  8) 拍摄时避免纯旋转（原地转头），要有平移分量，否则无法三角化。")


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="colmap_recon.py",
        description="从图片序列重建相机位姿与稀疏点云（优先 pycolmap，其次 colmap 命令行）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python pipeline/colmap_recon.py --images data/projects/case01/frames --workspace data/projects/case01/colmap\n"
            "  python pipeline/colmap_recon.py --images frames --workspace colmap --matcher sequential --use-gpu 0\n"
            "  python pipeline/colmap_recon.py --images frames --workspace colmap --dry-run\n"
        ),
    )
    parser.add_argument("--images", required=True, help="图片目录（例如 extract_frames.py 的输出目录）")
    parser.add_argument("--workspace", required=True, help="工作目录（输出 database.db 与 sparse/0）")
    parser.add_argument("--camera-model", default="OPENCV", help="COLMAP 相机模型（默认 OPENCV；畸变很小可用 PINHOLE/SIMPLE_RADIAL）")
    parser.add_argument("--matcher", default="auto", choices=["auto", "exhaustive", "sequential"], help="匹配策略（默认 auto：视频帧或 >150 张走 sequential，否则 exhaustive）")
    parser.add_argument("--mapper", default="incremental", choices=["incremental", "global", "auto"],
                        help="建图方式：incremental=经典增量式；global=COLMAP 的全局建图（原 GLOMAP，"
                             "已并入 3.12+）；auto=帧数>150 用 global。实测宿舍 211 帧：global 47 秒/注册 210 张，"
                             "增量 8.9 分钟/198 张，且增量法尺度爆炸（长宽比 2.94 vs 1.79）")
    parser.add_argument("--use-gpu", type=int, default=1, choices=[0, 1], help="是否使用 GPU（1 开 0 关，默认 1）")
    parser.add_argument(
        "--mask-dir", default=None,
        help="背景分离 mask 目录（与图片同名 .png，黑=忽略）。物体算法专有："
             "房间拍摄没有背景可言，不要给房间加 mask。生成见 tools/object_mask.py",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印将要执行的命令，不真正运行")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    config.ensure_utf8_stdio()
    config.require_python()
    args = build_parser().parse_args(argv)

    images_dir = Path(args.images)
    workspace = Path(args.workspace)
    config.log_rule("相机位姿重建（COLMAP / pycolmap）")
    config.log_info(f"图片目录：{images_dir}")
    config.log_info(f"工作目录：{workspace}")

    images = list_images(images_dir)
    if not images:
        config.log_error(f"图片目录不存在或没有图片：{images_dir}")
        config.log_info("请先运行：python pipeline/extract_frames.py --video <视频> --out <图片目录>")
        return 3
    # ⚠️ 图目录里若还有**含图片的子目录**，COLMAP 会递归读进去（它不听我们的 list_images）。
    #    这就是"模糊过滤白做"那个 bug 的入口，必须显式告警而不是静默继续。
    stray = find_image_subdirs(images_dir)
    if stray:
        total_stray = sum(n for _p, n in stray)
        config.log_warn(
            f"图片目录里有 {len(stray)} 个含图片的子目录（共 {total_stray} 张）："
            + "、".join(f"{p.name}({n})" for p, n in stray[:4])
        )
        config.log_warn(
            "  COLMAP 的 --image_path 会**递归扫描子目录**，这些图会被一起拿去重建"
            "（实测踩过：被模糊过滤剔除的帧就是这样又进了重建）。"
        )
        config.log_warn(
            f"  建议把它们移出图目录，例如：move \"{images_dir}\\{stray[0][0].name}\" \"{images_dir.parent}\""
        )
    matcher = choose_matcher(args.matcher, [p.name for p in images], len(images))
    if args.matcher == "auto":
        reason = "文件名形如 frame_XXXX（视频抽帧）" if matcher == "sequential" else "图片数量较少"
        config.log_info(f"自动选择匹配策略：{matcher}（依据：{reason}）")
    config.log_info(f"图片数量：{len(images)}；相机模型：{args.camera_model}；GPU：{'开' if args.use_gpu else '关'}")

    # ---- 背景分离 mask（可选；物体算法专有）---------------------------------
    # 约定：mask 与图片**同名**（扩展名换 .png）、黑 = 忽略该区域。
    # 生成工具：tools/object_mask.py（关键帧画框 → GrabCut → 逐帧传播）。
    mask_dir: Path | None = None
    if getattr(args, "mask_dir", None):
        candidate = Path(args.mask_dir)
        if not candidate.is_dir():
            config.log_error(f"--mask-dir 不是目录：{candidate}")
            return 3
        mask_names = {p.stem for p in candidate.glob("*.png")}
        missing = [p.stem for p in images if p.stem not in mask_names]
        if missing:
            # 缺 mask 的帧要么整帧参与（旧行为），要么整帧被忽略 —— 两种都不是用户想要的，
            # 所以这里**明确报错停下来**，而不是猜。
            config.log_error(
                f"mask 不完整：{len(missing)}/{len(images)} 张图片没有同名 mask（例如 {missing[:3]}）。"
            )
            config.log_info("请先运行：python tools/object_mask.py --images <帧目录> --out <mask目录> --box ...")
            return 3
        mask_dir = candidate
        config.log_ok(f"已加载背景分离 mask：{len(mask_names)} 个（黑=忽略）")

    pycolmap = config.try_import("pycolmap")
    colmap_exe = config.find_colmap()
    config.log_info(f"pycolmap：{'可用' if pycolmap is not None else '不可用'}；colmap 命令行：{colmap_exe or '未找到'}")

    if pycolmap is None and colmap_exe is None:
        if args.dry_run:
            config.log_warn("本机既没有 pycolmap 也没有 colmap 命令行；下面按默认方式打印将要执行的命令。")
            run_colmap_cli(
                images_dir,
                workspace,
                colmap="colmap",
                camera_model=args.camera_model,
                matcher=matcher,
                use_gpu=bool(args.use_gpu),
                dry_run=True,
                mask_dir=mask_dir,
            )
            config.log_ok("dry-run 结束：以上即为将要执行的命令。")
            return 0
        config.log_error("既没有 pycolmap 也没有 colmap 命令行，无法重建相机位姿。")
        config.log_info("安装方式（任选其一）：")
        config.log_info("  1) 官方 COLMAP：下载 https://github.com/colmap/colmap/releases 的 Windows 压缩包，"
                        "解压后把 COLMAP.bat 所在目录加入 PATH，或设置环境变量 COLMAP_EXE 指向它；")
        config.log_info("  2) pip install pycolmap（无需外部程序，但功能受版本限制）；")
        config.log_info("  3) 先跑 python pipeline/setup_env.py --check 查看完整环境自检表。")
        return 4

    # ---------------------------------------------------------------- 执行重建
    #
    # 为什么要「重试」：
    #   sequential 匹配快得多，但只在「一路单向拍下去」时才稳。手持现场视频常常
    #   一边走一边回头，顺序匹配就会把回环漏掉。本机实测（175 帧手持视频）：
    #       --matcher sequential → 注册  57/175（32.6%）
    #       --matcher exhaustive → 注册 178/178（100%）
    #   auto 模式对视频帧默认选 sequential，于是 CLI 的一键流水线会白跑一场。
    #   现在：**注册率不达标就自动用 exhaustive 再试一次**，用户不必自己去猜参数。
    attempts = [matcher]
    if matcher != "exhaustive" and not args.dry_run:
        attempts.append("exhaustive")

    summary = None
    summary_path = workspace / "colmap_summary.json"
    used = "未执行"
    used_matcher = matcher
    pycolmap_broken = False

    for attempt_index, attempt_matcher in enumerate(attempts):
        if attempt_index:
            config.log_warn(
                f"匹配策略 {attempts[0]} 注册率不达标（{summary['registeredImages']}/{len(images)}），"
                "自动改用 exhaustive 重试一次（更慢但更稳）。"
            )
            # 清掉上一次的稀疏模型，否则 pick_best_model 还会捡到那个差模型
            shutil.rmtree(workspace / "sparse", ignore_errors=True)

        started = time.time()
        used_matcher = attempt_matcher
        # 要背景分离时**优先走 COLMAP 命令行**：pycolmap 的 mask 参数在各版本间
        # 名字/支持情况不一致，而命令行有 `--ImageReader.mask_path` 这条稳定路径。
        # 静默丢掉 mask 是最坏的结果（用户以为做了背景分离，其实没有）。
        # --mapper global：直接走命令行 global_mapper。
        # 不用 pycolmap.global_mapping —— 它是 pybind 内建函数，签名探测不到，
        # 传参不匹配时会**静默返回空**（实测：日志里看不到任何报错，最后悄悄退回增量法）。
        if args.mapper == "global" and colmap_exe is not None:
            config.log_info("已指定 --mapper global：走命令行 global_mapper（原 GLOMAP）")
            pycolmap_broken = True
        if mask_dir is not None and pycolmap is not None and not pycolmap_broken and colmap_exe is not None:
            config.log_info("已指定 mask：本次改用 COLMAP 命令行（mask 支持更稳定），跳过 pycolmap。")
            pycolmap_broken = True
        if pycolmap is not None and not pycolmap_broken:
            ok = run_pycolmap(
                images_dir,
                workspace,
                camera_model=args.camera_model,
                matcher=attempt_matcher,
                use_gpu=bool(args.use_gpu),
                dry_run=args.dry_run,
                mapper=args.mapper,
            )
            used = "pycolmap"
            if not ok and not args.dry_run:
                pycolmap_broken = True
                config.log_warn("pycolmap 未能产出可用模型，改用 COLMAP 命令行重试。")
                if colmap_exe is None:
                    config.log_error("但本机没有 colmap 命令行，无法继续。请安装 COLMAP 或修复 pycolmap。")
                    return 4
                ok = run_colmap_cli(
                    images_dir,
                    workspace,
                    colmap=colmap_exe,
                    camera_model=args.camera_model,
                    matcher=attempt_matcher,
                    use_gpu=bool(args.use_gpu),
                    dry_run=False,
                    mask_dir=mask_dir,
                )
                used = "colmap-cli"
            if args.dry_run:
                config.log_ok("dry-run 结束：以上即为将要执行的步骤。")
                return 0
        else:
            ok = run_colmap_cli(
                images_dir,
                workspace,
                colmap=colmap_exe or "colmap",
                camera_model=args.camera_model,
                matcher=attempt_matcher,
                use_gpu=bool(args.use_gpu),
                dry_run=args.dry_run,
                mapper=args.mapper,
                mask_dir=mask_dir,
            )
            used = "colmap-cli"
            if args.dry_run:
                config.log_ok("dry-run 结束：以上即为将要执行的命令。")
                return 0

        sparse = workspace / "sparse"
        model_dir = pick_best_model(sparse)
        if model_dir is None:
            if attempt_index + 1 < len(attempts):
                summary = {"registeredImages": 0, "points3D": 0}
                continue
            if not ok:
                config.log_error("COLMAP 命令行执行失败（详细报错见上方输出）。")
                return 5
            config.log_error(f"没有在 {sparse} 下找到任何稀疏模型（sparse/0/images.bin）。")
            print_remediation(0, len(images))
            return 2

        if model_dir.name != "0":
            config.log_warn(f"sparse/{model_dir.name} 是最大的模型，按下游约定复制为 sparse/0。")
            target = sparse / "0"
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(model_dir, target)
            model_dir = target

        try:
            model = read_model(model_dir)
        except ColmapFormatError as exc:
            config.log_error(f"解析 COLMAP 模型失败：{exc}")
            return 5

        # 把图片目录接进工作区，形成标准的 COLMAP 数据集布局：
        #   <workspace>/images/ + <workspace>/sparse/0/
        # 下游（train_gpu.py / train_gsplat.py / 原版 3DGS）都按这个约定找图片，
        # 不接进来就会出现「只有 0 张可用图像」。
        link_dataset_images(images_dir, workspace)

        elapsed = time.time() - started
        summary = summarize(model, len(images), used_matcher, elapsed)
        summary["backend"] = used
        summary["attempts"] = attempt_index + 1
        config.dump_json(summary_path, summary)

        config.log_ok(
            f"重建完成：注册 {summary['registeredImages']}/{summary['totalImages']} 张图片"
            f"（{summary['registeredRatio'] * 100:.1f}%），稀疏点 {summary['points3D']} 个，"
            f"平均重投影误差 {summary['meanReprojectionError']} 像素，耗时 {config.format_duration(elapsed)}"
        )
        config.log_ok(f"已写出 {summary_path}")
        for cam in summary["cameras"]:
            config.log_info(f"相机 {cam['cameraId']}：{cam['model']} {cam['width']}×{cam['height']} 内参 {cam['intrinsics']}")

        registered = summary["registeredImages"]
        if registered >= 8 and registered >= 0.6 * len(images):
            config.log_info(
                "下一步：python pipeline/train_gsplat.py --data "
                f"\"{workspace.parent}\" --out \"{workspace.parent / 'train'}\""
            )
            return 0

        if attempt_index + 1 < len(attempts):
            continue  # 换 exhaustive 再来一次
        print_remediation(registered, len(images))
        return 2

    return 5


if __name__ == "__main__":
    raise SystemExit(main())
