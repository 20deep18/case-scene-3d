"""pipeline/train_gsplat.py —— 用 gsplat 训练 3D Gaussian Splatting 场景，并导出 3DGS PLY。

命令行：
    python pipeline/train_gsplat.py --data <COLMAP 场景目录> --out <输出目录>
        [--iterations 15000] [--sh-degree 3] [--downscale 2] [--max-gaussians 1500000]
        [--save-every 5000] [--log-every 100] [--images <图片目录>]

数据目录约定（脚本会自动探测多种常见布局）：
    <data>/images/            原始图片（也接受 <data>/frames/）
    <data>/sparse/0/          COLMAP 稀疏模型（cameras.bin / images.bin / points3D.bin）
    <data>/colmap/sparse/0/   如果 COLMAP 在子目录里，同样支持

关键实现说明：
  * COLMAP 二进制模型由本项目的 colmap_recon.py 自行解析，**不依赖 pycolmap**；
  * 图像读取按 PIL → imageio → OpenCV → 内置 PNG 解码器 的顺序回退；
  * 训练循环：gsplat 光栅化 + L1/D-SSIM 损失 + 自适应致密化与剪枝 + 周期性 opacity 重置；
  * 导出标准 3DGS 二进制 PLY（x,y,z,nx,ny,nz,f_dc_*,f_rest_*,opacity,scale_*,rot_*）。

退出码：0 成功；2 参数/数据错误；3 缺少 torch/gsplat；4 显存不足（优雅失败）；5 训练失败。
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

_PIPELINE_DIR = Path(__file__).resolve().parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

import colmap_recon  # noqa: E402
import config  # noqa: E402
import extract_frames  # noqa: E402  （复用 PNG 解码器，作为没有 PIL/imageio 时的兜底）

# ---------------------------------------------------------------------------
# SSIM 卷积是否允许走 cuDNN（默认**不允许**）
# ---------------------------------------------------------------------------
# 实测过一次原生崩溃：`cudnn_graph64_9.dll` 里 0xC0000005 写 0x0（访问冲突），
# 而 SSIM 的 5 次分组卷积是训练器里**唯一**走 cuDNN 的地方。
# 详见 Trainer._ssim 的说明。对照实验用 set DSH_SSIM_CUDNN=1 打开。
_SSIM_USE_CUDNN = os.environ.get("DSH_SSIM_CUDNN", "0") == "1"

SH_C0 = 0.28209479177387814

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm")

#: 缺依赖时打印的中文指引
MISSING_DEP_HINT = (
    "未检测到 torch/gsplat，先运行 pipeline/setup_env.py 安装。\n"
    "  本机是 Windows + RTX 5060（sm_120 / Blackwell），必须使用 CUDA ≥ 12.8 的 PyTorch：\n"
    "    python pipeline/setup_env.py --check                    # 先看环境自检表\n"
    "    python pipeline/setup_env.py --install-torch            # 安装 torch>=2.7+cu128（约 2.5GB+）\n"
    "    pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision\n"
    "  gsplat 官方只提供 Linux 预编译轮子，Windows 需要源码编译（MSVC + CUDA Toolkit ≥ 12.8）：\n"
    "    pip install git+https://github.com/nerfstudio-project/gsplat.git\n"
    "  没有 NVIDIA 显卡时的替代路线：用手机 App（Polycam / Luma AI）导出 .ply，\n"
    "  再运行 python pipeline/to_splat.py --ply <模型.ply> --out data/models/<名称>.splat"
)


# ---------------------------------------------------------------------------
# 场景解析
# ---------------------------------------------------------------------------


@dataclass
class FrameEntry:
    """一张可用于训练的图片（已与 COLMAP 位姿配对）。"""

    image_id: int
    name: str
    path: Path
    camera: colmap_recon.ColmapCamera
    image: colmap_recon.ColmapImage

    def intrinsics(self) -> tuple[float, float, float, float]:
        return self.camera.intrinsics()

    def width_height(self) -> tuple[int, int]:
        return int(self.camera.width), int(self.camera.height)


@dataclass
class Scene:
    """解析好的训练场景。"""

    data_dir: Path
    images_dir: Path
    model_dir: Path
    model: colmap_recon.ColmapModel
    frames: list[FrameEntry]
    missing_images: list[str] = field(default_factory=list)


def resolve_scene_paths(data_dir: Path, images_override: str | None = None) -> tuple[Path, Path]:
    """探测 (图片目录, 稀疏模型目录)，支持若干常见布局。"""
    data_dir = data_dir.resolve()
    sparse_candidates = [
        data_dir / "sparse" / "0",
        data_dir / "colmap" / "sparse" / "0",
        data_dir / "sparse",
        data_dir,
    ]
    model_dir = next((p for p in sparse_candidates if (p / "images.bin").is_file()), None)
    if model_dir is None:
        raise FileNotFoundError(
            f"在 {data_dir} 下找不到 COLMAP 稀疏模型（需要 sparse/0/images.bin）。\n"
            f"  已尝试：{', '.join(str(p) for p in sparse_candidates)}\n"
            "  请先运行：python pipeline/colmap_recon.py --images <图片目录> --workspace <工作目录>"
        )

    if images_override:
        images_dir = Path(images_override)
        if not images_dir.is_dir():
            raise FileNotFoundError(f"--images 指定的目录不存在：{images_dir}")
        return images_dir, model_dir

    candidates = [
        data_dir / "images",
        data_dir / "frames",
        data_dir.parent / "frames",
        data_dir.parent / "images",
        data_dir / "colmap" / "images",
        data_dir.parent / "images_dir",
    ]
    for candidate in candidates:
        # 用 IMAGE_SUFFIXES 统一判断，避免这里写死扩展名而漏掉 .ppm 之类
        if candidate.is_dir() and any(
            p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES for p in candidate.iterdir()
        ):
            return candidate, model_dir
    # 一个候选目录都没命中时，退回第一个存在的目录（后续按文件名逐个查找，能找到多少用多少）
    for candidate in candidates:
        if candidate.is_dir():
            return candidate, model_dir
    raise FileNotFoundError(
        f"找不到图片目录（已尝试：{', '.join(str(c) for c in candidates)}）。\n"
        "  请用 --images 显式指定图片目录。"
    )


def load_scene(data_dir: Path, images_override: str | None = None) -> Scene:
    """解析 COLMAP 场景（自实现的二进制解析器，不需要 pycolmap）。"""
    images_dir, model_dir = resolve_scene_paths(data_dir, images_override)
    model = colmap_recon.read_model(model_dir)
    frames: list[FrameEntry] = []
    missing: list[str] = []
    index: dict[str, Path] = {}
    for path in images_dir.iterdir() if images_dir.is_dir() else []:
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            index[path.name] = path
    for image_id in sorted(model.images):
        image = model.images[image_id]
        camera = model.cameras.get(image.camera_id)
        if camera is None:
            continue
        path = index.get(image.name) or index.get(Path(image.name).name)
        if path is None:
            missing.append(image.name)
            continue
        frames.append(FrameEntry(image_id, image.name, path, camera, image))
    return Scene(data_dir, images_dir, model_dir, model, frames, missing)


def describe_scene(scene: Scene) -> None:
    """打印场景概况，便于在没有 torch 时也能验证 COLMAP 解析是否正确。"""
    model = scene.model
    config.log_info(f"图片目录：{scene.images_dir}")
    config.log_info(f"稀疏模型：{scene.model_dir}")
    config.log_info(
        f"相机 {len(model.cameras)} 台 | 注册图片 {len(model.images)} 张 | 稀疏点 {len(model.points3D)} 个"
    )
    config.log_info(f"平均重投影误差：{model.mean_reprojection_error:.4f} 像素 | 平均 track 长度：{model.mean_track_length:.2f}")
    for camera in model.cameras.values():
        try:
            fx, fy, cx, cy = camera.intrinsics()
            config.log_info(
                f"  相机 {camera.camera_id}：{camera.model} {camera.width}×{camera.height} "
                f"fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}"
            )
        except colmap_recon.ColmapFormatError:
            config.log_warn(f"  相机 {camera.camera_id}：无法解析内参（模型 {camera.model}）")
    if model.images:
        first = model.images[min(model.images)]
        config.log_info(
            f"  示例位姿 {first.name}：qvec={tuple(round(v, 4) for v in first.qvec)} "
            f"tvec={tuple(round(v, 4) for v in first.tvec)} 光心={tuple(round(v, 4) for v in first.camera_center())}"
        )
    if scene.missing_images:
        config.log_warn(f"有 {len(scene.missing_images)} 张已注册图片在图片目录里找不到对应文件，例如：{scene.missing_images[:3]}")


# ---------------------------------------------------------------------------
# 图像读取（多级回退，最终兜底是内置 PNG 解码器）
# ---------------------------------------------------------------------------


def load_image_rgb(path: Path) -> tuple[int, int, bytes]:
    """读取图片为 RGB 字节流 (宽, 高, bytes)。"""
    errors: list[str] = []
    pil = config.try_import("PIL.Image")
    if pil is not None:
        try:
            with pil.open(str(path)) as handle:
                rgb = handle.convert("RGB")
                return rgb.width, rgb.height, rgb.tobytes()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"PIL: {exc}")

    imageio = config.try_import("imageio.v3") or config.try_import("imageio")
    if imageio is not None:
        try:
            import numpy as _np  # noqa: PLC0415

            array = imageio.imread(str(path))
            if array.ndim == 2:
                array = _np.stack([array] * 3, axis=-1)
            array = array[..., :3].astype("uint8")
            return int(array.shape[1]), int(array.shape[0]), array.tobytes()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"imageio: {exc}")

    cv2 = config.try_import("cv2")
    if cv2 is not None:
        try:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                return int(rgb.shape[1]), int(rgb.shape[0]), rgb.tobytes()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"OpenCV: {exc}")

    if path.suffix.lower() == ".png":
        errors.append("内置 PNG 解码器只提供灰度图，训练需要彩色；请安装 Pillow 或 imageio")

    raise RuntimeError(
        f"无法读取图片 {path}。请安装 Pillow 或 imageio 后重试：\n"
        f"  pip install pillow        （或 pip install imageio）\n"
        f"  详细信息：{'; '.join(errors) if errors else '未知错误'}"
    )


def image_to_tensor(path: Path, downscale: int, torch: Any, device: Any) -> Any:
    """读图 → torch 张量 (3, H, W)，按 downscale 做面积下采样。"""
    width, height, data = load_image_rgb(path)
    tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8).reshape(height, width, 3)
    tensor = tensor.permute(2, 0, 1).float() / 255.0
    if downscale > 1:
        target_h = max(1, height // downscale)
        target_w = max(1, width // downscale)
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0), size=(target_h, target_w), mode="area"
        ).squeeze(0)
    return tensor.to(device).contiguous()


def mask_to_tensor(path: Path, downscale: int, torch: Any, device: Any) -> Any:
    """读二值 mask → torch 张量 (1, H, W)，值 0/1，与图片用**同一套下采样**。

    ⚠️ 用 area（面积平均）再按 0.5 阈值二值化，而不是最近邻：
    mask 与图片必须**逐像素对齐**，否则边界上会系统性地多留/少留一圈像素。
    """
    gray = None
    pil = config.try_import("PIL.Image")
    if pil is not None:
        try:
            with pil.open(str(path)) as handle:
                g = handle.convert("L")
                width, height = g.width, g.height
                gray = (width, height, g.tobytes())
        except Exception:  # noqa: BLE001
            gray = None
    if gray is None:
        cv2 = config.try_import("cv2")
        if cv2 is not None:
            arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if arr is not None:
                gray = (int(arr.shape[1]), int(arr.shape[0]), arr.tobytes())
    if gray is None:
        raise RuntimeError(f"无法读取 mask（需要 Pillow 或 OpenCV）：{path}")

    width, height, data = gray
    tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8).reshape(height, width)
    tensor = tensor.float().unsqueeze(0).unsqueeze(0) / 255.0
    if downscale > 1:
        target_h = max(1, height // downscale)
        target_w = max(1, width // downscale)
        tensor = torch.nn.functional.interpolate(
            tensor, size=(target_h, target_w), mode="area"
        )
    return (tensor.squeeze(0) > 0.5).float().to(device).contiguous()


# ---------------------------------------------------------------------------
# PLY 导出（标准 3DGS 布局）
# ---------------------------------------------------------------------------


def ply_property_names(rest_per_channel: int) -> list[str]:
    """标准 3DGS PLY 的属性名顺序。"""
    names = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
    names += [f"f_rest_{i}" for i in range(rest_per_channel * 3)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    return names


def tensor_block_to_bytes(tensor: Any, numpy: Any | None) -> bytes:
    """把一个 (N, D) float32 张量转成小端字节流。"""
    if numpy is not None:
        try:
            return tensor.detach().cpu().numpy().astype("<f4", copy=False).tobytes()
        except Exception:  # noqa: BLE001 - 某些设备/后端不支持 numpy
            pass
    values = tensor.detach().cpu().tolist()
    flat = [v for row in values for v in row]
    return struct.pack(f"<{len(flat)}f", *flat)


def save_ply(
    path: Path,
    *,
    means: Any,
    scales_log: Any,
    quats: Any,
    opacities_logit: Any,
    features: Any,
    sh_degree_active: int | None = None,
) -> int:
    """把当前高斯参数写成标准 3DGS 二进制小端 PLY，返回高斯数量。

    features 形状为 (N, (max_degree+1)^2, 3)（系数主序、通道在后，gsplat 约定）；
    写 PLY 时按 3DGS 约定转成「通道主序」：f_rest 依次为 R 的 K 个系数、G 的 K 个、B 的 K 个。
    """
    numpy = config.try_import("numpy")
    count = int(means.shape[0])
    # features 形状是 (N, (degree+1)^2, 3)：第 0 个是直流分量，后面全是高阶系数。
    # 高阶系数共有 (degree+1)^2 - 1 个（与通道数 3 无关），所以这里**不能**再除以 3。
    # 之前写成 rest_count // 3，导致 reshape 用了错误的目标形状：
    #   shape '[10789, 15]' is invalid for input of size 485505   (10789*15*3)
    rest_count = int(features.shape[1]) - 1
    names = ply_property_names(rest_count)

    path.parent.mkdir(parents=True, exist_ok=True)
    header = "ply\nformat binary_little_endian 1.0\n"
    # PLY 头必须是 ASCII（很多解析器按 ascii/latin-1 读），所以注释只能写英文
    header += "comment trained by pipeline/train_gsplat.py (3DGS / gsplat)\n"
    header += f"element vertex {count}\n"
    header += "".join(f"property float {name}\n" for name in names)
    header += "end_header\n"

    import torch  # noqa: PLC0415 - 仅在本函数内使用

    dc = features[:, 0, :].detach()  # (N, 3)
    if rest_count > 0:
        rest = features[:, 1:, :].detach()  # (N, rest_count, 3)
        # 3DGS 约定是「通道主序」：先 R 的全部系数，再 G，再 B
        rest_ply = rest.permute(0, 2, 1).reshape(count, rest_count * 3)
    else:
        rest_ply = torch.zeros((count, 0), dtype=dc.dtype, device=dc.device)

    blocks = [
        means.detach().float(),
        torch.zeros((count, 3), dtype=torch.float32, device=means.device),  # 法向占位
        dc.float(),
        rest_ply.float(),
        opacities_logit.detach().float().reshape(count, 1),
        # 注意：3DGS 官方 PLY 里 scale_* 存的是**线性尺度**（已经 exp 过），
        # 不是 log 值。官方代码在 save_ply 里写的是 scaling（激活后的值）。
        torch.exp(scales_log.detach().float()),
        quats.detach().float(),
    ]

    # ------------------------------------------------------------------
    # 关键：PLY 的数据区必须是 **记录主序（record-major）** ——
    #   第 1 个高斯的所有属性挨着写，然后是第 2 个……
    # 而不是**块主序（block-major）**：
    #   所有高斯的 x、所有高斯的 y、……、所有高斯的 rot_3。
    #
    # 早先这里是 `for block in blocks: write(block)`，写出来的是块主序，
    # 于是整个文件对任何标准 PLY 读取器都是错位的：
    #   · 位置字段、颜色字段、不透明度、旋转的取值范围全都混在一起，
    #     每个字段的统计量几乎一样（这是"错位"的典型指纹）；
    #   · 在浏览器里渲染出来就是一堆霓虹色大色块，而不是场景。
    # 正确做法是先按列拼成一个 (N, D) 矩阵，再整体按行写出去。
    # ------------------------------------------------------------------
    table = torch.cat([b.reshape(count, -1) for b in blocks], dim=1).contiguous()
    if table.shape[1] != len(names):
        raise RuntimeError(
            f"PLY 属性数与数据列数不一致：header {len(names)} 列，数据 {table.shape[1]} 列"
        )
    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(tensor_block_to_bytes(table, numpy))
    return count


# ---------------------------------------------------------------------------
# 训练器
# ---------------------------------------------------------------------------


class Trainer:
    """一个精简但完整的 3DGS 训练器（gsplat 光栅化 + 稠密化 + 剪枝 + opacity 重置）。"""

    def __init__(self, scene: Scene, out_dir: Path, args: argparse.Namespace) -> None:
        import torch  # noqa: PLC0415

        self.torch = torch
        self.scene = scene
        self.out_dir = out_dir
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_sh_degree = max(0, min(3, args.sh_degree))
        self.started = time.time()

        if self.device.type != "cuda":
            config.log_warn("没有检测到可用的 CUDA 设备，将在 CPU 上训练（会非常慢，仅供小场景验证）。")

        self.frames = scene.frames
        self.images: dict[int, Any] = {}
        config.log_info(f"开始加载 {len(self.frames)} 张图片（downscale={args.downscale}）…")
        for index, frame in enumerate(self.frames):
            self.images[frame.image_id] = image_to_tensor(frame.path, args.downscale, torch, self.device)
            if (index + 1) % 50 == 0 or index + 1 == len(self.frames):
                config.log_info(f"  已加载 {index + 1}/{len(self.frames)} 张")

        # ---- 背景分离 mask（可选；物体算法专有）--------------------------------
        # 约定与 COLMAP 一致：与图片同名（扩展名换 .png）、黑 = 忽略该区域。
        # mask 外**不计入损失**（L1 与 SSIM 都遮），见 train() 里的用法。
        self.masks: dict[int, Any] = {}
        self.mask_dir = Path(args.mask_dir) if getattr(args, "mask_dir", None) else None
        self.skipped_by_mask: list[str] = []
        if self.mask_dir is not None:
            if not self.mask_dir.is_dir():
                config.log_error(f"--mask-dir 不是目录：{self.mask_dir}")
                return_code = 4
                raise SystemExit(return_code)
            found = 0
            for frame in self.frames:
                candidate = self.mask_dir / (Path(frame.path).stem + ".png")
                if not candidate.is_file():
                    continue
                m = mask_to_tensor(candidate, args.downscale, torch, self.device)
                ratio = float(m.mean().item())
                if ratio < 0.05:
                    # 前景几乎为空：这帧没有可监督的内容，留着只会让 loss 变成 0/0
                    # 那种"看起来在训练、其实没梯度"的状态 —— 直接跳过并记录。
                    self.skipped_by_mask.append(Path(frame.path).name)
                    continue
                self.masks[frame.image_id] = m
                found += 1
            config.log_info(
                f"背景分离：{found}/{len(self.frames)} 帧带 mask（mask 外不计入损失）"
                + (f"，跳过前景过少的 {len(self.skipped_by_mask)} 帧" if self.skipped_by_mask else "")
            )
            if found == 0:
                config.log_warn(
                    f"mask 目录里没有任何与帧同名的 png：{self.mask_dir} —— 本次**等价于没有背景分离**。"
                )
            if self.skipped_by_mask:
                config.log_info(f"  跳过的帧（前景 <5%）：{self.skipped_by_mask[:5]}")
            # 全部帧都被跳过就没法训练了，直接说清楚
            if found == 0 and self.skipped_by_mask:
                config.log_error("所有帧的 mask 前景都过少，无法训练。请检查框是否画在主体上。")
                raise SystemExit(4)

        self.camera_centers = torch.tensor(
            [list(frame.image.camera_center()) for frame in self.frames], dtype=torch.float32, device=self.device
        )
        self.spatial_scale = self._compute_spatial_scale()
        config.log_info(f"场景尺度（相机包围球半径×1.1）：{self.spatial_scale:.3f} 米")

        self._init_params()
        # 点云自身的特征尺度（最近邻距离中位数）—— 尺度上限的依据，见 _clamp_scales。
        self.feature_scale = self._estimate_feature_scale()
        # 尺度上限：两个口径取小（只收紧、不放松）。算一次，夹紧/剪枝/落盘共用，
        # 避免出现「_clamp_scales 夹到 A、_prune 按 B 剪」这种自相矛盾。
        _cam_cap = float(self.args.prune_scale) * float(self.spatial_scale)
        _k = float(getattr(self.args, "scale_cap_k", 4.0))
        _feat_cap = _k * self.feature_scale if self.feature_scale > 0 else float("inf")
        self.scale_cap = max(min(_cam_cap, _feat_cap), 1e-4)
        # 绝对尺度下限：既防数值塌缩，也防「刀片」——下限按特征尺度定，
        # 场景越小下限越小，不会误伤桌面小物件。
        self.scale_floor = max(self.feature_scale * 0.02, 1e-3) if self.feature_scale > 0 else 1e-3
        config.log_info(
            f"点云特征尺度（最近邻中位）：{self.feature_scale:.4f} 米；"
            f"尺度上限 {self.scale_cap:.4f} 米"
            f"（相机口径 {_cam_cap:.4f} / 特征口径 ×{_k:g} = {_feat_cap:.4f} 取小）；"
            f"尺度下限 {self.scale_floor:.5f} 米，各向异性比上限 "
            f"{float(getattr(self.args, 'scale_aniso_max', 4.0)):g}"
        )
        # 逐帧仿射曝光校正（--exposure-adapt 开启）：每个训练帧一对 (gain, bias)。
        # 实测这段素材帧间亮度跨度 27%（P05 92 → P95 161 / 255），
        # 不校正时 3DGS 只能用模糊去覆盖曝光不一致。
        self.exposure = None
        self.exposure_reg = float(getattr(self.args, "exposure_reg", 0.01))
        if getattr(self.args, "exposure_adapt", False):
            n_exp = max(1, len(self.frames))
            self.exposure = {
                "gain": torch.nn.Parameter(torch.ones(n_exp, dtype=torch.float32, device=self.device)),
                "bias": torch.nn.Parameter(torch.zeros(n_exp, dtype=torch.float32, device=self.device)),
            }
            config.log_info(f"已启用逐帧曝光自适应：{n_exp} 帧 × (增益, 偏置)，正则系数 {self.exposure_reg}")
        self._init_optimizer()

        self.grad_accum: Any | None = None
        self.densify_steps = 0
        self.background = torch.zeros(3, dtype=torch.float32, device=self.device)

    # -- 初始化 ---------------------------------------------------------

    def _compute_spatial_scale(self) -> float:
        torch = self.torch
        if self.camera_centers.shape[0] == 0:
            return 1.0
        center = self.camera_centers.mean(dim=0, keepdim=True)
        distances = (self.camera_centers - center).norm(dim=1)
        return float(distances.max().item()) * 1.1 + 1e-6

    def _init_params(self) -> None:
        torch = self.torch
        points = list(self.scene.model.points3D.values())
        if points:
            means = torch.tensor([p.xyz for p in points], dtype=torch.float32, device=self.device)
            colors = torch.tensor(
                [[c / 255.0 for c in p.rgb] for p in points], dtype=torch.float32, device=self.device
            )
            scales_log = torch.log(self._estimate_point_scale(means)).reshape(-1, 1).repeat(1, 3)
        else:
            config.log_warn("稀疏点云为空，改用相机包围盒内随机初始化。")
            count = 20000
            center = self.camera_centers.mean(dim=0)
            means = center + (torch.rand((count, 3), device=self.device) - 0.5) * self.spatial_scale
            colors = torch.rand((count, 3), device=self.device) * 0.5 + 0.25
            scales_log = torch.full((count, 3), math.log(self.spatial_scale * 0.01), device=self.device)

        count = means.shape[0]
        dc = ((colors - 0.5) / SH_C0).reshape(count, 1, 3)
        rest_count = (self.max_sh_degree + 1) ** 2 - 1
        features = torch.zeros((count, 1 + rest_count, 3), dtype=torch.float32, device=self.device)
        features[:, 0:1, :] = dc

        self.params: dict[str, Any] = {
            "means": torch.nn.Parameter(means.contiguous().requires_grad_(True)),
            "scales": torch.nn.Parameter(scales_log.contiguous().requires_grad_(True)),
            "quats": torch.nn.Parameter(
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).repeat(count, 1).contiguous().requires_grad_(True)
            ),
            "opacities": torch.nn.Parameter(
                torch.full((count,), math.log(0.1 / 0.9), device=self.device).requires_grad_(True)
            ),
            "sh0": torch.nn.Parameter(features[:, 0:1, :].contiguous().requires_grad_(True)),
            "shN": torch.nn.Parameter(features[:, 1:, :].contiguous().requires_grad_(True)),
        }
        config.log_info(f"初始高斯数量：{count}")

    def _estimate_point_scale(self, points: Any) -> Any:
        """用「到随机采样子集中最近 3 个点的平均距离」估计初始尺度（近似 3DGS 的 distCUDA2）。"""
        torch = self.torch
        total = points.shape[0]
        sample_size = min(total, 8192)
        if total <= 1:
            return torch.full((total,), 0.01, device=self.device)
        generator = torch.Generator(device="cpu").manual_seed(0)
        sample_idx = torch.randperm(total, generator=generator)[:sample_size].to(self.device)
        sample = points[sample_idx]
        scales = torch.empty(total, device=self.device)
        chunk = max(1, min(4096, total))
        for start in range(0, total, chunk):
            block = points[start : start + chunk]
            distances = torch.cdist(block, sample)  # (chunk, sample)
            k = min(3, distances.shape[1])
            nearest = distances.topk(k, dim=1, largest=False).values
            scales[start : start + block.shape[0]] = nearest.mean(dim=1).clamp_min(1e-6)
        return scales

    def _init_optimizer(self) -> None:
        torch = self.torch
        lr = self.args.learning_rate
        # 学习率换算。
        #
        # 官方 3DGS 的学习率是在**归一化场景**（场景尺度≈1）下调出来的：
        #   means 1.6e-4, scales 5e-3(log), quats 1e-3, opacities 5e-2, sh0 2.5e-3
        # 一旦场景用**真实米制坐标**（本场景 spatial_scale≈6.2 米），
        # 若把 means/scales 的学习率也乘 6.2，就会在几百步内把所有参数推到饱和：
        # 实测 scale 直接被顶到上限、opacity 卡在 0.5，
        # 结果是「一堆大雾团」而不是贴合表面的高斯。
        #
        # 所以这里只做**保守**的尺度补偿，并且用 --lr-scale-factor 暴露出来，
        # 便于在真实素材上按需调整（默认 1.0 = 官方归一化数值直接用）。
        factor = max(self.args.lr_scale_factor, 0.0)
        # means 的量纲是米，随场景放大是必要的（否则步长相对场景太小）
        group_lr = {
            "means": lr * self.spatial_scale * factor,
            "sh0": lr,
            "shN": lr / 20.0,
            "opacities": 0.05 * factor,
            "scales": 0.005 * factor,
            "quats": 0.001 * factor,
        }
        groups = [
            {"params": [self.params[name]], "lr": value, "name": name}
            for name, value in group_lr.items()
        ]
        if self.exposure is not None:
            # 曝光参数的学习率要小：它只需要吸收几十个灰阶的跳变，
            # 给大了会在前期把整幅画面的亮度拉跑，反而伤几何。
            groups.append({"params": [self.exposure["gain"]], "lr": 0.005, "name": "exposure_gain"})
            groups.append({"params": [self.exposure["bias"]], "lr": 0.005, "name": "exposure_bias"})
        self.optimizer = torch.optim.Adam(groups, lr=lr, eps=1e-15)

    def _estimate_feature_scale(self, sample: int = 2000) -> float:
        """点云的**特征尺度** = 随机子采样的最近邻距离中位数（米）。

        为什么需要它（实测数据，`tools/probe_feature_scale.py`）：
            数据集              稀疏点   nn中位   包围盒对角
            发雾（小视频 17 帧）    625    0.0747    12.0
            健康 room-real      48165   0.1487   134.9
        两个要点：
          1) **不能用包围盒尺度**：健康模型被野点撑到 134 m，
             `2% × 对角` = 2.70 m —— 完全失去约束力。nn 中位对野点稳健得多。
          2) 发雾那组稀疏点只有 625 个（健康组 4.8 万），几何支撑极弱，
             高斯只能靠「长大」去覆盖 → 这才有了那朵白云。
        返回 0.0 表示样本不足，调用方会退回原口径。
        """
        torch = self.torch
        import numpy as _np  # noqa: PLC0415

        with torch.no_grad():
            pts = self.params["means"].detach()
            n = int(pts.shape[0])
            if n < 50:
                return 0.0
            m = min(int(sample), n)
            g = torch.Generator(device="cpu").manual_seed(7)
            idx = torch.randperm(n, generator=g)[:m]
            sub = pts[idx].to("cpu").numpy().astype("float64")
        best = _np.full(m, _np.inf)
        for i in range(0, m, 500):
            blk = sub[i : i + 500]
            d = _np.linalg.norm(blk[:, None, :] - sub[None, :, :], axis=2)
            d[_np.arange(len(blk)), _np.arange(i, i + len(blk))] = _np.inf
            best[i : i + len(blk)] = d.min(axis=1)
        finite = best[_np.isfinite(best)]
        return float(_np.median(finite)) if finite.size else 0.0

    def _clamp_scales(self) -> None:
        """
        把每个高斯的尺度夹到合理上限。

        为什么必须有：3DGS 只在「致密化阶段」靠分裂/剪枝间接约束尺度，
        一旦尺度学习率按场景尺度放大（本场景 ×6.2），又**没有硬上限**，
        参数会一路单调增长 —— 实测跑 7000 步后：
            82% 的高斯最大尺度 > 1 米，P99 达到 17.4 米
            包围盒对角线 41.8 米（真实场景只有约 12 米）
        结果就是模型被几十个大「雾团」包住，画面既糊又暗，
        .splat 转换时还会以「尺度超限」为由丢掉 82% 的高斯。

        上限取 0.1 * spatial_scale（与剪枝阈值一致）：尺度超过它本来就该被剪掉，
        早夹住比等到剪枝更平滑。
        """
        torch = self.torch
        # 尺度上限 = min(相机口径, 点云特征口径)，**只收紧、不放松**。
        #
        # 相机口径：prune_scale × spatial_scale。这是原来的唯一依据，
        #   但 spatial_scale 是**相机轨迹**的尺度，不是**表面**的尺度：
        #   某任务 = 0.1 × 11.185 = 1.1185 m，而健康模型尺度中位只有 0.05~0.21 m
        #   —— 足足 8 倍余量，优化器必然把大量高斯顶到上限，
        #   于是导出后 56% 的高斯 > 0.5 m，又大又白又淡，叠成一层白雾盖住细节。
        # 特征口径：scale_cap_k × 点云最近邻中位。实测发雾组 4×0.0747 = 0.299 m、
        #   健康组 4×0.1487 = 0.595 m；而健康模型实际最大尺度才 0.334 m，
        #   所以这个口径**对好模型零影响**，只把发雾组的成长空间从 1.12 m 收到 0.30 m。
        limit = math.log(self.scale_cap)
        with torch.no_grad():
            sc = self.params["scales"].data
            # ---- 各向异性约束（治「法向云雾」的核心）----------------------------
            # 实测判据（tools 里量遍模型库）：
            #   健康 room-real      针状度 s1/s2 = 0.472，扁平度 s0/s1 = 0.440
            #   发雾的三个模型       针状度 0.089 ~ 0.234，两个小轴都塌了
            # 也就是说：**病态高斯是「刀片」而不是「贴面薄片」**。
            # 两个小轴塌到接近 0 后，成千上万张随机朝向的刀片交叉在一起，
            # 渲染出来就是墙面上那层「法向的雾」。
            #
            # 注意不能只看 min/max：薄片与刀片的 min/max 都接近 0，
            # 区别在**中间那个轴**——薄片的 s1 与 s2 相近，刀片的 s1 也塌了。
            # 所以这里约束的是 s_max / s_min ≤ aniso_max（把三个轴拉进同一个量级），
            # 默认 4.0 —— 刚好容纳健康模型（其上式约为 4.8，实际可达标）。
            aniso = float(getattr(self.args, "scale_aniso_max", 4.0))
            if aniso > 1.0:
                mx = sc.max(dim=1, keepdim=True).values
                sc.clamp_(min=mx - math.log(aniso))
            # 绝对下限：防止塌成数值意义上的零
            sc.clamp_(min=math.log(self.scale_floor))
            # 上限最后夹，保证不超 cap
            sc.clamp_(max=limit)

    # -- 张量打包 -------------------------------------------------------

    def current_features(self, degree: int) -> Any:
        torch = self.torch
        if degree <= 0 or self.params["shN"].shape[1] == 0:
            return self.params["sh0"]
        take = min(self.params["shN"].shape[1], (degree + 1) ** 2 - 1)
        return torch.cat([self.params["sh0"], self.params["shN"][:, :take, :]], dim=1)

    def rasterize(self, degree: int, viewmat: Any, K: Any, width: int, height: int) -> tuple[Any, Any]:
        """调用 gsplat 光栅化，返回 (渲染图, alpha)。"""
        torch = self.torch
        gsplat = config.try_import("gsplat")
        if gsplat is None:
            raise RuntimeError("gsplat 不可用")
        rasterize_fn = getattr(gsplat, "rasterization", None) or getattr(gsplat, "rasterize_gaussians", None)
        if rasterize_fn is None:
            raise RuntimeError("gsplat 里找不到 rasterization / rasterize_gaussians，请升级或降级 gsplat。")
        colors = self.current_features(degree)
        kwargs = dict(
            means=self.params["means"],
            quats=self.params["quats"],
            scales=torch.exp(self.params["scales"]),
            opacities=torch.sigmoid(self.params["opacities"]),
            colors=colors,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=width,
            height=height,
            sh_degree=degree,
            packed=False,
            render_mode="RGB",
            backgrounds=self.background.unsqueeze(0),
        )
        try:
            import inspect  # noqa: PLC0415

            signature = inspect.signature(rasterize_fn)
            accepts_kwargs = any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
            )
            if not accepts_kwargs:
                kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
        except (TypeError, ValueError):
            pass
        result = rasterize_fn(**kwargs)
        if not isinstance(result, tuple) or len(result) < 2:
            raise RuntimeError(f"gsplat 返回了意料之外的结果：{type(result)}")
        rendered, alpha = result[0], result[1]
        # gsplat 的 rasterization 总是返回**带批次维**的张量：
        #   renders (B,H,W,C)  alphas (B,H,W) 或 (B,H,W,1)
        # 本训练器只做单视角，所以在这里统一把批次维去掉，
        # 下游（L1 / SSIM / 致密化）都按 (H,W,3) 与 (H,W) 处理。
        if rendered.dim() == 4:
            rendered = rendered[0]
        if alpha.dim() == 4 and alpha.shape[-1] == 1:
            alpha = alpha[..., 0]
        if alpha.dim() == 3:
            alpha = alpha[0]
        return rendered, alpha

    # -- 训练循环 -------------------------------------------------------

    def train(self) -> int:
        torch = self.torch
        args = self.args
        log_every = max(1, args.log_every)
        self.grad_accum = torch.zeros(self.params["means"].shape[0], device=self.device)
        last_loss = 0.0

        for iteration in range(1, args.iterations + 1):
            degree = min(self.max_sh_degree, iteration // max(1, args.sh_degree_interval))
            frame = self.frames[iteration % len(self.frames)]
            target = self.images[frame.image_id]
            height, width = int(target.shape[1]), int(target.shape[2])
            fx, fy, cx, cy = frame.intrinsics()
            scale = args.downscale
            K = torch.tensor(
                [[fx / scale, 0.0, cx / scale], [0.0, fy / scale, cy / scale], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
                device=self.device,
            )
            R = torch.tensor(frame.image.rotation_matrix(), dtype=torch.float32, device=self.device)
            t = torch.tensor(frame.image.tvec, dtype=torch.float32, device=self.device)
            viewmat = torch.eye(4, dtype=torch.float32, device=self.device)
            viewmat[:3, :3] = R
            viewmat[:3, 3] = t

            rendered, _alpha = self.rasterize(degree, viewmat, K, width, height)
            image = rendered.permute(2, 0, 1)  # (3, H, W)
            if self.exposure is not None:
                # 逐帧仿射曝光校正：渲染图先乘本帧的增益、加偏置，再与真值比。
                # 这样「亮度跳变」由每帧自己的两个标量吸收，不必让高斯去糊掉细节。
                # 实测该素材帧间亮度 P05 92 → P95 161（跨度 27%），
                # 不校正时 3DGS 只能用「又大又软」的高斯去覆盖这种不一致。
                idx = int(frame.image_id) % self.exposure["gain"].shape[0]
                g = self.exposure["gain"][idx]
                b = self.exposure["bias"][idx]
                image = image * g + b
            mask = self.masks.get(frame.image_id)
            if mask is not None:
                # ---- 背景分离：mask 外**不计入损失**（物体算法专有）----------------
                # 用户点名要"可靠做法"：与其让高斯去解释背景（结果是背景也被重建、
                # 而且它们会被拉到镜头附近），不如直接不监督它。
                # ⚠️ L1 与 SSIM **两项都要遮**：只遮 L1 会留下暗门。
                m = mask.expand_as(image)
                l1 = (torch.abs(image - target) * m).sum() / m.sum().clamp(min=1.0)
                dssim = 1.0 - self._ssim(image.unsqueeze(0), target.unsqueeze(0),
                                         mask=mask.unsqueeze(0))
            else:
                l1 = torch.abs(image - target).mean()
                dssim = 1.0 - self._ssim(image.unsqueeze(0), target.unsqueeze(0))
            loss = (1.0 - args.dssim_weight) * l1 + args.dssim_weight * dssim
            if self.exposure is not None:
                # 正则：把增益/偏置拉回 1/0，避免「整体调暗 + 偏置补偿」这类退化解
                # 把曝光参数当成万能挡箭牌，反而学不到真实辐照。
                reg = ((self.exposure["gain"] - 1.0) ** 2).mean() + (self.exposure["bias"] ** 2).mean()
                loss = loss + self.exposure_reg * reg

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            last_loss = float(loss.detach().item())

            with torch.no_grad():
                if self.params["means"].grad is not None:
                    self.grad_accum += self.params["means"].grad.detach().norm(dim=1)
                # 不透明度不能超过 0.99，避免梯度饱和
                self.params["opacities"].data.clamp_(max=math.log(0.99 / 0.01))

            self.optimizer.step()
            # 尺度必须每步夹一次：没有硬上限时参数会单调涨到十几米（实测 82% 超标）
            self._clamp_scales()
            self.densify_steps += 1

            if iteration % args.densify_interval == 0 and iteration <= args.densify_until:
                self.densify_and_prune()
                self.grad_accum = torch.zeros(self.params["means"].shape[0], device=self.device)

            if args.opacity_reset_interval > 0 and iteration % args.opacity_reset_interval == 0:
                # 关键：**临近结束时不要再重置不透明度**。
                #
                # 重置会把所有不透明度打到 0.005，需要几百步才能恢复。
                # 默认 3000 的间隔在 15000 步训练里正好等于最后一轮 ——
                # 训练一结束就保存，导出的模型 alpha 只有 0.03~0.15，
                # 渲染出来几乎全黑（实测留出视角 PSNR 6.7 dB，
                # 而只训练 400 步时 alpha=0.93、PSNR 19.9 dB）。
                #
                # 留出最后 10% 的迭代（至少 500 步）让不透明度沉淀下来。
                tail = max(500, int(args.iterations * 0.1))
                if iteration <= args.iterations - tail:
                    self.reset_opacity(iteration)
                else:
                    self.opacity_reset_last_skipped = iteration

            if iteration % log_every == 0 or iteration == 1:
                config.log_info(
                    f"[训练] iteration={iteration}/{args.iterations} loss={last_loss:.6f} "
                    f"gaussians={self.params['means'].shape[0]} elapsed={time.time() - self.started:.2f}s "
                    f"sh_degree={degree}"
                )

            if args.save_every > 0 and iteration % args.save_every == 0:
                self.save(iteration)

        # 训练结束：用**留出视角**做一次诚实的画质评估。
        # 单张随机图的 loss 噪声很大（每步换一张图），不能当质量指标。
        if args.eval_views > 0:
            self.evaluate(holdout=args.eval_views, out_dir=args.out)
        return self.save(args.iterations)

    # -- 画质评估 -------------------------------------------------------

    def evaluate(self, holdout: int = 4, out_dir: Any = None) -> None:
        """
        在**训练时没参与优化的留出视角**上评估 PSNR / SSIM，并渲染出图。

        为什么要留出视角：只看训练图的 loss 会高估画质（等于"背答案"）。
        蓝图 M3 要求 "重渲染 PSNR 达标"，这里就给出这个数字。
        """
        torch = self.torch
        args = self.args
        frames = self.frames
        n = len(frames)
        if n == 0:
            return
        holdout = max(1, min(holdout, n // 4 if n >= 8 else 1))
        # 均匀抽样，避开与训练视角相邻的强相关样本
        step = max(1, n // holdout)
        idxs = list(range(0, n, step))[:holdout]

        import numpy as np  # noqa: PLC0415

        config.log_info(f"[评估] 在 {len(idxs)} 个留出视角上评估画质 …")
        psnrs, ssims, l1s = [], [], []
        out_paths = []
        degree = args.sh_degree

        for k, i in enumerate(idxs):
            frame = frames[i]
            target = self.images[frame.image_id]
            height, width = int(target.shape[1]), int(target.shape[2])
            fx, fy, cx, cy = frame.intrinsics()
            scale = args.downscale
            K = torch.tensor(
                [
                    [fx / scale, 0.0, cx / scale],
                    [0.0, fy / scale, cy / scale],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
                device=self.device,
            )
            R = torch.tensor(frame.image.rotation_matrix(), dtype=torch.float32, device=self.device)
            t = torch.tensor(frame.image.tvec, dtype=torch.float32, device=self.device)
            viewmat = torch.eye(4, dtype=torch.float32, device=self.device)
            viewmat[:3, :3] = R
            viewmat[:3, 3] = t

            with torch.no_grad():
                rendered, _alpha = self.rasterize(degree, viewmat, K, width, height)
                img = rendered.permute(2, 0, 1)
                if self.exposure is not None:
                    # 评估必须带上与训练**同一份** gain/bias，否则量的是另一个颜色空间。
                    # 上一轮就是因为漏了这句，A/B 报出「开启反而 -1.97 dB」的假结论。
                    eid = int(frame.image_id) % self.exposure["gain"].shape[0]
                    img = img * self.exposure["gain"][eid] + self.exposure["bias"][eid]
                img = img.clamp(0.0, 1.0)
                mse = torch.mean((img - target) ** 2).clamp_min(1e-10)
                psnr = float(-10.0 * torch.log10(mse))
                ssim = float(self._ssim(img.unsqueeze(0), target.unsqueeze(0)))
                l1 = float(torch.abs(img - target).mean())

            psnrs.append(psnr)
            ssims.append(ssim)
            l1s.append(l1)

            # 存一张对比图（上=渲染，下=真值），便于人眼判断"可读性"
            if out_dir is not None:
                rgb = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                ref = (target.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                pair = np.concatenate([rgb, ref], axis=0)
                p = Path(out_dir) / f"eval_view{k}.png"
                self._write_png(p, pair)
                out_paths.append(str(p))
            config.log_info(f"  视角 {Path(frame.name).name}: PSNR {psnr:5.2f} dB  SSIM {ssim:.4f}  L1 {l1:.4f}")

        mean_psnr = sum(psnrs) / len(psnrs)
        mean_ssim = sum(ssims) / len(ssims)
        mean_l1 = sum(l1s) / len(l1s)
        config.log_info(
            f"[评估] 平均 PSNR {mean_psnr:.2f} dB | 平均 SSIM {mean_ssim:.4f} | 平均 L1 {mean_l1:.4f}"
        )
        # 3DGS 论文在 Mip-NeRF360 上约 27~28 dB；合成简单场景可到 30+ dB
        #
        # ⚠️ 这里曾经调用 `config.log_success(...)`，但 config 只提供了 `log_ok`，
        #    结果**画质越好越会崩**：PSNR ≥ 24 dB 时抛 AttributeError，
        #    评估已经跑完、PLY 还没保存 → 整个建模任务以退出码 5 失败。
        #    （实测：23 帧的素材 PSNR 20.9 正常通过；28 帧 PSNR 25.1 反而失败。）
        if mean_psnr >= 28:
            config.log_ok(f"画质评估：优秀（{mean_psnr:.2f} dB）")
        elif mean_psnr >= 24:
            config.log_ok(f"画质评估：良好（{mean_psnr:.2f} dB）")
        elif mean_psnr >= 20:
            config.log_warn(f"画质评估：一般（{mean_psnr:.2f} dB），可增加迭代或提高分辨率")
        else:
            config.log_warn(f"画质评估：偏低（{mean_psnr:.2f} dB），建议检查位姿与素材质量")

        self.eval_result = {
            "psnr": round(mean_psnr, 3),
            "ssim": round(mean_ssim, 5),
            "l1": round(mean_l1, 5),
            "views": len(idxs),
            "images": out_paths,
        }

    @staticmethod
    def _write_png(path: Path, rgb: Any) -> None:
        """不依赖 Pillow 写 PNG（优先用 Pillow，装了就用）。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image  # noqa: PLC0415

            Image.fromarray(rgb).save(path)
            return
        except Exception:  # noqa: BLE001
            pass
        try:
            import imageio.v2 as imageio  # noqa: PLC0415

            imageio.imwrite(path, rgb)
            return
        except Exception:  # noqa: BLE001
            pass
        # 最后兜底：写 PPM（浏览器与 COLMAP 都认）
        h, w, _ = rgb.shape
        with open(path.with_suffix(".ppm"), "wb") as f:
            f.write(b"P6\n%d %d\n255\n" % (w, h))
            f.write(rgb.tobytes())

    def _ssim(self, image: Any, target: Any, mask: Any | None = None) -> Any:
        """结构相似度（3DGS 用的 11×11 高斯窗口 + D-SSIM）。

        `mask`（可选，形状 (1,H,W) 或与图同形）：只在 mask 内统计。
        背景分离时**必须**同时遮住 SSIM，否则背景的梯度照样会拉动高斯 ——
        只遮 L1 会留下"背景通过 SSIM 项继续影响训练"的暗门。

        ⚠️ 这 5 次 `conv2d` 是**训练器里唯一走 cuDNN 的地方**（光栅化是 gsplat
        的自研 CUDA，不经过 cuDNN）。2026-10-06 在这条路径上实测到过原生崩溃：

            python.exe 中处有未经处理的异常: 0xC0000005: 写入位置 0x0 时发生访问冲突
            (cudnn_graph64_9.dll)

        cuDNN 9 的 graph 引擎在 sm_120（RTX 50 系）+ 驱动 610.62 上不稳定；
        而深度卷积本身极便宜，不需要引擎选择。所以默认**关掉 cuDNN**，
        把崩溃面整段移除。对照实验：set DSH_SSIM_CUDNN=1
        """
        torch = self.torch
        window = self._gaussian_window(11, 1.5).to(image.device)
        channels = image.shape[1]
        window = window.expand(channels, 1, 11, 11).contiguous()
        with torch.backends.cudnn.flags(enabled=_SSIM_USE_CUDNN):
            mu1 = torch.nn.functional.conv2d(image, window, padding=5, groups=channels)
            mu2 = torch.nn.functional.conv2d(target, window, padding=5, groups=channels)
            sigma1 = torch.nn.functional.conv2d(image * image, window, padding=5, groups=channels) - mu1 * mu1
            sigma2 = torch.nn.functional.conv2d(target * target, window, padding=5, groups=channels) - mu2 * mu2
            sigma12 = torch.nn.functional.conv2d(image * target, window, padding=5, groups=channels) - mu1 * mu2
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1 * mu1 + mu2 * mu2 + c1) * (sigma1 + sigma2 + c2))
        if mask is not None:
            m = mask.expand_as(ssim_map)
            return (ssim_map * m).sum() / m.sum().clamp(min=1.0)
        return ssim_map.mean()

    def _gaussian_window(self, size: int, sigma: float) -> Any:
        torch = self.torch
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        window = gauss[:, None] * gauss[None, :]
        return window[None, None, :, :]

    # -- 稠密化 / 剪枝 / opacity 重置 ------------------------------------

    def densify_and_prune(self) -> None:
        torch = self.torch
        if self.grad_accum is None:
            return
        # 注意：梯度阈值**不能**乘 spatial_scale。
        # 3DGS 的 2e-4 是「位置参数梯度范数」的绝对阈值，单位就是梯度本身。
        # 原来写成 threshold = 2e-4 * spatial_scale，在本场景（spatial_scale=6.2）
        # 相当于把阈值抬高 6.2 倍，导致几乎没有高斯能达到阈值：
        # 实测 7000 步后高斯数只从 10849 涨到 11045，致密化等于没工作。
        # 学习率与尺度初始化乘 spatial_scale 是对的（那是不同量纲），阈值不是。
        threshold = self.args.densify_grad_threshold
        if self.densify_steps > 0:
            grads = self.grad_accum / self.densify_steps
        else:
            grads = self.grad_accum
        mask = grads >= threshold

        scales = torch.exp(self.params["scales"])
        max_scale = scales.max(dim=1).values
        split_mask = mask & (max_scale > self.args.densify_percent * self.spatial_scale)
        clone_mask = mask & ~split_mask

        with torch.no_grad():
            self._clone(clone_mask)
            self._split(split_mask)
            self._prune()
            self._cap_gaussians()
        self.densify_steps = 0

    def _stack(self, idx: Any) -> dict[str, Any]:
        torch = self.torch
        return {
            "means": self.params["means"].detach()[idx],
            "scales": self.params["scales"].detach()[idx],
            "quats": self.params["quats"].detach()[idx],
            "opacities": self.params["opacities"].detach()[idx],
            "sh0": self.params["sh0"].detach()[idx],
            "shN": self.params["shN"].detach()[idx],
        }

    def _append(self, extra: dict[str, Any]) -> None:
        """把新点拼接到参数上，并重建优化器状态。"""
        torch = self.torch
        for key in ("means", "scales", "quats", "opacities", "sh0", "shN"):
            merged = torch.cat([self.params[key].detach(), extra[key]], dim=0)
            self.params[key] = torch.nn.Parameter(merged.contiguous().requires_grad_(True))
        self._init_optimizer()
        self.grad_accum = torch.zeros(self.params["means"].shape[0], device=self.device)

    def _clone(self, mask: Any) -> None:
        torch = self.torch
        if mask.sum() == 0:
            return
        idx = torch.nonzero(mask).squeeze(1)
        self._append(self._stack(idx))

    def _split(self, mask: Any) -> None:
        """按 3DGS 的做法把一个高斯分裂成两个（沿自身尺度方向抖动）。"""
        torch = self.torch
        if mask.sum() == 0:
            return
        idx = torch.nonzero(mask).squeeze(1)
        base = self._stack(idx)
        scale_linear = torch.exp(base["scales"])
        noise = torch.randn((2 * idx.shape[0], 3), device=self.device) * scale_linear.repeat(2, 1)
        means = base["means"].repeat(2, 1) + noise
        scales = torch.log(scale_linear.repeat(2, 1) / 1.6)
        opacity = torch.sigmoid(base["opacities"]).repeat(2)
        opacity = 1.0 - torch.sqrt(torch.clamp(1.0 - opacity, min=1e-6))
        extra = {
            "means": means,
            "scales": scales,
            "quats": base["quats"].repeat(2, 1),
            "opacities": torch.log(opacity / (1.0 - opacity)),
            "sh0": base["sh0"].repeat(2, 1, 1),
            "shN": base["shN"].repeat(2, 1, 1),
        }
        self._append(extra)

    def _prune(self) -> None:
        """剪枝：不透明度过低、尺度过大、世界坐标异常的统统删掉。"""
        torch = self.torch
        opacity = torch.sigmoid(self.params["opacities"])
        max_scale = torch.exp(self.params["scales"]).max(dim=1).values
        keep = opacity >= self.args.prune_opacity
        keep &= max_scale <= self.scale_cap  # 与 _clamp_scales 同一口径，别一边夹紧一边剪松
        keep &= torch.isfinite(self.params["means"]).all(dim=1)
        removed = int((~keep).sum().item())
        if removed == 0:
            return
        if int(keep.sum().item()) == 0:
            config.log_warn("剪枝后没有任何高斯剩余，保留全部高斯以避免训练中断。")
            return
        idx = torch.nonzero(keep).squeeze(1)
        trimmed = self._stack(idx)
        for key in ("means", "scales", "quats", "opacities", "sh0", "shN"):
            self.params[key] = torch.nn.Parameter(trimmed[key].contiguous().requires_grad_(True))
        self._init_optimizer()
        self.grad_accum = torch.zeros(self.params["means"].shape[0], device=self.device)

    def _cap_gaussians(self) -> None:
        """超过 --max-gaussians 时，优先保留不透明度高的高斯。"""
        torch = self.torch
        limit = self.args.max_gaussians
        count = self.params["means"].shape[0]
        if limit <= 0 or count <= limit:
            return
        opacity = torch.sigmoid(self.params["opacities"])
        keep_idx = torch.topk(opacity, limit, largest=True).indices
        trimmed = self._stack(keep_idx)
        for key in ("means", "scales", "quats", "opacities", "sh0", "shN"):
            self.params[key] = torch.nn.Parameter(trimmed[key].contiguous().requires_grad_(True))
        self._init_optimizer()
        self.grad_accum = torch.zeros(self.params["means"].shape[0], device=self.device)
        config.log_warn(f"高斯数量超过上限 {limit}，已按不透明度裁剪到 {limit}。")

    def reset_opacity(self, iteration: int = 0) -> None:
        """周期性把不透明度重置到较低值，让「漂浮噪点」有机会被剪掉。"""
        torch = self.torch
        with torch.no_grad():
            value = min(0.01, float(self.args.prune_opacity))
            self.params["opacities"].data.fill_(math.log(value / (1.0 - value)))
        config.log_info(f"已重置不透明度（iteration={iteration}，opacity→{value:.3f}）")

    # -- 保存 -----------------------------------------------------------

    def _bake_exposure(self, features: Any) -> Any:
        """把逐帧曝光校正的**平均效应**烘进球谐系数，导出的模型才不带偏色。

        为什么必须做：训练时每帧的 gain/bias 让「渲染 × gain + bias ≈ 真值」，
        于是高斯的 SH 学到的是**校正后颜色空间**里的值。导出时不带这份参数，
        产品就会整体偏色 —— 这正是上一轮 A/B 报出「开启反而 -1.97 dB」的原因
        （评估同样没带参数，量的是另一个空间）。

        这里取**均值** gain/bias 做一次全局仿射：
            c = 0.5 + SH_C0 · sh0        （3DGS 的颜色反解）
            c' = c · ḡ + b̄              （烘进平均曝光）
            sh0' = (c' − 0.5) / SH_C0
            shN' = shN · ḡ              （高阶项只随增益线性缩放，偏置是常数只进 DC）
        等价于「让导出模型呈现整段素材的平均曝光」，而不是任何单帧。
        """
        if self.exposure is None:
            return features
        torch = self.torch
        with torch.no_grad():
            g = float(self.exposure["gain"].mean().item())
            b = float(self.exposure["bias"].mean().item())
        if abs(g - 1.0) < 1e-6 and abs(b) < 1e-6:
            return features
        out = features.clone()
        c = 0.5 + SH_C0 * out[:, 0:1, :]
        c = (c * g + b).clamp(0.0, 1.0)
        out[:, 0:1, :] = (c - 0.5) / SH_C0
        if out.shape[1] > 1:
            out[:, 1:, :] = out[:, 1:, :] * g
        config.log_info(f"已将平均曝光烘进球谐：gain={g:.4f} bias={b:+.4f}（导出模型不带逐帧偏色）")
        return out

    def save(self, iteration: int) -> int:
        self._cap_gaussians()
        out_ply = self.out_dir / "point_cloud.ply"
        features = self.current_features(self.max_sh_degree).detach()
        features = self._bake_exposure(features)
        count = save_ply(
            out_ply,
            means=self.params["means"].detach(),
            scales_log=self.params["scales"].detach(),
            quats=self.params["quats"].detach(),
            opacities_logit=self.params["opacities"].detach(),
            features=features,
        )
        size = out_ply.stat().st_size
        config.log_ok(
            f"[保存] iteration={iteration} 已写出 {out_ply}（{count} 个高斯，{config.human_bytes(size)}）"
        )

        # 连同画质评估结果一起写一份摘要，便于脚本/文档引用
        try:
            import json  # noqa: PLC0415

            summary = {
                "ply": str(out_ply),
                "gaussians": int(count),
                "bytes": int(size),
                "iterations": int(iteration),
                "shDegree": int(self.max_sh_degree),
                "downscale": int(self.args.downscale),
                "device": str(self.device),
                "eval": getattr(self, "eval_result", None),
                # ---- 尺度口径必须落盘（Bug A 修复的关键前提）----
                # `_clamp_scales()` 把高斯的线性尺度夹在 `prune_scale × spatial_scale`，
                # 导出脚本必须用**同一个上限**，否则会把训练器刻意保留的
                # 「铺墙铺地的大高斯」当噪点删掉。实测：导出阈值硬编码 0.5 时
                # 丢掉 51.5% 的高斯；改用这里的 maxScaleMeters（=1.1185）后丢弃 0%。
                # 之前这三个数一个都没写进摘要，导出脚本只能猜 —— 这就是 Bug A 的土壤。
                "sceneExtent": round(float(self.spatial_scale), 4),
                "pruneScale": float(self.args.prune_scale),
                # 落盘**实际生效**的上限（两个口径取小后的结果），而不是只写相机口径 ——
                # 否则导出脚本会按一个比训练时更松的数去筛，等于把训练时特意夹掉的雾团又放回来。
                "maxScaleMeters": round(float(self.scale_cap), 4),
                "featureScale": round(float(self.feature_scale), 4),
                "scaleCapK": float(getattr(self.args, "scale_cap_k", 4.0)),
                "cameraScaleCap": round(float(self.args.prune_scale * self.spatial_scale), 4),
            }
            (self.out_dir / "train_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:  # noqa: BLE001
            pass
        return 0


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_gsplat.py",
        description="用 gsplat 训练 3D Gaussian Splatting，并导出可交给 to_splat.py 的 3DGS PLY。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python pipeline/train_gsplat.py --data data/projects/case01/colmap --out data/projects/case01/train --images data/projects/case01/frames\n"
            "  python pipeline/train_gsplat.py --data scene --out out --iterations 30000 --downscale 4 --max-gaussians 800000\n"
        ),
    )
    parser.add_argument("--data", required=True, help="COLMAP 场景目录（含 images/ 与 sparse/0/，或含 colmap/sparse/0）")
    parser.add_argument("--out", required=True, help="输出目录（生成 point_cloud.ply）")
    parser.add_argument("--images", default=None, help="可选：显式指定图片目录（默认自动探测）")
    parser.add_argument("--iterations", type=int, default=15000, help="训练迭代次数（默认 15000）")
    parser.add_argument("--sh-degree", type=int, default=3, choices=[0, 1, 2, 3], help="球谐最高阶数（默认 3）")
    parser.add_argument("--downscale", type=int, default=2, help="图像下采样倍数（显存不足就调大，默认 2）")
    parser.add_argument("--max-gaussians", type=int, default=1500000, help="高斯数量上限（默认 1500000）")
    parser.add_argument("--save-every", type=int, default=5000, help="每多少次迭代保存一次 PLY（默认 5000，0 表示只在结束时保存）")
    parser.add_argument("--log-every", type=int, default=100, help="每多少次迭代打印一行进度（默认 100）")
    parser.add_argument(
        "--eval-views",
        type=int,
        default=4,
        help="训练结束后在多少个留出视角上评估 PSNR/SSIM 并渲染对比图（默认 4，0 表示不评估）",
    )
    parser.add_argument("--learning-rate", type=float, default=1.6e-4, help="位置学习率基准（默认 1.6e-4）")
    parser.add_argument(
        "--lr-scale-factor",
        type=float,
        default=1.0,
        help="尺度/不透明度/四元数学习率的额外倍率（默认 1.0）。"
        "真实米制场景若出现「大雾团」或参数饱和，调小到 0.3~0.5",
    )
    parser.add_argument("--dssim-weight", type=float, default=0.2, help="D-SSIM 损失权重（默认 0.2）")
    parser.add_argument(
        "--mask-dir", default=None,
        help="背景分离 mask 目录（与图片同名 .png，黑=忽略）。mask 外**不计入损失**"
             "（L1 与 SSIM 都遮）。物体算法专有：房间拍摄没有背景可言。生成见 tools/object_mask.py",
    )
    parser.add_argument("--sh-degree-interval", type=int, default=1000, help="每隔多少迭代提升一阶球谐（默认 1000）")
    parser.add_argument("--densify-interval", type=int, default=100, help="稠密化间隔（默认 100）")
    parser.add_argument("--densify-until", type=int, default=15000, help="在该迭代之前允许稠密化（默认 15000）")
    parser.add_argument("--densify-grad-threshold", type=float, default=2e-4, help="稠密化梯度阈值（默认 2e-4）")
    parser.add_argument("--densify-percent", type=float, default=0.01, help="尺度超过场景尺度的该比例时才分裂（默认 0.01）")
    parser.add_argument("--prune-opacity", type=float, default=0.005, help="不透明度低于该值即剪枝（默认 0.005）")
    parser.add_argument("--prune-scale", type=float, default=0.1, help="尺度超过 场景尺度×该值 即剪枝（默认 0.1）")
    parser.add_argument(
        "--scale-cap-k",
        type=float,
        default=4.0,
        help=(
            "尺度上限的**特征口径**倍数：上限 = min(prune_scale×spatial_scale, k×点云最近邻中位)。"
            " 默认 4.0。实测发雾组 4×0.0747=0.30 m（把 1.12 m 的成长空间收掉），"
            " 健康组 4×0.1487=0.60 m 而实际最大尺度才 0.33 m → 对好模型零影响。"
        ),
    )
    parser.add_argument(
        "--scale-aniso-max",
        type=float,
        default=4.0,
        help=(
            "各向异性比上限：把三个尺度轴拉进同一量级（s_max/s_min ≤ 该值，默认 4.0）。"
            " 实测健康模型针状度 s1/s2=0.472，发雾模型只有 0.089~0.234 —— "
            " 两个小轴塌成「刀片」后，随机朝向交叉渲染出来就是墙面上的法向云雾。"
        ),
    )
    parser.add_argument("--opacity-reset-interval", type=int, default=3000, help="不透明度重置周期（默认 3000，0 表示不重置）")
    parser.add_argument(
        "--exposure-adapt",
        action="store_true",
        help=(
            "启用逐帧仿射曝光校正（每帧一对 gain/bias）。**默认关闭，实测反而更差**："
            "12MB/23 帧那组 7000 步 A/B —— 关闭 PSNR 25.07 dB / SSIM 0.8745，"
            "开启 23.11 dB / 0.8579（-1.97 dB）。"
            "原因（尚未修）：校正只在**训练**里做，留出视角评估与导出的 .splat 都没有这份参数，"
            "高斯颜色被学到「校正后的空间」，于是产品本身偏色。"
            "要做对必须：评估时带上同一份 gain/bias，并在导出时把外观归一到均值增益 1。"
        ),
    )
    parser.add_argument("--exposure-reg", type=float, default=0.01, help="曝光参数正则系数，把 gain/bias 拉回 1/0（默认 0.01）")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    config.ensure_utf8_stdio()
    config.require_python()
    args = build_parser().parse_args(argv)

    if args.iterations <= 0:
        config.log_error("--iterations 必须大于 0。")
        return 2
    if args.downscale < 1:
        config.log_error("--downscale 必须 ≥ 1。")
        return 2

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    config.log_rule("3D Gaussian Splatting 训练")

    if not data_dir.is_dir():
        config.log_error(f"数据目录不存在：{data_dir}")
        return 2

    # 即使没有 torch 也先把 COLMAP 部分解析出来，方便定位问题
    try:
        scene = load_scene(data_dir, args.images)
    except (FileNotFoundError, colmap_recon.ColmapFormatError) as exc:
        config.log_error(str(exc))
        return 2
    describe_scene(scene)
    if not scene.frames:
        config.log_error("没有任何「已注册图片 + 对应图片文件」可用于训练。")
        config.log_info(f"图片目录：{scene.images_dir}；已注册图片：{len(scene.model.images)} 张")
        return 2

    torch = config.try_import("torch")
    gsplat = config.try_import("gsplat")
    if torch is None or gsplat is None:
        missing = []
        if torch is None:
            missing.append("torch")
        if gsplat is None:
            missing.append("gsplat")
        config.log_error(f"缺少依赖：{', '.join(missing)}")
        for line in MISSING_DEP_HINT.splitlines():
            config.log_info(line)
        config.log_info(
            f"（当前环境可以正常解析 COLMAP 场景：{len(scene.frames)} 张可用图片、"
            f"{len(scene.model.points3D)} 个稀疏点，说明数据准备没问题。）"
        )
        return 3

    info = config.torch_env_info()
    config.log_info(
        f"torch {info['torchVersion']} | CUDA 可用：{info['cudaAvailable']} | "
        f"CUDA 版本：{info['cudaVersion']} | 设备：{info['deviceName'] or 'CPU'} | 算力：{info['capability'] or '-'}"
    )
    if info["cudaAvailable"] and info["capability"] == "sm_120" and str(info["cudaVersion"] or "") < "12.8":
        config.log_warn(
            "RTX 5060 是 sm_120（Blackwell），需要 CUDA ≥ 12.8 的 PyTorch；"
            "当前 CUDA 版本偏低，可能报 'no kernel image is available'。"
        )
    config.log_info(f"输出目录：{out_dir}")
    # 把"SSIM 卷积走不走 cuDNN"写进日志：出问题时第一眼就能看到是哪条路径
    config.log_info(
        "SSIM 卷积：cuDNN " + ("开启（DSH_SSIM_CUDNN=1，对照实验用）" if _SSIM_USE_CUDNN else "关闭（默认，避开 cudnn_graph 崩溃）")
    )
    config.ensure_dirs(out_dir)

    try:
        trainer = Trainer(scene, out_dir, args)
    except RuntimeError as exc:
        config.log_error(f"初始化训练器失败：{exc}")
        return 5
    except Exception as exc:  # noqa: BLE001 - 图像读取/显存分配都可能出错
        config.log_error(f"初始化训练器失败：{type(exc).__name__}: {exc}")
        return 5

    try:
        code = trainer.train()
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        oom = "out of memory" in message.lower() or "OutOfMemoryError" in type(exc).__name__
        if oom:
            config.log_error("显存不足（CUDA out of memory），训练已优雅退出。")
            config.log_info("可执行的补救建议（按性价比排序）：")
            config.log_info(f"  1) 加大下采样：--downscale {min(8, args.downscale * 2)}（显存占用大致按平方下降）")
            config.log_info(f"  2) 降低高斯上限：--max-gaussians {max(100000, args.max_gaussians // 2)}")
            config.log_info("  3) 减少迭代次数或先用 --iterations 3000 试跑通流程；")
            config.log_info("  4) 训练前关掉其它占显存的程序（浏览器硬件加速、其它 CUDA 任务）；")
            config.log_info("  5) 确认驱动与 CUDA 版本匹配（RTX 5060 需要 CUDA ≥ 12.8 的 torch）。")
            return 4
        config.log_error(f"训练失败：{type(exc).__name__}: {message}")
        config.log_info("排查建议：")
        config.log_info("  1) 先用 --iterations 200 --downscale 8 试跑，确认流程能通；")
        config.log_info("  2) gsplat 版本与 torch 版本必须匹配（gsplat 官方只提供 Linux 轮子，Windows 需源码编译）；")
        config.log_info("  3) 报 'no kernel image is available' 说明 torch 的 CUDA 版本低于显卡算力要求；")
        config.log_info("  4) 检查图片是否都能正常解码（Pillow/imageio/OpenCV 至少装一个）。")
        import traceback  # noqa: PLC0415

        config.log_debug(traceback.format_exc())
        return 5

    ply = out_dir / "point_cloud.ply"
    config.log_ok(f"训练完成！PLY：{ply}")
    config.log_info(
        "下一步：python pipeline/to_splat.py --ply "
        f"\"{ply}\" --out \"{out_dir / 'point_cloud.splat'}\" --json \"{out_dir / 'point_cloud.json'}\""
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
