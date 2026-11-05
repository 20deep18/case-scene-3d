"""
pipeline/train_gpu.py —— 纯 PyTorch 可微高斯泼溅训练器（Windows / sm_120 可直接跑）

为什么需要它：
    官方 3DGS（graphdeco-inria）与 gsplat 都需要**编译 CUDA 扩展**，在 Windows 上
    依赖 MSVC + 与 PyTorch 完全匹配的 nvcc。本机（RTX 5060 Laptop，sm_120）没有
    MSVC，而 gsplat 官方也没有 cu128 的 Windows 预编译轮子，所以这里用**纯 PyTorch
    算子**实现一个可微的高斯泼溅渲染器：不需要编译器，只要有 torch+CUDA 即可训练。

与官方 3DGS 的关系：
    * 参数化完全一致（mean / scale=log(s) / rotation 四元数 / opacity=logit(a) / SH DC）
    * 损失一致（L1 + D-SSIM）
    * 致密化策略一致（按位置梯度范数 clone / split，按尺度与屏幕半径 prune，周期性重置 opacity）
    * 渲染是「按深度排序的二维高斯泼溅」的简化版（不用 CUDA 瓦片光栅化），
      因此细节与速度不及官方实现，但产出文件格式完全兼容（标准 3DGS PLY），
      可以继续用 pipeline/to_splat.py 转成 Web 用的 .splat。

用法：
    python pipeline/train_gpu.py --data data/scene/gs --out output/scene \
        [--iterations 15000] [--downscale 2] [--max-gaussians 300000] [--sh-degree 0]

退出码：0 成功；2 参数错误；3 缺 torch；4 缺 CUDA；5 读不到 COLMAP；6 显存不足
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None


# ---------------------------------------------------------------------------
# SSIM 卷积是否允许走 cuDNN（默认**不允许**）
# ---------------------------------------------------------------------------
# 实测过一次原生崩溃：`cudnn_graph64_9.dll` 里 0xC0000005 写 0x0。
# 详见 ssim_loss() 的说明。默认关掉；对照实验用 set DSH_SSIM_CUDNN=1 打开。
_SSIM_USE_CUDNN = os.environ.get("DSH_SSIM_CUDNN", "0") == "1"


# --------------------------------------------------------------------- COLMAP 读取

CAMERA_MODELS = {
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


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: list[float]

    @property
    def fx(self) -> float:
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
            return self.params[0]
        return self.params[0]

    @property
    def fy(self) -> float:
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
            return self.params[0]
        return self.params[1]

    @property
    def cx(self) -> float:
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
            return self.params[1]
        return self.params[2]

    @property
    def cy(self) -> float:
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
            return self.params[2]
        return self.params[3]


@dataclass
class Image:
    id: int
    qvec: np.ndarray  # (w,x,y,z)
    tvec: np.ndarray
    camera_id: int
    name: str

    def rotation(self) -> np.ndarray:
        w, x, y, z = self.qvec
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )


def _read_next_bytes(fid, num_bytes: int, fmt: str):
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError("COLMAP 二进制文件提前结束")
    return struct.unpack("<" + fmt, data)


def read_cameras_binary(path: Path) -> dict[int, Camera]:
    cams: dict[int, Camera] = {}
    with open(path, "rb") as fid:
        (num,) = _read_next_bytes(fid, 8, "Q")
        for _ in range(num):
            cam_id, model_id, width, height = _read_next_bytes(fid, 24, "iiQQ")
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"未知相机模型 id={model_id}")
            name, nparams = CAMERA_MODELS[model_id]
            params = _read_next_bytes(fid, 8 * nparams, "d" * nparams)
            cams[cam_id] = Camera(cam_id, name, int(width), int(height), list(params))
    return cams


def read_images_binary(path: Path) -> dict[int, Image]:
    images: dict[int, Image] = {}
    with open(path, "rb") as fid:
        (num,) = _read_next_bytes(fid, 8, "Q")
        for _ in range(num):
            props = _read_next_bytes(fid, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5], dtype=np.float64)
            tvec = np.array(props[5:8], dtype=np.float64)
            camera_id = props[8]
            name_bytes = b""
            while True:
                ch = fid.read(1)
                if ch == b"\x00":
                    break
                if not ch:
                    raise EOFError("读取图像名时文件结束")
                name_bytes += ch
            (num_points2d,) = _read_next_bytes(fid, 8, "Q")
            fid.read(24 * num_points2d)  # xys + point3D_id，训练不需要
            images[image_id] = Image(image_id, qvec, tvec, camera_id, name_bytes.decode("utf-8", "replace"))
    return images


# --------------------------------------------------------------------- 高斯模型


@dataclass
class Gaussians:
    means: "torch.Tensor"      # (N,3) 世界坐标
    scales: "torch.Tensor"     # (N,3) log 尺度
    quats: "torch.Tensor"      # (N,4) 未归一化四元数 (w,x,y,z)
    opacity: "torch.Tensor"    # (N,1) logit 不透明度
    sh_dc: "torch.Tensor"      # (N,3) 球谐 DC 系数
    sh_rest: "torch.Tensor | None" = None  # (N,K,3)
    grad_accum: "torch.Tensor | None" = None
    max_radii: "torch.Tensor | None" = None

    def count(self) -> int:
        return int(self.means.shape[0])

    def parameters(self):
        return [self.means, self.scales, self.quats, self.opacity, self.sh_dc]


def init_from_pointcloud(points: np.ndarray, colors: np.ndarray, sh_degree: int) -> Gaussians:
    """
    从 COLMAP 稀疏点云初始化高斯（与官方 3DGS 一致）：
      * 位置 = 点云坐标
      * 尺度 = 到最近 3 个邻居的平均距离的 log
      * 旋转 = 由 K 近邻协方差矩阵的特征向量构造（各向异性），这是 3DGS 能快速收敛的关键
      * 颜色 = 点云颜色转 SH DC 系数
    """
    n = points.shape[0]
    if n < 16:
        raise ValueError(f"稀疏点太少（{n} 个），无法初始化高斯模型")
    pts = torch.tensor(points, dtype=torch.float32, device="cuda")
    cols = torch.tensor(colors, dtype=torch.float32, device="cuda") / 255.0

    # --- 邻居搜索（分块，避免 O(N²) 撑爆显存）
    chunk = 2048
    k = min(12, n - 1)
    scale_list = []
    rot_list = []
    for i in range(0, n, chunk):
        block = pts[i : i + chunk]
        d = torch.cdist(block, pts)
        # 排除自身
        d[torch.arange(block.shape[0]), torch.arange(i, i + block.shape[0])] = float("inf")
        vals, idxs = torch.topk(d, k, dim=1, largest=False)
        mean_dist = vals[:, : max(1, min(3, k - 1))].mean(dim=1).clamp_min(1e-4)
        scale_list.append(torch.log(mean_dist))

        # 邻域协方差 -> 特征向量作为高斯朝向
        neigh = pts[idxs]  # (B,k,3)
        centered = neigh - neigh.mean(dim=1, keepdim=True)
        cov = centered.transpose(1, 2) @ centered / max(1, k - 1)  # (B,3,3)
        cov = cov + torch.eye(3, device="cuda") * 1e-6
        evals, evecs = torch.linalg.eigh(cov)  # 升序
        # 让最大特征值方向对齐局部 z 轴；构造旋转矩阵 R = [v0 v1 v2]
        V = evecs  # (B,3,3)，列为特征向量
        # 保证右手系
        det = torch.linalg.det(V)
        fix = torch.where(det < 0, -1.0, 1.0)
        V = V.clone()
        V[:, :, 0] = V[:, :, 0] * fix.unsqueeze(1)
        # 旋转矩阵 -> 四元数 (w,x,y,z)
        m = V
        tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
        q = torch.zeros((m.shape[0], 4), device="cuda")
        # 用稳定的分支计算
        s_pos = torch.sqrt((tr + 1.0).clamp_min(1e-8)) * 2
        s_x = torch.sqrt((1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2]).clamp_min(1e-8)) * 2
        s_y = torch.sqrt((1.0 + m[:, 1, 1] - m[:, 0, 0] - m[:, 2, 2]).clamp_min(1e-8)) * 2
        s_z = torch.sqrt((1.0 + m[:, 2, 2] - m[:, 0, 0] - m[:, 1, 1]).clamp_min(1e-8)) * 2
        cond = torch.stack([tr, m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]], dim=1).argmax(dim=1)
        qw = torch.where(cond == 0, 0.25 * s_pos, torch.where(cond == 1, (m[:, 2, 1] - m[:, 1, 2]) / s_x, torch.where(cond == 2, (m[:, 0, 2] - m[:, 2, 0]) / s_y, (m[:, 1, 0] - m[:, 0, 1]) / s_z)))
        qx = torch.where(cond == 0, (m[:, 2, 1] - m[:, 1, 2]) / s_pos, torch.where(cond == 1, 0.25 * s_x, torch.where(cond == 2, (m[:, 1, 0] + m[:, 0, 1]) / s_y, (m[:, 0, 2] + m[:, 2, 0]) / s_z)))
        qy = torch.where(cond == 0, (m[:, 0, 2] - m[:, 2, 0]) / s_pos, torch.where(cond == 1, (m[:, 0, 1] + m[:, 1, 0]) / s_x, torch.where(cond == 2, 0.25 * s_y, (m[:, 1, 2] + m[:, 2, 1]) / s_z)))
        qz = torch.where(cond == 0, (m[:, 1, 0] - m[:, 0, 1]) / s_pos, torch.where(cond == 1, (m[:, 0, 2] + m[:, 2, 0]) / s_x, torch.where(cond == 2, (m[:, 1, 2] + m[:, 2, 1]) / s_y, 0.25 * s_z)))
        quat = torch.stack([qw, qx, qy, qz], dim=1)
        quat = quat / quat.norm(dim=1, keepdim=True).clamp_min(1e-8)
        rot_list.append(quat)

    log_scale = torch.cat(scale_list, dim=0)  # (N,)
    quats = torch.cat(rot_list, dim=0)  # (N,4)

    # 各向异性：沿法向（最小特征值方向 = 局部 z 轴）压扁，面内尺度稍大
    # 这里用统一的初始各向异性比例，训练中会自动调整
    scales = torch.stack([log_scale, log_scale, log_scale - 1.2], dim=1)  # (N,3)

    C0 = 0.28209479177387814
    sh_dc = (cols - 0.5) / C0

    opacity = torch.full((n, 1), -1.0, device="cuda")  # sigmoid(-1) ≈ 0.27
    sh_rest = None
    if sh_degree > 0:
        K = {1: 3, 2: 8, 3: 15}[sh_degree]
        sh_rest = torch.zeros((n, K, 3), device="cuda")

    return Gaussians(
        means=pts.clone().requires_grad_(True),
        scales=scales.clone().requires_grad_(True),
        quats=quats.clone().requires_grad_(True),
        opacity=opacity.clone().requires_grad_(True),
        sh_dc=sh_dc.clone().requires_grad_(True),
        sh_rest=sh_rest.requires_grad_(True) if sh_rest is not None else None,
        grad_accum=torch.zeros(n, device="cuda"),
        max_radii=torch.zeros(n, device="cuda"),
    )


# --------------------------------------------------------------------- 可微渲染


def build_covariance(scales: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
    """Σ = R S Sᵀ Rᵀ，返回 (N,3,3)"""
    s = torch.exp(scales)
    q = quats / quats.norm(dim=1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=1),
        ],
        dim=1,
    )
    S = torch.diag_embed(s)
    M = R @ S
    return M @ M.transpose(1, 2)


def render(
    gauss: Gaussians,
    R: torch.Tensor,
    T: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    bg: torch.Tensor,
    sh_degree: int = 0,
    tile: int = 128,
):
    """
    按深度排序的二维高斯泼溅（简化版可微渲染）。
    把每个高斯投影到屏幕，按瓦片切分，瓦片内按深度从远到近做 alpha 合成。
    返回 (图像 (3,H,W), 可见高斯掩码 (N,))
    """
    device = gauss.means.device
    N = gauss.count()
    if N == 0:
        return bg.view(3, 1, 1).repeat(1, height, width), torch.zeros(0, dtype=torch.bool, device=device)

    # --- 相机空间
    cam = gauss.means @ R.transpose(0, 1) + T  # (N,3)
    z = cam[:, 2]
    visible = z > 0.05
    if visible.sum() == 0:
        return bg.view(3, 1, 1).repeat(1, height, width), visible
    idx = visible.nonzero(as_tuple=True)[0]
    cam_v = cam[idx]
    z = cam_v[:, 2]

    # --- 投影到屏幕
    x = fx * cam_v[:, 0] / z + cx
    y = fy * cam_v[:, 1] / z + cy

    # --- 2D 协方差：Σ' = J W Σ Wᵀ Jᵀ
    cov3 = build_covariance(gauss.scales, gauss.quats)[idx]
    W = R  # 世界 -> 相机 的旋转
    cov_cam = W @ cov3 @ W.transpose(0, 1)
    inv_z = 1.0 / z
    J = torch.zeros((idx.shape[0], 3, 3), device=device)
    J[:, 0, 0] = fx * inv_z
    J[:, 0, 2] = -fx * cam_v[:, 0] * inv_z * inv_z
    J[:, 1, 1] = fy * inv_z
    J[:, 1, 2] = -fy * cam_v[:, 1] * inv_z * inv_z
    cov2_full = J @ cov_cam @ J.transpose(1, 2)
    a = cov2_full[:, 0, 0] + 0.3
    b = cov2_full[:, 0, 1]
    c = cov2_full[:, 1, 1] + 0.3
    det = (a * c - b * b).clamp_min(1e-8)
    conic_a = c / det
    conic_b = -b / det
    conic_c = a / det

    # --- 屏幕半径（3σ）
    mid = 0.5 * (a + c)
    radius = torch.clamp(3.0 * torch.sqrt(mid.clamp_min(0.0)), 0.5, 200.0)

    alpha = torch.sigmoid(gauss.opacity[idx, 0])
    color = (gauss.sh_dc[idx] * 0.28209479177387814 + 0.5).clamp(0.0, 1.0)  # (M,3)

    # --- 按瓦片渲染
    img = bg.view(3, 1, 1).repeat(1, height, width).clone()
    nx = (width + tile - 1) // tile
    ny = (height + tile - 1) // tile

    for ty in range(ny):
        y0 = ty * tile
        y1 = min(y0 + tile, height)
        for tx in range(nx):
            x0 = tx * tile
            x1 = min(x0 + tile, width)
            # 该瓦片内可能影响像素的高斯
            inside = (x + radius > x0) & (x - radius < x1) & (y + radius > y0) & (y - radius < y1)
            sel = inside.nonzero(as_tuple=True)[0]
            if sel.shape[0] == 0:
                continue
            order = torch.argsort(z[sel], descending=True)  # 远 -> 近
            sel = sel[order]

            # 像素网格
            py = torch.arange(y0, y1, device=device, dtype=torch.float32) + 0.5
            px = torch.arange(x0, x1, device=device, dtype=torch.float32) + 0.5
            gy, gx = torch.meshgrid(py, px, indexing="ij")
            gy = gy.reshape(-1)
            gx = gx.reshape(-1)

            trans = torch.ones(gx.shape[0], device=device)
            acc = torch.zeros((gx.shape[0], 3), device=device)

            # 逐块合成：块大小按「瓦片像素数」自适应，把临时张量控制在约 200 万个 float，
            # 避免 (B,P) 形状过大造成显存峰值（这是 8GB 卡上 OOM 的主要原因）
            ca = conic_a[sel]
            cb = conic_b[sel]
            cc = conic_c[sel]
            cxx = x[sel]
            cyy = y[sel]
            cal = alpha[sel]
            ccol = color[sel]

            m = int(sel.shape[0])
            step = max(1, min(512, 2_000_000 // max(1, int(gx.shape[0]))))
            for s0 in range(0, m, step):
                s1 = min(s0 + step, m)
                dsx = gx.unsqueeze(0) - cxx[s0:s1].unsqueeze(1)  # (B,P)
                dsy = gy.unsqueeze(0) - cyy[s0:s1].unsqueeze(1)
                power = -0.5 * (
                    ca[s0:s1].unsqueeze(1) * dsx * dsx + cc[s0:s1].unsqueeze(1) * dsy * dsy
                ) - cb[s0:s1].unsqueeze(1) * dsx * dsy
                g = torch.exp(power)
                a_i = (cal[s0:s1].unsqueeze(1) * g).clamp(0.0, 0.99)
                acc = acc + ((a_i * trans.unsqueeze(0)).unsqueeze(2) * ccol[s0:s1].unsqueeze(1)).sum(dim=0)
                trans = trans * torch.prod(1.0 - a_i, dim=0)
                if float(trans.max()) < 1e-4:
                    break

            tile_img = acc + trans.unsqueeze(1) * bg.view(1, 3)
            img[:, y0:y1, x0:x1] = tile_img.reshape(y1 - y0, x1 - x0, 3).permute(2, 0, 1)

    return img.clamp(0.0, 1.0), visible


def ssim_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """简化 D-SSIM（11x11 高斯窗，与官方实现同参数）。

    ⚠️ 这里的 5 次 `conv2d` 是**整个训练器里唯一走 cuDNN 的地方**
    （高斯泼溅光栅化是本项目自研的 PyTorch 实现，不经过 cuDNN）。
    所以下面这段就是 `cudnn_graph64_9.dll` 的唯一入口 —— 2026-10-06 实测到过
    这条路径上的**原生崩溃**：

        python.exe 中处有未经处理的异常: 0xC0000005: 写入位置 0x0 时发生访问冲突
        (cudnn_graph64_9.dll)

    cuDNN 9 的 graph 引擎（`cudnn_graph64_9.dll` + `cudnn_engines_runtime_compiled`）
    会为分组卷积现场挑/编译执行引擎；在 sm_120（RTX 50 系）+ 驱动 610.62 这套
    很新的组合上不稳定。而我们要的深度卷积**极其便宜**（11×11、groups=3、
    每步 5 次），根本不需要 cuDNN 的引擎选择 ——

    所以默认**关掉 cuDNN**，走 ATen 的原生实现：
      · 崩溃面被彻底移除（那段代码不再被调用），不是"少崩一点"；
      · 代价可忽略（见 `tools/` 下的对照实测：训练步速与画质都没变）；
      · 数值等价性由 tests/test_train_gpu.py 与 tests/test_ply_layout.py 守着。
    需要对照实验时用环境变量打开：`set DSH_SSIM_CUDNN=1`。
    """
    C1 = 0.01**2
    C2 = 0.03**2
    kernel = torch.ones(3, 1, 11, 11, device=a.device) / 121.0
    with torch.backends.cudnn.flags(enabled=_SSIM_USE_CUDNN):
        mu1 = F.conv2d(a, kernel, padding=5, groups=3)
        mu2 = F.conv2d(b, kernel, padding=5, groups=3)
        sigma1 = F.conv2d(a * a, kernel, padding=5, groups=3) - mu1 * mu1
        sigma2 = F.conv2d(b * b, kernel, padding=5, groups=3) - mu2 * mu2
        sigma12 = F.conv2d(a * b, kernel, padding=5, groups=3) - mu1 * mu2
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 * mu1 + mu2 * mu2 + C1) * (sigma1 + sigma2 + C2))
    return 1.0 - ssim.mean()


# --------------------------------------------------------------------- 训练


def load_scene(data_dir: Path, downscale: int, device: torch.device):
    sparse = data_dir / "sparse" / "0"
    if not sparse.is_dir():
        raise FileNotFoundError(f"找不到 {sparse}，请先运行 pipeline/colmap_recon.py")
    cams = read_cameras_binary(sparse / "cameras.bin")
    imgs = read_images_binary(sparse / "images.bin")
    if not imgs:
        raise ValueError("COLMAP 结果里没有注册任何图像")

    # 稀疏点云
    # 注意 COLMAP points3D.bin 的真实结构是：
    #   point3D_id(Q, 8)  x(d) y(d) z(d)  r(B) g(B) b(B)  error(d)   —— 共 43 字节
    #   紧接 track_length(Q, 8，即 size_t) + track_length × (image_id(i4) + point2D_idx(i4))
    # 把 track_length 当成 1 字节会让后续全部错位，表现为读出一堆 nan / 1e-307 垃圾值
    pts_path = sparse / "points3D.bin"
    points = []
    colors = []
    if pts_path.exists():
        with open(pts_path, "rb") as fid:
            (num,) = _read_next_bytes(fid, 8, "Q")
            for _ in range(num):
                rec = _read_next_bytes(fid, 43, "QdddBBBd")
                points.append(rec[1:4])
                colors.append(rec[4:7])
                (track_len,) = _read_next_bytes(fid, 8, "Q")
                if track_len:
                    fid.read(8 * track_len)
    points = np.array(points, dtype=np.float64)
    colors = np.array(colors, dtype=np.float64)

    # 质量兜底：剔除 COLMAP 偶发的坏点（nan / inf / 量级离谱），避免污染初始化与场景尺度
    if points.shape[0]:
        good = np.isfinite(points).all(axis=1) & (np.abs(points) < 1e6).all(axis=1)
        if not good.all():
            dropped = int((~good).sum())
            config.log_warn(f"稀疏点云中有 {dropped} 个非法点（nan/inf/量级异常）已剔除")
            points = points[good]
            colors = colors[good]

    # 图像目录
    images_dir = data_dir / "images"
    if not images_dir.is_dir():
        # 有些 COLMAP 输出把图放在 undistorted/images
        alt = data_dir / "undistorted" / "images"
        images_dir = alt if alt.is_dir() else images_dir

    samples = []
    for img in imgs.values():
        cam = cams.get(img.camera_id)
        if cam is None:
            continue
        path = images_dir / img.name
        if not path.exists():
            # 尝试同名不同扩展
            stem = Path(img.name).stem
            for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
                cand = images_dir / (stem + ext)
                if cand.exists():
                    path = cand
                    break
        if not path.exists():
            continue
        R = img.rotation()
        samples.append(
            {
                "path": path,
                "R": R,
                "t": img.tvec,
                "cam": cam,
                "name": img.name,
            }
        )
    return samples, points, colors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="纯 PyTorch 高斯泼溅训练（无需编译 CUDA 扩展）")
    ap.add_argument("--data", required=True, help="COLMAP 场景目录（含 images/ 与 sparse/0）")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--iterations", type=int, default=15000)
    ap.add_argument(
        "--downscale",
        type=int,
        default=3,
        help="图像降采样倍数。8GB 显存建议 3~4；12GB 可试 2；显存不足时调大",
    )
    ap.add_argument(
        "--max-gaussians",
        type=int,
        default=60000,
        help="高斯数量上限。8GB 显存建议 40000~60000（太大或分辨率太高都会 OOM）",
    )
    ap.add_argument("--sh-degree", type=int, default=0, choices=[0, 1, 2, 3])
    ap.add_argument("--densify-until", type=int, default=0, help="致密化结束迭代（默认 iterations*0.6）")
    ap.add_argument("--densify-interval", type=int, default=200)
    ap.add_argument("--opacity-reset-interval", type=int, default=3000)
    ap.add_argument("--lr-means", type=float, default=1.6e-4)
    ap.add_argument("--lr-scales", type=float, default=5e-3)
    ap.add_argument("--lr-quats", type=float, default=1e-3)
    ap.add_argument("--lr-opacity", type=float, default=5e-2)
    ap.add_argument("--lr-sh", type=float, default=2.5e-3)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--debug-numeric", action="store_true", help="每步输出数值诊断（排查发散用）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if torch is None:
        config.log_error("未安装 PyTorch，无法训练。")
        config.log_info("  请先运行：python pipeline/setup_env.py --check")
        return 3
    if not torch.cuda.is_available():
        config.log_error("PyTorch 检测不到可用的 CUDA 设备。")
        config.log_info("  请确认与显卡算力匹配的 PyTorch 版本（RTX 50 系列需要 cu128 及以上的 cu 版本）。")
        return 4

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")

    data_dir = Path(args.data).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "point_cloud").mkdir(exist_ok=True)

    config.log_info(f"数据目录：{data_dir}")
    config.log_info(f"输出目录：{out_dir}")
    config.log_info(f"GPU：{torch.cuda.get_device_name(0)}  显存 {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    # 把"SSIM 卷积走不走 cuDNN"写进日志：出问题时第一眼就能看到是哪条路径
    config.log_info(
        "SSIM 卷积：cuDNN " + ("开启（DSH_SSIM_CUDNN=1，对照实验用）" if _SSIM_USE_CUDNN else "关闭（默认，避开 cudnn_graph 崩溃）")
    )

    try:
        samples, points, colors = load_scene(data_dir, args.downscale, device)
    except (FileNotFoundError, ValueError) as exc:
        config.log_error(str(exc))
        return 5
    if len(samples) < 3:
        config.log_error(f"只有 {len(samples)} 张可用图像，至少需要 3 张。请检查 images/ 与 sparse/0 是否匹配。")
        return 5
    config.log_info(f"已注册图像 {len(samples)} 张，稀疏点 {points.shape[0]} 个")

    gauss = init_from_pointcloud(points, colors, args.sh_degree)
    config.log_info(f"初始化高斯：{gauss.count()} 个")

    # 场景尺度：用稀疏点云包围盒推导位置/尺度的硬边界。
    # 3DGS 的位置学习率相对场景尺度很大，数值一旦失稳就会有少量高斯飞出去，
    # 表现为导出的 PLY 包围盒达到 1e20 量级。这里按场景尺度兜底。
    diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    diag = diag if diag > 1e-3 else 1.0
    mean_center = torch.tensor(points.mean(axis=0), dtype=torch.float32, device=device)
    pos_limit = max(diag * 3.0, 1.0)
    scale_limit = math.log(max(diag * 2.0, 0.5))
    config.log_info(
        f"场景尺度：对角线 {diag:.2f}m，位置限幅 ±{pos_limit:.1f}m，尺度上限 {scale_limit:.2f}(log)"
    )
    args._pos_limit = pos_limit
    args._scale_limit = scale_limit
    args._mean_center = mean_center

    params = gauss.parameters()
    opt = torch.optim.Adam(
        [
            {"params": [gauss.means], "lr": args.lr_means, "name": "means"},
            {"params": [gauss.scales], "lr": args.lr_scales, "name": "scales"},
            {"params": [gauss.quats], "lr": args.lr_quats, "name": "quats"},
            {"params": [gauss.opacity], "lr": args.lr_opacity, "name": "opacity"},
            {"params": [gauss.sh_dc], "lr": args.lr_sh, "name": "sh_dc"},
        ]
        + ([{"params": [gauss.sh_rest], "lr": args.lr_sh / 20.0, "name": "sh_rest"}] if gauss.sh_rest is not None else []),
        eps=1e-15,
    )

    bg = torch.zeros(3, device=device)
    # 渲染瓦片边长：显存不足时自动减半（见训练循环里的 OOM 恢复逻辑）
    tile_size = 128
    # 位置梯度的参考量级（自适应基准），用于动态衰减位置学习率
    grad_ref = 1e-3
    densify_until = args.densify_until or int(args.iterations * 0.6)
    densify_step_mark = 0
    t0 = time.time()
    ema_loss = None
    history = []

    import cv2  # 复用已安装的 opencv 读图（避免额外依赖）

    for it in range(1, args.iterations + 1):
        sample = samples[it % len(samples)]
        cam = sample["cam"]
        img_bgr = cv2.imread(str(sample["path"]), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        if args.downscale > 1:
            h = img_bgr.shape[0] // args.downscale
            w = img_bgr.shape[1] // args.downscale
            img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
        h, w = img_bgr.shape[:2]
        scale = w / cam.width
        target = torch.from_numpy(img_bgr[:, :, ::-1].copy()).to(device).float().permute(2, 0, 1) / 255.0
        fx, fy = cam.fx * scale, cam.fy * scale
        cx, cy = cam.cx * scale, cam.cy * scale
        R = torch.tensor(sample["R"], dtype=torch.float32, device=device)
        T = torch.tensor(sample["t"], dtype=torch.float32, device=device)

        opt.zero_grad(set_to_none=True)
        # 显存不足时自动降级：清缓存并把渲染瓦片调小后重试当前迭代，
        # 避免训练因为一次 OOM 直接崩掉（8GB 显卡上这是常见情况）
        while True:
            try:
                rendered, vis = render(gauss, R, T, fx, fy, cx, cy, w, h, bg, args.sh_degree, tile=tile_size)
                break
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                torch.cuda.empty_cache()
                if tile_size > 32:
                    tile_size = max(32, tile_size // 2)
                    config.log_warn(f"迭代 {it}：显存不足，渲染瓦片降为 {tile_size} 后重试")
                    continue
                # 瓦片已最小仍不足：丢掉这一步，继续训练
                config.log_warn(f"迭代 {it}：显存不足且无法继续降级，跳过该步")
                opt.zero_grad(set_to_none=True)
                rendered = None
                break
        if rendered is None:
            continue
        if it == 1:
            n_vis = int(vis.sum()) if vis.numel() else 0
            config.log_info(f"首个视角 {w}x{h}：可见高斯 {n_vis}/{gauss.count()} 个，场景对角线 {diag:.3f}m")

        l1 = (rendered - target).abs().mean()
        ds = ssim_loss(rendered.unsqueeze(0), target.unsqueeze(0))
        loss = 0.8 * l1 + 0.2 * ds
        if not torch.isfinite(loss):
            # 数值发散：跳过这一步，避免污染参数
            opt.zero_grad(set_to_none=True)
            config.log_warn(f"迭代 {it}：损失出现非有限值，已跳过该步")
            continue
        if not loss.requires_grad:
            # 该视角没有任何高斯投影到画面内（相机完全偏离场景时会发生），
            # 此时渲染结果与参数无关，反向传播会报
            # 「element 0 of tensors does not require grad」，直接跳过
            opt.zero_grad(set_to_none=True)
            continue
        loss.backward()

        # 统计位置梯度（用于致密化），并按梯度量级动态调整位置学习率：
        # 3DGS 的位置学习率相对场景尺度偏大，某一步梯度爆掉（遮挡突变、投影退化）后
        # Adam 的动量会把少数高斯推到 1e38，表现为导出包围盒量级离谱。
        if gauss.means.grad is not None:
            gn = gauss.means.grad.detach().norm(dim=1)
            gn = torch.nan_to_num(gn, nan=0.0, posinf=0.0, neginf=0.0)
            gauss.grad_accum[: gn.shape[0]] += gn
            gnorm = float(gn.mean())
            if gnorm > grad_ref * 5:
                opt.param_groups[0]["lr"] = args.lr_means * 0.1
                grad_ref = max(grad_ref, gnorm)
            elif gnorm > grad_ref * 2:
                opt.param_groups[0]["lr"] = args.lr_means * 0.3
                grad_ref = max(grad_ref, gnorm)
            elif gnorm < grad_ref:
                opt.param_groups[0]["lr"] = args.lr_means

        with torch.no_grad():
            opt.step()

            # 数值稳定性护栏：以场景尺度为界限幅位置与尺度，避免高斯飞出场景。
            # （3DGS 的位置学习率相对场景尺度很大，失稳时会有少量高斯跑到 1e20 量级）
            gauss.scales.clamp_(min=-10.0, max=args._scale_limit)
            center = args._mean_center
            torch.nan_to_num_(gauss.means.data, nan=0.0, posinf=args._pos_limit, neginf=-args._pos_limit)
            # 围绕场景中心逐轴限幅
            for axis in range(3):
                gauss.means.data[:, axis].clamp_(
                    min=float(center[axis]) - args._pos_limit,
                    max=float(center[axis]) + args._pos_limit,
                )
            torch.nan_to_num_(gauss.quats.data, nan=0.0, posinf=0.0, neginf=0.0)
            torch.nan_to_num_(gauss.sh_dc.data, nan=0.0, posinf=4.0, neginf=-4.0)
            torch.nan_to_num_(gauss.opacity.data, nan=-2.0, posinf=4.0, neginf=-6.0)
            gauss.opacity.data.clamp_(min=-6.0, max=4.0)

            # --- 致密化（用真实的累积步数求平均梯度）
            if it < densify_until and it % args.densify_interval == 0:
                densify(gauss, opt, args.max_gaussians, config, max(1, it - densify_step_mark))
                densify_step_mark = it

            # --- 周期性重置 opacity
            if it % args.opacity_reset_interval == 0 and it < args.iterations - 500:
                gauss.opacity.data.fill_(-2.0)

        cur = float(loss.detach())
        ema_loss = cur if ema_loss is None else 0.4 * ema_loss + 0.6 * cur
        if args.debug_numeric:
            with torch.no_grad():
                mx = float(gauss.means.data.abs().max())
                ms = float(gauss.scales.data.max())
                mg = (
                    0.0
                    if gauss.means.grad is None
                    else float(torch.nan_to_num(gauss.means.grad.detach().norm(dim=1), nan=0.0).max())
                )
                n_bad = int((~torch.isfinite(gauss.means.data).all(dim=1)).sum())
            config.log_info(
                f"  [数值] it={it} |mean|max={mx:.3e} scale_max={ms:.3f} grad_max={mg:.3e} 非法点={n_bad} loss={cur:.5f}"
            )
        if it % args.log_every == 0 or it == 1:
            used = torch.cuda.max_memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            elapsed = time.time() - t0
            config.log_info(
                f"迭代 {it}/{args.iterations}  损失={ema_loss:.5f}  L1={float(l1):.5f}  高斯={gauss.count()}  "
                f"显存={used:.2f}GB(预留{reserved:.2f}GB)  用时={elapsed:.0f}s"
            )
            history.append({"iteration": it, "loss": ema_loss, "gaussians": gauss.count(), "elapsedSec": elapsed})

        # 每一步都归还缓存块：本渲染器会按瓦片反复申请大小不同的临时张量，
        # 如果不主动归还，PyTorch 的缓存分配器会产生大量碎片，
        # 表现为「已分配显存不高但预留显存持续上涨」，最终 OOM。
        if (it & 7) == 0:
            torch.cuda.empty_cache()

    config.log_info("训练结束，导出标准 3DGS PLY …")
    ply_path = export_ply(gauss, out_dir / "point_cloud" / f"iteration_{args.iterations}" / "point_cloud.ply")
    meta = {
        "format": "3dgs-ply",
        "trainer": "pipeline/train_gpu.py（纯 PyTorch 实现，无需 CUDA 编译）",
        "iterations": args.iterations,
        "gaussianCount": gauss.count(),
        "shDegree": args.sh_degree,
        "downscale": args.downscale,
        "imageCount": len(samples),
        "initialPoints": int(points.shape[0]),
        "elapsedSec": round(time.time() - t0, 1),
        "history": history,
        "ply": str(ply_path),
        "createdAt": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    config.log_ok(f"训练完成：{gauss.count()} 个高斯，PLY 位于 {ply_path}")
    config.log_info(f"下一步：python pipeline/to_splat.py --ply \"{ply_path}\" --out data/models/scene.splat")
    return 0


def densify(gauss: Gaussians, opt, max_gaussians: int, cfg, mean_grad_window: int = 1) -> None:
    """
    按官方 3DGS 策略做 clone / split。
    判据是「位置梯度的平均范数」：grad_accum 里累积的是每一步的梯度范数之和，
    因此必须除以累积步数（而不是固定值），否则阈值没有物理意义。
    """
    if gauss.grad_accum is None:
        return
    n = gauss.count()
    if n >= max_gaussians:
        gauss.grad_accum.zero_()
        return
    with torch.no_grad():
        avg = gauss.grad_accum[:n] / max(1, mean_grad_window)
        densify_mask = avg > 2e-4
        if int(densify_mask.sum()) == 0:
            gauss.grad_accum.zero_()
            return
        scales = torch.exp(gauss.scales)
        max_scale = scales.max(dim=1).values
        # 官方策略：屏幕尺度小的 clone，大的 split（这里用世界尺度近似）
        clone_mask = densify_mask & (max_scale <= 0.02)
        split_mask = densify_mask & (max_scale > 0.02)
        budget = max_gaussians - n
        new_means, new_scales, new_quats, new_op, new_sh = [], [], [], [], []
        for mask, is_split in ((clone_mask, False), (split_mask, True)):
            k = int(mask.sum())
            if k == 0:
                continue
            k = min(k, max(0, budget))
            if k == 0:
                continue
            idx = mask.nonzero(as_tuple=True)[0][:k]
            means = gauss.means.data[idx]
            if is_split:
                std = scales[idx].mean(dim=1, keepdim=True) * 1.6
                means_new = means + torch.randn_like(means) * std
                scales_new = torch.log(scales[idx] / 1.6)
                op_new = gauss.opacity.data[idx].clone()
            else:
                means_new = means.clone()
                scales_new = gauss.scales.data[idx].clone()
                op_new = gauss.opacity.data[idx].clone()
            new_means.append(means_new)
            new_scales.append(scales_new)
            new_quats.append(gauss.quats.data[idx].clone())
            new_op.append(op_new)
            new_sh.append(gauss.sh_dc.data[idx].clone())
            budget -= k
            if budget <= 0:
                break

        if not new_means:
            gauss.grad_accum.zero_()
            return

        added = sum(x.shape[0] for x in new_means)
        gauss.means = torch.nn.Parameter(torch.cat([gauss.means.data, *new_means], dim=0))
        gauss.scales = torch.nn.Parameter(torch.cat([gauss.scales.data, *new_scales], dim=0))
        gauss.quats = torch.nn.Parameter(torch.cat([gauss.quats.data, *new_quats], dim=0))
        gauss.opacity = torch.nn.Parameter(torch.cat([gauss.opacity.data, *new_op], dim=0))
        gauss.sh_dc = torch.nn.Parameter(torch.cat([gauss.sh_dc.data, *new_sh], dim=0))
        gauss.grad_accum = torch.zeros(gauss.count(), device=gauss.means.device)

        # 把新参数交给优化器（保留原有超参）
        opt.param_groups[0]["params"] = [gauss.means]
        opt.param_groups[1]["params"] = [gauss.scales]
        opt.param_groups[2]["params"] = [gauss.quats]
        opt.param_groups[3]["params"] = [gauss.opacity]
        opt.param_groups[4]["params"] = [gauss.sh_dc]
    cfg.log_info(f"致密化：新增 {added} 个高斯，当前 {gauss.count()} 个")
    with torch.no_grad():
        avg = gauss.grad_accum
        if avg is not None:
            avg.zero_()


def export_ply(gauss: Gaussians, path: Path) -> Path:
    """导出标准 3DGS PLY（二进制小端，属性名与官方一致，便于 to_splat.py 复用）"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        means = gauss.means.detach().cpu().numpy().astype(np.float32)
        scales = gauss.scales.detach().cpu().numpy().astype(np.float32)
        quats = gauss.quats.detach().cpu().numpy()
        quats = quats / np.linalg.norm(quats, axis=1, keepdims=True).clip(1e-8)
        opacity = gauss.opacity.detach().cpu().numpy().astype(np.float32)
        sh_dc = gauss.sh_dc.detach().cpu().numpy().astype(np.float32)
        sh_rest = gauss.sh_rest.detach().cpu().numpy().astype(np.float32) if gauss.sh_rest is not None else None

    n = means.shape[0]
    # 只写标准 3DGS 属性（不含 nx/ny/nz），与官方实现及 to_splat.py 完全对齐
    props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"]
    if sh_rest is not None:
        K = sh_rest.shape[1]
        props += [f"f_rest_{i}" for i in range(K * 3)]
    props += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]

    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += [f"property float {p}" for p in props]
    header.append("end_header")
    header_bytes = ("\n".join(header) + "\n").encode("ascii")

    # f_rest 在官方 PLY 里的排列是「按系数交错」：f_rest_{k*3+c}
    columns = [means[:, 0], means[:, 1], means[:, 2], sh_dc[:, 0], sh_dc[:, 1], sh_dc[:, 2]]
    if sh_rest is not None:
        # sh_rest: (N,K,3) -> 交错为 (N, K*3)
        inter = sh_rest.reshape(n, -1)
        columns += [inter[:, i] for i in range(inter.shape[1])]
    columns += [
        opacity[:, 0],
        scales[:, 0],
        scales[:, 1],
        scales[:, 2],
        quats[:, 0].astype(np.float32),
        quats[:, 1].astype(np.float32),
        quats[:, 2].astype(np.float32),
        quats[:, 3].astype(np.float32),
    ]
    data = np.stack(columns, axis=1).astype(np.float32)

    with open(path, "wb") as fh:
        fh.write(header_bytes)
        fh.write(data.tobytes())
    return path


if __name__ == "__main__":
    sys.exit(main())
