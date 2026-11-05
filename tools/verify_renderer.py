#!/usr/bin/env python3
"""
tools/verify_renderer.py —— 用软件光栅化复刻 WebGL 着色器的数学，验证「浏览器端渲染路径」是否正确

为什么需要它：
    我们的 3DGS 渲染逻辑在 GLSL 里（app/web/js/splat-renderer.js）。没有浏览器就无法直接验证。
    本脚本用 NumPy 按**逐行对应**的方式复算同一套公式：
      1. 世界系 -> 相机系（COLMAP 约定，相机看 +z）
      2. 3D 协方差 Σ = M Mᵀ，M = R·S
      3. 相机系协方差 Σ' = W Σ Wᵀ，再用 Jacobian J 得到 2D 协方差
      4. conic = 逆矩阵，alpha = min(255)/255，exp(-0.5·二次型)
      5. 按视深度远→近排序，预乘 alpha 合成
    同时按 **Three.js 的换算**把场景坐标转过去，因此结果等价于「浏览器里那个视角看到的东西」。

用法：
    python tools/verify_renderer.py --splat data/models/sample-room.splat \
        --out output/verify/render.png [--camera-tag default] [--width 800] [--height 600] \
        [--compare data/scene-synth/images/view_004.png --compare-tag v4]
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

# ---------------------------------------------------------------- .splat 读取
# 与 app/web/js/splat-loader.js 的 parseSplat 完全一致：
#   0..11  float32 x,y,z
#   12..23 float32 scaleX,scaleY,scaleZ（线性尺度）
#   24..27 uint8   R,G,B,A（A 为 sigmoid(opacity)*255）
#   28..31 uint8   四元数 (w,x,y,z)，编码为 round(q*128+128)
SH_C0 = 0.28209479177387814


def load_splat(path: Path):
    raw = path.read_bytes()
    n = len(raw) // 32
    buf = np.frombuffer(raw[: n * 32], dtype=np.uint8).reshape(n, 32)
    f32 = buf[:, :24].copy().view(np.float32).reshape(n, 6)
    pos = f32[:, 0:3].astype(np.float64)
    lin_scale = f32[:, 3:6].astype(np.float64)
    rgba = buf[:, 24:28].astype(np.float64)
    quat = (buf[:, 28:32].astype(np.float64) - 128.0) / 128.0
    return pos, lin_scale, quat, rgba


def decode_color(rgba):
    """与着色器 computeColor 一致：linear = 0.5 + SH_C0 * (v/255)"""
    rgb = rgba[:, :3] / 255.0
    return np.clip(0.5 + SH_C0 * rgb, 0.0, 1.0)


def quat_to_matrix(q):
    """(w,x,y,z) -> 旋转矩阵，行优先 (n,3,3)；与着色器里的 R 构造一致"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = q.shape[0]
    R = np.empty((n, 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


# ---------------------------------------------------------------- 坐标换算

def colmap_to_scene_matrix():
    """
    复刻 app/web/js/camera-overlay.js 的 WORLD_MATRIX = R_x(-90°)。
    实测作用：(x,y,z)_local -> (x, z, -y)_scene
    """
    m = np.array(
        [
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0],
        ],
        dtype=np.float64,
    )
    return m


def make_camera_look_at(eye, target, up_scene):
    """在场景系里构造相机的旋转（Three.js 约定：相机朝 -z，上为 +y）"""
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up_scene)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    # 世界 -> 相机 的旋转（行 = 相机轴）
    R_wc = np.stack([r, u, -f], axis=0)
    return R_wc


def render(
    pos_local: np.ndarray,
    lin_scale: np.ndarray,
    quat: np.ndarray,
    color: np.ndarray,
    alpha: np.ndarray,
    R_cam: np.ndarray,
    C_scene: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    bg=(0.0, 0.0, 0.0),
    scale_mult: float = 1.0,
    opacity_mult: float = 1.0,
):
    """
    按 WebGL 着色器的公式做 CPU 光栅化。
    R_cam 是「世界(场景系) -> 相机」的旋转（行 = 相机轴），相机看向 +z（本函数内部约定）。
    """
    W = colmap_to_scene_matrix()
    # 位置换算到场景系
    pos_scene = pos_local @ W.T
    # 再到相机系
    cam = (pos_scene - C_scene) @ R_cam.T  # x_cam
    z = cam[:, 2]
    vis = z > 0.05
    idx = np.nonzero(vis)[0]
    if idx.size == 0:
        return np.zeros((height, width, 3), dtype=np.float64), 0

    cam_v = cam[idx]
    z = cam_v[:, 2]
    x = fx * cam_v[:, 0] / z + cx
    y = fy * cam_v[:, 1] / z + cy

    # 3D 协方差：Σ = W Σ_local Wᵀ（W 为场景->相机旋转），M = R·S
    Rq = quat_to_matrix(quat[idx])
    S = np.zeros((idx.size, 3, 3), dtype=np.float64)
    for k in range(3):
        S[:, k, k] = lin_scale[idx, k] * scale_mult
    M = Rq @ S
    Sigma_local = M @ M.transpose(0, 2, 1)
    Sigma_cam = R_cam @ Sigma_local @ R_cam.T  # 广播：(3,3) @ (n,3,3) @ (3,3)

    # Jacobian 投影
    inv_z = 1.0 / z
    J = np.zeros((idx.size, 3, 3), dtype=np.float64)
    J[:, 0, 0] = fx * inv_z
    J[:, 0, 2] = -fx * cam_v[:, 0] * inv_z * inv_z
    J[:, 1, 1] = fy * inv_z
    J[:, 1, 2] = -fy * cam_v[:, 1] * inv_z * inv_z
    cov2 = J @ Sigma_cam @ J.transpose(0, 2, 1)

    a = cov2[:, 0, 0] + 0.3
    b = cov2[:, 0, 1]
    c = cov2[:, 1, 1] + 0.3
    det = np.maximum(a * c - b * b, 1e-8)
    conic_a = c / det
    conic_b = -b / det
    conic_c = a / det

    mid = 0.5 * (a + c)
    radius = np.clip(3.0 * np.sqrt(np.maximum(mid, 0.0)), 0.6, 320.0)

    col = color[idx]
    alp = np.clip(alpha[idx, 3] / 255.0 * opacity_mult, 0.0, 1.0)

    # 远 -> 近（与渲染器一致：先画远的）
    order = np.argsort(-z)

    img = np.zeros((height, width, 3), dtype=np.float64)
    img[:, :] = np.array(bg, dtype=np.float64)

    # 简化：不做瓦片，直接按屏幕包围盒处理（CPU 验证够用）
    for k in order:
        cx0 = x[k]
        cy0 = y[k]
        rad = radius[k]
        if rad < 0.6:
            continue
        x0 = max(0, int(math.floor(cx0 - rad)))
        x1 = min(width, int(math.ceil(cx0 + rad)) + 1)
        y0 = max(0, int(math.floor(cy0 - rad)))
        y1 = min(height, int(math.ceil(cy0 + rad)) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx = xx + 0.5 - cx0
        dy = yy + 0.5 - cy0
        power = -2.0 * (conic_a[k] * dx * dx + conic_c[k] * dy * dy) - 2.0 * conic_b[k] * dx * dy
        g = np.where(power > 0.0, 0.0, np.exp(power))
        a_i = np.clip(alp[k] * g, 0.0, 1.0)
        if a_i.max() < 1.0 / 255.0:
            continue
        tile = img[y0:y1, x0:x1]
        # 预乘 alpha 合成（与 blendFuncSeparate(ONE, ONE_MINUS_SRC_ALPHA) 等价）
        img[y0:y1, x0:x1] = tile * (1.0 - a_i)[..., None] + col[k][None, None, :] * a_i[..., None]

    return np.clip(img, 0.0, 1.0), int(idx.size)


def main() -> int:
    ap = argparse.ArgumentParser(description="用 CPU 复刻 WebGL 渲染路径，验证 .splat 与相机换算")
    ap.add_argument("--splat", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=55.0)
    ap.add_argument("--eye", type=float, nargs=3, default=None, help="场景系相机位置")
    ap.add_argument("--target", type=float, nargs=3, default=None, help="场景系注视点")
    ap.add_argument("--compare", default=None, help="参考图（用于对比同一视角的合成数据集图像）")
    ap.add_argument("--compare-view-file", default=None, help="合成数据集的 dataset.json（读取内参）")
    ap.add_argument("--from-colmap-view", type=int, default=None, help="用合成数据集里第 N 个相机的真实位姿渲染（与参考图严格同一视角）")
    ap.add_argument("--colmap-scene", default=None, help="含 sparse/0/images.bin 的场景目录")
    ap.add_argument("--scale-mult", type=float, default=1.0)
    ap.add_argument("--opacity-mult", type=float, default=1.0)
    args = ap.parse_args()

    if cv2 is None:
        print("需要 opencv-python：pyruntime312\\python.exe -m pip install opencv-python")
        return 3

    splat = Path(args.splat)
    if not splat.exists():
        print(f"找不到模型：{splat}")
        return 2
    pos, scale, quat, rgba = load_splat(splat)
    color = decode_color(rgba)
    print(f"载入 {splat.name}：{pos.shape[0]} 个高斯")

    # 场景系包围盒，用于自动取景
    W = colmap_to_scene_matrix()
    pos_scene = pos @ W.T
    center = (pos_scene.max(axis=0) + pos_scene.min(axis=0)) / 2
    radius = float(np.linalg.norm(pos_scene.max(axis=0) - pos_scene.min(axis=0))) / 2

    eye = np.array(args.eye, dtype=np.float64) if args.eye else center + np.array([radius * 0.9, radius * 0.55, radius * 0.9])
    target = np.array(args.target, dtype=np.float64) if args.target else center
    up = np.array([0.0, 1.0, 0.0])

    # 可选：直接采用 COLMAP 里某个相机的真实位姿（与参考图严格同视角）
    if args.from_colmap_view is not None and args.colmap_scene:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
        import train_gpu as TG  # 复用 COLMAP 二进制读取器

        scene = Path(args.colmap_scene)
        cams = TG.read_cameras_binary(scene / "sparse" / "0" / "cameras.bin")
        imgs = TG.read_images_binary(scene / "sparse" / "0" / "images.bin")
        ordered = sorted(imgs.values(), key=lambda im: im.name)
        cam_entry = ordered[args.from_colmap_view]
        cam = cams[cam_entry.camera_id]
        R_cv = cam_entry.rotation()  # COLMAP: x_cam = R·X_local + t
        t_cv = cam_entry.tvec
        # 相机中心（local 系） -> 场景系
        C_local = -R_cv.T @ t_cv
        eye = C_local @ colmap_to_scene_matrix().T
        # 观察方向：local 系下 +z 轴，转到场景系
        dir_local = R_cv[2, :]
        dir_scene = dir_local @ colmap_to_scene_matrix().T
        target = eye + dir_scene
        # 直接用真实旋转，避免再由 look-at 反推带来误差
        R_wc_three = np.stack([R_cv[0, :], R_cv[1, :], R_cv[2, :]], axis=0) @ colmap_to_scene_matrix().T
        fx = fy = cam.fx
        cx, cy = cam.cx, cam.cy
        print(f"采用 COLMAP 相机 #{args.from_colmap_view}（{cam_entry.name}）")
        print(f"  相机中心(local) {np.round(C_local, 4)}  场景系 {np.round(eye, 4)}")
        print(f"  内参 fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}  原始分辨率 {cam.width}x{cam.height}")
    else:
        fx = None

    # WebGL 相机朝 -z，而我们的光栅函数用 +z 约定，因此把 R 的第 3 行取反
    if fx is None:
        R_wc_threejs = make_camera_look_at(eye, target, up)
        R_cam = R_wc_threejs.copy()
        R_cam[2, :] *= -1.0
        fov = math.radians(args.fov)
        fy = 0.5 * args.height / math.tan(fov / 2)
        fx = fy
        cx = args.width / 2
        cy = args.height / 2
    else:
        R_cam = R_wc_three.copy()
        R_cam[2, :] *= -1.0
        # 若请求分辨率与原始不同，按比例缩放内参
        scale_x = args.width / cam.width
        scale_y = args.height / cam.height
        fx *= scale_x
        fy *= scale_y
        cx *= scale_x
        cy *= scale_y

    print(f"相机位置 {np.round(eye, 3)}  注视 {np.round(target, 3)}")
    print(f"内参 fx=fy={fx:.1f} cx={cx} cy={cy}  分辨率 {args.width}x{args.height}")

    img, n_vis = render(
        pos, scale, quat, color, rgba, R_cam, eye, fx, fy, cx, cy,
        args.width, args.height, scale_mult=args.scale_mult, opacity_mult=args.opacity_mult,
    )
    print(f"可见高斯 {n_vis}/{pos.shape[0]}  非背景像素占比 {(img.sum(axis=2) > 0.02).mean() * 100:.1f}%")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), (img[:, :, ::-1] * 255).astype(np.uint8))
    print(f"已写出 {out}")

    # 若提供了合成数据集的参考图，用同一台相机再渲染一次并对比
    if args.compare and args.compare_view_file:
        dsinfo = json.loads(Path(args.compare_view_file).read_text(encoding="utf-8"))
        print(f"\n参考图：{args.compare}")
        ref = cv2.imread(args.compare, cv2.IMREAD_COLOR)
        if ref is None:
            print("  参考图读取失败")
            return 1
        print(f"  参考图尺寸 {ref.shape[1]}x{ref.shape[0]}，数据集内参 fx={dsinfo.get('fx')}")
        if ref.shape[1] != args.width or ref.shape[0] != args.height:
            ref = cv2.resize(ref, (args.width, args.height))
        a = img[:, :, ::-1]
        b = ref.astype(np.float64) / 255.0
        mae = float(np.abs(a - b).mean())
        print(f"  与参考图的平均绝对误差（MAE）= {mae:.4f}（0~1，越小越接近）")
        diff = np.abs(a - b)
        cv2.imwrite(str(out.with_name(out.stem + "-ref.png")), ref)
        cv2.imwrite(str(out.with_name(out.stem + "-diff.png")), np.clip(diff * 3, 0, 1)[:, :, ::-1] * 255)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
