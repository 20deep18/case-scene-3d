# 三维重建流水线（pipeline/）

把「手机 / 执法记录仪拍的视频」变成浏览器可直接加载的 **3D Gaussian Splatting（.splat）** 场景。

流水线一共五步，全部是命令行脚本，任何一步缺依赖都会给出**中文的下一步安装指引**，不会直接崩溃：

```
视频.mp4
   │  ① 抽帧 + 模糊过滤      pipeline/extract_frames.py
   ▼
frames/frame_0001.png …      + frames.json
   │  ② 相机位姿重建 SfM      pipeline/colmap_recon.py
   ▼
colmap/sparse/0/{cameras,images,points3D}.bin  + colmap_summary.json
   │  ③ 高斯训练              pipeline/train_gsplat.py
   ▼
train/point_cloud.ply
   │  ④ 导出 Web 格式         pipeline/to_splat.py
   ▼
data/models/<场景名>.splat   （浏览器通过 /models/<场景名>.splat 直接加载）
```

另外两个脚本：

| 脚本 | 用途 |
| --- | --- |
| `pipeline/config.py` | 公共配置：路径常量、依赖探测（numpy/cv2/torch/gsplat/pycolmap/colmap/ffmpeg）、中文日志 |
| `pipeline/make_test_scene.py` | **不需要任何第三方库**的合成房间场景，用于在真实数据到位前验证整条链路与前端显示 |
| `pipeline/setup_env.py` | 环境自检表（✅/❌/⚠️）与安装引导（含 RTX 5060 / sm_120 专项说明） |
| `tests/test_pipeline.py` | 不依赖第三方库的自测脚本，覆盖 PLY→splat、合成场景、模糊算法、COLMAP 二进制解析 |

---

## 0. 先跑一遍环境自检

```powershell
python pipeline/setup_env.py --check      # 只检查，不安装
python pipeline/setup_env.py              # 打印安装计划（wheels/ 里有离线轮子时会给出离线安装命令）
python pipeline/setup_env.py --yes        # 用 wheels/ 离线安装轻量依赖（numpy / imageio / opencv / pillow / plyfile / tqdm）
```

自检表会逐项告诉你还缺什么、下一条命令是什么。**本机实测约束（很重要）**：

* 显卡是 **RTX 5060 Laptop（Blackwell 架构，算力 sm_120）**，必须使用 **CUDA ≥ 12.8 的 PyTorch**，即
  `torch >= 2.7` 的 cu128 版本，否则会报 `no kernel image is available for execution on the device`：
  ```powershell
  python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
  ```
  本仓库 `wheels/` 里已经放了 `torch-2.7.1+cu128-cp312-cp312-win_amd64.whl`；
  注意它是 **cp312**，需要 Python 3.12 环境。
  同目录下的 `torch-2.4.0+cu124-cp310` 是 cu124 版本，**在 sm_120 上很可能不可用**，`setup_env.py --check` 会对它给出 ⚠️ 提示。
* **gsplat 官方只提供 Linux 预编译轮子**，Windows 上要么源码编译（需要 MSVC 生成工具 + CUDA Toolkit ≥ 12.8 + `CUDA_HOME`）：
  ```powershell
  python -m pip install git+https://github.com/nerfstudio-project/gsplat.git
  ```
  要么使用 `wheels/` 里已有的 Windows 轮子（如 `gsplat-1.5.3+pt24cu124-cp310-win_amd64.whl`，
  它**绑定 torch 2.4 + cu124**，选它就必须用对应的 torch）。
* 没有 NVIDIA 显卡时，请直接看本文最后的「替代路线」。

---

## 1. 数据采集要点（决定成败的关键，比算法重要得多）

| 要点 | 建议 |
| --- | --- |
| 拍摄方式 | **边走边拍**，横向平移为主；不要原地转头（纯旋转无法三角化） |
| 重叠率 | 相邻帧画面位移 ≤ 画面宽度的 30%；宁可多拍，抽帧可以再降 |
| 路径 | 绕着目标走 2~3 圈，一圈平视、一圈稍高、一圈稍低；重要物体至少被 3 个方向看到 |
| 速度 | 慢。每走一步停 0.3 秒，避免运动模糊（模糊帧会被抽帧脚本丢掉） |
| 曝光 | 关闭自动曝光剧烈变化、关闭美颜/防抖裁切；固定对焦（锁定 AE/AF）更好 |
| 场景 | 避免大面积纯白墙 / 玻璃 / 镜面 / 强反光；必要时贴临时标记纸 |
| 光照 | 尽量均匀，避免逆光与过曝；暗部细节不足会导致特征点匹配失败 |
| 视频参数 | 1080p 30fps 足够；4K 更清楚但抽帧更慢；不要用超广角边缘畸变过大的镜头 |
| 时长 | 一个房间 1~3 分钟即可（抽帧后 150~400 张） |

> 拍摄前先在现场放一把已知长度的尺子或标记物，重建后可以用它标定真实尺度（本流水线输出的单位按 COLMAP 的任意尺度，
> 若需要米制，可在导出后用已知长度换算 `--max-scale` 等阈值）。

---

## 2. 逐步命令

### ① 抽帧 + 模糊过滤

```powershell
python pipeline/extract_frames.py --video data\projects\case01\source.mp4 --out data\projects\case01\frames --fps 2
```

参数：`--fps 2`（每秒 2 帧）、`--long-side 1600`（长边上限）、`--max-frames 400`、`--blur-threshold 60`。

产物：`frames/frame_0001.png …` + `frames/frames.json`（记录保留/剔除数量、帧列表、模糊分数区间）。
被判为模糊的帧移到 `frames/_rejected/`，可以自己看一眼再决定要不要放回去。

### ② 相机位姿重建（COLMAP）

```powershell
python pipeline/colmap_recon.py --images data\projects\case01\frames --workspace data\projects\case01\colmap
```

参数：`--camera-model OPENCV`（手机视频默认用这个）、`--matcher auto|exhaustive|sequential`、`--use-gpu 1|0`、`--dry-run`。

* 有 `pycolmap` 时优先用 Python 接口，否则调用 `colmap` 命令行；两者都没有时会打印安装指引并退出码 4。
* `auto` 策略：文件名像 `frame_XXXX`（视频抽帧）或图片超过 150 张 → `sequential`，否则 `exhaustive`。
* 产物：`colmap/sparse/0/{cameras,images,points3D}.bin` + `colmap/colmap_summary.json`。
* **注册图片数 < 60% 或 < 8 张时退出码为 2**，并在 stderr 打印可直接执行的补救建议
  （提高重叠率 / 换 OPENCV 模型 / 开 guided matching / 用 sequential_matcher / 先提高模糊阈值 等）。

### ③ 高斯训练

```powershell
python pipeline/train_gsplat.py --data data\projects\case01 --out data\projects\case01\train --images data\projects\case01\frames
```

参数：`--iterations 15000`、`--sh-degree 3`、`--downscale 2`、`--max-gaussians 1500000`、`--save-every 5000`。

* 数据目录会自动探测：`<data>/images`（或 `frames`）+ `<data>/sparse/0`，也支持 `<data>/colmap/sparse/0`。
* COLMAP 的 `cameras.bin / images.bin / points3D.bin` 由本仓库**自己实现的二进制解析器**读取，**不需要 pycolmap**。
* 训练中每 `--log-every`（默认 100）次迭代向 stdout 打印一行，便于后端捕获展示：
  ```
  [12:03:44] [信息] [训练] iteration=800/15000 loss=0.041233 gaussians=120345 elapsed=63.21s sh_degree=0
  ```
* 显存不足会优雅退出（退出码 4）并给出 `--downscale`、`--max-gaussians` 的具体建议值。
* **没有 torch/gsplat 时退出码 3**，但会先把 COLMAP 场景解析结果打印出来（说明数据本身没问题）。
* 产物：`train/point_cloud.ply`（标准 3DGS 二进制小端，属性 `x,y,z,nx,ny,nz,f_dc_*,f_rest_*,opacity,scale_*,rot_*`）。

### ④ 导出 Web 格式（.splat）

```powershell
python pipeline/to_splat.py --ply data\projects\case01\train\point_cloud.ply --out data\models\case01.splat --json data\models\case01.json
```

参数：`--sh-degree 0|1|2|3`、`--min-opacity 0.005`、`--max-scale 1.0`、`--decimate 1.0`。

* **只有标准库也能跑**（自己解析 PLY 头 + `struct` 解包）；有 numpy 时自动走向量化路径，
  并在启动时用合成数据校验两条路径逐字节一致，不一致就自动退回纯 Python 路径（结果永远正确）。
* 性能：100 万高斯约 **3.7 秒**（纯标准库、无 numpy，实测）。
* 同时会写 `--sh-degree > 0` 时的高阶球谐系数 `<同名>.shcoeff.bin`（float32 小端）。

### ⑤ 放进 `data/models/`，前端即可加载

后端把 `data/models/` 映射到 URL `/models/`：

```
data/models/case01.splat   →   http://127.0.0.1:19387/models/case01.splat
data/models/case01.json    →   元数据（count / shDegree / bounds / axes …），可用于自动摆放与缩放
```

调用 `POST /api/projects/<id>/model` 时把 `url` 设为 `/models/case01.splat` 即可（见后端 `app/server/store.js` 的 `setModel`）。

### ⑥ 不拍照也能验证：合成测试场景

```powershell
python pipeline/make_test_scene.py                     # 6 万高斯，写入 data/models/
python pipeline/make_test_scene.py --count 200000 --ply data\models\test-scene.ply
```

产物：

| 文件 | 内容 |
| --- | --- |
| `data/models/test-scene.splat` | 合成房间（地面 / 后墙 / 左墙 / 桌子 / 球 / 盒子 / 门框），每个面用各向异性高斯铺满 |
| `data/models/test-scene.json` | 元数据（高斯数量、包围盒、半径、坐标系、生成参数） |
| `data/models/test-scene-cameras.json` | 8 个环绕相机位姿（position / target / R / t / c2w / w2c），供后端做演示轨迹 |
| `data/models/test-scene-observations.json` | 6 个已知三维点在若干相机中的投影像素坐标 + 内参（fx,fy,cx,cy，1920×1080），供 PnP / 重投影测试 |

---

## 3. 目录结构

```
default-workspace/
├─ pipeline/                     ← 本流水线（只看这里就能跑通全流程）
│  ├─ config.py                  公共配置 / 依赖探测 / 中文日志
│  ├─ extract_frames.py          ① 抽帧 + 模糊过滤
│  ├─ colmap_recon.py            ② 相机位姿（含自实现的 COLMAP 二进制读写）
│  ├─ train_gsplat.py            ③ gsplat 训练 + PLY 导出
│  ├─ to_splat.py                ④ PLY → .splat（+ .shcoeff.bin + JSON）
│  ├─ make_test_scene.py         合成测试场景
│  ├─ setup_env.py               环境自检与安装引导
│  └─ README.md                  本文件
├─ tests/test_pipeline.py        自测（不依赖第三方库）
├─ data/
│  ├─ models/                    ← 训练好的 .splat 放这里，URL 前缀 /models/
│  └─ projects/<案件>/
│     ├─ source.mp4              原始视频
│     ├─ frames/                 抽帧结果（frame_0001.png、frames.json、_rejected/）
│     ├─ colmap/                 COLMAP 工作目录（database.db、sparse/0/、colmap_summary.json）
│     └─ train/                  训练输出（point_cloud.ply）
├─ wheels/                       离线轮子（torch / gsplat / numpy / opencv / imageio-ffmpeg …）
├─ tools/fetch-wheels.mjs        轮子下载器（pip 临时目录受限时用它）
└─ app/                          前端与后端（**本流水线不改动这里**）
```

---

## 4. .splat 文件格式（前端加载依据）

每个高斯固定 **32 字节**，小端：

| 偏移 | 类型 | 含义 |
| --- | --- | --- |
| 0..11 | float32 ×3 | 世界坐标 `x, y, z`（米） |
| 12..23 | float32 ×3 | **线性**尺度 `scaleX, scaleY, scaleZ`（即 `exp(scale_raw)`） |
| 24..27 | uint8 ×4 | `R, G, B, A`；RGB 由 SH 直流项 `clamp(0.5 + 0.28209479177387814 * f_dc, 0, 1) * 255`，A 由 `sigmoid(opacity) * 255` |
| 28..31 | uint8 ×4 | 四元数 **按 (w, x, y, z) 顺序**，单位化后 `round(q * 128 + 128)` 并钳制到 0..255 |

坐标系约定：**COLMAP / 3DGS 世界系，y 轴向下，单位米**（JSON 里的 `axes` 字段会写明）。
文件长度必须严格等于 `数量 × 32`；`--json` 元数据里的 `bounds.radius` 可直接用来设置相机的远裁剪面与初始距离。

`.shcoeff.bin`（可选）：float32 小端，每个高斯 `3 × K` 个系数，`K = (阶数+1)² - 1`；
排列是**通道主序**（先 R 的 K 个系数，再 G 的 K 个，再 B 的 K 个），与 3DGS 的 PLY 约定一致。

---

## 5. 常见报错与处理

| 现象 / 报错 | 原因 | 处理 |
| --- | --- | --- |
| `未找到 ffmpeg，无法抽帧` | 没装 ffmpeg | `pip install imageio-ffmpeg`（最省事），或 `winget install Gyan.FFmpeg`，或设置 `DSH_FFMPEG` |
| 抽帧结果为空（退出码 5） | 视频损坏 / 编码不支持 | 先用 `ffmpeg -i 视频` 确认能解码；转成 H.264 mp4 再试 |
| 所有帧都被判为模糊（退出码 6） | 阈值过高或拍摄太抖 | 降低 `--blur-threshold`（如 30），或看 `frames/_rejected/` 里的图 |
| `图片目录不存在或没有图片`（退出码 3） | 目录为空或路径写错 | 先跑第 ① 步；路径含空格/中文时用引号包起来 |
| `既没有 pycolmap 也没有 colmap 命令行`（退出码 4） | 两者都没装 | 装 COLMAP 官方 Windows 包并加入 PATH / 设置 `COLMAP_EXE`，或 `pip install pycolmap` |
| 注册图片过少（退出码 2） | 重叠率低 / 模糊 / 纹理少 / 纯旋转 | 见脚本在 stderr 打印的 8 条补救建议（提高 fps、`--camera-model OPENCV`、`--matcher sequential`、开 guided matching…） |
| COLMAP 报 CUDA 相关错误 | 非 CUDA 版 COLMAP 却开了 GPU | 加 `--use-gpu 0`；反之 CPU 太慢就装 CUDA 版 |
| 训练报 `no kernel image is available` | torch 的 CUDA 版本低于显卡算力（RTX 5060 = sm_120） | 换 `torch>=2.7+cu128`：`pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision` |
| 训练报 `CUDA out of memory`（退出码 4） | 显存不足 | 加大 `--downscale`（4/8）、降低 `--max-gaussians`，或先用 `--iterations 3000` 试跑 |
| 训练报 `缺少依赖：torch, gsplat`（退出码 3） | 没装 | `python pipeline/setup_env.py --check`，再按提示安装（gsplat 在 Windows 需源码编译） |
| `PLY 解析失败`（退出码 4） | 不是 3DGS 的 PLY（缺 `f_dc_*` / `scale_*` / `rot_*`） | 用文本编辑器看文件头的 `property` 列表；手机 App 导出的 PLY 见下方替代路线 |
| 导入后模型黑屏 / 看不见 | 尺度或朝向不对 | 看 JSON 的 `bounds.radius`，把相机放到 `center + radius*2` 的位置；本流水线不做坐标轴翻转，前端 up 设为 `(0,-1,0)` |
| `numpy 快速路径自检未通过` 之类的警告 | 极端数值触发了差异 | 属正常保护，会自动改用纯 Python 路径，结果不受影响（仅速度略慢） |

---

## 6. 没有 NVIDIA 显卡时的替代路线（手机 App 导出 PLY）

没有 CUDA 就训不了 gsplat，但**导出步骤（to_splat.py）完全不依赖 GPU**，可以这样走：

1. 手机上安装 **Polycam**（LiDAR 版最好）或 **Luma AI**，绕着现场走一圈做扫描；
2. 在 App 里导出 **`.ply`**（Polycam：Export → Point Cloud → PLY；Luma：Export → Gaussian Splatting → PLY）。
   优先选「Gaussian Splatting / 3DGS 的 PLY」，而不是普通点云 —— 后者没有 `f_dc_/scale_/rot_` 属性，转换会报错；
3. 把文件拷到电脑，例如 `data/models/phone-scan.ply`；
4. 一行命令转换：
   ```powershell
   python pipeline/to_splat.py --ply data\models\phone-scan.ply --out data\models\phone-scan.splat --json data\models\phone-scan.json
   ```
5. 前端把模型 URL 设为 `/models/phone-scan.splat` 即可。

常见细节：

* 手机 App 的坐标系可能不是 COLMAP 的 y 向下约定 —— 但**本流水线不做任何轴翻转**，方向不对时在前端调整模型的 `rotation` / `world.up`，
  或先用 `python pipeline/to_splat.py --ply ... --out ... --json ...` 看 JSON 里的 `bounds` 判断朝向（例如"地面"应在 max.y 附近）。
* 若模型上下颠倒或旋转 90°，用 `store.js` 的 `setModel({ rotation: [...9 个数...] })` 传一个 3×3 旋转矩阵即可，不必重新转换。
* 若模型明显过大/过小，先用 `--max-scale` 过滤漂浮噪点（例如 `--max-scale 0.5`），再用 `--decimate 0.5` 抽稀以减小体积。
* 一些 App 导出的 PLY 是 `xyzw` 顺序的四元数，可加 `--quat-order xyzw`。

---

## 7. 自测

```powershell
python tests\test_pipeline.py
```

覆盖：to_splat 的逐字段往返与过滤、min-opacity/decimate、合成场景的包围盒与 6 点重投影精度（< 1px）、
`--ply` 往返一致性、模糊检测（PGM/PNG 两种容器与 Sub/Up 行过滤）、COLMAP 二进制读写往返（含按官方格式手工拼字节的独立校验）、
缺依赖与错误路径的退出码。**不需要任何第三方库**（有 numpy 时自动走 numpy 路径）。

自测全部在受限沙箱下可离线运行；如果某个依赖缺失导致用例跳过，输出里会明确写出跳过了什么。
