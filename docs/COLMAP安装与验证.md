# COLMAP 安装与验证记录

> 本机实测：**COLMAP 4.2.1 (Commit bd1fcf6, with CUDA)**，端到端跑通，注册率 100%，
> 平均重投影误差 **0.73 px**。

---

## 一、安装位置

用户提供的官方包 `colmap-x64-windows-cuda.zip`（395 MB）已解压到：

```
tools/colmap/
├── bin/
│   ├── colmap.exe          # 33.6 MB，主程序
│   └── *.dll               # 108 个依赖 DLL
├── plugins/
├── COLMAP.bat
└── RUN_TESTS.bat
```

**无需安装、无需 PATH、无需 MSVC** —— 官方包是自包含的，直接运行即可。

`pipeline/config.py` 的 `find_colmap()` 本来就会在
`COLMAP_EXE` 环境变量 → 系统 PATH → 常见安装目录 → **`tools/colmap/bin/colmap.exe`**
里依次查找，所以解压到这个位置后**无需任何配置**，流水线自动就能用：

```powershell
pyruntime312\python.exe -c "import sys; sys.path.insert(0,'pipeline'); import config; print(config.find_colmap())"
# C:\...\tools\colmap\bin\colmap.exe
```

CUDA 可用性已被实际调用验证（匹配阶段日志出现 `Bind FeatureMatcherWorker to GPU device 0`）。

---

## 二、一键验证

```powershell
# 需要一份有纹理的图片目录；下面用脚本生成的合成场景
pyruntime312\python.exe tools\make_textured_dataset.py --out data\scene-textured --views 24
pyruntime312\python.exe tools\verify_colmap.py --images data\scene-textured\images --workspace output\colmap-verify
```

`tools/verify_colmap.py` 会自动跑完
`feature_extractor → matcher → mapper`，然后用**官方 `model_analyzer`** 取权威统计，
再用自己实现的 `tools/colmap_model_io.py` 独立解析一遍做交叉校验。

实测输出（24 张 960×540 的纹理图）：

```
COLMAP: ...\tools\colmap\bin\colmap.exe
版本  : COLMAP 4.2.1 (Commit bd1fcf6 on 2026-09-29 with CUDA, boost hash maps)

→ 特征提取 … 完成
→ 穷举匹配 … 完成
→ 增量式重建 … 完成

=== 官方 model_analyzer（模型 0）===
  Cameras                      1
  Registered images            24
  Points                       10849
  Observations                 53422
  Mean track length            4.924140
  Mean observations per image  2225.916667
  Mean reprojection error      0.729196

=== tools/colmap_model_io.py 独立解析（交叉校验）===
  相机数            1
  注册图像          24
  三维点            10849
  观测数            53422
  平均 track 长度   4.9241
  平均重投影误差    0.729196 px
  相机 1: SIMPLE_RADIAL 960x540 fx=823.67 fy=823.67 cx=480.0 cy=270.0

=== 结论 ===
  注册率 24/24 = 100.0%
  平均重投影误差 0.7292 px（< 1.0px 属优秀）
  ✔ COLMAP 端到端可用
```

**自实现的解析器与官方工具在每一项统计上都完全一致**，说明二进制布局解析正确。

另外注意：我们生成的图像真值内参是 `fx=831.38`，COLMAP 解出的是 `fx=823.67`
（差 0.9%），**相机内参被正确恢复**。

---

## 三、完整链路：图片 → COLMAP → 训练 → .splat

```powershell
# 1) COLMAP 位姿（会自动把图片目录联接进工作区，形成标准布局）
pyruntime312\python.exe pipeline\colmap_recon.py --images data\scene-textured\images `
    --workspace output\colmap-pipeline --matcher exhaustive --use-gpu 1
#    → 注册 24/24，稀疏点 10797，平均重投影误差 0.72 px，耗时 54 秒

# 2) GPU 训练（自研纯 PyTorch 3DGS）
pyruntime312\python.exe pipeline\train_gpu.py --data output\colmap-pipeline `
    --out output\gs-textured --iterations 120 --downscale 3
#    → 已注册图像 24 张，稀疏点 10797 个（COLMAP 的点云被正确读取）

# 3) 转成 Web 用的 .splat
pyruntime312\python.exe pipeline\to_splat.py `
    --ply output\gs-textured\point_cloud\iteration_120\point_cloud.ply `
    --out data\models\colmap-textured.splat --json data\models\colmap-textured.json
#    → 10795 个高斯，337 KB
```

工作区布局（标准 COLMAP 数据集）：

```
output/colmap-pipeline/
├── images/            → 目录联接（mklink /J），不占额外空间
├── sparse/0/          cameras.bin / images.bin / points3D.bin / frames.bin
├── database.db        COLMAP 特征与匹配数据库
└── colmap_summary.json
```

---

## 四、这次为了接进流水线改了什么

| 问题 | 原因 | 处理 |
| --- | --- | --- |
| `unrecognised option '--SiftExtraction.use_gpu'` | **COLMAP 4.x 把选项改名了**：`SiftExtraction.*` → `FeatureExtraction.*`，`SiftMatching.*` → `FeatureMatching.*` | `pipeline/colmap_recon.py` 新增 `_supported_options()`：先读一次 `--help`，拿到该版本真实支持的选项集合，再从「4.x 写法 / 3.x 写法」里挑可用的。**两代版本都能跑** |
| `output_path is not a directory` | COLMAP 的 `mapper` 要求 `--output_path` 指向**已存在**的目录 | 调 mapper 前先 `mkdir` |
| 训练器报「只有 0 张可用图像」 | `colmap_recon.py` 只写了位姿，没把图片接进工作区；而 `train_gpu.py` 按标准约定找 `<workspace>/images/` | 新增 `link_dataset_images()`：优先 `mklink /J` 目录联接（不占空间），失败则软链，再失败才复制 |
| 不认识 `.ppm` | `IMAGE_SUFFIXES` 里没有 PPM | 加上 `.ppm` / `.pgm`（COLMAP 原生支持，`make_textured_dataset.py` 用它以避免依赖图像库） |
| `model_converter`（二进制→文本）崩溃 | COLMAP 4.2.1 在 Windows 上 `0xC0000409` 栈缓冲溢出 | 自己实现 `tools/colmap_model_io.py` 解析二进制；统计结果与官方 `model_analyzer` 完全一致 |

---

## 五、`images.bin` 的实际记录布局（用 hex dump 确认过）

网上流传的 COLMAP 3.x 布局说明与实际文件**需要核对**，这里记录本机实测结果：

```
images.bin
    uint64 num_images
    每张图像：
        int32   image_id
        double  qvec[4]        (w, x, y, z)
        double  tvec[3]
        int32   camera_id
        char    name[]         (\0 结尾)
        uint64  num_points2D
        num_points2D × (double x, double y, uint64 point3D_id)
                                    ↑ 共 24 字节/点
```

**两个容易读错的点**：

1. `camera_id` 在 4.2.1 里是**有**的（紧跟 tvec 之后），所以记录头是 64 字节而不是 60；
   一度按 60 字节解析，导致后面全部错位、`num_points2D` 读出天文数字。
   排查办法：`hex dump` 前几十字节，直接找 ASCII 文件名，数它前面有多少字节。
2. `point3D_id` 必须保持 **uint64**。无效值是 `0xFFFFFFFFFFFFFFFF`，
   转成 `float64` 会丢精度（实测出现 `RuntimeWarning: invalid value encountered in cast`），
   再转回整数就数错了观测数（94341 vs 官方的 53422）。
   注意 COLMAP 的 `Observations` 只统计**真的关联到三维点**的二维观测。

---

## 六、合成测试数据的坑（很重要）

`tools/make_test_dataset.py` 生成的房间是**纯色墙面**，SIFT 只能找到 **28~50** 个特征点，
匹配数为 **0**，`mapper` 直接报 `No images with matches` / `Failed to create any sparse model`。

**这不是 COLMAP 的问题**，是测试数据缺少纹理。为此新增
`tools/make_textured_dataset.py`：给每个表面贴上程序生成的多尺度纹理
（4 层棋盘 + 随机斑点 + 斜向条纹 + 噪声），特征点数量提升到 **3152/张**（约 60~110 倍）。

该脚本**不依赖任何图像库**（直接写 PPM P6），透视贴图用 numpy 逐像素反查单应完成。

### 相机约定的三个符号错误（同一个地方连踩三次）

写这个脚本时，相机模型连续写错了 3 次，**而且它们都能通过「相机中心在原点」这种自检**：

| 错误 | 后果 |
| --- | --- |
| 世界→相机写成 `A·x`（应为 `Aᵀ·x`） | 旋转方向反了 |
| 相机基第三列写成 `-f`（应为 `+f`） | z 变负，所有点都跑到相机背后 |
| `C = -Aᵀ·t` 漏了负号 | 相机中心算成「镜面对称」的位置 |
| `u = s × f`（应为 `f × s`） | 右手系变成左手系（det = -1），图像上下颠倒 |

**唯一有效的自检**（已写成 `self_check()`，不通过就直接退出、不生成废图）：

1. 原点必须落在相机正前方：`x_cam ≈ (0, 0, R)`，即 `z_cam > 0`
2. `C = -Aᵀ·t` 必须能精确还原 `eye`
3. `det(A) == +1`（纯旋转，不能是镜像）

教训：**坐标系约定不要靠推理，要拿已知正确的实现（`tests/test_pnp.mjs`）对照，
并且自检必须验证「前方点落在前方」，而不是只验证「相机中心在原点」。**

---

## 七、后续

- **材质/纹理真实度**：当前是程序化纹理的立方体，仅用于验证 COLMAP 可用。
  真实重建请用实际拍摄的照片（见部署文档「采集要点」）。
- **匹配策略**：24 张图用穷举匹配（`exhaustive_matcher`）约 3 秒；
  图片上百张时建议改用 `sequential_matcher`（`--matcher sequential`）或
  `vocab_tree_matcher`（需要额外下载词汇树文件）。
- **`model_converter` 崩溃**：如需二进制→文本，用 `tools/colmap_model_io.py`，
  或改用更低版本的 COLMAP。
