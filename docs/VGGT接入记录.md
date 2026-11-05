# VGGT 接入记录（2026-11-02）

> 目标：用 VGGT 替代 COLMAP，拿到**稠密点云 + 位姿**，再走现有的家具分割流水线。
> 动机：本机 COLMAP 是 CPU 版，**PatchMatch 稠密重建是 CUDA-only，这个版本根本做不到**；
> 而白瓷砖 + 反光让 SIFT 特征太少（实测厕所两段视频 PSNR 18.8~21.2 dB、33% 高斯近白）。

## 一、环境（已就绪）

| 项 | 状态 |
| --- | --- |
| 代码 | `models/vggt/repo/vggt-main/`（62 MB，从 `gh-proxy.com` 下的 GitHub zip，GitHub 直连被重置） |
| 权重 | `models/vggt/model.pt`（**3.93 GB**，从 `hf-mirror.com` 下，约 2~3 MB/s） |
| 依赖 | 装进 `pydeps312`（未动 torch 2.7.1+cu128）：torchvision 0.22.1、timm 1.0.30、huggingface_hub 2.4.0、safetensors、einops、**httpx2/httpcore2**（hub 2.4 的新依赖名）、trimesh |
| 导入自检 | 17 项里 16 项通过（见 `.buildtmp/check_vggt_imports.py`）；唯一注意点：`import vggt` 必须**显式 `sys.path.insert`** 仓库路径 —— 项目 python 是精简版，`PYTHONPATH` 不生效 |
| 输入帧 | `output/bathroom/vggt-in/`（40 帧，518px，来自视频 2） |

## 二、运行器

`tools/vggt_cloud.py`：
- `--frames N` 抽帧（**显存随帧数平方增长**，8 GB 先试 8~12）
- `--resolution 518`、`--conf 5.0`、`--max-points 200000`
- 输出 `*.ply` + `*.poses.json`（OpenCV 约定，任意尺度）
- `--dry-run`：不加载权重，验证"取帧 → 预处理(1024²) → 写 PLY → 家具流水线读取"全链路

**已离线验证**：
- 干跑：12 帧 → `(12,3,1024,1024)` → 写出 PLY ✓
- 该 PLY 被 `furniture_lines.py` 正确读取（"没有尺度信息 → 跳过尺度过滤"）✓
- **无尺度点云的真值回归**：把合成房转成只有 xyz 的 PLY（模拟 VGGT 产物），仍是 **4/4 命中** ✓

## 三、本轮修的三个 bug（都有实测依据）

| 现象 | 原因 | 修法 |
| --- | --- | --- |
| PLY 过滤后 **0 个点** | 3DGS 的 `scale_*` 是 log 尺度、高斯普遍 ~1.1 单位，按"跨度 3%"取阈值（0.69）全被滤掉 | 阈值改为 `min(跨度×0.5, max(跨度×3%, 该点云尺度的 P75))`；无尺度点云（MVS/VGGT）直接跳过尺度过滤 |
| 合成房从 4/4 **退化为 0** | 平面剥离把家具自己的大面（床面/柜门）当墙剥了 | 只在**房间盒不可信**时才剥平面（房间盒可信时剥壳已足够） |
| 盒高普遍 **偏短 8~12%** | 家具底面贴地，被剥壳一起削掉 | 底面离地面 ≤（剥壳边距+1.5 体素）就**吸附到地面**；实测衣柜 1.907→2.027（真值 2.00，误差 −4.6% → **+1.3%**） |

## 四、恢复命令

```bat
REM 1) 续传权重（可中断，自动续传）
curl -L -C - -o models\vggt\model.pt https://hf-mirror.com/facebook/VGGT-1B/resolve/main/model.pt

REM 2) 跑 VGGT（先小帧数试显存）
pyruntime312\python.exe tools\vggt_cloud.py --frames-dir output\bathroom\vggt-in --out output\vggt\bathroom.ply --frames 8

REM 3) 判定（合成房 4/4 = 0.34；厕所现有模型 0.33~0.40 = 房间级）
pyruntime312\python.exe tools\compare_clouds.py --cloud "VGGT=output/vggt/bathroom.ply"
```

判定标准：`最大盒尺寸 / 场景跨度` **< 0.15** 才算"像家具"。

## 五、实测结果（2026-11-02）：三条路线在这间厕所上都不成立

| 路线 | 关键数字 | 结果 |
| --- | --- | --- |
| COLMAP(CPU)+3DGS | 219/263 注册、7k~30k 步、39,741 高斯、PSNR 19.3~21.2 dB | 雾团，家具 0 个 |
| **VGGT 原始点图** | 8/12/16/24 帧全部跑通（无 OOM）、200k 点、置信度 P50≈1.0 | 云是稀疏带状的噪点；"盒子"是 0.17×0.15×0.05 薄片、每块约 1,200 点 |
| **VGGT 位姿 + 3DGS**（新） | 12 帧、1,135,650 高斯、PSNR **14.43 dB** | 高斯密度上来了，但仍是白雾；10 个盒子互相重叠在同一个团里（见 `output/vggt/boxes-over-cloud.png`） |

**关键发现**：
1. **帧间距比帧数更重要**：均匀抽帧（间隔 4.8 s）最大盒/跨度 0.42~0.45；连续 1 s 帧 0.16；24 帧反而 0.53。
2. **VGGT 逐帧内参很不稳**（fx 200~441、fy 285~829，方形图里 fx 必须等于 fy）→ 训练时改用"全帧共享、各向同性"内参（f=599.35）。
3. `colmap model_converter` 在本机崩溃（0xC0000409）；pycolmap 读"带占位 track 的文本模型"会 IndexError → 改为**直接写二进制 COLMAP 模型**（`tools/colmap_bin_writer.py`，pycolmap 复核通过）。
4. 三条路线的共同失败原因**不是算力**，是**这间厕所的画面**：白瓷砖 + 强反光 → 无光度约束 → SfM/3DGS/VGGT 都只能长出"白雾"；雾里没有可分的物体，任何分割都只能切出碎片。

**结论：家具级线稿需要"有纹理、有家具"的场景。** 路线（GPU/MVS/VGGT）都绕不过这一点。
演示兜底：合成房出家具级线稿（已验证 4/4），真实素材出房间级，PPT 里如实标注。
## 六、家具盒接入网页 UI（2026-11-02）

### 做了什么

| 位置 | 改动 |
| --- | --- |
| `app/web/js/structure-layer.js` | 新增 `setFurniture()` / `setFurnitureVisible()` / `_addFurniture()` / `_renderFurnitureOnly()`；新色 `COLOR_FURN`（深橙，与"物体盒"的橙区分）；标签只在**点数最多的前 5 个盒子**上画（10 个一起标会糊成一团，实测） |
| `app/web/js/app.js` | `applyStructure()` 开头先 `setFurniture(null)`（换模型必须清掉上一个模型的橙框）；新增 `structureToFurnitureUrl()` + `loadFurnitureSidecar()`；`#cFurniture` 开关接线；提示条报"家具 N 件（橙框）" |
| `app/web/index.html` | 新增勾选框「显示家具盒」 |
| `tests/browser_furniture_layer.mjs`（新） | **11 项断言**：数据加载、线段数=盒数×12、开关可关可开、换到无家具模型必须清空、提示条报数 |

### 数据来源约定

家具盒是**独立侧车文件**：与模型同名 `*.furniture.json`（由 `tools/furniture_lines.py` 产出）。
结构文件优先同名的 `*.structure.json` → 没有就用**模型 URL** 推导（`.splat` → `.furniture.json`）。

### 接的时候踩到的三个坑（都是真 bug，已修）

1. **没有结构文件时家具永远画不出来**：`applyStructure()` 在 `!structureUrl` 时提前 `return`，
   而 VGGT/稠密重建的模型恰恰没有结构文件（实测 `structureUrl: null`）。
   → 家具改成在提前返回**之前**就加载。
2. **`update()` 里的 `if (!this.data) return;`**：线段数虽然是 120 条（测试通过），
   但**每帧投影被整个跳过**，画面上一条橙线都看不到（截图为空）。
   → 改成"没有任何线段/折线/点/标签时才返回"。
3. **家具 URL 从 structureUrl 推导**：该模型 structureUrl 为 null，推出来是 null。
   → 增加"从模型 URL 推导"的兜底。

> 这三点说明一件事：**"测试通过"不等于"画面对"**。所以这轮每次都截图目视核对。

### 回归

- 新增 `browser_furniture_layer` **11/11**
- 既有套件：structure_layer 57/57、pnp_panel 44/44、recon_panel 17/17、recon_gate 32/32、
  measure_panel 18/18、delete 14/14、render_check 18/18
- `browser_recon_cancel_live`：4/5（失败项是"取消按钮 disabled"—— 属**重建启动链路**，
  与本次改动无关：structure-layer 里 0 处 recon 代码，其余 recon 套件全绿；待单独排查）