# 案发现场三维重建与协同研判系统

> 把二维画面（监控、照片、勘查视频）与三维空间精准联系起来的警务研判「空间信息交互中枢」。
> 一线民警用手机拍摄现场 → AI 自动重建 1:1 三维数字现场 → 研判人员打开浏览器即可在同一三维场景中
> 实时漫游、打点标注、同步轨迹、贴回监控画面，实现跨时空协同研判。

本项目是《案发现场三维重建与协同研判系统 · 项目蓝图》的 MVP 实现，走的是**B 路线（开源自部署）**。

---

## 一、30 秒跑起来

```powershell
# 1) 启动系统（双击也可以）
.\start.ps1 -OpenBrowser

# 2) 浏览器打开 http://localhost:8787/
#    首次进入先「新建案件」，再选一个三维现场模型，点「载入三维现场」
```

仓库里已经带了 3 个可直接加载的演示场景（`data/models/`）：

| 文件 | 内容 |
| --- | --- |
| `sample-room.splat` | 合成案发现场：房间 + 桌子 + 物证箱 + 地面血迹（12 万高斯） |
| `synth-trained.splat` | 由本项目训练器从合成数据集训练出来的场景 |
| `test-scene.splat` | 流水线自测用的合成房间 |

> 打开两个浏览器窗口（或两台电脑访问同一地址），用不同昵称进入**同一个案件**，
> 就能立刻看到多人实时协同的效果：视角同步、打点标注实时出现、文字交流。

---

## 二、能做什么

| 能力 | 说明 | 对应蓝图模块 |
| --- | --- | --- |
| **三维现场重建** | 视频/照片 → COLMAP 位姿 → GPU 高斯泼溅训练 → `.splat` | 模块一 |
| **浏览器端流畅渲染** | 自研 WebGL2 3DGS 渲染器：球谐 SH、EWA 走样消除、分块基数排序、实时漫游 | 模块三 |
| **多源画面对齐** | 纯 JS 实现 PnP（DLT + Hartley 归一化 + LM 精修），把监控画面按真实透视贴回三维现场 | 模块二 |
| **交互式相机标定** | 可视化标定面板：两个子选项分别「点画面像素」与「点三维对应点」，实时显示重投影误差与残差，解算后自动贴回 | 模块二 |
| **图层管理** | 按类型分组、可展开查看**具体条目**、一键显隐、点条目定位 | 三、核心作用 |
| **多人实时协同** | 零依赖 WebSocket 后端：视角同步、在线名单、打点/轨迹/区域/测距标注实时同步、文字交流 | 模块三 |
| **研判工具** | 物证打点、嫌疑人行动轨迹、区域圈定、两点测距（真实米制）、相机标注与视频贴图、标注顶点拖拽 | 三、核心作用 |
| **取证与留痕** | 案件目录化存储、原子写入防损坏、操作事件日志、缩略图、案件导出/导入 | 三、核心作用 |

---

## 三、目录结构

```
.
├── start.ps1                  # PowerShell 一键启动（推荐）
├── 启动系统.bat                # 双击启动
├── 重建流水线.bat              # 双击跑完整重建：视频 → .splat
├── app/
│   ├── server/                # 后端：零依赖 Node（静态资源 + REST + WebSocket）
│   │   ├── index.js           #   HTTP 路由、媒体分片上传、模型索引
│   │   ├── ws.js              #   自实现 RFC6455 WebSocket 服务端
│   │   └── store.js           #   案件库：原子写入、标注校验、事件日志
│   └── web/                   # 前端：原生 ES 模块 + Three.js
│       ├── index.html
│       ├── css/app.css
│       └── js/
│           ├── app.js             # 主程序：场景、交互、协同全流程
│           ├── splat-renderer.js  # ★ 自研 WebGL2 3DGS 渲染器
│           ├── splat-loader.js    # .splat / .ply 解码器
│           ├── picker.js          # 高斯点云拾取（网格索引 + 邻域 PCA 法向）
│           ├── annotations.js     # 三维标注渲染
│           ├── drawing.js         # 绘制工具状态机
│           ├── camera-overlay.js  # 监控画面贴回现场（视频纹理 + 视锥）
│           ├── pnp.js             # ★ 纯 JS PnP 相机位姿解算
│           ├── pnp-panel.js       # ★ 交互式相机标定面板
│           ├── collab.js          # 协同客户端（自动重连）
│           ├── ui.js              # 界面层
│           └── api.js             # REST 客户端
├── pipeline/                  # Python 三维重建流水线
│   ├── setup_env.py           #   环境自检与安装引导
│   ├── extract_frames.py      #   ffmpeg 抽帧 + 拉普拉斯模糊过滤
│   ├── colmap_recon.py        #   COLMAP 位姿估计（pycolmap / 命令行）
│   ├── train_gpu.py           #   ★ 纯 PyTorch GPU 高斯泼溅训练（免编译）
│   ├── train_gsplat.py        #   gsplat 版训练（有 gsplat 时更快）
│   ├── to_splat.py            #   ★ 3DGS PLY → .splat（Web 格式）
│   ├── make_test_scene.py     #   合成测试场景
│   └── README.md              #   流水线详细文档
├── tools/                     # 环境与数据工具
├── tests/                     # 自动化测试
├── docs/                      # 文档
│   ├── 部署与使用说明.md        #   ★ 完整部署、使用、排错
│   ├── 技术架构.md
│   └── 开源插件清单.md
├── data/
│   ├── models/                # 三维现场文件（.splat/.ply + .json 元数据）
│   └── projects/              # 案件数据
└── vendor/three/              # 本地化的 Three.js（无 CDN 依赖）
```

---

## 四、环境要求

| 组件 | 版本 | 用途 | 是否必需 |
| --- | --- | --- | --- |
| Node.js | ≥ 18（本机用 24） | 后端服务与前端构建无关（原生 ESM） | **必需** |
| 浏览器 | Chrome/Edge ≥ 113（需 WebGL2） | 三维渲染 | **必需** |
| Python | 3.10 / 3.12 | 重建流水线 | 训练时必需 |
| PyTorch | ≥ 2.7 + CUDA 12.8 | GPU 训练（RTX 50 系需 cu128+） | 训练时必需 |
| ffmpeg | 任意（已内置） | 视频抽帧 | 抽帧时必需 |
| COLMAP | 4.x | 相机位姿估计 | 真实拍摄时必需 |

**本机实测配置**：Windows 11 25H2 / RTX 5060 Laptop（8GB，sm_120）/ CUDA Toolkit 12.8。
环境已就绪：`pyruntime312`（Python 3.12.10 + torch 2.7.1+cu128）、`pydeps312`（numpy / opencv / ffmpeg / plyfile / pycolmap）、`vendor/three`。

---

## 五、文档索引

- **[部署与使用说明](docs/部署与使用说明.md)** —— 从这里开始：环境安装、逐步命令、排错、验收
- [技术架构](docs/技术架构.md) —— 坐标系、数据流、渲染管线、协同协议
- [开源插件清单](docs/开源插件清单.md) —— 用了哪些开源项目、各自的许可证与替代方案
- [流水线文档](pipeline/README.md) —— 重建流水线的详细命令与报错处理

---

## 六、测试

全部 **565 项断言 + 6 项可达性检查**通过。详细输出见
[验证与测试报告](docs/验证与测试报告.md)。

```powershell
$NODE = "C:\Users\HP\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\node\bin\node.exe"

# ★ 真实浏览器渲染回归（最重要，需先启动服务；会拉起无头 Edge）
& $NODE tests\browser_render_check.mjs

# ★ PnP 标定面板（浏览器端：抽帧、点击配对、真实高斯拾取、解算一致性）
& $NODE tests\browser_pnp_panel.mjs

# ★ 现场建模面板：素材清单、体积体检、启动任务、日志轮询
& $NODE tests\browser_recon_panel.mjs

# ★ 素材体积门槛 + 取消/关闭可靠性（体积不足先提示；取消在任何内核下都关得掉）
& $NODE tests\browser_recon_gate.mjs

# 素材体积体检与中文名路径解析（服务端）
& $NODE tests\test_recon_materials.mjs

# 标定面板纯逻辑（不需要浏览器）
& $NODE tests\test_pnp_panel.mjs

# 用案件里真实的视频跑一遍「导入 → 贴回现场」链路
& $NODE tests\verify_real_video.mjs

# 案件可用性自检（只读，会自行清理临时数据）
& $NODE tests\verify_user_case.mjs

# 后端 API + WebSocket 协同
& $NODE tests\smoke.mjs

# 标注幂等性（防止同一对象被重复创建）
& $NODE tests\test_annotation_idempotent.mjs

# PnP 位姿解算 / 相机坐标换算
& $NODE tests\test_pnp.mjs
& $NODE tests\test_camera_pose.mjs

# 模型索引与 .splat 格式 + 服务可达性
& $NODE tests\test_model_index.mjs
& $NODE tests\test_final_check.mjs

# ★ 房间结构图层（房间盒/地面网格/墙面/相机轨迹，含**像素级**验证）
#   同时验证「判定不是 room 时不许画房间盒」（柜子曾被糊上假房间盒的那个回归）
& $NODE tests\browser_structure_layer.mjs

# 重建流水线（Python）
pyruntime312\python.exe tests\test_pipeline.py
pyruntime312\python.exe tests\test_train_gpu.py
pyruntime312\python.exe tests\test_ply_layout.py

# ★ 水平层检测（合成数据 + 已知真值；防"自己挖出来的谐波峰"复发）
pyruntime312\python.exe tests\test_layer_detection.py

# ★ 模块拆分回归（场景算法 / 物体算法 / 通用几何必须保持分开，防单文件长回来）
pyruntime312\python.exe tests\test_module_split.py

# 判据的实测依据（会打印三态判定表；新素材判不准时先跑它）
pyruntime312\python.exe tools\diag_scene_mode_table.py
```

`tests/browser_render_check.mjs` 与 `tests/browser_pnp_panel.mjs` 通过 Edge 的调试协议在
**真实 WebGL2 环境**里加载页面、操作渲染器与标定面板、读回画布像素，并断言：
画布尺寸、四个顶点属性均已启用、绘制数量、像素覆盖率、最大不透明度、帧率上限、
控制台无报错，以及标定面板的抽帧/点击配对/解算一致性。
这类「只看代码看不出来」的问题（见[报告 §6](docs/验证与测试报告.md)与
[报告 §4.2](docs/验证与测试报告.md)）只能靠它们拦住。

---

## 七、合规边界（严格遵守蓝图第四节）

- ✅ 只做**视觉重建与空间对齐**：把监控画面、勘查视频「贴」到三维模型里，直观表达空间关系
- ❌ **不做**人脸识别、行为预测等后端安防 AI，规避技术风险与隐私合规问题
- ✅ 最终交付形态是**浏览器 Web 端**，一线民警无需安装专业软件
- ✅ 输入端**极度轻量化**：普通手机、执法记录仪、普通监控摄像头均可
- ❌ **不依赖**激光雷达或专业三维扫描仪

---

## 八、已知限制

1. **COLMAP 已装好并验证**：官方 CUDA 版解压在 `tools/colmap/`，实测 24/24 注册、平均重投影误差 **0.73 px**。
   但**尚未用真实拍摄的照片验证**（目前跑的是程序化纹理的合成数据）。详见 [COLMAP 安装与验证](docs/COLMAP安装与验证.md)。
2. **gsplat 无 sm_120 轮子**：gsplat 官方只有 Linux 或 cu124 及以下的 Windows 轮子，RTX 5060（sm_120）跑不了。
   因此本项目自带 `pipeline/train_gpu.py`——**纯 PyTorch 实现的可微高斯泼溅训练器**，不需要编译器。
   代价是速度与细节不如官方 CUDA 光栅化实现。
3. **渲染质量**：自研 WebGL2 渲染器采用「分块 + 基数排序 + 逐高斯绘制」，属于 3DGS 的 Web 端实用实现，
   与官方 CUDA 光栅化在极端近景/大量重叠时会有细微差别。
4. **单机部署**：当前 WebSocket 与案件库是单进程内存 + 文件存储，适合单派出所/单案组使用；
   跨网段、多节点需要另接 Redis/数据库（见技术架构文档 §6）。
5. **训练分辨率受显存限制**：自研训练器在 8GB 显存的机器上只能用 240×135 左右的分辨率训练
   （`--downscale 3~4`），细节偏少。有 MSVC 的机器建议编译 gsplat 或原版 3DGS，
   两者导出的 PLY 与本项目完全兼容。详见[报告 §7.1](docs/验证与测试报告.md)。
6. **浏览器测试依赖无头 Edge**：`tests/browser_render_check.mjs` 与
   `tests/browser_pnp_panel.mjs` 需要允许启动浏览器进程，并且只能使用软件渲染
   （SwiftShader），因此它测到的帧率（约 34 FPS）是软件渲染瓶颈，不代表真实显卡上的表现。
7. **PnP 标定需要人工点选对应点**：至少 4 组、建议 6~8 组，且要分散在画面各处。
   监控摄像头的真实内参通常拿不到，面板默认按视场角估算（可在右侧手动填写）。
   重投影误差 > 5px 时建议检查是否有误点。

---

## 九、交互小贴士

- **双击标注列表项 = 定位/聚焦**该对象；删除要按行尾的 `✕` 并二次确认（防止误删）。
- 导入监控/勘查视频后，点画面卡片上的 **「贴回现场」** 会站到该机位，用来核对画面与
  三维模型是否重合；点 **「PnP 标定」** 打开标定面板精确求解机位。
- 标定面板里的工作顺序是「先在画面上点像素 → 再到三维视口里点同一位置」，
  面板会在三维那一步给出提示。
- 左下角 **「高斯尺寸」** 滑块可以放大高斯球，低分辨率训练的模型调大后更容易看清结构。
- 渲染循环有 **60FPS 上限**，页面切到后台会自动暂停。


---

## 十、仓库与交付（2026-11-05 补充）

### 只读演示（手机/外部访问，无需服务器）

`https://case-scene.pages.dev/` —— Cloudflare Pages 托管 `dist-static/`，**本机不用开机**。
它只做"看"：模型实时渲染、结构线、家具盒、测量；上传/建模/协同需本地服务。

重新生成静态包：

```bat
python tools\build_static.py --out dist-static --project-id <案件id>
```

`build_static.py` 会：策展模型（只打演示用的 3 个）、按代码里的 `import` **精确拷贝 vendor 并按依赖闭包补齐**、
注入 `__DSH_STATIC__`（只读模式）、生成 `DEPLOY.txt`。
⚠️ 静态包必须 **≤1000 个文件**（Cloudflare Pages 拖拽上限），当前 32 个 / 约 21 MB。

### 哪些内容不进仓库

见 `.gitignore`。仓库只放**源码 + 文档 + 前端依赖 + 演示包**（约 53 MB）；以下体积过大，按需自取：

| 目录 | 体积 | 说明 |
| --- | --- | --- |
| `models/` | 5.7 GB | VGGT / Grounding DINO / Depth Anything 权重 |
| `pydeps312/`、`pydeps/`、`wheels/` | ~15 GB | 第三方 Python 依赖与 torch 轮子 |
| `tools/colmap/` | 764 MB | COLMAP 二进制（自行安装 COLMAP 3.12+，需含 `global_mapper`） |
| `tools/cloudflared/` | 53 MB | 内网穿透 |
| `data/`、`output/` | ~9 GB | 本机实验数据与产物 |

### 重建环境（可选，仅训练需要）

Python 3.12 嵌入式运行时 + CUDA 12.8 的 torch 2.7.1；训练必须通过 MSVC 环境脚本调用：

```bat
call tools\msvc-env.bat && python pipeline\train_gsplat.py --data <工作目录> --out <工作目录>\train ...
```