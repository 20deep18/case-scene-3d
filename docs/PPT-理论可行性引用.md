# PPT 用：立项可行性引用（可直接贴进"理论可行性"页）

> 用法：答辩/评审讲"这个项目不是拍脑袋，是先查了文献确认路线可行"。
> 每条给出：**要证明的结论 → 学术依据（可贴）→ 我们实测到哪一步**。
> 原则：**只贴我们真做过或真验证过的**；没做的放"展望"页，别混进来。

## 一、总览：五个关键结论 + 对应依据

| # | 要证明的结论 | 学术依据（贴这行就够） | 我们的实测 |
| --- | --- | --- | --- |
| 1 | 手机视频能恢复相机位姿与稀疏点云 | **COLMAP**: Schönberger & Frahm, *Structure-from-Motion Revisited*, CVPR 2016 · https://demuc.de/colmap/ | 卫生间 219/263 注册、1.12 px；宿舍 198/211、1.15 px |
| 2 | 建图可以做到"秒级~分钟级"且不漂 | **GLOMAP**: Pan, Baráth, Pollefeys, Schönberger, *Global Structure-from-Motion Revisited*, ECCV 2024 · https://demuc.de/papers/pan2024glomap.pdf | 宿舍 211 帧：**47 秒 / 210 张**（增量法 8.9 分钟 / 198 张，且尺度爆炸） |
| 3 | **不依赖 SfM** 也能从少量帧秒级出几何（含位姿/深度/点云） | **VGGT**: Wang et al., *Visual Geometry Grounded Transformer*, CVPR 2025 **Best Paper** · https://arxiv.org/abs/2503.11651 | 12/24 帧直接出点云+位姿+深度；COLMAP 漂移的宿舍，VGGT 仍分出床板/立柱/墙 |
| 4 | 三维高斯能实时渲染、可交互 | **3DGS**: Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH 2023 · https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ | 本机 RTX 5060 训练；卫生间 5.1 万高斯、宿舍 85.7 万高斯，网页端实时可交互 |
| 5 | 手机上"实时三维重建"学术上成立（我们选择只做查看是工程取舍） | **Mobile3DRecon**: 杨幸彬等, *Real-time Monocular 3D Reconstruction on a Mobile Phone*, ISMAR 2020 **Best Paper** · https://ieeexplore.ieee.org/document/9201064 | 未做端侧重建；但已在手机浏览器实时渲染 85 万高斯模型（WebGL） |

## 二、支撑"AI 原生"（A 类要求）的两条

| 结论 | 依据 | 我们的实现 |
| --- | --- | --- |
| 场景几何可由**基础模型**直接推理，不是手写规则 | VGGT（CVPR 2025 Best Paper） | `tools/vggt_cloud.py`：点云/位姿/深度，宿舍 24 帧 PSNR 16.58 dB（12 帧 12.61 dB） |
| 物体级框可由**开放词表检测**得到 | Grounding DINO: Liu et al., *Grounding DINO: Marrying DINO with Grounded Pre-training for Open-Set Object Detection*, ECCV 2024 · https://arxiv.org/abs/2303.05499 | `tools/detect_furniture.py`：宿舍 171 原始框 → 7 件物体，`bed` 9 票 / 置信 0.83 |

## 三、"我们查过资料才立项"的叙事（建议这样讲）

1. **需求拆解**：手机视频 →（位姿）→（稠密表达）→（结构/物体）→（浏览器可交互）→（多人同看）。
2. **每一步先找成熟方案**：位姿有 COLMAP（2016 起工业标准）、稠密表达有 3DGS（2023 SIGGRAPH）、
   快速几何有 VGGT（2025 最佳论文）、物体有开放词表检测（2024）。
3. **再做可行性验证**：我们用真实手机视频跑通了一遍，量了注册率/PSNR/尺寸误差
   （见 `docs/技术现状与待决问题-给专家.md`），**确认"能出东西"才开始写系统**。
4. **诚实边界**：手机端实时重建学术上可行（Mobile3DRecon），但我们本期**只做手机端查看**；
   无位姿长视频重建（LongSplat 等）是研究级复现，本期不做。

## 四、引用格式（复制即用）

```text
[1] J. L. Schönberger, J.-M. Frahm. Structure-from-Motion Revisited. CVPR 2016.
[2] L. Pan, D. Baráth, M. Pollefeys, J. L. Schönberger. Global Structure-from-Motion Revisited (GLOMAP). ECCV 2024.
[3] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, D. Novotny. Visual Geometry Grounded Transformer (VGGT). CVPR 2025 (Best Paper).
[4] B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. SIGGRAPH 2023.
[5] 杨幸彬, 周立阳, 姜翰青, 鲍虎军, 章国锋. Mobile3DRecon: Real-time Monocular 3D Reconstruction on a Mobile Phone. ISMAR 2020 (Best Paper).
[6] S. Liu, Z. Zeng, T. Ren, et al. Grounding DINO: Marrying DINO with Grounded Pre-training for Open-Set Object Detection. ECCV 2024.
[7] P. Lindenberger, P.-E. Sarlin, V. Larsson, M. Pollefeys. Pixel-Perfect Structure-from-Motion with Featuremetric Refinement. ICCV 2021 (Oral).
[8] N. Keetha, et al. SplaTAM / COLMAP-Free 3D Gaussian Splatting. CVPR 2024. (仅展望)
```

> 说明：第 7、8 条只出现在"展望/相关工作"，不要放在"我们实现了"的页面上。
