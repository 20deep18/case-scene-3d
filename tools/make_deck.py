#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成讲解用 PPT：`作品讲解-案发现场三维重建.pptx`

以"作品"为中心：立意 → 一句话流程 → 成果 → 作用 → 收束。
配图全部来自本机实测截屏与渲染（output/deck/*.png），不用网图。

用法（用 DSH 自带 Python，含 python-pptx）：
    <python> tools/make_deck.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_intro_slide import add_intro  # noqa: E402
ROOT = Path(__file__).resolve().parent.parent
DECK = ROOT / "output" / "deck"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else (DECK / "作品讲解-案发现场三维重建.pptx")

FONT = "微软雅黑"
INK = RGBColor(0x1B, 0x22, 0x2B)      # 正文深灰蓝
ACCENT = RGBColor(0x0E, 0x6B, 0x8A)   # 主色（青蓝）
MUTED = RGBColor(0x6B, 0x76, 0x82)    # 次要文字
LINE = RGBColor(0xD8, 0xDF, 0xE6)

W, H = Inches(13.333), Inches(7.5)


def add_text(slide, x, y, w, h, runs, *, align=PP_ALIGN.LEFT, spacing=1.15):
    """runs: [(文本, 字号, 加粗, 颜色)]；每条一行。"""
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, (text, size, bold, color) in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        p.space_after = Pt(6)
        r = p.add_run()
        r.text = text
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
        r.font.name = FONT
    return box


def add_title_bar(slide, title, subtitle=None):
    add_text(slide, Inches(0.75), Inches(0.42), W - Inches(1.5), Inches(0.8),
             [(title, 30, True, INK)])
    y = Inches(1.18)
    if subtitle:
        add_text(slide, Inches(0.78), y, W - Inches(1.6), Inches(0.4),
                 [(subtitle, 14, False, MUTED)])
        y = Inches(1.55)
    ln = slide.shapes.add_shape(1, Inches(0.78), y, W - Inches(1.56), Emu(9525))
    ln.fill.solid()
    ln.fill.fore_color.rgb = LINE
    ln.line.fill.background()
    ln.shadow.inherit = False
    return y + Inches(0.18)


def add_image_fit(slide, path, x, y, w, h, *, shadow=False):
    """按比例放入图片并居中于给定框。"""
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    left = int(x + (w - nw) / 2)
    top = int(y + (h - nh) / 2)
    pic = slide.shapes.add_picture(str(path), left, top, width=nw, height=nh)
    if shadow:
        pic.shadow.inherit = False
    return pic


def add_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def bullet_slide(prs, title, bullets, *, subtitle=None, image=None, caption=None,
                 notes="", image_right=True, img_frac=0.56):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    y = add_title_bar(s, title, subtitle)
    body_w = W - Inches(1.56)
    if image:
        iw = int(body_w * img_frac)
        tx = Inches(0.78) + iw + Inches(0.35) if image_right else Inches(0.78)
        ix = Inches(0.78) if image_right else W - Inches(0.78) - iw
        add_image_fit(s, image, ix, y + Inches(0.1), iw, H - y - Inches(1.05))
        tw = body_w - iw - Inches(0.35)
    else:
        tw, tx = body_w, Inches(0.78)
    add_text(s, tx, y + Inches(0.25), tw, H - y - Inches(1.2),
             [(b, 17, False, INK) for b in bullets], spacing=1.35)
    if caption:
        add_text(s, Inches(0.78), H - Inches(0.82), body_w, Inches(0.4),
                 [(caption, 11.5, False, MUTED)])
    add_notes(s, notes)
    return s


def section_slide(prs, kicker, title, notes=""):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bar = s.shapes.add_shape(1, Inches(0.9), Inches(3.0), Emu(38100), Inches(1.4))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    bar.shadow.inherit = False
    add_text(s, Inches(1.15), Inches(2.95), W - Inches(2.3), Inches(0.5),
             [(kicker, 15, True, ACCENT)])
    add_text(s, Inches(1.15), Inches(3.38), W - Inches(2.3), Inches(1.2),
             [(title, 34, True, INK)])
    add_notes(s, notes)
    return s


def main() -> int:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    # ⚠️ 只改尺寸不够：python-pptx 会把 `p:sldSz` 的 **type 属性**留成 `screen4x3`，
    #    而渲染器（实测 LibreOffice）会照 type 把内容按 4:3 缩放，于是整页内容
    #    被放大 1.5 倍、右侧文字被切掉（我用一条已知位置的红线量出来的：
    #    声明 1.000 in → 渲染到 1.50 in，两种幻灯片尺寸下都恰好 1.5×）。
    #    把 type 一起改成 screen16x9 即可。
    _sldSz = prs._element.find(qn("p:sldSz"))
    if _sldSz is not None:
        _sldSz.set("type", "screen16x9")

    # ---- 0 自我介绍（第 1 页）------------------------------------------------
    add_intro(prs)
    print("  0 自我介绍")

    # ---- 1 封面 -------------------------------------------------------------
    s = prs.slides.add_slide(prs.slide_layouts[6])
    band = s.shapes.add_shape(1, 0, 0, W, Inches(2.35))
    band.fill.solid()
    band.fill.fore_color.rgb = RGBColor(0x0B, 0x2A, 0x38)
    band.line.fill.background()
    band.shadow.inherit = False
    add_text(s, Inches(0.9), Inches(0.62), W - Inches(1.8), Inches(1.0),
             [("案发现场三维重建与协同研判系统", 40, True, RGBColor(0xFF, 0xFF, 0xFF))])
    add_text(s, Inches(0.95), Inches(1.62), W - Inches(1.9), Inches(0.6),
             [("拍一段视频，把现场“带回来”—— 在浏览器里走进去、标记、量测、一起看",
               17, False, RGBColor(0xB6, 0xD6, 0xE2))])
    add_image_fit(s, DECK / "01-overview.png", Inches(0.9), Inches(2.7),
                  W - Inches(1.8), Inches(4.0))
    add_notes(s, "各位好。今天我用几分钟，介绍我做的一套系统：案发现场三维重建与协同研判。先说明一句：这套系统还在创作阶段，能跑通、也在真实素材上验证过，但离理想效果还有距离，今天我把做到的、和还没做到的都如实讲。"
                 "一句话说清它做什么——用一部手机拍一段视频，就能把现场变成一个"
                 "可以在浏览器里走进去的三维模型。")
    print("  1 封面")

    # ---- 2 立意 -------------------------------------------------------------
    bullet_slide(
        prs, "立意：现场“看完就没了”",
        ["现场是不可复现的：勘查只有一次机会，回来之后只剩照片和平面图",
         "照片是平面的：站位、遮挡、通道、高度关系，说不清、也复看不了",
         "多人研判缺共同视角：办案人各自看图，很难确认“我们说的是同一个角落”",
         "我想解决的就是这三件事：把现场留下来、把它变成能看懂的三维、让多人对着同一个现场看"],
        subtitle="为什么做这套系统",
        notes="先说立意。我做这个系统的出发点很朴素：案发现场最大的问题是"
              "“看完就没了”。现场只能看一次，回来之后剩下的就是照片和平面图。"
              "照片是平的，很多空间关系说不清；多人讨论时也没有一个共同的视角。"
              "所以我想做的，就是把现场变成能被反复走进去的三维。立意我越想越觉得对，但从立意到落地，我目前只走完了一部分 —— 后面几页会具体说差在哪。",
        image=DECK / "video-frames.png",
        caption="上图为真实拍摄的手机视频抽帧（同一现场，用于重建的原始素材）",
    )
    print("  2 立意")

    # ---- 3 一句话流程 -------------------------------------------------------
    bullet_slide(
        prs, "它怎么工作：一条流水线，四步",
        ["① 拍：手机围着现场拍一段视频（不需要专业设备、不需要标定板）",
         "② 抽帧：自动挑出清晰帧，剔除模糊帧",
         "③ 定位：算出每一帧相机的位置与朝向（即“当时站在哪里、朝哪看”）",
         "④ 成形：训练成三维高斯模型，导出成浏览器能直接打开的 .splat",
         "",
         "最后你得到的是：一个能旋转、能走进去、能标注、能量测的三维现场"],
        subtitle="拍摄视频 → 生成三维模型",
        notes="流程只有四步：拍视频、抽帧、算相机位姿、生成三维模型。"
              "对使用者来说，只有第一步需要动手，后面都是自动的。老实说流程是通的，但成功率还不理想：快走、甩镜头、对着白墙停几秒，重建就会失败或断成几段；让任何人都能一次拍成功，我还没做到。"
              "得到的产物是一个在浏览器里直接打开的三维现场。",
        image=DECK / "04-recon.png",
        caption="界面里的“现场建模”：导入视频 → 一键开始建模（左下角为流程说明）",
        image_right=False,
        img_frac=0.44,
    )
    print("  3 流程")

    # ---- 4 成果一 -----------------------------------------------------------
    bullet_slide(
        prs, "成果 · 房间现场",
        ["旋转、缩放、贴地漫游：像站在现场里一样看",
         "相机轨迹（红色）留在模型里：一眼看出“当时是怎么走的”",
         "房间结构线（青/黄）自动叠加：房间盒、地面网格、可信墙面",
         "顶栏常驻“结构状态牌”：当前按房间还是物体呈现、画了没有、为什么"],
        subtitle="一个真实房间的重建结果（浏览器界面）",
        notes="这是我重建的一个真实房间。可以旋转、缩放，也可以贴地漫游。"
              "红色的线是相机轨迹，一眼就能看出当时是怎么走的；"
              "青色的房间盒和黄色的墙是系统自动叠加的结构线。不足也很明显：结构线还比较工程化、信息太密，不像能直接汇报的图；对乱一点的现场，拟合结果还会开始飘。这块我还在改。",
        image=DECK / "01-overview.png",
        caption="房间尺寸 17.1 × 9.2 × 2.9 m；59,919 个高斯；结构线 229 条",
    )
    print("  4 成果一")

    # ---- 5 成果二 -----------------------------------------------------------
    bullet_slide(
        prs, "成果 · 狭小空间也能量",
        ["同一个流程，窄小、贴身的现场同样适用",
         "判定为“贴着物体拍”时，只呈现相机轨迹，绝不画假的房间盒",
         "几何不可信就什么都不画，并在界面上写明原因 —— 宁可说不可信，也不装成功"],
        subtitle="柜子 / 狭小空间这类“贴身拍”的现场",
        notes="第二个例子是狭小空间。这类现场系统会判断成“贴着物体拍”，"
              "这时它只显示相机轨迹，不会硬画一个房间盒。"
              "我这里坚持一个原则：几何不可信，就什么都不画，并且把原因写在界面上。但必须承认：这一页其实是「我还没做到」—— 柜子这类贴身拍摄的现场，我目前基本还重建不出能用的结构，数据退化时我只能选择什么都不画。",
        image=DECK / "03-cabinet.png",
        caption="判定为 object（贴拍）：不叠加房间盒，只保留轨迹，并在左栏说明原因",
    )
    print("  5 成果二")

    # ---- 6 空间线条 ---------------------------------------------------------
    bullet_slide(
        prs, "空间线条：让结构一眼读懂",
        ["把三维结构投影成二维直线稿（白底细线），适合汇报与讨论",
         "自动做“娃娃屋”剖切：挡在前面的近侧墙与天花切掉，露出室内",
         "遮挡关系按几何解析计算 —— 后面的线被前面的面挡住就不画",
         "图上直接标注尺寸与统计，线稿本身就能读"],
        subtitle="同一现场的四视角线稿（离线渲染）",
        notes="这是同一现场的空间线条稿。它把三维结构画成二维直线，"
              "并且自动做了剖切和遮挡处理，所以能直接看懂空间的构成。要说清它现在的水平：只能给房间级线条，家具级的线条完全没有；剖切和遮挡偶尔也会算错。这一块我刚起步，离「一眼读懂」还有明显距离。"
              "尺寸标在图上，可以边讲边指。",
        image=DECK / "lineart.png",
        caption="房间 17.14 × 9.19 × 2.88 m；每张图右下角为线数与被遮挡比例",
    )
    print("  6 线条")

    # ---- 7 作用一 -----------------------------------------------------------
    bullet_slide(
        prs, "作用一 · 把物证标在空间里",
        ["打点 / 测距 / 尺寸标注 / 区域 / 轨迹 / 相机位：一套工具就地记录",
         "标注带真实比例尺信息：模型里的距离可以直接读成米",
         "标注与物证列表可筛选、可追溯（谁标的、什么时候标的）",
         "现场怎么摆，模型里就怎么标 —— 不依赖记忆和照片角度"],
        subtitle="模型不只是“看”，更是记录与研判的载体",
        notes="模型的价值不只是看。这是标注工具：打点、测距、尺寸标注、区域、轨迹。短板也绕不开：绝对尺度目前还要靠人工实测那一步来定，让系统自己算准我还没做到，现在只能算「半自动」。"
              "标注是带真实比例尺的，模型里量出来的距离可以直接读成米。"
              "右侧是标注与物证列表，可以筛选、可以追溯。",
        image=DECK / "02-annotate.png",
        caption="示例：物证点 2 个 + 一条空间距离（4.771 m）+ 一条带实测值的尺寸标注",
    )
    print("  7 作用一")

    # ---- 8 作用二：多人协同 --------------------------------------------------
    bullet_slide(
        prs, "作用二 · 多人对着同一个现场看",
        ["同一案件多人同时在线，视角、标注、状态实时同步",
        "底部协同区：在线人员、消息、系统事件（谁进入 / 谁离开了研判）",
        "反复复看同一现场：不同时间、不同人，看到的是同一份三维证据",
        "把“我说的那个角落”变成“我们一起转到的那个角落”"],
        subtitle="把现场变成可以反复复看、共同讨论的三维证据",
        notes="第三点是协同。同一案件可以多人同时在线，标注和状态实时同步。坦白讲协同这块还是毛坯：多人改同一条标注的冲突处理、权限、操作记录都还很粗糙，只是「能用」，离「敢在真实案件里依赖」还有一段路。"
              "它的意义是把“我说的那个角落”变成“我们一起转到的那个角落”。",
        image=DECK / "02-annotate.png",
        caption="顶栏右上为在线协同人员，左下为协同消息与系统事件",
    )
    print("  8 作用二")

    # ---- 9 收束 -------------------------------------------------------------
    section_slide(
        prs, "一句话总结",
        "从“看照片”到“走进现场”",
        notes="总结一句：这套系统做的事，是把现场从照片变成可以走进去的三维空间，"
              "并且让多人能在同一个空间里标、量、讨论。"
              "但今天我更想说它还没做到的地方：重建成功率不够高、结构线不够干净、狭小空间和家具级结构还没攻下来、尺度还要靠人工量、协同还是毛坯 —— 这些我都记着，也正在做。为什么愿意一直磨：现场不可复现，但它应该可以被更完整地保存下来；我希望有一天，办案人员不用靠回忆去说明现场，而是能把它原样带回来，让每个参与研判的人站在同一个空间里看同一件事。今天离那个目标还很远，但这条路是走得通的。我会继续做下去。谢谢。",
    )
    print("  9 收束")

    prs.save(OUT)
    print(f"\n已写出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
