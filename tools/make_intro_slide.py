#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""「自我介绍」页模板：同时出**预览图**与**可编辑单页 PPTX**（同一套坐标）。

为什么两样一起出：这台机器上渲染器预览不可靠（LibreOffice 会把内容放大 1.5×），
所以预览图我用 PIL 直接画（像素级可信），PPTX 用同一组英寸坐标生成（PowerPoint 里可编辑）。

用法: <python> tools/make_intro_slide.py
产出: output/deck/intro-template.png 与 output/deck/01-自我介绍-模板.pptx
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "output" / "deck"
PNG = OUT_DIR / "intro-template.png"
PPTX = OUT_DIR / "01-自我介绍-模板.pptx"

F_REG = r"C:\Windows\Fonts\msyh.ttc"
F_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"

# ===== 文案（要改就改这一段）=====================================================
KICKER = "汇报人"
NAME = "黄祖飞"
TAGLINE = "计算机 · 网络 · AI 思维"
BLOCKS = [
    ("兴趣", "喜欢把计算机拆开看它怎么运转，也着迷于网络里信息怎么连起来；"
             "习惯用 AI 的方式思考 —— 先问一句「能不能让模型来做」。"),
    ("展望", "用 AI 学得更快、想得更远，把创造思维做成看得见的作品。"),
    ("这次", "这套系统，就是我把兴趣变成作品的一次尝试。"),
]
FOOTER = "案发现场三维重建与协同研判系统 · 作品讲解"

# ===== 配色 =====================================================================
INK = (0x1B, 0x22, 0x2B)
TEAL = (0x0E, 0x6B, 0x8A)
MUTED = (0x6B, 0x76, 0x82)
LINE = (0xD8, 0xDF, 0xE6)
SOFT = (0xE9, 0xF2, 0xF6)

# ===== 版式（英寸；PNG 与 PPTX 共用）=============================================
LEFT = 1.05
Y_KICKER = 0.80
Y_NAME = 1.06
Y_TAGLINE = 2.50
Y_RULE = 3.00
Y_BLOCKS = 3.28
LABEL_H = 0.34
LINE_H = 0.40
BLOCK_GAP = 0.20
COL_W = 6.30          # 正文栏宽
FOOTER_Y = 7.00
ART = (10.55, 3.95, 2.05)   # 装饰圆心与半径

PNG_DPI = 144


def px(inch: float) -> int:
    return int(round(inch * PNG_DPI))


def wrap(draw: ImageDraw.ImageDraw, text: str, font, max_px: int) -> list[str]:
    """按像素宽度折行；避免标点落到行首（中文排版常见毛病）。"""
    lines, cur = [], ""
    for ch in text:
        if draw.textlength(cur + ch, font=font) <= max_px or not cur:
            cur += ch
        elif ch in "，。；：、！？”）":
            cur += ch          # 标点跟着上一行走，不留孤零零一行
            lines.append(cur)
            cur = ""
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


def draw_art(img: Image.Image) -> None:
    """右侧极简线稿：等轴测盒子 + 一台相机 —— 呼应“拍摄 → 三维”。"""
    d = ImageDraw.Draw(img, "RGBA")
    cx, cy, r = px(ART[0]), px(ART[1]), px(ART[2])
    for i in range(22):                                  # 淡淡的同心圆底
        rad = r - i * 2.4
        if rad > 0:
            d.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), outline=(*SOFT, 30), width=2)

    w, h, dz = r * 1.20, r * 0.35, r * 0.50

    def iso(x, y, z):
        return (cx + (x - y) * w, cy + (x + y - 1) * h - z * dz)

    base = [(0, 0), (1, 0), (1, 1), (0, 1)]
    top = [iso(x, y, 0.0) for x, y in base]
    bot = [iso(x, y, -1.0) for x, y in base]
    for i in range(4):
        d.line((bot[i], bot[(i + 1) % 4]), fill=(*TEAL, 95), width=3)
        d.line((top[i], top[(i + 1) % 4]), fill=(*TEAL, 60), width=2)
        d.line((top[i], bot[i]), fill=(*TEAL, 80), width=3)
    for k in (0.25, 0.5, 0.75):                          # 地面网格
        d.line((iso(k, 0, -1), iso(k, 1, -1)), fill=(*TEAL, 30), width=2)
        d.line((iso(0, k, -1), iso(1, k, -1)), fill=(*TEAL, 30), width=2)
    cam = (cx + r * 1.02, cy - r * 1.05)                 # 相机在盒外上方
    d.ellipse((cam[0] - 8, cam[1] - 8, cam[0] + 8, cam[1] + 8), fill=(*TEAL, 220))
    for c in (top[0], top[1], top[2]):
        d.line((cam, c), fill=(*TEAL, 55), width=2)


def render_png() -> None:
    W, H = px(13.333), px(7.5)
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw_art(img)
    d = ImageDraw.Draw(img)
    f_kicker = ImageFont.truetype(F_BOLD, px(0.24))
    f_name = ImageFont.truetype(F_BOLD, px(0.92))
    f_tag = ImageFont.truetype(F_BOLD, px(0.30))
    f_label = ImageFont.truetype(F_BOLD, px(0.20))
    f_body = ImageFont.truetype(F_REG, px(0.215))
    f_foot = ImageFont.truetype(F_REG, px(0.155))

    x = px(LEFT)
    d.text((x, px(Y_KICKER)), KICKER, font=f_kicker, fill=TEAL)
    d.text((x, px(Y_NAME)), NAME, font=f_name, fill=INK)
    d.text((x, px(Y_TAGLINE)), TAGLINE, font=f_tag, fill=TEAL)
    d.line((x, px(Y_RULE), x + px(5.6), px(Y_RULE)), fill=LINE, width=3)

    y = Y_BLOCKS
    for label, body in BLOCKS:
        d.text((x, px(y)), label, font=f_label, fill=MUTED)
        lines = wrap(d, body, f_body, px(COL_W))
        ty = y + LABEL_H
        for ln in lines:
            d.text((x, px(ty)), ln, font=f_body, fill=INK)
            ty += LINE_H
        y = ty + BLOCK_GAP
    d.text((x, px(FOOTER_Y)), FOOTER, font=f_foot, fill=MUTED)
    img.save(PNG)
    print(f"已写出 {PNG}  ({img.size[0]}x{img.size[1]})")


def _font(run, size_pt, bold=False, color=INK):
    run.font.size = Pt(size_pt)
    run.font.bold = bold
    run.font.color.rgb = RGBColor(*color)
    run.font.name = "微软雅黑"
    rpr = run._r.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rf)
    for attr in ("w:eastAsia", "w:ascii", "w:hAnsi"):
        rf.set(qn(attr), "微软雅黑")


INTRO_NOTES = (
    "开场先介绍自己：我叫黄祖飞，兴趣是计算机、网络和 AI 思维 —— 喜欢拆开看它怎么运转、"
    "着迷于网络里信息怎么连起来，也习惯用 AI 的方式思考。往后我想用 AI 学得更快、想得更远，"
    "把创造思维做成看得见的作品；今天这套系统就是一次尝试。"
)


def add_intro(prs, *, notes: str = INTRO_NOTES):
    """把「自我介绍」页加到给定 Presentation 的最前面（调用方保证顺序）。"""
    s = prs.slides.add_slide(prs.slide_layouts[6])

    # 装饰：与预览图同一段线稿（裁出来当图片，保证两边一致）
    full = Image.new("RGBA", (px(13.333), px(7.5)), (255, 255, 255, 0))
    draw_art(full)
    half = 2.35
    crop = full.crop((px(ART[0] - half), px(ART[1] - half), px(ART[0] + half), px(ART[1] + half)))
    art_path = OUT_DIR / "_intro-art.png"
    crop.save(art_path)
    s.shapes.add_picture(str(art_path), Inches(ART[0] - half), Inches(ART[1] - half),
                         width=Inches(half * 2), height=Inches(half * 2))

    def tb(x, y, w, h, text, size, bold=False, color=INK, spacing=1.28):
        box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.line_spacing = spacing
        run = p.add_run()
        run.text = text
        _font(run, size, bold, color)
        return box

    tb(LEFT, Y_KICKER, 5.0, 0.40, KICKER, 16, True, TEAL)
    tb(LEFT - 0.03, Y_NAME, 7.0, 1.40, NAME, 58, True, INK)
    tb(LEFT, Y_TAGLINE, 6.6, 0.50, TAGLINE, 22, True, TEAL)

    measure = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    f_body_px = ImageFont.truetype(F_REG, px(0.215))
    y = Y_BLOCKS
    for label, body in BLOCKS:
        tb(LEFT, y, 1.4, 0.32, label, 13, True, MUTED)
        lines = wrap(measure, body, f_body_px, px(COL_W))
        tb(LEFT, y + 0.33, COL_W, LINE_H * len(lines) + 0.15, body, 15.5, False, INK)
        y += 0.33 + LINE_H * len(lines) + BLOCK_GAP
    tb(LEFT, FOOTER_Y, 8.4, 0.35, FOOTER, 11, False, MUTED)

    if notes:
        s.notes_slide.notes_text_frame.text = notes
    art_path.unlink(missing_ok=True)
    return s


def build_pptx() -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    sldSz = prs._element.find(qn("p:sldSz"))
    if sldSz is not None:
        sldSz.set("type", "screen16x9")
    add_intro(prs)
    prs.save(PPTX)
    print(f"已写出 {PPTX}")


if __name__ == "__main__":
    render_png()
    build_pptx()
