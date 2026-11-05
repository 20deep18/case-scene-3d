#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把演讲稿 markdown 转成**手机上好读**的 Word/PDF。

为什么不用原 markdown：手机上打开 .md 不方便（多数手机没有 markdown 阅读器，
纯文本又丢掉了标题层级与强调）。所以排成 A4 大字版：
正文 14pt、行距 1.5、窄边距，手机上放大后依然清楚。

用法:
    <python> tools/make_speech_doc.py
产出:
    output/deck/演讲稿-作品讲解.docx
（之后用 LibreOffice 转 PDF 供手机直接阅读）
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Cm

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "output" / "deck" / "演讲稿-作品讲解.md"
OUT = ROOT / "output" / "deck" / "演讲稿-作品讲解.docx"

BODY_PT = 14.0      # 手机阅读：正文放大到 14pt
LINE = 1.5
ACCENT = RGBColor(0x0E, 0x6B, 0x8A)
MUTED = RGBColor(0x6B, 0x76, 0x82)


def set_font(run, *, size=BODY_PT, bold=False, color=None, italic=False, mono=False):
    name = "Consolas" if mono else "微软雅黑"
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color is not None:
        run.font.color.rgb = color
    # 中文字形必须显式指定 eastAsia，否则字体设置对中文不生效
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rf)
    rf.set(qn("w:eastAsia"), name)
    rf.set(qn("w:ascii"), name)
    rf.set(qn("w:hAnsi"), name)


def add_rich(par, text, *, size=BODY_PT, color=None):
    """处理 **加粗**、`代码`、〔提示〕三种行内标记。"""
    text = text.rstrip()
    for chunk in re.split(r"(\*\*[^*]+\*\*|`[^`]+`|〔[^〕]*〕)", text):
        if not chunk:
            continue
        if chunk.startswith("**") and chunk.endswith("**"):
            set_font(par.add_run(chunk[2:-2]), size=size, bold=True, color=color)
        elif chunk.startswith("`") and chunk.endswith("`"):
            set_font(par.add_run(chunk[1:-1]), size=size - 1, color=color, mono=True)
        elif chunk.startswith("〔") and chunk.endswith("〕"):
            set_font(par.add_run(chunk), size=size - 1, color=MUTED, italic=True)
        else:
            set_font(par.add_run(chunk), size=size, color=color)


def body_par(doc, text, *, style=None, space_after=8, indent=0.0):
    p = doc.add_paragraph(style=style)
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = LINE
    pf.space_after = Pt(space_after)
    if indent:
        pf.left_indent = Cm(indent)
    add_rich(p, text)
    return p


def main() -> int:
    if not SRC.exists():
        print(f"找不到讲稿：{SRC}")
        return 1
    lines = SRC.read_text(encoding="utf-8").splitlines()

    doc = Document()
    # 窄边距 + 大字，手机上放大后仍然好读
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Cm(1.6)
        s.left_margin = s.right_margin = Cm(1.8)

    # 标题
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.LEFT
    set_font(t.add_run("案发现场三维重建与协同研判系统"), size=24, bold=True)
    sub = doc.add_paragraph()
    set_font(sub.add_run("作品讲解 · 演讲稿"), size=16, color=ACCENT, bold=True)

    in_quote = False
    seen_h1 = False
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            in_quote = False
            continue
        if line.startswith("# "):
            # 文档开头那行 H1 与我自己加的封面标题重复，跳过（手机上看不要两个大标题）
            if not seen_h1:
                seen_h1 = True
                continue
            h = doc.add_paragraph()
            h.paragraph_format.space_before = Pt(18)
            h.paragraph_format.space_after = Pt(8)
            set_font(h.add_run(line[2:].strip()), size=19, bold=True, color=ACCENT)
            continue
        if line.startswith("## "):                      # 页标题
            h = doc.add_paragraph()
            h.paragraph_format.space_before = Pt(16)
            h.paragraph_format.space_after = Pt(6)
            set_font(h.add_run(line[3:].strip()), size=17, bold=True)
            continue
        if line.startswith("### "):
            h = doc.add_paragraph()
            set_font(h.add_run(line[4:].strip()), size=15, bold=True)
            continue
        if line.startswith("---"):                      # 分隔线
            p = doc.add_paragraph()
            set_font(p.add_run("· · ·"), size=12, color=MUTED)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            continue
        if line.startswith("> "):                       # 说明／提示块
            body_par(doc, line[2:].strip(), space_after=4, indent=0.6)
            continue
        if line.startswith("- "):                       # 列表
            body_par(doc, "• " + line[2:].strip(), space_after=4, indent=0.5)
            continue
        if re.match(r"^\d+\.\s", line):                 # 有序列表
            body_par(doc, line, space_after=4, indent=0.5)
            continue
        body_par(doc, line)

    doc.save(OUT)
    print(f"已写出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
