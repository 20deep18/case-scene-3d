#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Markdown → DOCX（通用，支持标题/段落/列表/表格/代码块/引用）。

为什么自己写：本机没有 pandoc，而文档里有**表格**（实测数据），
`python-docx` 转表格要手动建行建列、逐个 run 设中文字体，所以封装成工具复用。

用法:
    <python> tools/md_to_docx.py docs/技术现状与待决问题-给专家.md
    （同目录生成同名 .docx；之后用 LibreOffice 转 PDF）
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

FONT = "微软雅黑"
MONO = "Consolas"
ACCENT = RGBColor(0x0E, 0x6B, 0x8A)
MUTED = RGBColor(0x5A, 0x66, 0x72)


def set_font(run, *, size=10.5, bold=False, color=None, mono=False, italic=False):
    name = MONO if mono else FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color is not None:
        run.font.color.rgb = color
    run.font.name = name
    rpr = run._r.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rf)
    for attr in ("w:eastAsia", "w:ascii", "w:hAnsi"):
        rf.set(qn(attr), name)


def add_rich(par, text, *, size=10.5, base_color=None):
    """处理 **加粗**、`代码`、〔提示〕。"""
    for chunk in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text):
        if not chunk:
            continue
        if chunk.startswith("**") and chunk.endswith("**"):
            set_font(par.add_run(chunk[2:-2]), size=size, bold=True, color=base_color)
        elif chunk.startswith("`") and chunk.endswith("`"):
            set_font(par.add_run(chunk[1:-1]), size=size - 0.5, mono=True, color=base_color)
        else:
            set_font(par.add_run(chunk), size=size, color=base_color)


def add_table(doc, rows: list[list[str]]):
    """rows[0] 是表头。列宽按内容长度粗分，避免中文挤成一列。"""
    ncol = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=ncol)
    table.style = "Table Grid"
    table.autofit = True
    for i, row in enumerate(rows):
        for j in range(ncol):
            cell = table.cell(i, j)
            cell.text = ""
            par = cell.paragraphs[0]
            par.paragraph_format.space_after = Pt(2)
            txt = row[j] if j < len(row) else ""
            add_rich(par, txt, size=9)
            for run in par.runs:
                run.font.bold = i == 0 or run.font.bold
    doc.add_paragraph()


def convert(md_path: Path, out_path: Path) -> None:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    doc = Document()
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Cm(1.8)
        s.left_margin = s.right_margin = Cm(1.9)

    i = 0
    in_code = False
    code_buf: list[str] = []
    table_buf: list[list[str]] = []

    def flush_table():
        nonlocal table_buf
        if table_buf:
            add_table(doc, table_buf)
            table_buf = []

    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()

        # 代码块
        if line.startswith("```"):
            if in_code:
                p = doc.add_paragraph()
                p.paragraph_format.space_after = Pt(8)
                p.paragraph_format.left_indent = Cm(0.4)
                set_font(p.add_run("\n".join(code_buf)), size=8.5, mono=True, color=MUTED)
                code_buf = []
                in_code = False
            else:
                flush_table()
                in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(raw)
            i += 1
            continue

        # 表格
        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                i += 1
                continue  # 分隔行
            table_buf.append(cells)
            i += 1
            continue
        flush_table()

        if not line.strip():
            i += 1
            continue

        # 标题
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14 if level <= 2 else 10)
            p.paragraph_format.space_after = Pt(6)
            size = {1: 18, 2: 14, 3: 12, 4: 11}[level]
            # 标题里的行内代码（反引号）要剥掉，否则会原样显示成 `tools/xxx.py`
            head_txt = m.group(2).strip().replace("`", "")
            set_font(p.add_run(head_txt), size=size, bold=True,
                     color=ACCENT if level <= 2 else None)
            i += 1
            continue

        # 分隔线
        if re.fullmatch(r"-{3,}|\*{3,}", line.strip()):
            i += 1
            continue

        # 引用
        if line.startswith("> "):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.6)
            p.paragraph_format.space_after = Pt(4)
            add_rich(p, line[2:].strip(), size=9.5, base_color=MUTED)
            i += 1
            continue

        # 列表
        m = re.match(r"^(\s*)[-*]\s+(.*)$", line)
        if m:
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.space_after = Pt(3)
            add_rich(p, m.group(2), size=10.5)
            i += 1
            continue
        m = re.match(r"^(\s*)\d+\.\s+(.*)$", line)
        if m:
            p = doc.add_paragraph(style="List Number")
            p.paragraph_format.space_after = Pt(3)
            add_rich(p, m.group(2), size=10.5)
            i += 1
            continue

        # 普通段落
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        p.paragraph_format.line_spacing = 1.25
        add_rich(p, line.strip(), size=10.5)
        i += 1

    flush_table()
    doc.save(out_path)
    print(f"已写出 {out_path}")


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python tools/md_to_docx.py <input.md> [output.docx]")
        return 2
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".docx")
    if not src.exists():
        print(f"找不到 {src}")
        return 1
    convert(src, dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
