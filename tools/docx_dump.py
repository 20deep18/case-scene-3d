# -*- coding: utf-8 -*-
"""把 DOCX 全文（含表格、样式）导出为便于阅读的文本/Markdown。"""
import sys
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn


def iter_block_items(parent):
    from docx.document import Document as _Doc
    if isinstance(parent, _Doc):
        parent_elm = parent.element.body
    else:
        parent_elm = parent._element
    for child in parent_elm.iterchildren():
        if child.tag == qn('w:p'):
            yield Paragraph(child, parent)
        elif child.tag == qn('w:tbl'):
            yield Table(child, parent)


def table_to_md(tbl):
    rows = []
    for r in tbl.rows:
        cells = []
        for c in r.cells:
            txt = ' '.join(p.text.strip() for p in c.paragraphs if p.text.strip())
            cells.append(txt.replace('|', '\\|'))
        rows.append(cells)
    if not rows:
        return ''
    width = max(len(r) for r in rows)
    out = []
    for i, r in enumerate(rows):
        r = r + [''] * (width - len(r))
        out.append('| ' + ' | '.join(r) + ' |')
        if i == 0:
            out.append('|' + '---|' * width)
    return '\n'.join(out)


def main(path, out):
    doc = Document(path)
    lines = []
    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            t = block.text.strip()
            if not t:
                continue
            style = (block.style.name or '')
            if style.startswith('Heading') or style.startswith('标题'):
                lvl = ''.join(ch for ch in style if ch.isdigit()) or '1'
                lines.append('#' * min(int(lvl) + 1, 6) + ' ' + t)
            elif style.startswith('Title') or style == 'Title':
                lines.append('# ' + t)
            else:
                lines.append(t)
            lines.append('')
        else:
            lines.append(table_to_md(block))
            lines.append('')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'段落/表格块已导出到 {out}')
    print(f'总字符数: {len(chr(10).join(lines))}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
