import sys
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn


def iter_block_items(parent):
    body = parent.element.body
    for child in body.iterchildren():
        if child.tag == qn('w:p'):
            yield Paragraph(child, parent)
        elif child.tag == qn('w:tbl'):
            yield Table(child, parent)


def dump(path):
    doc = Document(path)
    print("=" * 100)
    print("FILE:", path)
    print("=" * 100)
    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            style = block.style.name if block.style is not None else ""
            prefix = ""
            if style.startswith("Heading") or style.startswith("标题"):
                prefix = "### "
            elif style in ("List Bullet", "List Paragraph", "List Number"):
                prefix = "- "
            print(f"{prefix}{text}")
        else:
            print("[TABLE]")
            for row in block.rows:
                cells = [c.text.strip().replace("\n", " / ") for c in row.cells]
                print(" | ".join(cells))
            print("[/TABLE]")
    print()


for p in sys.argv[1:]:
    dump(p)
