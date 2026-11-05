#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""给访问地址生成二维码，并生成一个 /qr 页面（大图 + 地址文字）。

为什么要它：评审现场用**手机扫码**进系统，二维码必须是"打开就能看到模型"的入口；
而临时隧道每次重启都会换域名 —— 所以这里支持**一次生成多张**（内网 + 公网），
并在页面上把地址写清楚，避免拿到旧码对不上新域名。

用法:
    <python> tools/make_qr.py --url http://192.168.100.5:8787 ^
        --url https://xxx.trycloudflare.com --title "案发现场三维研判系统"

产出:
    app/web/qr/<slug>.png      每个地址一张二维码
    app/web/qr/index.html      汇总页（浏览器打开 /qr/ 即可投屏给别人扫）
"""

from __future__ import annotations

import argparse
import html
import re
from datetime import datetime
from pathlib import Path

import segno

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "app" / "web" / "qr"


def slugify(url: str) -> str:
    s = re.sub(r"^[a-z]+://", "", url.strip())
    s = s.replace(":", "_").replace("/", "_").replace(".", "-")
    s = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fa5]", "", s)
    return s or "qr"


def main() -> int:
    ap = argparse.ArgumentParser(description="生成访问二维码 + /qr 汇总页")
    ap.add_argument("--url", action="append", required=True, help="访问地址（可重复）")
    ap.add_argument("--title", default="案发现场三维重建与协同研判系统")
    ap.add_argument("--note", default="手机扫码 → 打开现场（只看模型与结构线条，无需安装 App）")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for url in args.url:
        url = url.strip()
        if not url:
            continue
        slug = slugify(url)
        png = OUT_DIR / f"{slug}.png"
        # scale=10 + border=3：投屏/打印都能扫（经验值，太小会扫不出来）
        qr = segno.make(url, error="m")
        qr.save(png, scale=10, border=3, dark="#0b1014", light="#ffffff")
        entries.append({"url": url, "file": png.name, "slug": slug})
        print(f"已生成 {png.name}  →  {url}")

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    cards = "\n".join(
        f"""      <figure class="card">
        <img src="{html.escape(e['file'])}" alt="二维码 {html.escape(e['url'])}" />
        <figcaption>
          <a href="{html.escape(e['url'])}">{html.escape(e['url'])}</a>
          <span class="hint">扫码或点链接</span>
        </figcaption>
      </figure>"""
        for e in entries
    )

    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>扫码访问 · {html.escape(args.title)}</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{
    margin: 0; min-height: 100vh; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 18px;
    background: #0b1014; color: #e8eef4;
    font-family: "Microsoft YaHei", "PingFang SC", system-ui, sans-serif;
    padding: 28px;
  }}
  h1 {{ font-size: clamp(20px, 3.4vw, 30px); margin: 0; text-align: center; }}
  p.note {{ color: #9fb3c2; margin: 0; text-align: center; }}
  .grid {{ display: flex; flex-wrap: wrap; gap: 28px; justify-content: center; }}
  .card {{
    margin: 0; padding: 16px; border-radius: 14px; background: #fff;
    box-shadow: 0 10px 30px rgba(0,0,0,.45); text-align: center;
  }}
  .card img {{ display: block; width: min(46vw, 300px); height: auto; }}
  figcaption {{ margin-top: 10px; font-size: 13px; color: #123; word-break: break-all; }}
  figcaption a {{ color: #0e6b8a; text-decoration: none; font-weight: 600; }}
  .hint {{ display: block; color: #667; font-size: 12px; margin-top: 2px; }}
  footer {{ color: #5d7284; font-size: 12px; text-align: center; }}
</style>
</head>
<body>
  <h1>{html.escape(args.title)}</h1>
  <p class="note">{html.escape(args.note)}</p>
  <div class="grid">
{cards}
  </div>
  <footer>生成于 {stamp}　·　地址变了就要重新生成二维码（见 docs/下一轮计划与外部访问方案.md）</footer>
</body>
</html>
"""
    (OUT_DIR / "index.html").write_text(page, encoding="utf-8")
    print(f"已写出 {OUT_DIR / 'index.html'}（浏览器打开 /qr/ 即可）")
    print(f"共 {len(entries)} 个地址")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
