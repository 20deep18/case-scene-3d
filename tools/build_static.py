#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""构建「只读演示版」静态包（部署到 Cloudflare Pages / Vercel / GitHub Pages）。

为什么要它：评审只需要**看**模型和结构线条，不需要上传/建模/协同。
静态托管的好处是 URL 永久稳定（二维码不会失效）、免费、且不依赖任何一台机器开着。

做法：
  1. 从**正在运行的本地服务**取三样只读数据：
     · /api/projects           → 案件列表（演示用只有一个）
     · /api/projects/<id>      → 该案件的详情（含标注，看的时候能显示物证点）
     · /models/index.json      → 模型清单
  2. 过滤掉超过 Cloudflare Pages 单文件上限（25 MB）的文件
  3. 拷贝 app/web 与选中的模型文件
  4. 在 index.html 注入 `window.__DSH_STATIC__ = true` 和静态案件地址
  5. （可选）生成二维码指向最终域名

用法:
    <python> tools/build_static.py --out dist-static
    <python> tools/build_static.py --out dist-static --public-url https://xxx.pages.dev
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "app" / "web"
MODELS = ROOT / "data" / "models"
PAGES_FILE_LIMIT = 25 * 1024 * 1024      # Cloudflare Pages：单文件 25 MB

# 演示用模型**策展列表**（按展示顺序）。
# 为什么需要：data/models 里有 20 多个文件，一半是调试残留（尺度修复验证/各向异性修复…），
# 评审点开二维码看到一长串怪名字，根本不知道点哪个 —— 第一印象就废了。
# 这里只保留"每个能力一个代表"，其余在本地留着，不进静态包。
DEMO_MODELS = [
    # 三个模型各代表一种技术内涵（截图与讲解见 docs/模型讲解-三张截图.md）：
    "宿舍现场-彩色-精简",   # ① AI：VGGT 几何 + 开放词表检测 → 5 个带类别/置信度的盒子
    "room-real",            # ② 结构线：RANSAC 墙面 + 曼哈顿正交 → 房间盒（误差 0.9%）
    "柜子现场-致密",         # ③ 渲染质量：近距离稠密 3DGS，细节与颜色最接近实物
]


def http_json(url: str):
    with urllib.request.urlopen(url, timeout=20) as r:  # noqa: S310
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist-static")
    ap.add_argument("--server", default="http://127.0.0.1:8787", help="本地在跑的服务地址")
    ap.add_argument("--public-url", default=None, help="部署后的域名（给了就顺便生成二维码）")
    ap.add_argument("--all-models", action="store_true",
                    help="不做策展，把 data/models 里所有模型都打进静态包（默认只打 DEMO_MODELS 里的）")
    ap.add_argument("--project-id", default=None, help="指定演示用案件 id（默认自动挑）")
    ap.add_argument("--keep-large", action="store_true", help="保留超过 25 MB 的文件（GitHub Pages 能接受，Pages 不接受）")
    args = ap.parse_args()

    out = (ROOT / args.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # ---- 1) 只读数据 ------------------------------------------------
    try:
        projects = http_json(f"{args.server}/api/projects")
        plist = projects.get("projects") or projects
    except Exception as exc:  # noqa: BLE001
        print(f"✗ 取案件列表失败（{args.server} 上的服务没跑？）：{exc}")
        return 1
    if not plist:
        print("✗ 没有任何案件")
        return 1
    # 挑案件：显式指定 > 名字不像测试的 > 第一个。
    # ⚠️ 不处理的话会抓到回归测试留下的 `delete-test`，评委打开就看到"案件：delete-test"（实测踩到）。
    if args.project_id:
        pid = args.project_id
    else:
        candidates = [p for p in plist if isinstance(p, dict)]
        # ⚠️ 排除词只写"测试残留"的**特征**，不要写"测试"二字 ——
        #    演示案件本身就叫「测试名称」，写进去会把它一起排掉（实测踩到）。
        bad_tokens = ("delete-test", "selftest", "browser-test", "tmp-")
        goods = [p for p in candidates
                 if not any(k in str(p.get("id", "")).lower() for k in bad_tokens)]
        pid = (goods[0] if goods else candidates[0])["id"]
    detail = http_json(f"{args.server}/api/projects/{pid}")
    (out / "project.json").write_text(json.dumps(detail, ensure_ascii=False), encoding="utf-8")
    anns = (detail.get("project") or detail).get("annotations") or []
    print(f"只读案件：{pid}（标注 {len(anns)} 条）")

    index = http_json(f"{args.server}/models/index.json")
    models = index.get("models") or []

    if not args.all_models:
        by_name = {}
        for m in models:
            src_name = urllib_request_unquote(m.get("url", "")).rsplit("/", 1)[-1]
            by_name[Path(src_name).stem] = m
        picked = [by_name[k] for k in DEMO_MODELS if k in by_name]
        missing = [k for k in DEMO_MODELS if k not in by_name]
        print(f"模型策展：保留 {len(picked)}/{len(models)} 个"
              + (f"（缺少：{', '.join(missing)}）" if missing else ""))
        models = picked
    kept, dropped = [], []
    for m in models:
        name = m.get("name") or ""
        # ⚠️ 必须从 url 里取**真实文件名**：索引里的 name 可能是去扩展名的显示名
        #    （实测 "柜子现场" 对应 "柜子现场.splat"，按 name 找文件会全部报"不存在"）。
        rel = urllib_request_unquote(m.get("url", "")).rsplit("/", 1)[-1]
        src = MODELS / rel
        if not src.exists():
            dropped.append((name, "文件不存在"))
            continue
        size = src.stat().st_size
        if size > PAGES_FILE_LIMIT and not args.keep_large:
            dropped.append((name, f"{size / 1048576:.1f} MB > 25 MB"))
            continue
        kept.append((m, src, size))
    print(f"模型：保留 {len(kept)} 个，排除 {len(dropped)} 个")
    for name, why in dropped:
        print(f"  排除 {name}（{why}）")

    # 只保留仍存在的模型条目
    index["models"] = [m for m, _src, _sz in kept]
    (out / "models").mkdir(exist_ok=True)
    (out / "models" / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")

    # ---- 2) 拷贝模型文件（含结构/家具/场景模式 sidecar）--------------
    copied = 0
    for m, src, _sz in kept:
        shutil.copy2(src, out / "models" / src.name)
        copied += 1
        stem = src.name.rsplit(".", 1)[0]
        for side in (".structure.json", ".furniture.json", ".scenemode.json"):
            s = MODELS / f"{stem}{side}"
            if s.exists() and s.stat().st_size <= PAGES_FILE_LIMIT:
                shutil.copy2(s, out / "models" / s.name)
                copied += 1
    total_mb = sum(f.stat().st_size for f in (out / "models").rglob("*") if f.is_file()) / 1048576
    print(f"拷贝 {copied} 个文件，模型目录共 {total_mb:.1f} MB")

    # ---- 3a) 前端依赖 vendor/（three.js 等）------------------------
    # ⚠️ 服务端把 /vendor 映射到仓库根的 vendor/，**不在 app/web 里** ——
    #    第一版只拷了 app/web，结果静态版打开就 404（three.module.js 加载失败 → 整页起不来）。
    vendor = ROOT / "vendor"
    if not vendor.exists():
        print("✗ 找不到 vendor/（three.js 等前端依赖），静态版无法运行")
        return 1
    # ⚠️ 只拷**代码里真正 import 的** vendor 文件。
    #    原因：Cloudflare Pages 的「拖拽上传」上限是 **1,000 个文件**，
    #    而整个 vendor/ 有 1,117 个（three.js 的 examples 目录几乎都没用到），
    #    整目录拷过去会直接超限、拖不上去（实测文档：Drag and drop | 1,000 files）。
    needed = {"three/build/three.module.js"}
    import re as _re
    for js in (WEB / "js").glob("*.js"):
        for m in _re.finditer(r"three/addons/([^'\")\s]+)", js.read_text(encoding="utf-8")):
            needed.add(f"three/examples/jsm/{m.group(1)}")
    # **依赖闭包**：three.module.js 会 import './three.core.js'、addons 也可能 import 兄弟文件，
    # 只拷入口文件会 404（实测：three.core.js 缺失 → 整个 app.js 起不来）。
    # 这里反复扫描已收集文件的相对 import，直到不再新增。
    import re as _re2
    rel_import = _re2.compile(r"""(?:from\s*|import\s*\()['\"](\.[^'\"]+)['\"]""")
    changed = True
    while changed:
        changed = False
        for rel in list(needed):
            src = vendor / rel
            if not src.exists():
                continue
            try:
                text = src.read_text(encoding="utf-8", errors="ignore")
            except Exception:  # noqa: BLE001
                continue
            for m2 in rel_import.finditer(text):
                dep = (Path(rel).parent / m2.group(1)).as_posix()
                # 规范化 ./ ../ 与缺扩展名
                parts: list[str] = []
                for seg in dep.split("/"):
                    if seg == "..":
                        if parts:
                            parts.pop()
                    elif seg in (".", ""):
                        continue
                    else:
                        parts.append(seg)
                dep = "/".join(parts)
                if not dep.endswith(".js"):
                    dep += ".js"
                if dep not in needed:
                    needed.add(dep)
                    changed = True

    copied_v, missing_v = 0, []
    for rel in sorted(needed):
        src = vendor / rel
        if not src.exists():
            missing_v.append(rel)
            continue
        dst = out / "vendor" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied_v += 1
    print(f"拷贝 vendor/：{copied_v} 个文件（按 import 精确拷贝）"
          + (f"，缺 {missing_v}" if missing_v else ""))

    # ---- 3) 前端 ------------------------------------------------
    for item in WEB.iterdir():
        if item.name == "qr":
            continue
        if item.is_dir():
            shutil.copytree(item, out / item.name)
        else:
            shutil.copy2(item, out / item.name)

    # ---- 4) 注入静态模式标记 ------------------------------------
    idx = out / "index.html"
    html = idx.read_text(encoding="utf-8")
    boot = (
        "  <script>\n"
        "    // 静态演示版（只读）：没有服务端。api.js / collab.js 据此降级。\n"
        "    window.__DSH_STATIC__ = true;\n"
        "    document.documentElement.classList.add('static-mode');\n"
        "    window.__STATIC_PROJECT_URL__ = '/project.json';\n"
        "  </script>\n"
        "  <script type=\"importmap\">"
    )
    assert '<script type="importmap">' in html, "index.html 里找不到 importmap 锚点"
    html = html.replace('  <script type="importmap">', boot, 1)
    idx.write_text(html, encoding="utf-8")

    # ---- 5) 部署说明 + 可选二维码 -------------------------------
    (out / ".nojekyll").write_text("", encoding="utf-8")
    (out / "_headers").write_text(
        "# Cloudflare Pages：模型文件不可变，可长缓存\n/models/*\n  Cache-Control: public, max-age=86400\n",
        encoding="utf-8")

    qr_note = ""
    if args.public_url:
        qr_note = f"\n二维码：pyruntime312\\python.exe tools\\make_qr.py --url {args.public_url}\n"
    (out / "DEPLOY.txt").write_text(
        "只读演示版（静态）\n"
        "==================\n"
        "部署（Cloudflare Pages，免费，不需要域名）：\n"
        "  1. 打开 https://dash.cloudflare.com → Workers & Pages → Create → Pages\n"
        "  2. 选 Upload assets，把本目录整个拖进去\n"
        "  3. 得到 https://<项目名>.pages.dev（HTTPS）\n"
        f"{qr_note}"
        "注意：这里只有「看」——模型、结构线、家具盒、标注。\n"
        "      上传视频、建模、协同编辑在本版本中不可用（会提示只读演示版）。\n",
        encoding="utf-8")
    print(f"\n已生成静态包：{out}")
    print("部署：把该目录拖到 Cloudflare Pages 的 Upload assets（见 DEPLOY.txt）")
    return 0


def urllib_request_unquote(url: str) -> str:
    from urllib.parse import unquote

    return unquote(url)


if __name__ == "__main__":
    raise SystemExit(main())
