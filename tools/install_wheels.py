"""
把 wheel 直接解包到目标目录（不依赖 pip）。
用法:
  python tools/install_wheels.py --dest pydeps wheels/a.whl wheels/b.whl
  python tools/install_wheels.py --dest pydeps --all wheels/
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

# Windows 控制台默认 GBK，统一 UTF-8 输出，避免中文/符号乱码
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass


def wheel_targets(zf: zipfile.ZipFile, dist_info: str) -> list[tuple[str, str]]:
    """
    解析 WHEEL/RECORD，返回 (zip 内成员, 目标相对路径) 列表。
    处理 *.data/purelib|platlib|scripts|data 布局。
    """
    wheel_meta = None
    for name in zf.namelist():
        if name.endswith(".dist-info/WHEEL"):
            wheel_meta = name
            break
    root_is_purelib = True
    if wheel_meta:
        text = zf.read(wheel_meta).decode("utf-8", "replace")
        for line in text.splitlines():
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip().lower() == "root-is-purelib":
                root_is_purelib = value.strip().lower() == "true"

    data_prefix = dist_info[:-len(".dist-info")] + ".data/"
    out: list[tuple[str, str]] = []
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        if name.startswith(data_prefix):
            rest = name[len(data_prefix):]
            category, _, tail = rest.partition("/")
            if category in ("purelib", "platlib"):
                out.append((name, tail))
            elif category == "scripts":
                out.append((name, f"Scripts/{tail}"))
            elif category == "data":
                out.append((name, f"data/{tail}"))
            elif category == "headers":
                continue
            else:
                out.append((name, rest))
        else:
            out.append((name, name))
    return out


def install_wheel(path: Path, dest: Path, verbose: bool = True) -> dict:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        dist_info = next((n.split("/")[0] for n in names if n.endswith(".dist-info/METADATA")), None)
        if dist_info is None:
            raise ValueError(f"{path.name}: 不是合法的 wheel（缺少 METADATA）")
        targets = wheel_targets(zf, dist_info)
        count = 0
        total = 0
        for member, rel in targets:
            info = zf.getinfo(member)
            total += info.file_size
            out_path = dest / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and out_path.stat().st_size == info.file_size:
                count += 1
                continue
            with zf.open(member) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            count += 1
        # 记录已安装的发行版，便于自检
        record_dir = dest / ".installed"
        record_dir.mkdir(parents=True, exist_ok=True)
        (record_dir / f"{dist_info}.txt").write_text(
            f"{path.name}\nmembers={count}\nbytes={total}\n", encoding="utf-8"
        )
        if verbose:
            print(f"  ✓ {dist_info:<52} {count:>6} 文件 {total / 1048576:>8.1f} MB")
        return {"dist": dist_info, "files": count, "bytes": total}


def main() -> int:
    ap = argparse.ArgumentParser(description="离线安装 wheel 到指定目录")
    ap.add_argument("--dest", required=True, help="目标目录（会加入 sys.path）")
    ap.add_argument("--all", action="store_true", help="安装目录下所有 .whl")
    ap.add_argument("wheels", nargs="*", help="wheel 文件路径")
    args = ap.parse_args()

    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for w in args.wheels:
        p = Path(w)
        if p.is_dir():
            files.extend(sorted(p.glob("*.whl")))
        elif p.suffix == ".whl":
            files.append(p)
        else:
            print(f"!! 跳过非 wheel 文件: {w}")
    if not files:
        print("没有找到任何 wheel 文件")
        return 2

    # 依赖顺序：先装纯 python 的（如 filelock/typing_extensions），再装 numpy/torch 等
    def weight(p: Path) -> tuple:
        n = p.name.lower()
        pure = 0 if ("py3-none-any" in n or "py2.py3" in n) else 1
        return (pure, n)

    files.sort(key=weight)

    ok = 0
    fail = 0
    for f in files:
        try:
            install_wheel(f, dest)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {f.name}: {type(exc).__name__}: {exc}")
            fail += 1
    print(f"\n完成：{ok} 个 wheel 安装到 {dest}，失败 {fail} 个")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
