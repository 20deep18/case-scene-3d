"""列出 PE 文件（DLL/EXE）的导入依赖，用于诊断 Windows DLL 加载失败。"""
from __future__ import annotations

import struct
import sys
from pathlib import Path


def imports(path: Path) -> list[str]:
    data = path.read_bytes()
    if data[:2] != b"MZ":
        raise ValueError("不是 PE 文件")
    pe_off = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe_off:pe_off + 4] != b"PE\0\0":
        raise ValueError("PE 签名不正确")
    coff = pe_off + 4
    num_sections = struct.unpack_from("<H", data, coff + 2)[0]
    opt_size = struct.unpack_from("<H", data, coff + 16)[0]
    opt_off = coff + 20
    magic = struct.unpack_from("<H", data, opt_off)[0]
    is_pe32p = magic == 0x20B
    dd_off = opt_off + (112 if is_pe32p else 96)
    import_rva, import_size = struct.unpack_from("<II", data, dd_off + 8)
    if import_rva == 0:
        return []

    sections = []
    sec_off = opt_off + opt_size
    for i in range(num_sections):
        off = sec_off + i * 40
        name = data[off:off + 8].rstrip(b"\0").decode("latin1")
        vsize, vaddr, rawsize, rawptr = struct.unpack_from("<IIII", data, off + 8)
        sections.append((name, vaddr, vsize, rawptr, rawsize))

    def rva_to_off(rva: int) -> int:
        for _name, vaddr, vsize, rawptr, rawsize in sections:
            if vaddr <= rva < vaddr + max(vsize, rawsize):
                return rawptr + (rva - vaddr)
        return rva

    out = []
    off = rva_to_off(import_rva)
    while True:
        entry = data[off:off + 20]
        if len(entry) < 20 or entry == b"\0" * 20:
            break
        name_rva = struct.unpack_from("<I", entry, 12)[0]
        if name_rva == 0:
            break
        n_off = rva_to_off(name_rva)
        end = data.index(b"\0", n_off)
        out.append(data[n_off:end].decode("latin1"))
        off += 20
    return out


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    for arg in sys.argv[1:]:
        p = Path(arg)
        try:
            deps = imports(p)
        except Exception as exc:  # noqa: BLE001
            print(f"{p.name}: ERROR {exc}")
            continue
        print(f"{p.name} 依赖 {len(deps)} 个 DLL:")
        for d in deps:
            found = ""
            if not d.lower().startswith(("api-ms-", "ext-ms-")):
                found = "  [系统/已见]"
            print(f"    {d}{found}")
