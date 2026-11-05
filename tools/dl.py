"""Robust downloader with resumable progress and candidate fallback."""
import os
import ssl
import sys
import time
import urllib.request
import urllib.error

CTX = ssl.create_default_context()
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"


def human(n):
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def download(url, dest, min_size=1024 * 64, tries=3):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(1, tries + 1):
        try:
            existing = os.path.getsize(dest) if os.path.exists(dest) else 0
            headers = {"User-Agent": UA}
            if existing:
                headers["Range"] = f"bytes={existing}-"
            req = urllib.request.Request(url, headers=headers)
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
                total = int(r.headers.get("Content-Length", 0))
                if r.status == 200 and existing:
                    existing = 0
                total = total + existing
                mode = "ab" if existing else "wb"
                got = existing
                last = 0
                with open(dest, mode) as f:
                    while True:
                        chunk = r.read(1024 * 256)
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
                        if time.time() - last > 3:
                            speed = (got - existing) / max(time.time() - t0, 1e-6)
                            pct = f"{100*got/total:.1f}%" if total else "?"
                            print(f"    {pct}  {human(got)}/{human(total)}  {human(speed)}/s", flush=True)
                            last = time.time()
            size = os.path.getsize(dest)
            if size < min_size:
                raise RuntimeError(f"file too small ({size} bytes) - probably an error page")
            print(f"  OK {dest}  {human(size)}  ({time.time()-t0:.0f}s)", flush=True)
            return True
        except Exception as e:
            print(f"  attempt {attempt}/{tries} failed for {url}: {type(e).__name__}: {e}", flush=True)
            time.sleep(2)
    return False


if __name__ == "__main__":
    dest = sys.argv[1]
    urls = sys.argv[2:]
    for u in urls:
        print(f"==> trying {u}", flush=True)
        if download(u, dest):
            sys.exit(0)
    print("ALL CANDIDATES FAILED")
    sys.exit(1)
