import json
import urllib.request

MIRROR = "https://pypi.tuna.tsinghua.edu.cn/pypi"

PKGS = ["torch", "torchvision", "gsplat", "pycolmap", "opencv-python", "plyfile", "numpy", "scipy", "tqdm", "imageio", "opencv-contrib-python"]


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "probe/1.0"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read().decode("utf-8"))


for name in PKGS:
    try:
        info = get(f"{MIRROR}/{name}/json")
        ver = info["info"]["version"]
        urls = info["urls"]
        tags = []
        for u in urls:
            fn = u["filename"]
            if fn.endswith(".whl") and ("cp313" in fn or "cp312" in fn or "py3-none" in fn or "abi3" in fn):
                tags.append(fn)
        print(f"### {name} latest={ver}  files={len(urls)}")
        for t in tags[-8:]:
            print("    ", t)
    except Exception as e:
        print(f"### {name} ERROR {type(e).__name__}: {e}")
