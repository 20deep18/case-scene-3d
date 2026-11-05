import json
import urllib.request

IDX = "https://pypi.org/pypi"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "probe/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


for name in ["torch", "gsplat", "pycolmap", "opencv-python", "numpy", "plyfile", "scipy", "tqdm", "imageio", "torchvision"]:
    try:
        info = get(f"{IDX}/{name}/json")
        print(f"### {name} latest={info['info']['version']}")
        rels = sorted(info["releases"].keys())
        print("    n_releases:", len(rels), "last:", rels[-6:])
    except Exception as e:
        print(f"### {name} ERROR {type(e).__name__}: {e}")

print()
print("=== torch cp313 win wheels in recent versions ===")
info = get(f"{IDX}/torch/json")
for v in ["2.7.0", "2.7.1", "2.8.0", "2.9.0", "2.9.1", "2.10.0"]:
    files = info["releases"].get(v, [])
    hits = [f["filename"] for f in files if "cp313" in f["filename"] and "win_amd64" in f["filename"]]
    print(f"  torch {v}: {hits}")

print()
print("=== gsplat releases ===")
g = get(f"{IDX}/gsplat/json")
for v in sorted(g["releases"].keys())[-10:]:
    files = g["releases"][v]
    print(f"  {v}: {[f['filename'] for f in files]}")
