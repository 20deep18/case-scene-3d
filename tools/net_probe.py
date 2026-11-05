import socket, ssl, sys, urllib.request, os

print("proxy env:", {k: v for k, v in os.environ.items() if 'proxy' in k.lower()})

for host, port in [("pypi.org", 443), ("github.com", 443), ("download.pytorch.org", 443), ("registry.npmjs.org", 443), ("mirrors.aliyun.com", 443), ("pypi.tuna.tsinghua.edu.cn", 443)]:
    try:
        ip = socket.gethostbyname(host)
        s = socket.create_connection((ip, port), timeout=8)
        s.close()
        print(f"TCP OK   {host} ({ip}):{port}")
    except Exception as e:
        print(f"TCP FAIL {host}:{port} -> {type(e).__name__}: {e}")

for url in ["https://pypi.org/simple/", "https://mirrors.aliyun.com/pypi/simple/"]:
    try:
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(url, timeout=15, context=ctx) as r:
            print(f"HTTPS OK {url} -> {r.status}")
    except Exception as e:
        print(f"HTTPS FAIL {url} -> {type(e).__name__}: {e}")
