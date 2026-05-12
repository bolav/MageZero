"""
Check that the MageZero inference server is up and responding correctly.

Usage:
  python scripts/ping_server.py [http://127.0.0.1:50052]
"""
import sys
import urllib.request
import msgpack

url = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:50052")

# Health check
try:
    with urllib.request.urlopen(f"{url}/healthz", timeout=5) as r:
        print(f"healthz: {r.status} {r.read().decode()}")
except Exception as e:
    print(f"healthz FAILED: {e}")
    sys.exit(1)

# Evaluate with dummy indices
payload = msgpack.packb({"indices": [1, 2, 3], "offsets": [0]}, use_bin_type=True)
req = urllib.request.Request(
    f"{url}/evaluate",
    data=payload,
    headers={"Content-Type": "application/x-msgpack"},
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        result = msgpack.unpackb(r.read(), raw=False)
    print(f"evaluate: OK  value={result['value']:.4f}  "
          f"policy_player len={len(result['policy_player'])}")
except Exception as e:
    print(f"evaluate FAILED: {e}")
    sys.exit(1)

print("Server OK")
