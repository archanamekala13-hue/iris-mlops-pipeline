"""Send sample prediction requests so the monitoring dashboards have data.

Usage:
    python send_traffic.py                       # 60 requests to localhost:8000
    python send_traffic.py https://YOUR-SERVICE.onrender.com 100
    python send_traffic.py http://localhost:8000 100 --drift   # simulate drifted inputs
"""
import json
import random
import sys
import time
import urllib.request

args = [a for a in sys.argv[1:] if not a.startswith("--")]
base = args[0].rstrip("/") if args else "http://localhost:8000"
count = int(args[1]) if len(args) > 1 else 60
drift = "--drift" in sys.argv

# Rough per-species ranges of the Iris measurements.
RANGES = {
    "setosa":     {"sepal_length": (4.3, 5.8), "sepal_width": (2.3, 4.4), "petal_length": (1.0, 1.9), "petal_width": (0.1, 0.6)},
    "versicolor": {"sepal_length": (4.9, 7.0), "sepal_width": (2.0, 3.4), "petal_length": (3.0, 5.1), "petal_width": (1.0, 1.8)},
    "virginica":  {"sepal_length": (4.9, 7.9), "sepal_width": (2.2, 3.8), "petal_length": (4.5, 6.9), "petal_width": (1.4, 2.5)},
}

ok = 0
for i in range(count):
    ranges = RANGES[random.choice(list(RANGES))]
    body = {k: round(random.uniform(*v), 1) for k, v in ranges.items()}
    if drift:  # shift every measurement up by 50% to trigger the drift check
        body = {k: round(v * 1.5, 1) for k, v in body.items()}
    req = urllib.request.Request(
        f"{base}/predict",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            ok += resp.status == 200
    except Exception as exc:
        print("request failed:", exc)
    time.sleep(0.1)

print(f"Sent {count} requests, {ok} succeeded.")
print(f"See {base}/monitoring/summary")
