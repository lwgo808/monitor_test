"""
Monitor Control Software (MCS) stub.

Right now this prints recommendations to stdout and logs them to a JSON file.
Future: swap send_to_monitor() to write serial commands to an Arduino/ESP32,
call a display-scaling API, or push to a brightness daemon.
"""

import json
import time
from pathlib import Path
from .posture import AdjustmentRecommendation, ErgonomicMetrics

LOG_PATH = Path(__file__).resolve().parent.parent / "ergonomics_log.json"


def send_to_monitor(metrics: ErgonomicMetrics, rec: AdjustmentRecommendation) -> None:
    payload = {
        "timestamp": time.time(),
        "distance_cm": round(metrics.distance_cm, 1),
        "horizontal_offset_px": round(metrics.horizontal_offset_px, 1),
        "vertical_offset_pct": round(metrics.vertical_offset_pct, 3),
        "pitch": round(metrics.pitch, 1),
        "yaw": round(metrics.yaw, 1),
        "roll": round(metrics.roll, 1),
        "recommendations": rec.messages,
    }

    # Log to file for later analysis / MCS integration
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(payload) + "\n")

    # Print live feedback
    if rec.messages:
        print("\n[Ergonomics Alert]")
        for msg in rec.messages:
            print(f"  • {msg}")
    else:
        print(f"\r[OK] dist={metrics.distance_cm:.0f}cm  "
              f"h={metrics.horizontal_offset_px:+.0f}px  "
              f"pitch={metrics.pitch:+.1f}°  yaw={metrics.yaw:+.1f}°  roll={metrics.roll:+.1f}°",
              end="", flush=True)
