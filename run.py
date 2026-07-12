"""
One command, whole system:  python run.py

    Reality -> Data -> Understanding -> Prediction -> Decision -> Action

Each stage writes to artifacts/ or reports/, so stages are cached and
independently inspectable.
"""
import subprocess, sys, time
from pathlib import Path

SRC = Path(__file__).parent / "src"

STAGES = [
    ("Box 1+2   Data, churn target, feature separation", "understand.py"),
    ("Box 3     Logistic baseline -> RF -> GBM + calibration", "models.py"),
    ("Box 4     Cost-aware decision layer", "decide.py"),
    ("Box 5     Build the dashboard", "dashboard.py"),
]

for name, script in STAGES:
    print(f"\n{'=' * 74}\n  {name}\n{'=' * 74}")
    t = time.time()
    if subprocess.run([sys.executable, "-u", script], cwd=SRC).returncode != 0:
        sys.exit(f"\nFAILED at: {name}")
    print(f"  ...{time.time() - t:.1f}s")

print("\n\n" + "=" * 74)
print("  DONE.  Open docs/index.html in a browser.")
print("=" * 74)
