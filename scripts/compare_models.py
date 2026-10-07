"""Compare ResNet50 with the trained LeafLiteNet runs on this machine.

    python scripts/compare_models.py

For every model: parameters, multiply-accumulate operations, saved size (weights
only, no optimizer state), CPU inference time (batch 1 and batch 32, median of
repeated runs) and the test metrics from its evaluation report. Accuracy numbers
are read from the reports; nothing is estimated. Writes
artifacts/reports/model_comparison.json and model_comparison.md.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

from src.model import lite_cnn  # noqa: E402
from src.preprocessing import config  # noqa: E402

REPORTS = config.ROOT / "artifacts" / "reports"


def cpu_ms_per_image(model, batch, repeats=30):
    infer = tf.function(lambda x: model(x, training=False), autograph=False)
    x = tf.constant(np.random.default_rng(0).uniform(-120, 150, (batch, 224, 224, 3)),
                    dtype=tf.float32)
    for _ in range(5):
        infer(x)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        infer(x).numpy()
        times.append(time.perf_counter() - start)
    return float(np.median(times) * 1000 / batch)


def weights_size_mb(model):
    path = Path(tempfile.mkdtemp()) / "model.keras"
    model.save(path)
    return round(path.stat().st_size / 1e6, 1)


def describe(name, model, metrics):
    test = metrics["evaluation"]["test"]
    row = {"model": name, "params": model.count_params(), "macs": lite_cnn.count_macs(model),
           "size_mb": weights_size_mb(model),
           "cpu_ms_batch1": round(cpu_ms_per_image(model, 1), 1),
           "cpu_ms_batch32": round(cpu_ms_per_image(model, 32), 1),
           "test_accuracy": test["accuracy"], "test_f1_macro": test["f1_macro"],
           "test_top3": test["top3_accuracy"],
           "val_accuracy": metrics["evaluation"]["val"]["accuracy"]}
    print(json.dumps(row), flush=True)
    return row


def main():
    rows = [describe("ResNet50 (transfer learning)",
                     keras.models.load_model(ROOT / "models" / "resnet50" / "best_resnet50.keras",
                                             compile=False),
                     json.loads((REPORTS / "model_metrics.json").read_text()))]
    runs = sorted((ROOT / "models" / "lite_cnn").glob("*/best.keras"))
    for model_path in runs:
        run = model_path.parent.name
        report = REPORTS / "lite_cnn" / run / "model_metrics.json"
        if run.endswith("_smoke") or not report.exists():
            continue
        rows.append(describe(run, keras.models.load_model(model_path, compile=False),
                             json.loads(report.read_text())))
    if len(rows) == 1:
        print("No trained LeafLiteNet runs found under models/lite_cnn/ - train them first.")

    machine = {"cpu_threads": os.cpu_count(), "tensorflow": tf.__version__}
    (REPORTS / "model_comparison.json").write_text(
        json.dumps({"machine": machine, "models": rows}, indent=2))
    lines = ["| Model | Params (M) | MACs (M) | Size (MB) | CPU ms/img (b=1) | CPU ms/img (b=32) "
             "| Test acc. | Macro F1 | Top-3 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['params'] / 1e6:.2f} | {r['macs'] / 1e6:.0f} | "
                     f"{r['size_mb']} | {r['cpu_ms_batch1']} | {r['cpu_ms_batch32']} | "
                     f"{r['test_accuracy']:.4f} | {r['test_f1_macro']:.4f} | {r['test_top3']:.4f} |")
    (REPORTS / "model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
