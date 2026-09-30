"""Evaluate models/resnet50/best_resnet50.keras on train/val/test.

Writes artifacts/plots/{accuracy_curve,loss_curve,confusion_matrix}.png and
artifacts/reports/{classification_report.txt,model_metrics.json}.

    python scripts/evaluate_resnet50.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.evaluate import evaluate  # noqa: E402

if __name__ == "__main__":
    evaluate()
