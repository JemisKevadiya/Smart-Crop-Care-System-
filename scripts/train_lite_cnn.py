"""Train and evaluate LeafLiteNet (src/model/lite_cnn.py) from scratch.

    python scripts/train_lite_cnn.py                              # main model
    python scripts/train_lite_cnn.py --run leaflitenet_noSE --no-se   # ablation
    python scripts/train_lite_cnn.py --run leaflitenet_w0.5 --width 0.5
    python scripts/train_lite_cnn.py --smoke                      # a few steps, quick check

Outputs (under --root, default: the project folder):
    models/lite_cnn/<run>/best.keras
    artifacts/reports/lite_cnn/<run>/  training_log.csv, model_metrics.json,
                                       classification_report.txt, confusion matrix, curves

Built for a GPU (Google Colab: notebooks/04_leaflitenet_colab.ipynb); on a CPU an
epoch takes about an hour. An interrupted run resumes when started again with
the same --run and --root.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from tensorflow import keras  # noqa: E402

from src.model import lite_cnn  # noqa: E402
from src.preprocessing import config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", default="leaflitenet", help="name of this training run")
    parser.add_argument("--width", type=float, default=1.0, help="channel width multiplier")
    parser.add_argument("--no-se", action="store_true", help="disable squeeze-and-excitation")
    parser.add_argument("--epochs", type=int, default=lite_cnn.TRAIN["epochs"])
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--root", default=str(config.ROOT),
                        help="where models/ and artifacts/ are written (e.g. a Google Drive folder)")
    parser.add_argument("--cache-dir", default=None,
                        help="cache decoded images here after the first epoch (speeds up Colab)")
    parser.add_argument("--fresh", action="store_true", help="discard saved progress of this run")
    parser.add_argument("--no-evaluate", action="store_true", help="skip test-set evaluation")
    parser.add_argument("--smoke", action="store_true",
                        help="2 epochs x 3 steps, evaluated on 64 images per split")
    args = parser.parse_args()

    keras.utils.set_random_seed(config.SEED)
    run = f"{args.run}_smoke" if args.smoke else args.run
    if args.fresh:
        shutil.rmtree(lite_cnn.run_paths(run, args.root)["state"], ignore_errors=True)

    lite_cnn.train(run=run, width=args.width, use_se=not args.no_se,
                   epochs=2 if args.smoke else args.epochs,
                   batch_size=8 if args.smoke else args.batch_size,
                   root=args.root, cache_dir=args.cache_dir,
                   steps_per_epoch=3 if args.smoke else None,
                   validation_steps=2 if args.smoke else None)
    if not args.no_evaluate:
        lite_cnn.evaluate(run=run, root=args.root, limit=64 if args.smoke else None)


if __name__ == "__main__":
    main()
