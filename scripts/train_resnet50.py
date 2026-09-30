"""Train the ResNet50 disease classifier (both stages) from the command line.

Full training on CPU takes many hours; this script is the non-notebook way to
run it in the background:

    python scripts/train_resnet50.py
    python scripts/train_resnet50.py --smoke   # a few steps, to check everything runs

An interrupted run resumes from its last completed epoch when started again.
Pass --fresh to discard saved training state and start over.
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tensorflow as tf  # noqa: E402

from src.model import resnet50  # noqa: E402
from src.preprocessing import config  # noqa: E402


def keep_system_awake():
    """Stop Windows from sleeping while this process runs (reverts on exit)."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-epochs", type=int, default=resnet50.TRAIN["stage1_epochs"])
    parser.add_argument("--stage2-epochs", type=int, default=resnet50.TRAIN["stage2_epochs"])
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--smoke", action="store_true",
                        help="2 epochs x 3 steps per stage, saved under models/resnet50/smoke_test/")
    parser.add_argument("--evaluate", action="store_true",
                        help="run scripts/evaluate_resnet50.py's evaluation after training")
    parser.add_argument("--fresh", action="store_true",
                        help="delete saved training state and train from the start")
    args = parser.parse_args()

    keep_system_awake()
    tf.keras.utils.set_random_seed(config.SEED)
    out = resnet50.MODEL_DIR / "smoke_test" if args.smoke else resnet50.MODEL_DIR
    if args.fresh:
        shutil.rmtree(out / "training_state", ignore_errors=True)
    if args.smoke:
        resnet50.train(stage1_epochs=2, stage2_epochs=2, batch_size=8, steps_per_epoch=3,
                       validation_steps=2, model_path=out / "best_resnet50.keras",
                       reports_dir=out)
    else:
        resnet50.train(stage1_epochs=args.stage1_epochs, stage2_epochs=args.stage2_epochs,
                       batch_size=args.batch_size)
        if args.evaluate:
            from src.model.evaluate import evaluate
            evaluate()


if __name__ == "__main__":
    main()
