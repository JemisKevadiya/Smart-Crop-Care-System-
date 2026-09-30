"""End-to-end test of src/disease_detection on real dataset images.

1. Threshold analysis on the validation split (how the low-confidence cut-off
   was chosen).
2. Consistency: predictor output == evaluation-pipeline output for the same files.
3. One held-out test image per class through DiseasePredictor.predict(path).
4. The 33 supplied Dataset/raw/test images (demo only; they duplicate labelled files).
5. Input validation / error handling cases.
6. Non-leaf inputs, to show how low-confidence handling behaves.

Writes artifacts/reports/prediction_test.json.

    python scripts/test_prediction.py
"""

import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src.disease_detection import (  # noqa: E402
    DiseasePredictor,
    ImageValidationError,
    ModelLoadError,
)
from src.disease_detection.predictor import DEFAULT_CONFIDENCE_THRESHOLD  # noqa: E402
from src.disease_detection.preprocessing import preprocess_image  # noqa: E402
from src.preprocessing import config, pipeline, splits  # noqa: E402

REPORT = ROOT / "artifacts" / "reports" / "prediction_test.json"


def section(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}", flush=True)


def threshold_analysis(predictor):
    section("1. Confidence threshold analysis (validation split)")
    val = splits.load_split("val")
    probs = predictor.model.predict(pipeline.make_dataset("val", batch_size=64, training=False),
                                    verbose=0)
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    correct = pred == val["label_idx"].to_numpy()
    rows = []
    for t in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        kept = conf >= t
        rows.append({
            "threshold": t,
            "flagged_low_conf": float(1 - kept.mean()),
            "accuracy_when_confident": float(correct[kept].mean()),
            "errors_caught": float((~correct & ~kept).sum() / max(1, (~correct).sum())),
            "correct_flagged": float((correct & ~kept).sum() / correct.sum()),
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"val accuracy {correct.mean():.4f}; errors {int((~correct).sum())}; "
          f"median confidence correct {np.median(conf[correct]):.3f} / "
          f"wrong {np.median(conf[~correct]):.3f}")
    return {"table": rows, "val_accuracy": float(correct.mean()),
            "median_conf_correct": float(np.median(conf[correct])),
            "median_conf_wrong": float(np.median(conf[~correct]))}


def consistency_check(predictor):
    section("2. Predictor vs evaluation pipeline (same files)")
    sample = splits.load_split("test").sample(64, random_state=0)
    ref = predictor.model.predict(
        pipeline.make_dataset("test", batch_size=64, training=False, manifest=sample), verbose=0)
    ours = np.stack([predictor.predict_probabilities(preprocess_image(ROOT / p))[0]
                     for p in sample["path"]])
    diff = float(np.abs(ref - ours).max())
    same_top1 = bool((ref.argmax(1) == ours.argmax(1)).all())
    print(f"64 test images: max |prob difference| = {diff:.2e}; identical top-1: {same_top1}")
    assert same_top1 and diff < 1e-4
    return {"images": 64, "max_abs_prob_diff": diff, "identical_top1": same_top1}


def per_class_test(predictor):
    section("3. One held-out test image per class via predict(path)")
    sample = splits.load_split("test").groupby("label").sample(1, random_state=config.SEED)
    rows, times = [], []
    for _, r in sample.iterrows():
        t0 = time.perf_counter()
        p = predictor.predict(ROOT / r["path"])
        times.append((time.perf_counter() - t0) * 1000)
        rows.append({"true": r["label"], "predicted": p.class_name,
                     "confidence": round(p.confidence, 4), "correct": p.class_name == r["label"],
                     "is_confident": p.is_confident})
    df = pd.DataFrame(rows)
    for r in rows:
        mark = "OK " if r["correct"] else "BAD"
        flag = "" if r["is_confident"] else "  [low confidence]"
        print(f"{mark} {r['confidence']:.3f}  {r['true']:52s} -> {r['predicted']}{flag}")
    print(f"correct {df['correct'].sum()}/{len(df)}; low-confidence {(~df['is_confident']).sum()}; "
          f"end-to-end time per image: median {np.median(times):.0f} ms")
    return {"images": len(df), "correct": int(df["correct"].sum()),
            "low_confidence": int((~df["is_confident"]).sum()),
            "median_ms_per_image": float(np.median(times)), "results": rows}


def supplied_test_images(predictor):
    section("4. Supplied Dataset/raw/test images (demo only)")
    inv = pd.read_csv(config.INVENTORY_CSV)
    parts = inv["path"].str.split("/", expand=True)
    labelled = inv[parts[2].notna()].assign(label=parts[1])
    md5_label = labelled.drop_duplicates("md5").set_index("md5")["label"]
    supplied = pd.read_csv(config.SPLITS_DIR / "supplied_test_samples.csv")
    rows = []
    for _, r in supplied.iterrows():
        p = predictor.predict(ROOT / r["path"])
        true = md5_label[r["md5"]]
        rows.append({"file": Path(r["path"]).name, "true": true, "predicted": p.class_name,
                     "confidence": round(p.confidence, 4), "correct": p.class_name == true,
                     "twin_split": r["twin_split"]})
        print(f"{'OK ' if rows[-1]['correct'] else 'BAD'} {p.confidence:.3f}  "
              f"{rows[-1]['file']:28s} -> {p.class_name}  (twin in {r['twin_split']})")
    df = pd.DataFrame(rows)
    print(f"correct {df['correct'].sum()}/{len(df)} (26 of these have an identical copy in "
          f"the training split, so this is a demo, not an accuracy measurement)")
    return {"images": len(df), "correct": int(df["correct"].sum()), "results": rows}


def encode(image, fmt, **kw):
    buf = io.BytesIO()
    image.save(buf, format=fmt, **kw)
    return buf.getvalue()


def error_handling(predictor):
    section("5. Validation and error handling")
    leaf_path = ROOT / splits.load_split("test")["path"].iloc[0]
    leaf = Image.open(leaf_path)
    jpeg = leaf_path.read_bytes()
    rotated_big = leaf.resize((1200, 900))
    cases = {
        "empty bytes": (b"", False),
        "text file": (b"this is not an image", False),
        "truncated JPEG": (jpeg[: len(jpeg) // 3], False),
        "tiny 16x16 image": (encode(leaf.resize((16, 16)), "PNG"), False),
        "missing file path": (ROOT / "no_such_file.jpg", False),
        "GIF format": (encode(leaf, "GIF"), False),
        "unsupported type (int)": (12345, False),
        "PNG with alpha (RGBA)": (encode(leaf.convert("RGBA"), "PNG"), True),
        "grayscale JPEG": (encode(leaf.convert("L"), "JPEG"), True),
        "non-square 1200x900 JPEG": (encode(rotated_big, "JPEG", quality=95), True),
        "file-like object": (io.BytesIO(jpeg), True),
        "PIL image": (leaf, True),
        "solid grey image": (Image.new("RGB", (256, 256), (128, 128, 128)), False),
        "black image": (Image.new("RGB", (256, 256), (0, 0, 0)), False),
        "solid green image": (Image.new("RGB", (256, 256), (90, 140, 60)), False),
    }
    results = {}
    for name, (source, should_pass) in cases.items():
        try:
            p = predictor.predict(source)
            outcome = f"predicted {p.class_name} ({p.confidence:.3f})"
            passed = should_pass
        except ImageValidationError as exc:
            outcome = f"rejected: {exc}"
            passed = not should_pass
        results[name] = {"expected": "accept" if should_pass else "reject",
                         "outcome": outcome, "as_expected": passed}
        print(f"{'PASS' if passed else 'FAIL'}  {name:28s} {outcome}")

    try:
        DiseasePredictor(model_path=ROOT / "models" / "missing.keras")
        results["missing model file"] = {"as_expected": False, "outcome": "loaded?!"}
    except ModelLoadError as exc:
        results["missing model file"] = {"as_expected": True, "outcome": f"ModelLoadError: {exc}"}
    print(f"{'PASS' if results['missing model file']['as_expected'] else 'FAIL'}  "
          f"{'missing model file':28s} {results['missing model file']['outcome']}")
    assert all(r["as_expected"] for r in results.values()), "error handling case failed"
    return results


def non_leaf_inputs(predictor):
    section("6. Non-leaf inputs (low-confidence handling)")
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:256, 0:256]
    inputs = {
        "random noise": Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)),
        "colour gradient": Image.fromarray(np.stack([xx, yy, 255 - xx], -1).astype(np.uint8)),
        "checkerboard": Image.fromarray(
            (((xx // 32 + yy // 32) % 2) * 255).astype(np.uint8)).convert("RGB"),
    }
    results = {}
    for name, img in inputs.items():
        p = predictor.predict(img)
        results[name] = {"predicted": p.class_name, "confidence": round(p.confidence, 4),
                         "flagged_low_confidence": not p.is_confident}
        print(f"{name:14s} -> {p.class_name} ({p.confidence:.3f}) "
              f"{'FLAGGED low confidence' if not p.is_confident else 'NOT flagged'}")
    example = predictor.predict(inputs["random noise"])
    print("\nExample message:", example.message)
    return results


def main():
    t0 = time.perf_counter()
    predictor = DiseasePredictor()
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s; {len(predictor.class_names)} classes; "
          f"confidence threshold {predictor.confidence_threshold}")
    report = {"confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD}
    report["threshold_analysis"] = threshold_analysis(predictor)
    report["consistency"] = consistency_check(predictor)
    report["per_class_test"] = per_class_test(predictor)
    report["supplied_test_images"] = supplied_test_images(predictor)
    report["error_handling"] = error_handling(predictor)
    report["non_leaf_inputs"] = non_leaf_inputs(predictor)
    example = predictor.predict(ROOT / splits.load_split("test")["path"].iloc[0])
    report["example_prediction"] = example.to_dict()
    section("Example Prediction object")
    print(json.dumps(example.to_dict(), indent=2))
    REPORT.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()
