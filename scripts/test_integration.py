"""End-to-end test: leaf image -> ResNet50 -> disease -> fertilizer/treatment advice.

Uses held-out test-split images (never seen in training) and writes
artifacts/reports/integration_test.json.

    python scripts/test_integration.py
"""

import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src.disease_detection import DiseasePredictor  # noqa: E402
from src.fertilizer.recommendation import NOT_AVAILABLE  # noqa: E402
from src.integration import CropCareAnalyzer, format_report  # noqa: E402
from src.preprocessing import config, splits  # noqa: E402

REPORT = ROOT / "artifacts" / "reports" / "integration_test.json"
PER_CLASS = 10
ADVICE = ["fertilizer", "treatment", "eco_friendly_treatment", "chemical_treatment"]


def section(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}", flush=True)


def batch_run(analyzer):
    section(f"1. {PER_CLASS} held-out test images per class")
    sample = (splits.load_split("test").groupby("label")
              .sample(PER_CLASS, random_state=config.SEED).reset_index(drop=True))
    rows, t0 = [], time.perf_counter()
    for _, r in sample.iterrows():
        res = analyzer.analyze(ROOT / r["path"])
        p, rec = res.prediction, res.recommendation
        row = {"path": r["path"], "true": r["label"], "status": res.status,
               "predicted": p.class_name if p else None,
               "confidence": round(p.confidence, 4) if p else None,
               "correct": bool(p and p.class_name == r["label"]),
               "advice_attached": rec is not None,
               "advice_matches_prediction": bool(rec and p and rec.disease == p.class_name),
               "advice_complete": bool(rec and all(getattr(rec, f) != NOT_AVAILABLE
                                                   for f in ADVICE))}
        rows.append(row)
    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(rows)

    ok = df[df["status"] == "ok"]
    low = df[df["status"] == "low_confidence"]
    summary = {
        "images": len(df),
        "status_counts": df["status"].value_counts().to_dict(),
        "top1_accuracy_all": float(df["correct"].mean()),
        "accuracy_when_ok": float(ok["correct"].mean()),
        "accuracy_when_low_confidence": float(low["correct"].mean()) if len(low) else None,
        "wrong_but_confident": int((~ok["correct"]).sum()),
        "ok_with_matching_complete_advice": int((ok["advice_matches_prediction"]
                                                 & ok["advice_complete"]).sum()),
        "low_confidence_with_advice": int(low["advice_attached"].sum()),
        "seconds_per_image": round(elapsed / len(df), 3),
    }
    for k, v in summary.items():
        print(f"{k:34s} {v}")
    assert summary["ok_with_matching_complete_advice"] == len(ok), "advice missing for ok result"
    assert summary["low_confidence_with_advice"] == 0, "advice attached to uncertain prediction"
    assert set(df["status"]) <= {"ok", "low_confidence"}

    wrong = df[(df["status"] == "ok") & ~df["correct"]]
    if len(wrong):
        print("\nConfident but wrong:")
        for _, r in wrong.iterrows():
            print(f"  {r['confidence']:.3f}  {r['true']} -> {r['predicted']}")
    return summary, df


def show_examples(analyzer, df):
    section("2. Full results for three images")
    picks = [df[df["true"] == "Tomato___Late_blight"].iloc[0],
             df[df["true"] == "Potato___healthy"].iloc[0],
             df[df["true"] == "Tomato___Tomato_Yellow_Leaf_Curl_Virus"].iloc[0]]
    examples = []
    for r in picks:
        res = analyzer.analyze(ROOT / r["path"])
        print(f"\n--- {Path(r['path']).name} (true: {r['true']})")
        print(format_report(res))
        examples.append(res.to_dict())
    return examples


def low_confidence_cases(analyzer, df):
    section("3. Low-confidence handling on real images")
    low = df[df["status"] == "low_confidence"].sort_values("confidence").head(5)
    lenient = CropCareAnalyzer(predictor=analyzer.predictor, advise_when_uncertain=True)
    results = []
    for _, r in low.iterrows():
        strict = analyzer.analyze(ROOT / r["path"])
        opt_in = lenient.analyze(ROOT / r["path"])
        assert strict.recommendation is None
        assert opt_in.status == "low_confidence" and opt_in.recommendation is not None
        results.append({"true": r["true"], "predicted": r["predicted"],
                        "confidence": r["confidence"], "correct": bool(r["correct"]),
                        "message": strict.message})
        print(f"{r['confidence']:.3f}  true {r['true']}  predicted {r['predicted']}  "
              f"{'(correct)' if r['correct'] else '(WRONG)'}")
    if results:
        print("\nMessage shown to the user:\n ", results[0]["message"])
        print("Advice withheld by default: yes; attached with advise_when_uncertain=True: yes")
    return results


def error_cases(analyzer):
    section("4. Error handling")
    leaf = ROOT / splits.load_split("test")["path"].iloc[0]
    data = leaf.read_bytes()
    checks = {}

    def check(name, result, expected):
        checks[name] = {"status": result.status, "expected": expected,
                        "message": result.message, "passed": result.status == expected}
        print(f"{'PASS' if result.status == expected else 'FAIL'}  {name:34s} "
              f"{result.status:26s} {result.message[:90]}")

    check("empty upload", analyzer.analyze(b""), "invalid_image")
    check("text file", analyzer.analyze(b"hello"), "invalid_image")
    check("truncated JPEG", analyzer.analyze(data[:3000]), "invalid_image")
    check("blank image", analyzer.analyze(Image.new("RGB", (300, 300), "white")), "invalid_image")
    check("missing file", analyzer.analyze(ROOT / "missing.jpg"), "invalid_image")
    check("valid file-like upload", analyzer.analyze(io.BytesIO(data)), "ok")

    missing_csv = CropCareAnalyzer(predictor=analyzer.predictor,
                                   recommendations_path=ROOT / "no_such.csv")
    check("recommendation file missing", missing_csv.analyze(leaf), "recommendation_unavailable")

    tmp = ROOT / "artifacts" / "reports" / "_tmp_bad_recs.csv"
    tmp.write_text("disease,crop\nApple___Apple_scab,Apple\n")
    try:
        bad_cols = CropCareAnalyzer(predictor=analyzer.predictor, recommendations_path=tmp)
        check("recommendation CSV missing columns", bad_cols.analyze(leaf),
              "recommendation_unavailable")
    finally:
        tmp.unlink(missing_ok=True)

    broken = DiseasePredictor()
    broken.model = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated failure"))
    check("model failure", CropCareAnalyzer(predictor=broken).analyze(leaf), "model_error")

    assert all(c["passed"] for c in checks.values()), "error handling check failed"
    return checks


def main():
    t0 = time.perf_counter()
    analyzer = CropCareAnalyzer()
    print(f"Analyzer ready in {time.perf_counter() - t0:.1f}s "
          f"(confidence threshold {analyzer.predictor.confidence_threshold})")
    summary, df = batch_run(analyzer)
    report = {"batch": summary}
    report["examples"] = show_examples(analyzer, df)
    report["low_confidence"] = low_confidence_cases(analyzer, df)
    report["error_handling"] = error_cases(analyzer)
    report["per_image"] = df.to_dict(orient="records")
    REPORT.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()
