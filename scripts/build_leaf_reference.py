"""Build the reference leaves used to reject non-leaf images, and evaluate it.

    python scripts/build_leaf_reference.py [--per-class 60] [--threshold 0.60]

1. Extracts ResNet50 features (gap layer) for --per-class training images per
   class and saves them to models/resnet50/leaf_reference.npz.
2. Scores real leaves (validation and test samples) and a set of out-of-scope
   images (faces, scenes, plots, synthetic patterns, the cartoon banner) and
   writes artifacts/reports/leaf_scope_check.json.

The threshold sits between the lowest real-leaf score and the highest
out-of-scope score. Needs the dataset in Dataset/raw (about 15 minutes on CPU).
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from src.disease_detection import DiseasePredictor  # noqa: E402
from src.disease_detection.preprocessing import preprocess_image  # noqa: E402
from src.disease_detection.scope_check import (  # noqa: E402
    DEFAULT_K,
    DEFAULT_SCOPE_THRESHOLD,
    REFERENCE_PATH,
    knn_similarity,
)
from src.preprocessing import config, splits  # noqa: E402

REPORT_PATH = config.ROOT / "artifacts" / "reports" / "leaf_scope_check.json"


def sample_split(name, per_class, seed=7):
    df = splits.load_split(name)
    return df.groupby("label").sample(per_class, random_state=seed)


def features_for(predictor, sources, batch_size=32):
    feats, probs = [], []
    for i in range(0, len(sources), batch_size):
        batch = np.concatenate([preprocess_image(s) for s in sources[i:i + batch_size]])
        f, p = predictor.extract(batch)
        feats.append(f)
        probs.append(p)
        print(f"  {min(i + batch_size, len(sources))}/{len(sources)}", flush=True)
    return np.concatenate(feats), np.concatenate(probs)


def out_of_scope_images(seed=0):
    """Non-leaf pictures available in every install, plus random crops of them."""
    from matplotlib import cbook
    from sklearn.datasets import load_sample_images

    rng = np.random.default_rng(seed)
    scenes = load_sample_images().images
    pictures = {
        "face_photo": Image.open(cbook.get_sample_data("grace_hopper.jpg")).convert("RGB"),
        "city_scene": Image.fromarray(scenes[0]),
        "flower_photo": Image.fromarray(scenes[1]),
        "cartoon_leaves_banner": Image.open(config.ROOT / "assets" / "hero_leaves.jpg").convert("RGB"),
        "plot": Image.open(config.ROOT / "artifacts" / "plots" / "confusion_matrix.png").convert("RGB"),
    }
    images = {}
    for name, im in pictures.items():
        images[name] = im
        w, h = im.size
        for j in range(4):
            side = int(min(w, h) * rng.uniform(0.4, 0.8))
            x, y = rng.integers(0, w - side + 1), rng.integers(0, h - side + 1)
            images[f"{name}_crop{j}"] = im.crop((x, y, x + side, y + side))
    for j in range(5):
        images[f"noise{j}"] = Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))
    yy, xx = np.indices((224, 224))
    images["checkerboard"] = Image.fromarray(
        np.repeat(((yy + xx) // 28 % 2 * 255)[..., None], 3, -1).astype(np.uint8))
    images["stripes"] = Image.fromarray(
        np.repeat((xx // 16 % 2 * 255)[..., None], 3, -1).astype(np.uint8))
    green = np.stack([rng.normal(60, 30, (224, 224)), rng.normal(140, 40, (224, 224)),
                      rng.normal(60, 30, (224, 224))], -1)
    images["green_noise"] = Image.fromarray(np.clip(green, 0, 255).astype(np.uint8))
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--per-class", type=int, default=60, help="reference images per class")
    parser.add_argument("--eval-per-class", type=int, default=50, help="validation images per class")
    parser.add_argument("--threshold", type=float, default=DEFAULT_SCOPE_THRESHOLD)
    args = parser.parse_args()

    predictor = DiseasePredictor(scope_checker=False)  # no scope check while building it
    paths = lambda df: [config.ROOT / p for p in df["path"]]  # noqa: E731

    print("Reference (train) images")
    train = sample_split("train", args.per_class)
    ref_feats, _ = features_for(predictor, paths(train))
    ref = ref_feats / np.linalg.norm(ref_feats, axis=1, keepdims=True)

    results = {}
    for split, per_class in (("val", args.eval_per_class), ("test", 10)):
        print(f"Real leaves ({split})")
        df = sample_split(split, per_class)
        feats, probs = features_for(predictor, paths(df))
        scores = knn_similarity(feats, ref, DEFAULT_K)
        correct = np.array(predictor.class_names)[probs.argmax(1)] == df["label"].to_numpy()
        results[split] = {"images": len(df), "min_score": round(float(scores.min()), 4),
                          "rejected": int((scores < args.threshold).sum()),
                          "rejected_rate": round(float(np.mean(scores < args.threshold)), 4),
                          "accuracy": round(float(correct.mean()), 4)}

    print("Out-of-scope images")
    ood = out_of_scope_images()
    feats, probs = features_for(predictor, list(ood.values()))
    scores = knn_similarity(feats, ref, DEFAULT_K)
    results["out_of_scope"] = {
        "images": len(ood), "max_score": round(float(scores.max()), 4),
        "caught": int((scores < args.threshold).sum()),
        "caught_rate": round(float(np.mean(scores < args.threshold)), 4),
        "mean_softmax_confidence": round(float(probs.max(1).mean()), 4),
        "per_image": {n: {"score": round(float(s), 4), "confidence": round(float(p), 4)}
                      for n, s, p in zip(ood, scores, probs.max(1))},
    }

    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REFERENCE_PATH, features=ref.astype(np.float16),
                        threshold=np.float32(args.threshold), k=np.int32(DEFAULT_K),
                        layer=np.array("gap"), per_class=np.int32(args.per_class))
    report = {"method": f"mean cosine similarity to {DEFAULT_K} nearest reference leaves "
                        "(ResNet50 gap features)",
              "reference_images": int(len(ref)), "threshold": args.threshold, **results}
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {REFERENCE_PATH} and {REPORT_PATH}")
    for key in ("val", "test"):
        r = results[key]
        print(f"{key}: {r['rejected']}/{r['images']} real leaves rejected (min score {r['min_score']})")
    o = results["out_of_scope"]
    print(f"out of scope: {o['caught']}/{o['images']} caught (max score {o['max_score']}, "
          f"mean confidence {o['mean_softmax_confidence']:.1%})")


if __name__ == "__main__":
    main()
