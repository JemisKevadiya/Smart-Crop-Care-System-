"""Read-only inspection of the raw plant disease dataset.

Walks Dataset/raw/{train,valid,test}, opens every file with Pillow and records
format, dimensions, colour mode, file size, MD5 and perceptual hash. Nothing in
the dataset is modified. Results are written to artifacts/reports/.

Usage:
    python scripts/inspect_dataset.py [--workers N]
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "Dataset" / "raw"
REPORT_DIR = ROOT / "artifacts" / "reports"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def inspect_file(path_str):
    path = Path(path_str)
    rec = {
        "path": path.relative_to(DATASET_DIR).as_posix(),
        "ext": path.suffix,
        "bytes": path.stat().st_size,
        "md5": None,
        "phash": None,
        "format": None,
        "width": None,
        "height": None,
        "mode": None,
        "error": None,
    }
    try:
        data = path.read_bytes()
        rec["md5"] = hashlib.md5(data).hexdigest()
        # verify() checks structure; a second open + load() decodes all pixels
        # and catches truncated files that verify() lets through.
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
            rec["format"] = im.format
            rec["width"], rec["height"] = im.size
            rec["mode"] = im.mode
            rec["phash"] = str(imagehash.phash(im))
    except Exception as exc:  # noqa: BLE001 - any failure means unreadable
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def collect_files():
    files = []
    for dirpath, _, filenames in os.walk(DATASET_DIR):
        for name in filenames:
            files.append(os.path.join(dirpath, name))
    return sorted(files)


def split_and_class(rel_path):
    parts = rel_path.split("/")
    split = parts[0]
    cls = parts[1] if len(parts) > 2 else None  # flat test folder has no class
    return split, cls


def summarise(records):
    splits = defaultdict(lambda: Counter())
    split_totals = Counter()
    ext_counts = Counter()
    format_counts = Counter()
    mode_counts = Counter()
    dim_counts = Counter()
    non_images = []
    corrupted = []
    ext_format_mismatch = []
    zero_byte = []
    sizes = []

    for r in records:
        split, cls = split_and_class(r["path"])
        ext_counts[r["ext"]] += 1
        if r["bytes"] == 0:
            zero_byte.append(r["path"])
        if r["ext"].lower() not in IMAGE_EXTS:
            non_images.append(r["path"])
        if r["error"]:
            corrupted.append({"path": r["path"], "error": r["error"]})
            continue
        split_totals[split] += 1
        splits[split][cls] += 1
        format_counts[r["format"]] += 1
        mode_counts[r["mode"]] += 1
        dim_counts[f'{r["width"]}x{r["height"]}'] += 1
        sizes.append(r["bytes"])
        expected = {"JPEG": {".jpg", ".jpeg"}, "PNG": {".png"}}.get(r["format"])
        if expected and r["ext"].lower() not in expected:
            ext_format_mismatch.append({"path": r["path"], "format": r["format"]})

    ok = [r for r in records if not r["error"]]

    # Exact duplicates (identical bytes)
    by_md5 = defaultdict(list)
    for r in ok:
        by_md5[r["md5"]].append(r["path"])
    exact_groups = [p for p in by_md5.values() if len(p) > 1]

    # Near-duplicates (identical 64-bit perceptual hash, different bytes)
    by_phash = defaultdict(set)
    for r in ok:
        by_phash[r["phash"]].add(r["md5"])
    near_groups = []
    for ph, md5s in by_phash.items():
        if len(md5s) > 1:
            near_groups.append(sorted(by_md5[m][0] for m in md5s))

    def classify(groups):
        within_class = cross_class = cross_split = 0
        examples = {"cross_class": [], "cross_split": []}
        for g in groups:
            sc = [split_and_class(p) for p in g]
            splits_in = {s for s, _ in sc}
            classes_in = {c for _, c in sc if c}
            if len(splits_in) > 1:
                cross_split += 1
                if len(examples["cross_split"]) < 10:
                    examples["cross_split"].append(g)
            if len(classes_in) > 1:
                cross_class += 1
                if len(examples["cross_class"]) < 10:
                    examples["cross_class"].append(g)
            elif len(splits_in) == 1:
                within_class += 1
        return {
            "groups": len(groups),
            "files_involved": sum(len(g) for g in groups),
            "redundant_copies": sum(len(g) - 1 for g in groups),
            "within_same_class_and_split": within_class,
            "spanning_splits": cross_split,
            "spanning_classes": cross_class,
            "examples": examples,
        }

    # Class-name consistency between train and valid
    train_classes = set(splits.get("train", {}))
    valid_classes = set(splits.get("valid", {}))

    # Test images: flat folder, label inferred from file name only
    test_files = sorted(r["path"] for r in ok if split_and_class(r["path"])[0] == "test")

    sizes.sort()
    return {
        "dataset_dir": str(DATASET_DIR),
        "total_files": len(records),
        "readable_images": len(ok),
        "split_totals": dict(split_totals),
        "num_classes": {s: len([c for c in v if c]) for s, v in splits.items()},
        "images_per_class": {
            s: dict(sorted(v.items(), key=lambda kv: str(kv[0]))) for s, v in splits.items()
        },
        "classes_only_in_train": sorted(train_classes - valid_classes),
        "classes_only_in_valid": sorted(valid_classes - train_classes),
        "extensions": dict(ext_counts),
        "decoded_formats": dict(format_counts),
        "colour_modes": dict(mode_counts),
        "dimensions": dict(dim_counts.most_common()),
        "file_size_bytes": {
            "min": sizes[0] if sizes else None,
            "median": sizes[len(sizes) // 2] if sizes else None,
            "max": sizes[-1] if sizes else None,
            "total_mb": round(sum(sizes) / 1e6, 1),
        },
        "corrupted": corrupted,
        "zero_byte_files": zero_byte,
        "non_image_files": non_images,
        "extension_format_mismatch": ext_format_mismatch,
        "exact_duplicates": classify(exact_groups),
        "near_duplicates_same_phash": classify(near_groups),
        "test_files": test_files,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    if not DATASET_DIR.is_dir():
        sys.exit(f"Dataset not found: {DATASET_DIR}")

    files = collect_files()
    print(f"Found {len(files)} files under {DATASET_DIR}", flush=True)

    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, rec in enumerate(pool.map(inspect_file, files, chunksize=64), 1):
            records.append(rec)
            if i % 5000 == 0:
                print(f"  inspected {i}/{len(files)}", flush=True)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / "dataset_inventory.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    summary = summarise(records)
    with open(REPORT_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {REPORT_DIR / 'dataset_inventory.csv'}")
    print(f"Wrote {REPORT_DIR / 'dataset_summary.json'}")


if __name__ == "__main__":
    main()
