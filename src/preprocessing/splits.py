"""Leakage-free train/val/test split of the raw dataset.

The raw dataset was augmented *before* it was split: a single leaf photo
appears as the original plus flipped/rotated/recoloured copies, and these
copies are scattered across train/ and valid/. The supplied test/ folder is
33 byte-identical copies of valid/ images. Splitting file-by-file would
therefore put the same leaf on both sides of the split.

This module groups every image with all other images of the same physical
leaf and assigns whole groups to one split. Two images are in the same group
if they share any of:

* the source leaf name (file name with UUID prefix and augmentation suffix removed)
* the UUID prefix
* identical bytes (MD5)
* identical perceptual hash (near-duplicates)

Groups never span classes, so the split is also stratified per class.
Nothing is copied: the output is CSV manifests of paths into Dataset/raw.
"""

import json
import re

import numpy as np
import pandas as pd

from . import config

AUG_SUFFIX = re.compile(r"_(flipLR|flipTB|\d+deg|new\w*)$", re.IGNORECASE)
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def parse_name(filename):
    """Return (uuid or None, source leaf key) for a raw file name."""
    stem = filename.rsplit(".", 1)[0]
    uuid = None
    if "___" in stem:
        prefix, rest = stem.split("___", 1)
        if UUID.match(prefix):
            uuid, stem = prefix.lower(), rest
    return uuid, AUG_SUFFIX.sub("", stem).strip().lower()


def load_inventory(path=config.INVENTORY_CSV):
    """Load the per-image inventory produced by scripts/inspect_dataset.py."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run scripts/inspect_dataset.py first")
    inv = pd.read_csv(path)
    if inv["error"].notna().any():
        raise ValueError("Inventory contains unreadable images; resolve them before splitting")
    parts = inv["path"].str.split("/", expand=True)
    inv["source_split"] = parts[0]
    inv["label"] = parts[1].where(parts[2].notna())  # flat test/ folder has no class
    inv["filename"] = inv["path"].str.rsplit("/", n=1).str[-1]
    parsed = inv["filename"].map(parse_name)
    inv["uuid"] = parsed.str[0]
    inv["leaf_key"] = parsed.str[1]
    return inv


def assign_groups(df):
    """Union-find over shared leaf key / UUID / MD5 / pHash within each class."""
    parent = np.arange(len(df))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for column in ["leaf_key", "uuid", "md5", "phash"]:
        for _, idx in df.groupby(["label", column], dropna=True).indices.items():
            root = find(idx[0])
            for j in idx[1:]:
                parent[find(j)] = root

    roots = np.array([find(i) for i in range(len(df))])
    _, group_ids = np.unique(roots, return_inverse=True)
    return group_ids


def split_groups(df, ratios=config.SPLIT_RATIOS, seed=config.SEED):
    """Assign whole groups to splits, per class, to match `ratios` by image count."""
    rng = np.random.default_rng(seed)
    names = list(ratios)
    assignment = {}
    for label in sorted(df["label"].unique()):
        sizes = df[df["label"] == label].groupby("group_id").size()
        order = rng.permutation(sizes.index.to_numpy())
        target = {s: ratios[s] * sizes.sum() for s in names}
        filled = dict.fromkeys(names, 0)
        for gid in order:
            # Put the group where the split is furthest below its target.
            s = max(names, key=lambda n: (target[n] - filled[n]) / target[n])
            assignment[gid] = s
            filled[s] += sizes[gid]
    return df["group_id"].map(assignment)


def build_splits(inventory=None, ratios=config.SPLIT_RATIOS, seed=config.SEED):
    """Return (manifest DataFrame, class_names, report dict)."""
    inv = load_inventory() if inventory is None else inventory

    labelled = inv[inv["label"].notna()].copy()
    flat_test = inv[inv["label"].isna()].copy()

    # Exact byte duplicates inside the labelled pool add nothing; keep one copy.
    before = len(labelled)
    labelled = labelled.drop_duplicates("md5", keep="first").reset_index(drop=True)
    dropped_dupes = before - len(labelled)

    labelled["group_id"] = assign_groups(labelled)
    labelled["split"] = split_groups(labelled, ratios, seed)

    class_names = sorted(labelled["label"].unique())
    labelled["label_idx"] = labelled["label"].map({c: i for i, c in enumerate(class_names)})
    labelled["path"] = "Dataset/raw/" + labelled["path"]

    # The supplied test/ images duplicate labelled files; record where each
    # twin landed so they are only ever used as demo images, never for metrics.
    twin_split = labelled.set_index("md5")["split"]
    flat_test["twin_split"] = flat_test["md5"].map(twin_split)
    flat_test["path"] = "Dataset/raw/" + flat_test["path"]

    report = {
        "seed": seed,
        "ratios": ratios,
        "images_in_pool": int(before),
        "exact_duplicates_dropped": int(dropped_dupes),
        "images_after_dedup": int(len(labelled)),
        "num_groups": int(labelled["group_id"].nunique()),
        "largest_group": int(labelled.groupby("group_id").size().max()),
        "split_sizes": labelled["split"].value_counts().reindex(list(ratios)).astype(int).to_dict(),
        "split_fractions": (labelled["split"].value_counts(normalize=True)
                            .reindex(list(ratios)).round(4).to_dict()),
        "supplied_test_images_excluded": int(len(flat_test)),
        "supplied_test_twin_splits": flat_test["twin_split"].value_counts().to_dict(),
    }
    columns = ["path", "label", "label_idx", "group_id", "split", "source_split", "md5", "phash"]
    return labelled[columns], class_names, report, flat_test[["path", "md5", "twin_split"]]


def check_no_leakage(manifest):
    """Return the number of groups/hashes/leaf keys that appear in more than one split."""
    result = {}
    for column in ["group_id", "md5", "phash"]:
        per_value = manifest.groupby(column)["split"].nunique()
        result[column] = int((per_value > 1).sum())
    keys = manifest["path"].str.rsplit("/", n=1).str[-1].map(lambda f: parse_name(f)[1])
    per_leaf = manifest.assign(k=manifest["label"] + "|" + keys).groupby("k")["split"].nunique()
    result["leaf_key"] = int((per_leaf > 1).sum())
    return result


def save_splits(manifest, class_names, report, supplied_test, out_dir=config.SPLITS_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in manifest.groupby("split"):
        part.drop(columns="split").to_csv(out_dir / f"{name}.csv", index=False)
    supplied_test.to_csv(out_dir / "supplied_test_samples.csv", index=False)
    (out_dir / "class_names.json").write_text(json.dumps(class_names, indent=2))
    (out_dir / "split_report.json").write_text(json.dumps(report, indent=2))


def load_split(name, splits_dir=config.SPLITS_DIR):
    return pd.read_csv(splits_dir / f"{name}.csv")


def load_class_names(splits_dir=config.SPLITS_DIR):
    return json.loads((splits_dir / "class_names.json").read_text())


def summarise_per_class(manifest):
    """Images per class and split, as a wide table."""
    table = manifest.pivot_table(index="label", columns="split", values="path",
                                 aggfunc="count", fill_value=0)
    return table[[s for s in config.SPLIT_RATIOS if s in table.columns]]


if __name__ == "__main__":
    manifest, class_names, report, supplied = build_splits()
    leaks = check_no_leakage(manifest)
    if any(leaks.values()):
        raise SystemExit(f"Leakage detected: {leaks}")
    save_splits(manifest, class_names, report, supplied)
    print(json.dumps(report, indent=2))
    print("Leakage check:", leaks)
    print(f"Wrote manifests to {config.SPLITS_DIR}")
