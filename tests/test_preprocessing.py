import numpy as np
import pandas as pd
import pytest

from src.preprocessing import config, splits


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG", "frec_scab 3417"),
        ("2f668a95-80ab-40ac-98ef-657bd85b668b___FREC_Scab 3247_270deg.JPG", "frec_scab 3247"),
        ("ab7d0d06-b45c-4472-a708-c2a1d65ee748___FREC_Scab 3152_new30degFlipLR.JPG",
         "frec_scab 3152"),
        ("RS_Rust 1564_flipLR.JPG", "rs_rust 1564"),
        ("48c55974-9fe9-4f4b-94f7-c8cd127d1e05___GHLB_PS Leaf 23.7 Day 13.jpg",
         "ghlb_ps leaf 23.7 day 13"),
    ],
)
def test_parse_name_strips_uuid_and_augmentation(filename, expected):
    assert splits.parse_name(filename)[1] == expected


def _toy_inventory():
    rows = []
    for label in ["A___x", "B___y"]:
        for leaf in range(20):
            for aug in ["", "_flipLR", "_180deg"]:
                split = "train" if aug != "_180deg" else "valid"
                name = f"{leaf:08x}-0000-0000-0000-000000000000___SRC {leaf}{aug}.JPG"
                rows.append({
                    "path": f"{split}/{label}/{name}",
                    "md5": f"{label}{leaf}{aug}",
                    "phash": f"{label}{leaf}{aug}",
                    "error": np.nan,
                })
    return pd.DataFrame(rows)


def test_augmented_copies_never_cross_splits():
    inv = _toy_inventory()
    parts = inv["path"].str.split("/", expand=True)
    inv["source_split"], inv["label"] = parts[0], parts[1]
    inv["filename"] = parts[2]
    parsed = inv["filename"].map(splits.parse_name)
    inv["uuid"], inv["leaf_key"] = parsed.str[0], parsed.str[1]

    manifest, class_names, report, _ = splits.build_splits(inv)

    assert report["num_groups"] == 40  # 20 leaves x 2 classes
    assert splits.check_no_leakage(manifest) == {"group_id": 0, "md5": 0, "phash": 0,
                                                 "leaf_key": 0}
    assert set(manifest["split"]) == set(config.SPLIT_RATIOS)
    assert class_names == ["A___x", "B___y"]


@pytest.mark.skipif(not (config.SPLITS_DIR / "train.csv").exists(),
                    reason="splits not generated")
def test_saved_splits_are_disjoint():
    parts = {s: splits.load_split(s) for s in config.SPLIT_RATIOS}
    manifest = pd.concat([df.assign(split=s) for s, df in parts.items()])
    assert splits.check_no_leakage(manifest) == {"group_id": 0, "md5": 0, "phash": 0,
                                                 "leaf_key": 0}
    assert manifest["label"].nunique() == 38


@pytest.mark.skipif(not (config.SPLITS_DIR / "val.csv").exists(),
                    reason="splits not generated")
def test_val_pipeline_output():
    tf = pytest.importorskip("tensorflow")
    from src.preprocessing import pipeline

    x, y = next(iter(pipeline.make_dataset("val", batch_size=4)))
    assert x.shape == (4, 224, 224, 3)
    assert x.dtype == tf.float32 and y.dtype == tf.int32
    assert float(tf.reduce_min(x)) >= -123.68 - 1e-3
    assert float(tf.reduce_max(x)) <= 255 - 103.939 + 1e-3
