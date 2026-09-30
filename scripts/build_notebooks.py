"""Generate notebooks/01_dataset_analysis.ipynb and 02_data_preprocessing.ipynb.

Run, then execute them:
    python scripts/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

SETUP = """\
import sys
from pathlib import Path

ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", 60)
pd.set_option("display.width", 140)

# Chart styling: light surface, recessive axes, fixed categorical order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # train, val/valid, test
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#b5b4ad", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": "#0b0b0b",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold",
})
"""


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def analysis_notebook():
    cells = [
        md("""
# 01 · Dataset analysis

Read-only analysis of the plant disease dataset at `Dataset/raw/`.
Nothing in the dataset is modified.

The per-image inventory (format, size, MD5, perceptual hash, decode errors) is
produced by `scripts/inspect_dataset.py`, which fully decodes every image. It is
regenerated here only if it is missing, because the full scan takes several minutes.
"""),
        code(SETUP + """
from src.preprocessing import config

if not config.INVENTORY_CSV.exists():
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "scripts" / "inspect_dataset.py")], check=True)

inv = pd.read_csv(config.INVENTORY_CSV)
parts = inv["path"].str.split("/", expand=True)
inv["split"] = parts[0]
inv["label"] = parts[1].where(parts[2].notna())
inv["crop"] = inv["label"].str.split("___").str[0]
inv["condition"] = inv["label"].str.split("___").str[1]
print(f"{len(inv):,} files inventoried under {config.DATASET_DIR}")
"""),
        md("## 1 · Structure and split sizes"),
        code("""
overview = inv.groupby("split").agg(images=("path", "size"), classes=("label", "nunique"))
overview.loc["total"] = [overview["images"].sum(), inv["label"].nunique()]
overview
"""),
        md("""
* `train/` and `valid/` each contain one folder per class (38 folders, identical names).
* `test/` is a flat folder of 33 images; the label is only encoded in the file name.
"""),
        md("## 2 · Images per class"),
        code("""
per_class = inv[inv["label"].notna()].pivot_table(
    index="label", columns="split", values="path", aggfunc="count", fill_value=0)
per_class["total"] = per_class.sum(axis=1)
per_class["valid/train"] = (per_class["valid"] / per_class["train"]).round(3)
per_class
"""),
        code("""
stats = per_class["train"].describe()[["min", "max", "mean", "std"]].round(1)
print("Train images per class:", stats.to_dict())
print(f"Imbalance ratio (max/min): {per_class['train'].max() / per_class['train'].min():.2f}")
"""),
        code("""
order = per_class.sort_values("total").index
fig, ax = plt.subplots(figsize=(9, 10))
y = np.arange(len(order))
ax.barh(y, per_class.loc[order, "train"], color=SERIES[0], height=0.8, label="train")
ax.barh(y, per_class.loc[order, "valid"], left=per_class.loc[order, "train"] + 8,
        color=SERIES[1], height=0.8, label="valid")
ax.set_yticks(y, order)
ax.set_xlabel("Images")
ax.set_title("Images per class (raw folders)")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(config.PLOTS_DIR / "01_images_per_class.png", dpi=120)
plt.show()
"""),
        md("## 3 · Crops and conditions"),
        code("""
crops = (inv[inv["label"].notna()].groupby("crop")
         .agg(classes=("label", "nunique"), images=("path", "size"),
              healthy_class=("condition", lambda s: (s == "healthy").any()))
         .sort_values("images", ascending=False))
print(f"{len(crops)} crops, {inv['label'].nunique()} classes "
      f"({(inv.drop_duplicates('label')['condition'] == 'healthy').sum()} healthy, "
      f"{(inv.drop_duplicates('label')['condition'] != 'healthy').sum() - 1} disease)")
crops
"""),
        md("## 4 · Formats, colour modes, dimensions, file sizes"),
        code("""
for col in ["ext", "format", "mode"]:
    print(col, inv[col].value_counts(dropna=False).to_dict())
inv["dims"] = inv["width"].astype("Int64").astype(str) + "x" + inv["height"].astype("Int64").astype(str)
print("dimensions", inv["dims"].value_counts().to_dict())
print("file size (KB):", (inv["bytes"] / 1024).describe()[["min", "50%", "max"]].round(1).to_dict())
"""),
        code("""
fig, ax = plt.subplots(figsize=(8, 3))
ax.hist(inv["bytes"] / 1024, bins=60, color=SERIES[0], edgecolor="#fcfcfb", linewidth=0.5)
ax.set_xlabel("File size (KB)")
ax.set_ylabel("Images")
ax.set_title("JPEG file size distribution (all images are 256×256)")
ax.grid(axis="x", visible=False)
fig.tight_layout()
plt.show()
"""),
        md("## 5 · Integrity"),
        code("""
integrity = {
    "decode errors (corrupted)": int(inv["error"].notna().sum()),
    "zero-byte files": int((inv["bytes"] == 0).sum()),
    "non-JPEG content": int((inv["format"] != "JPEG").sum()),
    "non-RGB images": int((inv["mode"] != "RGB").sum()),
    "not 256x256": int((inv["dims"] != "256x256").sum()),
}
integrity
"""),
        md("## 6 · Sample image per class"),
        code("""
from PIL import Image

samples = inv[inv["split"] == "train"].groupby("label").head(1).sort_values("label")
cols = 7
rows = int(np.ceil(len(samples) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.4))
for ax in axes.flat:
    ax.axis("off")
for ax, (_, r) in zip(axes.flat, samples.iterrows()):
    ax.imshow(Image.open(config.DATASET_DIR / r["path"]))
    ax.set_title(r["label"].replace("___", "\\n").replace("_", " ")[:40], fontsize=7)
fig.tight_layout()
fig.savefig(config.PLOTS_DIR / "01_sample_per_class.png", dpi=100)
plt.show()
"""),
        md("## 7 · Duplicates"),
        code("""
dupes = inv[inv.duplicated("md5", keep=False)].sort_values("md5")
groups = dupes.groupby("md5")["split"].agg(lambda s: " + ".join(sorted(set(s))))
print("Exact-duplicate groups by splits involved:", groups.value_counts().to_dict())

test = inv[inv["split"] == "test"]
twin = inv[inv["split"] != "test"].drop_duplicates("md5").set_index("md5")["path"]
test_twins = test.assign(identical_to=test["md5"].map(twin))[["path", "identical_to"]]
print(f"Supplied test images with a byte-identical copy in valid/: "
      f"{test_twins['identical_to'].str.startswith('valid/').sum()} / {len(test)}")
test_twins.head(10)
"""),
        code("""
near = inv[inv["split"] != "test"]
near = near[near.duplicated("phash", keep=False) & ~near.duplicated("md5", keep=False)]
near_groups = near.groupby("phash").agg(n=("path", "size"), labels=("label", "nunique"),
                                        splits=("split", lambda s: " + ".join(sorted(set(s)))))
print(f"Near-duplicate groups (same perceptual hash, different bytes): {len(near_groups)}")
print("Groups spanning more than one class:", int((near_groups['labels'] > 1).sum()))
near_groups["splits"].value_counts()
"""),
        md("""
## 8 · Leakage between train/ and valid/

File names follow `<uuid>___<source leaf id>[_<augmentation>].JPG`. The dataset was
augmented offline (flips, rotations, colour changes) **before** it was divided into
train/ and valid/, so augmented copies of one physical leaf are spread across both.
"""),
        code("""
from src.preprocessing.splits import parse_name, AUG_SUFFIX

lab = inv[inv["label"].notna()].copy()
lab["filename"] = lab["path"].str.rsplit("/", n=1).str[-1]
lab["leaf_key"] = lab["filename"].map(lambda f: parse_name(f)[1])
lab["aug"] = lab["filename"].str.rsplit(".", n=1).str[0].str.extract(AUG_SUFFIX, expand=False)

print("Augmentation suffixes:", lab["aug"].value_counts().to_dict())
print(f"Augmented files: {lab['aug'].notna().sum():,} of {len(lab):,}")

leaf_splits = lab.groupby(["label", "leaf_key"])["split"].nunique()
print(f"Distinct source leaves: {len(leaf_splits):,}; "
      f"present in BOTH train and valid: {(leaf_splits > 1).sum():,} "
      f"({(leaf_splits > 1).mean():.1%})")
"""),
        code("""
shared = leaf_splits[leaf_splits > 1].index[:3]
fig, axes = plt.subplots(len(shared), 4, figsize=(9, 2.5 * len(shared)))
for row, (label, key) in zip(axes, shared):
    files = lab[(lab["label"] == label) & (lab["leaf_key"] == key)].sort_values("split").head(4)
    for ax in row:
        ax.axis("off")
    for ax, (_, r) in zip(row, files.iterrows()):
        ax.imshow(Image.open(config.DATASET_DIR / r["path"]))
        ax.set_title(f"{r['split']} · {r['aug'] if isinstance(r['aug'], str) else 'original'}",
                     fontsize=8)
fig.suptitle("Same leaf in train/ and valid/", fontweight="bold")
fig.tight_layout()
plt.show()
"""),
        md("""
## Conclusions

* **38 classes, 14 crops, 52,747 images**, all readable 256×256 RGB JPEGs, well
  balanced (train imbalance ratio ≈ 1.24).
* **The supplied `test/` folder is not independent**: all 33 images are byte copies of
  `valid/` files and cover only 8 classes.
* **`train/` and `valid/` leak into each other**: augmented copies of the same leaf
  are on both sides, so validation accuracy on the original split would be inflated.
* A handful of exact duplicates exist inside the labelled data.

➡ Preprocessing (notebook 02) pools train/ + valid/, groups images by physical leaf
and re-splits whole groups into train/val/test.
"""),
    ]
    return cells


def preprocessing_notebook():
    cells = [
        md("""
# 02 · Data preprocessing

Builds a **leakage-free** train/val/test split and a TensorFlow `tf.data` input
pipeline for ResNet50 (224×224). No model is trained here and no image files are
copied: the splits are CSV manifests of paths into `Dataset/raw/`.

Why the dataset is re-split (see notebook 01):
* augmented copies of the same leaf are spread across `train/` and `valid/`;
* the supplied `test/` images are byte copies of `valid/` images.
"""),
        code(SETUP + """
import time
import tensorflow as tf

from src.preprocessing import config, splits, pipeline

tf.keras.utils.set_random_seed(config.SEED)
print("TensorFlow", tf.__version__, "| Keras", tf.keras.__version__)
print("Image size:", config.IMAGE_SIZE, "| batch:", config.BATCH_SIZE, "| seed:", config.SEED)
print("Split ratios:", config.SPLIT_RATIOS)
"""),
        md("""
## 1 · Group images by physical leaf and split

Images are linked into one group if they share a source leaf name (UUID prefix and
augmentation suffix removed), a UUID, identical bytes or an identical perceptual
hash. Whole groups are assigned to a split, class by class, so the split is
stratified and no leaf appears in two splits.
"""),
        code("""
manifest, class_names, report, supplied_test = splits.build_splits()
print(json.dumps(report, indent=2))
"""),
        md("## 2 · Leakage checks"),
        code("""
leaks = splits.check_no_leakage(manifest)
print("Values shared between splits:", leaks)
assert all(v == 0 for v in leaks.values()), "leakage detected"

paths_per_split = [set(manifest.loc[manifest["split"] == s, "path"]) for s in config.SPLIT_RATIOS]
assert not (paths_per_split[0] & paths_per_split[1] or paths_per_split[0] & paths_per_split[2]
            or paths_per_split[1] & paths_per_split[2])
assert manifest["md5"].is_unique
assert manifest.groupby("group_id")["label"].nunique().max() == 1, "a group spans classes"
print("OK: no shared files, hashes, near-duplicates or source leaves across splits")
"""),
        code("""
print("Supplied test/ images are excluded from all splits (duplicates of labelled files).")
print("Split holding each one's identical twin:", supplied_test["twin_split"].value_counts().to_dict())
"""),
        md("## 3 · Class balance per split"),
        code("""
per_class = splits.summarise_per_class(manifest)
fractions = per_class.div(per_class.sum(axis=1), axis=0)
print("Per-class split fraction range:")
print(fractions.agg(["min", "max"]).round(3))
assert per_class.shape[0] == 38 and (per_class > 0).all().all()
per_class
"""),
        code("""
order = per_class.sum(axis=1).sort_values().index
fig, ax = plt.subplots(figsize=(9, 10))
y = np.arange(len(order))
left = np.zeros(len(order))
for color, split in zip(SERIES, config.SPLIT_RATIOS):
    values = per_class.loc[order, split].to_numpy()
    ax.barh(y, values, left=left, color=color, height=0.8, label=split,
            edgecolor="#fcfcfb", linewidth=1)
    left += values
ax.set_yticks(y, order)
ax.set_xlabel("Images")
ax.set_title("Images per class after leaf-grouped split")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(config.PLOTS_DIR / "02_split_per_class.png", dpi=120)
plt.show()
"""),
        md("## 4 · Save manifests"),
        code("""
splits.save_splits(manifest, class_names, report, supplied_test)
for f in sorted(config.SPLITS_DIR.iterdir()):
    print(f"{f.name:32s} {f.stat().st_size / 1024:8.1f} KB")
"""),
        md("""
## 5 · The tf.data pipeline

| Step | train | val / test |
|---|---|---|
| shuffle (whole split, reshuffled every epoch) | ✓ | – |
| read → decode JPEG (RGB) → resize to 224×224 (bilinear, antialiased) | ✓ | ✓ |
| batch (32) | ✓ | ✓ |
| random flip (H+V), rotation ±10 %, zoom ±10 %, translation ±10 %, brightness ±10 %, contrast ±10 % | ✓ | – |
| `resnet50.preprocess_input` (RGB→BGR, subtract ImageNet mean) | ✓ | ✓ |
| prefetch | ✓ | ✓ |

`preprocess_input` uses fixed ImageNet statistics, so nothing is fitted on the data.
"""),
        code("""
train_ds = pipeline.make_dataset("train")
val_ds = pipeline.make_dataset("val")
test_ds = pipeline.make_dataset("test")
print(train_ds.element_spec)
class_names = splits.load_class_names()
print(len(class_names), "classes; index 0 =", class_names[0], "| index 37 =", class_names[-1])
"""),
        md("## 6 · Verify batch shapes, dtypes and value ranges"),
        code("""
for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
    x, y = next(iter(ds))
    x = x.numpy()
    print(f"{name:5s} x{tuple(x.shape)} {x.dtype}  y{tuple(y.shape)} {y.dtype}  "
          f"range [{x.min():7.2f}, {x.max():7.2f}]  "
          f"channel means (BGR) {np.round(x.mean(axis=(0, 1, 2)), 1)}  "
          f"labels {y.numpy().min()}-{y.numpy().max()}")
    assert x.shape[1:] == (224, 224, 3) and x.dtype == np.float32
    assert y.dtype == tf.int32 and 0 <= y.numpy().min() and y.numpy().max() < 38
    # caffe-mode bounds: 0 - max(mean) .. 255 - min(mean)
    assert x.min() >= -123.68 - 1e-3 and x.max() <= 255 - 103.939 + 1e-3
"""),
        md("## 7 · Verify preprocessing against an independent reference"),
        code("""
from PIL import Image

val_df = splits.load_split("val")
row = val_df.iloc[0]
ref = Image.open(ROOT / row["path"]).convert("RGB")
ref = np.asarray(ref, dtype=np.float32)
ref = tf.image.resize(ref, config.IMAGE_SIZE, antialias=True).numpy()
ref_bgr = ref[..., ::-1] - np.array([103.939, 116.779, 123.68], dtype=np.float32)

x_val, y_val = next(iter(pipeline.make_dataset("val", batch_size=1)))
single = pipeline.preprocess_image_file(ROOT / row["path"])
print("label:", class_names[int(y_val[0])], "| manifest label:", row["label"])
print("max |pipeline - reference|:", float(np.abs(x_val[0].numpy() - ref_bgr).max()))
print("max |pipeline - single-image inference path|:", float(np.abs(x_val.numpy() - single.numpy()).max()))
assert class_names[int(y_val[0])] == row["label"]
assert np.allclose(x_val[0].numpy(), ref_bgr, atol=1.0)
assert np.allclose(x_val.numpy(), single.numpy(), atol=1e-4)
"""),
        md("## 8 · Determinism: val/test fixed, train randomised"),
        code("""
a = np.concatenate([x.numpy() for x, _ in val_ds.take(3)])
b = np.concatenate([x.numpy() for x, _ in val_ds.take(3)])
print("val identical across passes:", np.array_equal(a, b))
assert np.array_equal(a, b)

ya = np.concatenate([y.numpy() for _, y in train_ds.take(3)])
yb = np.concatenate([y.numpy() for _, y in train_ds.take(3)])
print("train order differs across passes:", not np.array_equal(ya, yb))
assert not np.array_equal(ya, yb)
"""),
        md("## 9 · Augmentation examples (train only)"),
        code("""
one = splits.load_split("train").sample(1, random_state=config.SEED)
aug_ds = pipeline.make_dataset("train", batch_size=1, manifest=pd.concat([one] * 8))
clean = pipeline.to_display(pipeline.make_dataset("val", batch_size=1, manifest=one).get_single_element()[0])

fig, axes = plt.subplots(1, 9, figsize=(16, 2.3))
axes[0].imshow(clean[0]); axes[0].set_title("val/test view", fontsize=8)
for ax, (x, _) in zip(axes[1:], aug_ds):
    ax.imshow(pipeline.to_display(x)[0]); ax.set_title("train (augmented)", fontsize=8)
for ax in axes:
    ax.axis("off")
fig.suptitle(one["label"].iloc[0], fontweight="bold")
fig.tight_layout()
fig.savefig(config.PLOTS_DIR / "02_augmentation_examples.png", dpi=100)
plt.show()
"""),
        code("""
x, y = next(iter(train_ds))
imgs = pipeline.to_display(x).numpy()
fig, axes = plt.subplots(2, 8, figsize=(16, 4.6))
for ax, img, label in zip(axes.flat, imgs, y.numpy()):
    ax.imshow(img); ax.axis("off")
    ax.set_title(class_names[label].replace("___", "\\n").replace("_", " ")[:36], fontsize=7)
fig.suptitle("One training batch (shuffled + augmented)", fontweight="bold")
fig.tight_layout()
plt.show()
"""),
        md("## 10 · Full pass: every image in every split decodes through TensorFlow"),
        code("""
counts = {}
for split in config.SPLIT_RATIOS:
    n, t0 = 0, time.perf_counter()
    for x, y in pipeline.make_dataset(split, batch_size=256, training=False):
        assert np.isfinite(x.numpy()).all()
        n += int(x.shape[0])
    counts[split] = n
    print(f"{split:5s} {n:6,d} images decoded in {time.perf_counter() - t0:5.1f}s")
    assert n == len(splits.load_split(split))
"""),
        code("""
n, t0 = 0, time.perf_counter()
for x, _ in train_ds.take(50):
    n += int(x.shape[0])
print(f"Augmented training throughput: {n / (time.perf_counter() - t0):.0f} images/s on CPU")
"""),
        md("""
## Summary

* Train, val and test come from `Dataset/raw/train` + `valid` pooled, deduplicated,
  and split **by physical leaf** (80/10/10, stratified per class). No file, hash,
  near-duplicate or source leaf is shared between splits.
* The supplied `test/` folder is excluded from evaluation; it duplicates labelled files.
* Every image is resized to 224×224 and passed through `resnet50.preprocess_input`;
  training batches are additionally shuffled and augmented.
* val/test preprocessing is deterministic and identical to the single-image
  inference path (`pipeline.preprocess_image_file`).

Outputs: `data/splits/{train,val,test}.csv`, `class_names.json`, `split_report.json`,
`supplied_test_samples.csv`.
"""),
    ]
    return cells


def training_notebook():
    cells = [
        md("""
# 03 · ResNet50 training

Transfer-learning classifier built from scratch on the leaf-grouped splits from
notebook 02.

```
Input 224×224×3 (resnet50.preprocess_input applied by the pipeline)
  → ResNet50 (ImageNet weights, include_top=False)
  → GlobalAveragePooling2D → Dropout → Dense (ReLU) → Dense softmax (num_classes)
```

* **Stage 1:** the backbone is frozen and only the head is trained (Adam, lr 1e-3).
* **Stage 2:** `conv5_x`, the last residual stage, is unfrozen with BatchNorm kept
  frozen, and training continues (Adam, lr 1e-5).
* **Callbacks:** ModelCheckpoint saves the best `val_accuracy` to
  `models/resnet50/best_resnet50.keras`. EarlyStopping and ReduceLROnPlateau watch
  `val_loss`.

This notebook verifies the model and runs a **smoke test**, which is a few steps
only and not a performance measurement. Set `RUN_FULL_TRAINING = True` to train
for real, or run `python scripts/train_resnet50.py` in the background.
"""),
        code(SETUP + """
import time
import tensorflow as tf
from tensorflow import keras

from src.preprocessing import config, pipeline, splits
from src.model import resnet50

tf.keras.utils.set_random_seed(config.SEED)
print("TensorFlow", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU") or "none (CPU)")
print(json.dumps(resnet50.TRAIN, indent=2))
"""),
        md("## 1 · Classes come from the dataset"),
        code("""
assert (config.SPLITS_DIR / "train.csv").exists(), "run notebook 02 first"
class_names = resnet50.load_class_names()
num_classes = len(class_names)
train_labels = set(splits.load_split("train")["label"])
assert train_labels == set(class_names), "class list does not match training data"

resnet50.save_class_names(class_names)
reloaded = np.load(resnet50.CLASS_NAMES_PATH)
assert list(reloaded) == class_names
print(f"num_classes = {num_classes} (from {config.SPLITS_DIR / 'class_names.json'})")
print(f"saved {resnet50.CLASS_NAMES_PATH}")
"""),
        md("## 2 · Build the model (stage 1: frozen backbone)"),
        code("""
model, backbone = resnet50.build_model(num_classes)
resnet50.compile_model(model, resnet50.TRAIN["stage1_lr"])
model.summary()
print("Stage 1:", resnet50.count_params(model))
assert not backbone.trainable
assert model.output_shape == (None, num_classes)
"""),
        md("## 3 · Stage 2 layer selection"),
        code("""
probe, probe_backbone = resnet50.build_model(num_classes)
start = resnet50.unfreeze_upper_layers(probe_backbone)
trainable = [l for l in probe_backbone.layers if l.trainable]
bn_trainable = [l.name for l in trainable if isinstance(l, keras.layers.BatchNormalization)]
print(f"Backbone layers: {len(probe_backbone.layers)}; fine-tuned from index {start} "
      f"({probe_backbone.layers[start].name}); trainable layers: {len(trainable)}")
print("first / last trainable:", trainable[0].name, "/", trainable[-1].name)
print("BatchNorm layers left trainable:", bn_trainable)
resnet50.compile_model(probe, resnet50.TRAIN["stage2_lr"])
print("Stage 2:", resnet50.count_params(probe))
assert not bn_trainable and all(not l.trainable for l in probe_backbone.layers[:start])
del probe, probe_backbone
"""),
        md("## 4 · Forward pass on real data (untrained head)"),
        code("""
x, y = next(iter(pipeline.make_dataset("val", batch_size=8)))
probs = model.predict(x, verbose=0)
print("output shape:", probs.shape, "| row sums:", np.round(probs.sum(axis=1), 4))
assert probs.shape == (8, num_classes) and np.allclose(probs.sum(axis=1), 1, atol=1e-4)
"""),
        md("## 5 · Estimated training time on this machine"),
        code("""
def time_steps(m, steps=5, batch_size=config.BATCH_SIZE):
    ds = pipeline.make_dataset("train", batch_size=batch_size).take(steps + 1)
    it = iter(ds)
    m.train_on_batch(*next(it))                      # warm-up / graph build
    t0 = time.perf_counter()
    for xb, yb in it:
        m.train_on_batch(xb, yb)
    return (time.perf_counter() - t0) / steps

steps_per_epoch = int(np.ceil(len(splits.load_split("train")) / config.BATCH_SIZE))
s1 = time_steps(model)
probe, probe_backbone = resnet50.build_model(num_classes)
resnet50.unfreeze_upper_layers(probe_backbone)
resnet50.compile_model(probe, resnet50.TRAIN["stage2_lr"])
s2 = time_steps(probe)
del probe, probe_backbone
print(f"{steps_per_epoch} steps/epoch at batch {config.BATCH_SIZE}")
print(f"stage 1: {s1:.2f} s/step  -> ~{s1 * steps_per_epoch / 60:.0f} min/epoch (+ validation)")
print(f"stage 2: {s2:.2f} s/step  -> ~{s2 * steps_per_epoch / 60:.0f} min/epoch (+ validation)")
"""),
        md("""
## 6 · Smoke test of the full two-stage loop

Runs **2 epochs × 3 steps per stage** at batch 8 and writes to
`models/resnet50/smoke_test/`. This checks that training, callbacks, checkpointing
and reloading work end to end. **Its metrics say nothing about model quality.**
"""),
        code("""
smoke_dir = resnet50.MODEL_DIR / "smoke_test"
smoke_model, smoke_hist = resnet50.train(
    stage1_epochs=2, stage2_epochs=2, batch_size=8, steps_per_epoch=3, validation_steps=2,
    model_path=smoke_dir / "best_resnet50.keras", reports_dir=smoke_dir)

ckpt = smoke_dir / "best_resnet50.keras"
assert ckpt.exists()
restored = keras.models.load_model(ckpt)
assert restored.output_shape == (None, num_classes)
assert list(np.load(smoke_dir / "class_names.npy")) == class_names
for f in sorted(smoke_dir.iterdir()):
    print(f"{f.name:28s} {f.stat().st_size / 1e6:8.2f} MB")
print("Smoke test passed: both stages ran, checkpoint saved and reloaded.")
"""),
        md("""
## 7 · Full training

Off by default. On CPU, expect the per-epoch times estimated in section 5. When
enabled, this cell writes `models/resnet50/best_resnet50.keras`,
`models/resnet50/class_names.npy` and the per-stage logs in `artifacts/reports/`.
"""),
        code("""
RUN_FULL_TRAINING = False

if RUN_FULL_TRAINING:
    tf.keras.utils.set_random_seed(config.SEED)
    model, history = resnet50.train()
else:
    print("Skipped. Set RUN_FULL_TRAINING = True or run: python scripts/train_resnet50.py")
"""),
        code("""
hist_path = resnet50.REPORTS_DIR / "training_history.json"
if hist_path.exists() and resnet50.BEST_MODEL_PATH.exists():
    h = json.loads(hist_path.read_text())
    n1 = len(h["stage1"]["loss"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for ax, metric in zip(axes, ["accuracy", "loss"]):
        for key, color, name in [(metric, SERIES[0], "train"), (f"val_{metric}", SERIES[1], "val")]:
            values = h["stage1"][key] + h["stage2"][key]
            ax.plot(range(1, len(values) + 1), values, color=color, linewidth=2, label=name)
        ax.axvline(n1 + 0.5, color="#b5b4ad", linestyle="--", linewidth=1)
        ax.set_title(metric); ax.set_xlabel("epoch")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "03_training_curves.png", dpi=120)
    plt.show()
else:
    print("No full training run yet, so there are no curves or accuracy figures to report.")
"""),
    ]
    return cells


def write(cells, name):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "smart-crop-care", "display_name": "Python (smart-crop-care)", "language": "python"}
    NB_DIR.mkdir(exist_ok=True)
    nbf.write(nb, NB_DIR / name)
    print("wrote", NB_DIR / name)


if __name__ == "__main__":
    write(analysis_notebook(), "01_dataset_analysis.ipynb")
    write(preprocessing_notebook(), "02_data_preprocessing.ipynb")
    write(training_notebook(), "03_resnet50_training.ipynb")
