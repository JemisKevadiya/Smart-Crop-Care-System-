"""Evaluate the trained ResNet50 checkpoint and write plots and reports.

Every number comes from running models/resnet50/best_resnet50.keras on the
leaf-grouped splits. Train accuracy is measured the same way as val/test
(no augmentation, no dropout); the accuracy logged during training, which
includes augmentation and dropout, is reported separately.
"""

import json
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)
from tensorflow import keras  # noqa: E402

from src.model import resnet50  # noqa: E402
from src.preprocessing import config, pipeline, splits  # noqa: E402

PLOTS_DIR = config.PLOTS_DIR
REPORTS_DIR = resnet50.REPORTS_DIR

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
TRAIN_COLOR, VAL_COLOR = "#2a78d6", "#eb6834"


def predict_split(model, split, batch_size=64):
    """Return (y_true, probabilities) for a split, in manifest order."""
    df = splits.load_split(split)
    ds = pipeline.make_dataset(split, batch_size=batch_size, training=False)
    probs = model.predict(ds, verbose=2)
    return df["label_idx"].to_numpy(), probs


def split_metrics(y_true, probs, num_classes):
    y_pred = probs.argmax(axis=1)
    out = {"images": int(len(y_true)),
           "accuracy": float(accuracy_score(y_true, y_pred)),
           "top3_accuracy": float(top_k_accuracy_score(y_true, probs, k=3,
                                                       labels=range(num_classes))),
           "loss": float(-np.mean(np.log(np.clip(probs[np.arange(len(y_true)), y_true],
                                                 1e-7, 1.0))))}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg,
                                                     zero_division=0)
        out[f"precision_{avg}"], out[f"recall_{avg}"], out[f"f1_{avg}"] = map(float, (p, r, f))
    return out


def plot_curves(history):
    """accuracy_curve.png and loss_curve.png across both stages."""
    s1, s2 = history["stage1"], history["stage2"]
    n1 = len(s1.get("loss", []))
    paths = {}
    for metric, fname in [("accuracy", "accuracy_curve.png"), ("loss", "loss_curve.png")]:
        fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        for key, color, label in [(metric, TRAIN_COLOR, "train (augmented)"),
                                  (f"val_{metric}", VAL_COLOR, "validation")]:
            values = s1.get(key, []) + s2.get(key, [])
            epochs = np.arange(1, len(values) + 1)
            ax.plot(epochs, values, color=color, linewidth=2, marker="o", markersize=4,
                    label=label)
            ax.annotate(f"{values[-1]:.4f}", (epochs[-1], values[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK_2)
        ax.axvline(n1 + 0.5, color="#b5b4ad", linestyle="--", linewidth=1)
        ax.text(n1 + 0.6, 0.02, "stage 2: fine-tuning conv5", transform=ax.get_xaxis_transform(),
                fontsize=8, color=INK_2)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Epoch", color=INK_2)
        ax.set_ylabel(metric.capitalize(), color=INK_2)
        ax.set_title(f"ResNet50 {metric} per epoch", color=INK, fontweight="bold")
        ax.grid(color=GRID, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=INK_2)
        ax.legend(frameon=False)
        fig.tight_layout()
        paths[metric] = PLOTS_DIR / fname
        fig.savefig(paths[metric], dpi=130)
        plt.close(fig)
    return paths


def plot_confusion(cm, class_names, title):
    """Row-normalised confusion matrix (recall on the diagonal), counts in cells."""
    norm = cm / cm.sum(axis=1, keepdims=True)
    short = [c.replace("___", " · ").replace("_", " ") for c in class_names]
    fig, ax = plt.subplots(figsize=(15, 13.5), facecolor=SURFACE)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(short)), short, rotation=90, fontsize=7, color=INK_2)
    ax.set_yticks(range(len(short)), short, fontsize=7, color=INK_2)
    for i, j in zip(*np.nonzero(cm)):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=5.5,
                color="white" if norm[i, j] > 0.5 else INK)
    ax.set_xlabel("Predicted class", color=INK_2)
    ax.set_ylabel("True class", color=INK_2)
    ax.set_title(title, color=INK, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="Share of true class")
    fig.tight_layout()
    path = PLOTS_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def evaluate(model_path=resnet50.BEST_MODEL_PATH):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model = keras.models.load_model(model_path)
    class_names = [str(c) for c in np.load(resnet50.CLASS_NAMES_PATH)]
    assert class_names == splits.load_class_names(), "class_names.npy does not match the splits"
    num_classes = len(class_names)

    results, predictions = {}, {}
    for split in ("train", "val", "test"):
        print(f"Evaluating {split} ...", flush=True)
        y_true, probs = predict_split(model, split)
        predictions[split] = (y_true, probs)
        results[split] = split_metrics(y_true, probs, num_classes)
        print(split, json.dumps(results[split]), flush=True)

    y_true, probs = predictions["test"]
    y_pred = probs.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    report_test = classification_report(y_true, y_pred, labels=range(num_classes),
                                        target_names=class_names, digits=4, zero_division=0)
    report_val = classification_report(predictions["val"][0],
                                       predictions["val"][1].argmax(axis=1),
                                       labels=range(num_classes), target_names=class_names,
                                       digits=4, zero_division=0)
    per_class = classification_report(y_true, y_pred, labels=range(num_classes),
                                      target_names=class_names, output_dict=True,
                                      zero_division=0)

    history = {s: resnet50.read_log(REPORTS_DIR / f"training_{s}.csv")
               for s in ("stage1", "stage2")}
    curve_paths = plot_curves(history)
    cm_path = plot_confusion(
        cm, class_names,
        f"Confusion matrix · test split ({len(y_true):,} images, "
        f"accuracy {results['test']['accuracy']:.4f})")
    # Class names contain commas (e.g. "Pepper,_bell___..."), so let pandas quote them.
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        REPORTS_DIR / "confusion_matrix_test.csv", index_label="true \\ predicted")

    with open(REPORTS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("ResNet50 plant disease classifier - classification report\n")
        f.write(f"Model: {model_path}\n\n")
        f.write(f"=== TEST split ({len(y_true)} images) ===\n{report_test}\n")
        f.write(f"=== VALIDATION split ({len(predictions['val'][0])} images) ===\n{report_val}\n")

    all_logs = history["stage1"].get("val_accuracy", []) + history["stage2"].get("val_accuracy", [])
    best_epoch = int(np.argmax(all_logs)) + 1 if all_logs else None
    last = {s: {k: v[-1] for k, v in h.items() if k != "epoch"} for s, h in history.items() if h}

    worst = sorted(((c, per_class[c]["f1-score"]) for c in class_names), key=lambda t: t[1])[:5]
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_path": str(model_path),
        "num_classes": num_classes,
        "evaluation": results,
        "training_log_last_epoch": last,
        "training_epochs": {s: len(h.get("loss", [])) for s, h in history.items()},
        "best_val_accuracy_epoch_overall": best_epoch,
        "test_per_class": {c: {k: per_class[c][k] for k in ("precision", "recall", "f1-score",
                                                             "support")}
                           for c in class_names},
        "test_lowest_f1_classes": dict(worst),
        "notes": {
            "train": "Clean evaluation of the saved checkpoint on the train split "
                     "(no augmentation, no dropout).",
            "training_log_last_epoch": "Metrics logged during training (train accuracy "
                                       "there includes augmentation and dropout).",
            "splits": "Leaf-grouped 80/10/10 split; no leaf, file or near-duplicate is shared "
                      "between train, val and test.",
        },
        "artifacts": {"accuracy_curve": str(curve_paths["accuracy"]),
                      "loss_curve": str(curve_paths["loss"]),
                      "confusion_matrix": str(cm_path)},
    }
    with open(REPORTS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Wrote", REPORTS_DIR / "model_metrics.json")
    return metrics
