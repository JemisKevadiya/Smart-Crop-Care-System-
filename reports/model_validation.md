# Model Validation

**Date:** 30 October 2026 · **Code:** commit `7c343af` plus the Part 2 fixes in `fix_log.md`.
No model was retrained, replaced or changed. All numbers below were measured in this session unless
marked "Kaggle run".

## 1. Which model the app uses

| Check | Result |
|---|---|
| Model selection | `src/disease_detection/model_config.py`, `select_model()` |
| `DISEASE_MODEL` in `.env` / environment | not set |
| Model used by the app | **EfficientNetB0** (`models/efficientnetb0/best_efficientnetb0.keras`) |
| Seen in the running app | Sidebar "Test accuracy 98.5%". After an upload the caption reads "Identified from the image by the EfficientNetB0 model" (real browser, port 8501). |
| Fallback | `DISEASE_MODEL=resnet50`, or automatic when the EfficientNetB0 files are missing (tested: `select_model()` returns ResNet50) |
| DenseNet121 | Not selectable: `DISEASE_MODEL=densenet121` falls back to EfficientNetB0. Comparison only. |

## 2. Inputs, preprocessing, outputs and class order

All three models were loaded and checked.

| Model | Input | Output | Preprocessing used by the app | Classes | Same class order as the others |
|---|---|---|---|---|---|
| EfficientNetB0 | (None, 224, 224, 3) | (None, 38) softmax | raw 0–255 pixels (normalisation is inside the model) | 38 | yes |
| ResNet50 | (None, 224, 224, 3) | (None, 38) softmax | `resnet50.preprocess_input` | 38 | yes |
| DenseNet121 | (None, 224, 224, 3) | (None, 38) softmax | `densenet.preprocess_input` (used only in this validation) | 38 | yes |

- Each `class_names.json` matches its `class_names.npy`.
- Every model has the `gap` layer that the non-leaf check needs.
- **App image path:**
  1. Validate the file.
  2. Fix the photo's EXIF rotation.
  3. Convert to RGB and take a centre square crop.
  4. Resize to 224×224 (anti-aliased).
  5. Apply the model's own preprocessing.
- **Confidence:** the highest softmax probability. The app also keeps the top 3. Below 0.80, the advice is withheld.

## 3. Does the app reproduce the training results?

The three models were trained on Kaggle on the same leakage-free split (fingerprint
`13afce0ee189dd7d`). The local dataset holds **5,273 of its 8,799 test images** (matched by MD5). Each
model was run on these images through the **app's own image path**, then compared with the probabilities
saved by the Kaggle run for the same images.

| Model | Local accuracy | Errors | Kaggle accuracy, same images | Top-1 agreement with Kaggle | Largest probability difference |
|---|---|---|---|---|---|
| EfficientNetB0 | 98.33% | 88 | 98.33% | **100%** | 1.0 × 10⁻⁵ |
| ResNet50 | 98.39% | 85 | 98.39% | **100%** | 3.7 × 10⁻⁶ |
| DenseNet121 | 96.91% | 163 | 96.91% | **100%** | 4.2 × 10⁻⁶ |

**Conclusion:** the app's preprocessing, class order and confidence match the training pipeline exactly
for all three models. No mapping or preprocessing bug exists.

## 4. Existing evaluation results (Kaggle run, all 8,799 test leaves)

From `artifacts/reports/<model>/model_metrics.json` and `artifacts/reports/model_comparison.csv`.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Top-3 | Errors | Parameters |
|---|---|---|---|---|---|---|---|
| EfficientNetB0 | 98.45% | 0.9847 | 0.9844 | 0.9844 | 99.93% | 136 | 4.39 M |
| ResNet50 | 98.35% | 0.9839 | 0.9833 | 0.9834 | 99.97% | 145 | 24.1 M |
| DenseNet121 | 96.95% | 0.9705 | 0.9695 | 0.9695 | 99.81% | 268 | 7.31 M |

Paired McNemar tests on the same images:
- EfficientNetB0 vs ResNet50: p = 0.55, so there is no real difference.
- Both are significantly better than DenseNet121 (p < 0.0001).

**The comparison is valid:** the same images were used for all three, and the leakage check is 0 on all seven checks.

## 5. Production model behaviour (measured this session, EfficientNetB0)

| Test | Result | Source |
|---|---|---|
| 5,430 held-out validation images | accuracy 98.27%, 94 errors | `scripts/test_prediction.py` |
| 80% confidence threshold (validation) | flags 3.28% of images, catches 64.9% of errors; confident predictions 99.37% correct | same |
| Evaluation path vs app path (64 images) | identical top-1, largest probability difference 1.25 × 10⁻⁶ | same |
| One held-out image per class | 37 / 38 correct; the miss (Septoria → Leaf Mold) was flagged low-confidence (0.45) | same |
| 380 held-out images (10 per class) | top-1 98.42%; 99.72% when confident; 1 wrong but confident | `scripts/test_integration.py` |
| Non-leaf check | 33/33 non-leaf images rejected and 0 of 10,703 held-out real leaves rejected (threshold 0.44) | earlier check on this commit (`leaf_scope_check_app.json`) |

## 6. Speed (this PC, CPU only)

| Model | Single image (model only) | 5,273 images in batches of 64 |
|---|---|---|
| EfficientNetB0 | 119 ms | 113 s |
| ResNet50 | 209 ms | 310 s |
| DenseNet121 | 210 ms | 332 s |

The Streamlit app and a browser test were running at the same time, so these times are higher than on an
idle PC. Use them only to compare the models with each other.

## 7. Recommendation

Keep **EfficientNetB0** as the production model. It is as accurate as ResNet50 (p = 0.55) and about 5×
smaller. It was also the fastest in this test. Nothing in this validation calls for a change.

## 8. Limitations

- **Local subset:** only 5,273 of the 8,799 test images are on this PC, so the full-test numbers in section 4 come from the Kaggle run.
- **Image type:** every test image is a lab-style single-leaf photo; accuracy on farm photos was not measured.
- **Class coverage:** the 33 images in the supplied `Dataset/raw/test` folder are copies of training/validation images, so they are a demo only.
