# Fix Log

**Date:** 30 October 2026 · **Base:** commit `7c343af` · **Changes:** not yet committed.

Each fix is small and targeted. No model, model artifact or dataset file was changed, and nothing was
retrained. After the fixes, all 178 tests passed with EfficientNetB0 and again with ResNet50.

## Fixes made

| # | Audit ID | Problem (confirmed) | Fix | Files | Retest |
|---|---|---|---|---|---|
| F1 | P1 | `test_integration.py`, `test_prediction.py` and `build_leaf_reference.py` took "held-out" images from the old local split `data/splits/`. About 80% of those images are in the current models' training set. | New `splits.load_model_split(name)`: it returns the local images of the split the models were trained on (`data/splits_kaggle/`), matched by MD5. The three scripts now use it. | `src/preprocessing/splits.py`, `src/preprocessing/config.py` (`KAGGLE_SPLITS_DIR`), the three scripts | Found 42,006 / 5,430 / 5,273 local train/val/test images; both report scripts ran on them |
| F2 | P1 | The three scripts wrote to `artifacts/reports/*.json` at the top level, which no longer exists, and described ResNet50 only. | Reports now go to `artifacts/reports/<active model>/`; docstrings updated. | the three scripts | Reports written to `artifacts/reports/efficientnetb0/` |
| F3 | new | `test_prediction.py` fed `pipeline.make_dataset()` into the active model. That pipeline always applies **ResNet50** preprocessing, so the numbers were wrong for EfficientNetB0. The threshold analysis also read the old split. | New `evaluation_probs()`: TensorFlow decode and resize plus the active model's own preprocessing; used for the threshold analysis and the consistency check. | `scripts/test_prediction.py` | Validation accuracy 98.27% (Kaggle validation 98.26%); consistency difference 1.25 × 10⁻⁶ |
| F4 | new | `test_integration.py`'s "model failure" check replaced `predictor.model`, but `predict()` uses `_feature_model`. The simulated failure never happened, and the check **FAILED**. | The check now breaks `_feature_model`. The app's handling was already correct. | `scripts/test_integration.py` | 9/9 error checks pass; a broken model gives `model_error` |
| F5 | B1 | The summary line said "Latest agriculture update", but it shows the top-ranked headline (crop mentions first), not always the newest. | Renamed to "Top agriculture update" (both lines); the two test assertions updated. | `app_pages/disease_detection.py`, `tests/test_app.py` | 178 tests pass; seen in the real browser |
| F6 | B2 | "Built with" listed scikit-learn (training only) and left out LangChain/Groq, the weather services and the news feeds. | The list now names what the app uses. | `app.py` | App starts; 178 tests pass |
| F7 | B3 | Unused `exc` variables in two `except` lines. | Removed `as exc`. | `src/weather/common.py` | Weather tests pass; real no-network test gives the right message |
| F8 | B4 | `groq` and `langchain_core` are imported directly but were not in `requirements.txt`. | Added `langchain-core` and `groq`, unpinned like the rest. | `requirements.txt` | Both already installed; nothing changed in `.venv` |
| F9 | — | `.env.example` didn't say that DenseNet121 can't be selected. | Added one comment line. | `.env.example` | — |

The Docker image was rebuilt with these fixes and tested: 12/12 in a real browser (see `end_to_end_test.md`).

New output files from the reruns (not code):
- `artifacts/reports/efficientnetb0/integration_test.json`
- `artifacts/reports/efficientnetb0/prediction_test.json`

## Not changed: needs your approval

| Audit ID | Item | Why not done automatically |
|---|---|---|
| S1 | **Session link sharing:** opening a copied `?sid=` link restores that user's leaf, results and chat for up to 12 hours, and two tabs from one link overwrite each other's state. | A real fix changes how sessions are identified, for example by rotating the id after each restore. That changes refresh behaviour and needs a design decision. Rotating the id alone would not stop the first person who opens a leaked link. |
| B4 (part) | Pin `requirements.txt` to the tested versions | Changes what every future install gets. `requirements-docker.txt` is already pinned. |
| B5 | Delete the leftover `config/`, `src/utils/`, empty `ui/` and `logs/` folders | Deleting files. They are git-ignored and unused, so they do no harm. |
| B6 | Old Docker Compose container `newsmartcropcare-smart-crop-care-1` on port 8501 | Deleting a Docker object; it is outside the code. |
| P2 | Report, PPT and viva guide show old numbers | Documents outside the repo; a separate task. |
| — | `build_leaf_reference.py` was fixed but **not run** | Running it overwrites `models/<model>/leaf_reference.npz`, which is a model artifact. |
