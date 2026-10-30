# Smart Crop Care — Project Audit

**Date:** 29 October 2026 · **Commit audited:** `7c343af` (main, in sync with GitHub) · **Scope:** read-only
inspection. No application code, models or dataset were changed; only this report was added.

Every status below comes from a check run during this audit, unless the row says otherwise.

| Status | Meaning |
|---|---|
| VERIFIED WORKING | Tested during this audit and it worked |
| IMPLEMENTED BUT NOT TESTED | Code exists, but it was not run in this audit (reason given) |
| FAILING | Tested and it did not work |
| MISSING | Not present in the project |
| BLOCKED | Could not be tested now (reason given) |

## 1. Summary

The application works end to end. All 178 automated tests pass with both the default model
(EfficientNetB0) and the fallback (ResNet50). A live run with real services also passed every stage:
image → disease → advice → Crop Context → weather → news → chatbot. No secrets are in the repository
or its history.

**No FAILING items were found.** The main problems are:

- **Old local split (medium):** two report scripts and `build_leaf_reference.py` still use the old local split in `data/splits/`. About 80% of its "test" images are in the new models' training set, so those scripts would report optimistic results and could overwrite the leaf reference.
- **Documents outside the repo (medium):** the report, PPT and viva guide still show the old model numbers.
- **Small code issues (low):** the session link can be shared, the "Built with" list is incomplete, there are unused variables, and some direct dependencies are unpinned.

## 2. Architecture and folder structure

```
Farmer → Streamlit UI (app.py + app_pages/)
        → Disease detection (src/disease_detection: validate → preprocess → model → scope check)
        → CropCareAnalyzer (src/integration) → fertilizer/treatment lookup (src/fertilizer)
        → Crop Context (src/context, one per session, restorable for 12 h)
        → Weather (src/weather) · Live news (src/news) · Chatbot (src/chatbot, Groq via LangChain)
        → Smart summary on the dashboard
```

| Path | Contents |
|---|---|
| `app.py` | Streamlit entry point: sidebar, navigation, session restore |
| `app_pages/` | `home`, `disease_detection` (dashboard), `weather`, `news`, `assistant` |
| `src/disease_detection/` | `model_config` (model choice), `model_loader`, `preprocessing`, `predictor`, `scope_check` |
| `src/integration/analyzer.py` | Image → prediction → recommendation |
| `src/fertilizer/` | Advice table loader and `get_recommendation()` |
| `src/context/` | `CropContext` and per-session storage |
| `src/weather/` | Location search, OpenWeather (current), Open-Meteo (forecast, backup) |
| `src/news/news_service.py` | RSS reading, farmer filtering, topics |
| `src/chatbot/` | Groq model setup, prompt, `AgricultureAssistant` |
| `src/ui/` | Shared Streamlit parts: location picker, news panel, chat panel |
| `src/preprocessing/`, `src/model/` | Local dataset pipeline and ResNet50 training code |
| `models/<model>/` | Model (Git LFS), class names, leaf reference |
| `artifacts/reports/<model>/`, `artifacts/plots/<model>/` | Metrics, reports, curves for each model |
| `data/fertilizer/` | `fertilizer_recommendations.csv` (38 rows) |
| `data/splits/` | **Old** local split (half of the data) used by scripts and tests |
| `data/splits_kaggle/` | Common split used to train all three current models |
| `notebooks/` | 01 dataset analysis, 02 preprocessing, 04 ResNet50, 05 EfficientNetB0, 06 DenseNet121 |
| `scripts/` | Dataset inspection, training, evaluation, leaf reference, config check, test scripts |
| `tests/` | 10 pytest files, 178 tests |
| `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `requirements-docker.txt` | Deployment |
| `config/`, `src/utils/`, `ui/`, `logs/` | Leftovers: git-ignored and imported by nothing (`ui/` is empty) |

## 3. Module-by-module status

| # | Module | Status | Evidence |
|---|---|---|---|
| 1 | Streamlit frontend (`app.py`, 5 pages) | VERIFIED WORKING | 40 Streamlit AppTest tests pass: full flow, reruns, page switches, refresh restore, all-services-down case. The real-browser test was not repeated in this audit. |
| 2 | Disease detection: loading, preprocessing, labels, confidence | VERIFIED WORKING | 24 tests pass. Live: Grape Black rot leaf → `Grape___Black_rot`, confidence 1.00, in scope, 2.0 s including model load. |
| 2a | Non-leaf scope check | VERIFIED WORKING | 7 tests pass. Earlier today on this commit's models: EfficientNetB0 rejected 33/33 non-leaf images and 0/10,703 held-out real leaves (threshold 0.44). |
| 3 | Fertilizer recommendation | VERIFIED WORKING | 23 tests pass; `check_config.py`: 38 entries, every model class covered. Live: status `found`. |
| 4 | Treatment recommendation | VERIFIED WORKING | Same table and tests as fertilizer (treatment, eco-friendly and chemical columns). |
| 5 | Crop Context | VERIFIED WORKING | 10 tests pass. Live: disease, recommendation, weather and news all stored; no API key found in the context. |
| 6 | Current weather | VERIFIED WORKING | 14 tests pass (mocked failures). Live: Surat, OpenWeather, 26.3 °C, no errors. |
| 6a | Forecast | VERIFIED WORKING | Live: 7 forecast days from Open-Meteo. |
| 7 | Live agriculture news | VERIFIED WORKING | 17 tests pass. Live: 100 items from The Economic Times, The Hindu BusinessLine and Krishi Jagran; 0 feed errors; 100/100 links are https. |
| 8 | Agriculture chatbot | VERIFIED WORKING | 31 tests pass (fake models). Live Groq call: answered in 1.3 s with the main model; named "Black rot" and the confidence from the context. |
| 9 | Session state / refresh restore | VERIFIED WORKING | Covered by AppTest tests. See risk S1. |
| 10 | Configuration and environment variables | VERIFIED WORKING | `scripts/check_config.py`: 12 OK, 0 WARN, 0 FAIL. Both keys set; values were not displayed. |
| 11 | Docker | VERIFIED WORKING (earlier today); BLOCKED now | Image rebuilt today at 17:43 from the current runtime files and tested: healthy, both models load, leaf predicted, banner rejected. Docker Desktop is stopped now, so it was not re-run. |
| 12 | DVC | MISSING | No `.dvc` folder or `.dvc` files. Large files use Git LFS instead. |
| 13 | `scripts/test_integration.py`, `scripts/test_prediction.py` | IMPLEMENTED BUT NOT TESTED | Not run: they write report files into `artifacts/reports/`, and their "held-out" images are no longer held out (finding P1). |
| 14 | Local training pipeline (`src/model`, `scripts/train_resnet50.py`) | IMPLEMENTED BUT NOT TESTED | Not run: training was out of scope. The current models were trained on Kaggle instead. |
| 15 | Running app instances | BLOCKED | Nothing is listening on ports 8501–8503 now; the copies started earlier have stopped. |

## 4. Model inventory

All three models were loaded in this audit. Each takes a 224×224×3 input, gives 38 outputs and has a
`gap` layer. All three use the same class order, and each `class_names.json` matches its `.npy`. They
were trained on the **same** leakage-free Kaggle split (fingerprint `13afce0ee189dd7d`).

| Model | Role | File | Parameters | Test accuracy (8,799 leaves) | Macro F1 | Leaf reference / threshold |
|---|---|---|---|---|---|---|
| **EfficientNetB0** | **Production (default)** | `models/efficientnetb0/best_efficientnetb0.keras`, 46 MB | 4,387,273 | **98.45%** | 0.984 | 11,400 × 1,280, 0.44 |
| ResNet50 | Fallback (`DISEASE_MODEL=resnet50`) | `models/resnet50/best_resnet50.keras`, 221 MB | 24,122,022 | 98.35% | 0.983 | 11,400 × 2,048, 0.60 |
| DenseNet121 | Comparison only | `models/densenet121/best_densenet121.keras`, 50 MB | 7,309,670 | 96.95% | 0.970 | 11,400 × 1,024, 0.60 |

- **Production model:** `model_config.select_model()` returns EfficientNetB0. `.env` does not set `DISEASE_MODEL`.
- **DenseNet121 in the app:** it can't be selected; `DISEASE_MODEL=densenet121` falls back to EfficientNetB0. That is by design, but it isn't documented in `.env.example`.
- **McNemar tests** (`artifacts/reports/model_comparison.csv`): EfficientNetB0 vs ResNet50 p = 0.55 (equal); both are better than DenseNet121 (p < 0.0001).

## 5. Dependencies and paths

| Check | Result |
|---|---|
| Code compiles (`compileall` on app, pages, src, scripts, tests) | OK |
| Every third-party import installed in `.venv` (Python 3.11.9) | OK: TensorFlow 2.21.0, Keras 3.15.1, Streamlit 1.64.0, and the rest |
| Direct imports missing from `requirements.txt` | `groq` and `langchain_core` (in `src/chatbot/chatbot.py`), `httpx` (tests). They are installed through `langchain-groq`, so nothing breaks now. `requirements-docker.txt` pins `groq` and `langchain-core`. |
| `requirements.txt` pinning | Unpinned (`tensorflow>=2.16`, the rest without versions). A future install may get different versions. |
| Files the Dockerfile copies | All exist (checked today). |
| README links to files | 15 of 15 exist. Model numbers match the metrics files. |
| Stale path | `config/config.yaml` points to `models/resnet50/resnet50_plant_disease.keras`, which doesn't exist. The file is git-ignored and nothing imports it. |
| `data/splits/` vs `data/splits_kaggle/` | Both exist. The app uses neither. Scripts and tests use the old `data/splits/` (finding P1). |

## 6. Potential bugs and security risks

**Security checks that passed:**
- `.env` was never committed.
- No API-key patterns appear in tracked files or in git history.
- `.env.example` has placeholders only.
- Weather errors leave out the request URL, which contains the key.
- No key appears in the Crop Context.
- The Docker image runs as a non-root user and has no `.env`.

| ID | Type | Finding | Effect |
|---|---|---|---|
| P1 | Bug (evaluation) | `scripts/test_integration.py`, `scripts/test_prediction.py` and `scripts/build_leaf_reference.py` sample "held-out" images from the old `data/splits/`. Measured: 4,218 of 5,291 (80%) of its test images and 4,263 of 5,304 (80%) of its val images are in the current models' **training** set. | Their reports would be optimistic. Rerunning `build_leaf_reference.py` would **overwrite** `models/<model>/leaf_reference.npz` using leaves the model was trained on. |
| P2 | Documentation | The report, PPT and viva guide (outside the repo) still show the old numbers (98.09%, 97.62%, 97.94%). | Wrong numbers could be presented at the viva. |
| S1 | Security / privacy | The session id in the URL (`?sid=`) restores the leaf photo, results and chat for anyone who opens a copied link, for up to 12 hours. Two tabs opened from one link also overwrite each other's saved state. | Data leaks between people who share a link. |
| B1 | Minor bug | The summary's "Latest agriculture update" shows the first item after headlines about the detected crop have been moved to the top, so it may not be the newest. | Misleading label. |
| B2 | Minor | The "Built with" list in `app.py` (lines 230–233) leaves out LangChain/Groq, OpenWeather/Open-Meteo and the RSS feeds. | Incomplete information. |
| B3 | Code quality | `exc` is assigned but not used in `src/weather/common.py` lines 35 and 39. | Cosmetic. |
| B4 | Maintainability | Unpinned `requirements.txt`; `groq`/`langchain_core` used directly but not listed. | A future install may break. |
| B5 | Housekeeping | Leftover git-ignored folders `config/`, `src/utils/`, empty `ui/`, a root `__pycache__/` and `logs/`. | Confusing for readers of the folder. |
| B6 | Deployment | The old Docker Compose container `newsmartcropcare-smart-crop-care-1` maps port 8501, which clashes with a local app on 8501 (seen today). | It fails to start while the local app runs. |
| L1 | Limitation | Trained on lab-style single-leaf photos; field accuracy is not measured. Only 14 crops / 38 classes. | Results may be lower on farm photos. |
| L2 | Limitation | Weather, news and the chatbot depend on outside services; Groq has free-tier rate limits. | Those sections can be unavailable (handled with messages). |

## 7. Tests and results

| Test file | Tests | EfficientNetB0 | ResNet50 |
|---|---|---|---|
| `test_app.py` | 40 | pass | pass |
| `test_chatbot.py` | 31 | pass | pass |
| `test_disease_detection.py` | 24 | pass | pass |
| `test_fertilizer.py` | 23 | pass | pass |
| `test_news.py` | 17 | pass | pass |
| `test_weather.py` | 14 | pass | pass |
| `test_crop_context.py` | 10 | pass | pass |
| `test_preprocessing.py` | 8 | pass | pass |
| `test_scope_check.py` | 7 | pass | pass |
| `test_integration.py` | 4 | pass | pass |
| **Total** | **178** | **178 passed (94 s)** | **178 passed (177 s)** |

Other checks run in this audit:
- `scripts/check_config.py`: 12 OK.
- Live end-to-end run with real services: all stages passed.
- All three models loaded and predicted.

The weather, news and chatbot tests use mocks; the live run covers the real services.

## 8. Recommended fixes (priority order)

1. **P1:** point `scripts/test_integration.py`, `scripts/test_prediction.py` and `scripts/build_leaf_reference.py` at the common split (`data/splits_kaggle/` val/test, matched to local files by MD5). Until then, don't run `build_leaf_reference.py`.
2. **P2:** update the report, PPT and viva guide with the current results (98.45% / 98.35% / 96.95% on the same 8,799 leaves).
3. **S1:** after a session is restored, give it a new id and delete the old saved entry, so each link works only once.
4. **B4:** pin `requirements.txt` to the tested versions and list `groq` and `langchain-core` directly.
5. **B6:** remove the old Compose container, or document using another port when a local app is running.
6. **B1, B2, B3:** rename or fix the summary news label, complete "Built with", and remove the unused `exc` variables.
7. **B5:** delete the leftover `config/`, `src/utils/`, empty `ui/` and `logs/` folders, or move them to an archive.
8. Optional: document in `.env.example` that DenseNet121 is not selectable.
