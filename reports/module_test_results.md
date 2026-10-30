# Module Test Results

**Date:** 30 October 2026 · **Code:** commit `7c343af` plus the fixes in `fix_log.md` · **Production model:** EfficientNetB0

Every test below was executed in this session. The **Type** column shows how each test ran:

| Type | Meaning |
|---|---|
| **Unit (mocked)** | pytest with fake services or recorded data; no internet needed |
| **Real** | the real module with real files, models and live internet services |
| **Real failure** | a real failure condition: a wrong key, no network (traffic sent to a dead proxy) or a missing model file |
| **Simulated** | a failure produced inside the test, for example by hiding the `.env` keys or replacing a function |

## 1. Automated test suite (pytest)

`python -m pytest`, after the fixes:

| File | Tests | Area | EfficientNetB0 | ResNet50 |
|---|---|---|---|---|
| `test_app.py` | 40 | Streamlit UI flow (AppTest), session state; news uses a test feed | 40 pass | 40 pass |
| `test_chatbot.py` | 31 | Prompt, context, fallback, error mapping (fake chat models) | 31 pass | 31 pass |
| `test_disease_detection.py` | 24 | Validation, preprocessing, prediction, labels | 24 pass | 24 pass |
| `test_fertilizer.py` | 23 | Advice table loading and lookup | 23 pass | 23 pass |
| `test_news.py` | 17 | RSS parsing, filtering, topics (recorded feeds) | 17 pass | 17 pass |
| `test_weather.py` | 14 | OpenWeather, Open-Meteo, fallbacks (mocked HTTP) | 14 pass | 14 pass |
| `test_crop_context.py` | 10 | Crop Context updates | 10 pass | 10 pass |
| `test_preprocessing.py` | 8 | Local split, input pipeline | 8 pass | 8 pass |
| `test_scope_check.py` | 7 | Non-leaf check | 7 pass | 7 pass |
| `test_integration.py` | 4 | Image → advice | 4 pass | 4 pass |
| **Total** | **178** | | **178 passed (68.6 s)** | **178 passed (67.9 s)** |

## 2. Results by module

| Module | Status | Type | Evidence |
|---|---|---|---|
| **Disease detection** | VERIFIED WORKING | Real | 5,430 held-out validation images: 98.27% accuracy. 380 held-out test images: 98.42%. One per class: 37/38. Real browser: Grape Black rot at 100.0%. See `model_validation.md`. |
| Invalid images | VERIFIED WORKING | Real | 16/16 cases handled correctly in `test_prediction.py`: empty, text, truncated JPEG, 16×16, missing path, GIF, wrong input type, RGBA PNG, grayscale, 1200×900, file object, PIL image, grey/black/green blanks, missing model file. Browser: a fake `.jpg` was rejected with "not a valid image". |
| Non-leaf images | VERIFIED WORKING | Real | Noise, gradient and checkerboard rejected ("does not look like a leaf", score 0.24 < 0.44). Browser: the cartoon leaf banner was rejected as "Not a supported leaf". |
| Low confidence | VERIFIED WORKING | Real | Below 80%, the advice is withheld by default and attached only with `advise_when_uncertain=True`. The browser test used a confident leaf. |
| **Fertilizer recommendation** | VERIFIED WORKING | Unit + Real | 23 unit tests; `check_config.py`: all 38 classes covered. Real: shown on the dashboard. |
| **Treatment recommendation** | VERIFIED WORKING | Unit + Real | Same table: treatment, eco-friendly and chemical lines shown in the browser. |
| Missing or broken advice file | VERIFIED WORKING | Real | Missing CSV and CSV without columns both give `recommendation_unavailable` with a message (`test_integration.py`). |
| **Crop Context** | VERIFIED WORKING | Unit + Real | 10 unit tests. Real: disease, advice, weather and news stored; no API key in the context. |
| **Current weather** | VERIFIED WORKING | Real | Surat: OpenWeather, 26.3 °C (Part 1 live run); browser search "Surat" shows °C. |
| **Forecast** | VERIFIED WORKING | Real | 7 days from Open-Meteo. |
| Weather, no key | VERIFIED WORKING | Simulated (`.env` hidden) | Falls back to Open-Meteo with the note "OPENWEATHER_API_KEY is not set…"; 7-day forecast shown. |
| Weather, wrong key | VERIFIED WORKING | Real failure | OpenWeather rejected the key; Open-Meteo used, with a note; the key isn't shown. |
| Weather, no network | VERIFIED WORKING | Real failure | "Could not connect to Open-Meteo. Check your internet connection." for current and forecast. |
| **Live farmer news** | VERIFIED WORKING | Real | 100 items from 3 feeds, 0 feed errors, 100/100 https links. Shown on the dashboard and on the news page. |
| News, no network | VERIFIED WORKING | Real failure | "Live farmer news is temporarily unavailable. Please try again later." |
| **Chatbot** | VERIFIED WORKING | Real (live Groq) | Browser: "What disease was detected?" answered in 2.6 s with "Black rot on grape … confidence of 100% … fungal disease". |
| Chatbot, no key | VERIFIED WORKING | Simulated (`.env` hidden) | "GROQ_API_KEY is not set. Add GROQ_API_KEY=your_key to the .env file…" |
| Chatbot, wrong key | VERIFIED WORKING | Real failure | "Groq rejected the API key; check GROQ_API_KEY in .env." The backup model is not tried, which is correct for a bad key. |
| Chatbot, no network | VERIFIED WORKING | Real failure | "Could not connect to Groq; check your internet connection." |
| Detection offline | VERIFIED WORKING | Real failure | With no network, the leaf was still analysed correctly (`ok`). |
| **Missing model files** | VERIFIED WORKING | Simulated | EfficientNetB0 files missing → `select_model()` returns ResNet50. The real loader with a missing file → `ModelLoadError` with a clear message. Dashboard (AppTest): "The disease detection model could not be loaded. Model file not found…", no crash. |
| Model inference failure | VERIFIED WORKING | Simulated | Analyzer returns `model_error`: "Analysis failed: Model inference failed…" |
| **Streamlit workflow** | VERIFIED WORKING | Real browser | 12/12 stages, see `end_to_end_test.md`. |
| Session state / refresh | VERIFIED WORKING | Real browser + Unit | The result is kept after a page reload (`?sid=` restore); 40 AppTest tests. |
| Configuration | VERIFIED WORKING | Real | `scripts/check_config.py`: 12 OK, 0 WARN, 0 FAIL. |
| Docker | VERIFIED WORKING | Real browser | Rebuilt with the Part 2 fixes (`smart-crop-care`, 3.35 GB, built in 254 s). The container was healthy, contains every fix, runs as non-root `appuser` and has no `.env`. Real-browser test against it on port 8503: 12/12. Memory about 547 MB after a prediction. No errors in the container log. |
| `build_leaf_reference.py` | IMPLEMENTED BUT NOT TESTED | — | Not run on purpose: it overwrites a model artifact. |

**FAILING items:** none remain. One check failed during testing (the model-failure check in
`scripts/test_integration.py`). The cause was the test script itself, not the app, and it was fixed (fix F4).

## 3. Remaining limitations

- **Mocked unit tests:** the weather, news and chatbot unit tests use mocks. Their real behaviour was checked separately (rows marked Real or Real failure above).
- **Missing keys:** the "no key" cases hide the `.env` file inside the test process, because the real `.env` has keys.
- **Supplied test folder:** in `test_prediction.py`, the 33 supplied images (31/33 correct) are labelled "twin in train/val/test" from the **old** local split, so that label is out of date. These images are a demo, not an accuracy measurement.
- **Field photos:** accuracy on farm photos is not measured.
