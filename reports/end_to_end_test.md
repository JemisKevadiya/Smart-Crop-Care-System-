# End-to-End Test

**Date:** 30 October 2026 · **Code:** commit `7c343af` plus the fixes in `fix_log.md` · **Model:** EfficientNetB0

## 1. Entry point

`app.py` is the Streamlit entry point:
- It calls `st.navigation` with the five pages in `app_pages/`.
- The README, the Dockerfile `CMD` and `docker-compose.yml` all use it.

Command used:

```
.venv\Scripts\streamlit.exe run app.py --server.headless true --server.port 8501
```

The app reported healthy (`/_stcore/health` = ok) **5 seconds** after starting.

## 2. Real-browser test (live services, nothing mocked)

**Setup:**
- **Browser:** headless Microsoft Edge, controlled through its DevTools protocol against the running app at http://localhost:8501.
- **Services:** the real OpenWeather, Open-Meteo, RSS feeds and Groq.
- **Test leaf:** `Dataset/raw/valid/Grape___Black_rot/00090b0f-…FAM_B.Rot 0376.JPG`.

| # | Stage | Result | Evidence |
|---|---|---|---|
| 1 | Home page loads | PASS | 3.0 s; "How it works" shown |
| 2 | Weather page: search "Surat" | PASS | 1.6 s; °C and forecast shown |
| 3 | Upload leaf → disease + confidence | PASS | 1.0 s after upload: Grape – Black rot, 100.0% |
| 4 | Fertilizer + treatment | PASS | Fertilizer, treatment, eco-friendly and chemical lines shown |
| 5 | Weather on the dashboard | PASS | The Surat location set on the Weather page carried over through the Crop Context |
| 6 | Live farmer news | PASS | News cards with "Read full article" links |
| 7 | Smart summary | PASS | Disease, advice, weather and "Top agriculture update" lines |
| 8 | Chatbot (live Groq) | PASS | 2.6 s: "the detected disease is Black rot on grape, with a model confidence of 100%. This is a fungal disease…" |
| 9 | Browser refresh | PASS | The result is still shown after a reload (URL has `?sid=`) |
| 10 | Non-leaf photo | PASS | The cartoon leaf banner was rejected: "Not a supported leaf" |
| 11 | Invalid image file (`.jpg` with text inside) | PASS | "not a valid image" |
| 12 | Farmer news page | PASS | Articles listed |

**12 / 12 passed.** The app's error log had no errors or tracebacks. No API key appeared in any log,
report or test output (checked against the real key values).

**Test-script issues, not app bugs:** the first two attempts failed stages 2, 5 and 6, and one attempt
stopped while uploading the non-leaf photo.
- **Stage 2:** the script typed into the search box before the browser had focused it. The screenshot showed an empty box, and the app's correct "Enter a city or village name" message.
- **Stage 5:** failed only because stage 2 had set no location.
- **Stage 6:** the check ran before the news panel finished loading. The screenshot shows the news cards present.
- **Upload:** once, the file input wasn't found for the non-leaf upload; the next run uploaded it without problems.

The script was fixed: it now focuses the box through the browser and waits for the news panel and the
chat reply. The final run passed 12/12.

### Same test against Docker

The image was rebuilt with the Part 2 fixes and run as `docker run -p 8503:8501 --env-file .env
smart-crop-care`. The same 12 stages were run against http://localhost:8503: **12 / 12 passed**.

- **Container:** healthy, runs as `appuser`, and has no `.env` file.
- **Memory:** about 547 MB after the test.
- **Log:** no errors.
- **Local app:** the run against port 8501 was repeated with the final script and also passed 12 / 12.

**Script fix:** in two earlier Docker attempts the script looked for the file input or the search box
while the page was still redrawing after a navigation or a reload. The script now waits for those
elements. This was a test-script timing problem; the app worked.

## 3. Pipeline check of each stage (Python, live services)

This run used the same modules the app calls, in the Part 1 live run, on the same code apart from the
cosmetic fixes:

| Stage | Result |
|---|---|
| Image → disease | `Grape___Black_rot`, confidence 1.00, a supported leaf |
| Advice | status `found` |
| Crop Context | disease, advice, weather and news filled; no key inside |
| Weather | OpenWeather current (26.3 °C), 7-day Open-Meteo forecast |
| News | 100 items, 3 sources, 0 errors |
| Chatbot | Groq main model, 1.3 s; mentions "Black rot" and the confidence |

## 4. Failure end-to-end

The failure scenarios were run on the same code; details are in `module_test_results.md`.

| Scenario | Detection and advice | Weather | News | Chatbot |
|---|---|---|---|---|
| No API keys (`.env` hidden) | works | Open-Meteo fallback | works | "GROQ_API_KEY is not set…" |
| Wrong API keys (real rejection) | works | Open-Meteo fallback, key not shown | works | "Groq rejected the API key…" |
| No network (dead proxy) | works | "Could not connect…" | "temporarily unavailable" | "Could not connect to Groq…" |
| Missing model file | dashboard shows "model could not be loaded", no crash | — | — | — |

## 5. Blocked or not covered

- **Low-confidence path:** checked in the scripts and AppTest, not in the browser run.
- **Hindi and Gujarati answers, IP location, coordinates entry:** covered by unit tests only, not in the browser run.
- **Network exposure:** the app binds to all network interfaces (Streamlit default). Its startup log shows a Network URL and an External URL, so other machines on the network can reach it while it runs.
