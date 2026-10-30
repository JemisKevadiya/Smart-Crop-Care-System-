# Smart Crop Care

## 1. Project Overview

Crop diseases are often noticed late, and many farmers cannot get an expert to look at a
sick plant quickly. Even after a disease is identified, they still have to work out the
right fertilizer and treatment, decide whether the weather allows spraying, and keep up
with schemes, prices and advisories.

Smart Crop Care brings these steps into one Streamlit web app. A farmer uploads a photo of
a single leaf. The app identifies the likely disease, suggests fertilizer and treatment,
shows the local weather and current farmer news, and lets the farmer ask an AI assistant
about the combined result.

The system combines:

- **Plant disease detection**: an EfficientNetB0 image model (ResNet50 as fallback), 38 classes across 14 crops
- **Fertilizer recommendation**: from a curated table for each disease
- **Treatment recommendation**: general, eco-friendly and chemical options
- **Weather information**: current conditions and a 7-day forecast
- **Live farmer news**: from the RSS feeds of Indian agriculture publishers
- **Agriculture chatbot**: Groq LLM through LangChain, using the app's results

It is a decision-support tool, not a replacement for an agriculture expert.

## 2. Features and Verification Status

These statuses come from the test run on 30 October 2026, reported in
[`reports/module_test_results.md`](reports/module_test_results.md) and
[`reports/end_to_end_test.md`](reports/end_to_end_test.md).
- **Real:** tested with real files, models and live services.
- **Mocked:** tested only with fake services in pytest.

| Feature | Status | How it was verified |
|---|---|---|
| Disease detection (EfficientNetB0) | Verified working | Real: held-out images, real browser upload |
| Invalid and non-leaf image rejection | Verified working | Real: 16 invalid-input cases; non-leaf photos rejected |
| Fertilizer and treatment advice | Verified working | Real, plus 23 unit tests |
| Current weather and 7-day forecast | Verified working | Real (OpenWeather, Open-Meteo); real no-key, wrong-key and no-network cases |
| Live farmer news | Verified working | Real (3 RSS feeds); real no-network case |
| Agriculture Assistant (English) | Verified working | Real (live Groq); real wrong-key and no-network cases |
| Hindi and Gujarati answers | Mocked only | Unit tests check the language instruction; real answers not checked |
| Location by coordinates | Mocked only | Unit tests; the real browser test used city search |
| Approximate location from IP (ipinfo.io) | Not tested | No unit or real test |
| Combined dashboard and summary | Verified working | Real browser, 12/12 stages |
| Session kept across refresh | Verified working | Real browser reload, plus AppTest tests |
| ResNet50 fallback model | Verified working | All 178 tests with `DISEASE_MODEL=resnet50`; app predictions match the Kaggle results |
| Docker image | Verified working | Rebuilt 30 October 2026; real browser against the container, 12/12 stages |

**Disease detection**
- Upload a JPEG, PNG, BMP or WEBP leaf photo (up to 15 MB). The app shows the crop, the
  disease, the model's confidence and the top 3 candidates.
- Empty, corrupted, blank, oversized and unsupported files are rejected with a clear
  message.
- Photos that are not leaves (faces, scenes, objects) are rejected as "not a supported
  leaf" instead of getting a made-up diagnosis.
- When confidence is below 80%, the app asks for a clearer photo and withholds the advice.

**Fertilizer and treatment**
- Fertilizer, cultural treatment, eco-friendly treatment, chemical treatment and notes
  for all 38 classes.

**Weather**
- Choose a location by city or village search, by coordinates, or approximately from
  your IP address.
- Current temperature, humidity, wind, condition and rain.
- A 7-day forecast: temperature, rain amount and chance of rain, humidity, wind and
  evapotranspiration (ET₀), with charts.
- If OpenWeather is unavailable, current conditions come from Open-Meteo instead.

**Live farmer news**
- Current headlines from Indian agriculture publishers, filtered to farmer-relevant
  topics.
- Topic categories, keyword search, a refresh button and links to the full articles.

**Agriculture Assistant**
- A chatbot that answers in **English, Hindi or Gujarati**.
- It uses the current disease result, advice, weather and news, and says "I don't have
  that information currently." when something is missing.

**One dashboard**
- The Disease detection page shows everything for the uploaded leaf on one page: result,
  crop care, weather, news, a summary and the assistant.

**Session kept across refresh**
- Results, chat and the last leaf survive page switches and a browser refresh (F5).
- The session id is in the page address (`?sid=`), so anyone who opens a copied link sees
  that session (see [Known Limitations](#15-known-limitations)).

**Docker**
- A ready-to-run container image (see [Run with Docker](#run-with-docker)).

## 3. System Architecture and Workflow

```
Farmer
  ↓  uploads a leaf photo, chooses a location, asks questions
Streamlit app  (app.py + app_pages/)
  ↓
Disease Detection        src/disease_detection  (validation → EfficientNetB0 → scope check)
  ↓
Crop Context             src/context  (one shared record per user session)
  ↓
Fertilizer + Treatment   src/fertilizer  (lookup in fertilizer_recommendations.csv)
  ↓
Weather                  src/weather  (OpenWeather + Open-Meteo)
  ↓
Live Farmer News         src/news  (publishers' RSS feeds)
  ↓
Agriculture Chatbot      src/chatbot  (Groq via LangChain, reads the crop context)
  ↓
Smart Crop Care          combined dashboard and summary
```

The modules do not import each other. They exchange plain data through the **Crop
Context** (`src/context/crop_context.py`):

- the analyzer stores the disease, confidence and advice;
- the weather and news modules store their latest results;
- the chatbot reads the whole context as text.

The context contains no images, models or API keys. `src/integration/analyzer.py`
(`CropCareAnalyzer`) chains image → disease → advice, and the pages only call these
modules. Each module is described in [Appendix A](#appendix-a-module-details).

App pages (`app_pages/`):

| Page | What it does |
|---|---|
| Home | What the app does, how the modules connect, location choice and "Try now" |
| Disease detection | Upload a leaf; the combined dashboard (result, crop care, weather, news, summary, assistant) |
| Weather | Current weather, 7-day forecast table and charts |
| Farmer news | All live news with categories, search and refresh |
| Agriculture Assistant | Full-page chat, with or without the results from the other pages |

## 4. Technology Stack

| Area | Technology |
|---|---|
| Language | Python 3.11 |
| Deep learning | TensorFlow / Keras; EfficientNetB0 (default) and ResNet50 (fallback), both ImageNet transfer learning |
| Data and images | NumPy, pandas, Pillow, ImageHash, scikit-learn (metrics, splits) |
| Plots and notebooks | Matplotlib, Jupyter (nbformat, nbconvert, ipykernel) |
| Web app | Streamlit (multipage navigation, `st.cache_data` / `st.cache_resource`) |
| Weather | OpenWeather (current conditions); Open-Meteo (forecast, geocoding, fallback current); ipinfo.io (approximate location) |
| News | RSS feeds of The Economic Times, The Hindu BusinessLine and Krishi Jagran, read with `requests` and `xml.etree` |
| Chatbot | Groq LLM API through LangChain (`langchain-groq`); default model `openai/gpt-oss-120b`, fallback `openai/gpt-oss-20b` |
| Testing | pytest, Streamlit AppTest |
| Deployment | Docker, Docker Compose |
| Model storage | Git LFS |

## 5. Repository Structure

```
app.py                       Streamlit entry point: navigation, sidebar, shared styles
app_pages/
  home.py                    overview and location choice
  disease_detection.py       upload + combined dashboard
  weather.py                 current weather and 7-day forecast
  news.py                    live farmer news
  assistant.py               Agriculture Assistant chat
src/
  disease_detection/         model choice, image validation, preprocessing, model loading,
                             prediction, non-leaf scope check
  fertilizer/                recommendation table loader and get_recommendation()
  integration/               CropCareAnalyzer: image -> disease -> advice
  context/                   CropContext and per-session storage (survives refresh)
  weather/                   location search, OpenWeather, Open-Meteo, combined report
  news/                      RSS reading, filtering and topic categories
  chatbot/                   Groq/LangChain model, system prompt, AgricultureAssistant
  ui/                        shared Streamlit parts: location picker, news panel, chat panel
  preprocessing/             dataset config, leaf-grouped splits, tf.data training pipeline
  model/                     ResNet50 definition, two-stage training, evaluation (local pipeline)
  env.py                     reads settings from the environment or .env
data/
  fertilizer/                fertilizer_recommendations.csv (archive/ holds the old table)
  splits/                    local dataset split, the older local pipeline's split, still used by some tests
  splits_kaggle/             split used to train the current models (val/test lists, split report, leakage check)
models/efficientnetb0/       best_efficientnetb0.keras (Git LFS), class_names, leaf_reference.npz (default model)
models/resnet50/             best_resnet50.keras (Git LFS), class_names, leaf_reference.npz (fallback)
models/densenet121/          best_densenet121.keras (Git LFS), class_names, leaf_reference.npz
                             (comparison only, not used by the app)
notebooks/                   01 dataset analysis, 02 preprocessing, 04 ResNet50, 05 EfficientNetB0
                             and 06 DenseNet121 (Kaggle, all on the common split)
scripts/                     dataset inspection, training, evaluation, leaf reference,
                             config check, prediction and integration tests
artifacts/
  plots/                     dataset plots; <model>/ training curves and confusion matrix
  reports/                   dataset reports; <model>/ metrics, classification report, test results;
                             model_comparison.csv
reports/                     project audit, fix log, test results, model validation, readiness review
assets/hero_leaves.jpg       banner image
tests/                       pytest suite, including Streamlit AppTest UI tests
.streamlit/config.toml       Streamlit theme and server settings
.env.example                 template for API keys
Dockerfile, docker-compose.yml, .dockerignore, requirements-docker.txt
requirements.txt
LICENSE
```

`Dataset/raw/` (the Kaggle images) is not in the repository. See [Dataset](#7-dataset).

## 6. Plant Disease Model

```
Image
  → Validation      format (JPEG/PNG/BMP/WEBP), size ≤ 15 MB, not empty,
                    not corrupted, not blank
  → Preprocessing   EXIF rotation fix, RGB, centre square crop, resize to 224×224,
                    model-specific scaling
                    (EfficientNetB0: raw 0–255 pixels; ResNet50: resnet50.preprocess_input)
  → EfficientNetB0  softmax over 38 classes → disease + confidence (top 3 kept)
  → Scope check     features compared with real training leaves;
                    too different → "not a supported leaf"
  → Confidence      ≥ 80%: advice shown;
                    < 80%: low confidence, advice withheld, clearer photo requested
```

The code is in `src/disease_detection/` (`model_config.py`, `preprocessing.py`,
`model_loader.py`, `predictor.py`, `scope_check.py`). `model_config.py` selects the model
(`DISEASE_MODEL`), its preprocessing, its leaf reference and its reports.

### Production model: EfficientNetB0 · fallback: ResNet50

**Production model check (30 October 2026):**
- With no `DISEASE_MODEL` set, the app uses EfficientNetB0. The running app showed "Identified from the image by the EfficientNetB0 model".
- All three models take 224×224×3 input, output 38 classes, and share the same class order.
- On the 5,273 test images available locally, the app's image path gave **the same top prediction as the Kaggle evaluation for 100% of images**, for every model.

Details: [`reports/model_validation.md`](reports/model_validation.md).

Both models were trained on Kaggle (2× T4 GPU) with the same notebook, which differs only in
the model name: [`notebooks/05_smart_crop_care_efficientnetb0.ipynb`](notebooks/05_smart_crop_care_efficientnetb0.ipynb)
and [`notebooks/04_smart_crop_care_resnet50.ipynb`](notebooks/04_smart_crop_care_resnet50.ipynb).
They use **exactly the same train / validation / test images** (identical split fingerprints), so
their results can be compared directly. A third model, DenseNet121, was trained the same way for
comparison only ([`notebooks/06_smart_crop_care_densenet121.ipynb`](notebooks/06_smart_crop_care_densenet121.ipynb)).

- **Head:** global average pooling → dropout 0.3 → Dense 256 (ReLU) → Dense 38 (softmax), on
  ImageNet weights.
- **Two-stage training:** the head for 5 epochs with the backbone frozen (Adam 1e-3), then 5 epochs
  fine-tuning the last stage with BatchNorm frozen (Adam 1e-5). EfficientNetB0 fine-tunes
  `block6`–`block7`; ResNet50 fine-tunes `conv5`.
- **Augmentation:** random flips, rotation, zoom, translation, brightness and contrast. ModelCheckpoint,
  EarlyStopping and ReduceLROnPlateau were used.
- **Preprocessing:** EfficientNetB0 normalises pixels inside the model, so it takes raw 0–255 input.
  ResNet50 uses `resnet50.preprocess_input`.

**Results on the same 8,799 held-out test leaves (Kaggle evaluation):**

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Macro ROC-AUC | Top-3 | Parameters | Training |
|---|---|---|---|---|---|---|---|---|
| **EfficientNetB0** | **98.45%** | 0.9847 | 0.9844 | 0.9844 | 0.9999 | 99.93% | 4.39 M (46 MB) | 100.5 min |
| ResNet50 | 98.35% | 0.9839 | 0.9833 | 0.9834 | 0.9999 | 99.97% | 24.1 M (221 MB) | 106.9 min |
| DenseNet121 (comparison) | 96.95% | 0.9705 | 0.9695 | 0.9695 | 0.9997 | 99.81% | 7.3 M (50 MB) | 102.9 min |

| Split | Images | EfficientNetB0 accuracy | ResNet50 accuracy | DenseNet121 accuracy |
|---|---|---|---|---|
| Train | 70,138 | 98.84% | 98.97% | 97.67% |
| Validation | 8,904 | 98.26% | 98.24% | 96.86% |
| **Test** | **8,799** | **98.45%** | **98.35%** | **96.95%** |

On the same test leaves EfficientNetB0 made 136 errors, ResNet50 145 and DenseNet121 268. Paired
McNemar tests: EfficientNetB0 vs ResNet50 p = 0.55, so **these two are equally accurate**; both are
significantly better than DenseNet121 (p < 0.0001). The full table is in
[`artifacts/reports/model_comparison.csv`](artifacts/reports/model_comparison.csv). EfficientNetB0 is the default because it is
about 5× smaller and about 2× faster on a CPU. The weakest classes for both are tomato diseases
(EfficientNetB0: Early blight F1 0.94, Target Spot 0.94; ResNet50: Target Spot 0.90, Early blight
0.93). Reports and plots: [`artifacts/reports/efficientnetb0/`](artifacts/reports/efficientnetb0/),
[`artifacts/reports/resnet50/`](artifacts/reports/resnet50/), [`artifacts/plots/`](artifacts/plots/).

**ResNet50 (fallback) — full results.** Trained on Kaggle (2× T4) in 106.9 min (stage 1 53.2 min,
stage 2 53.7 min); best validation accuracy 0.9824 after fine-tuning `conv5`.

| Split | Images | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | Macro ROC-AUC | Top-3 |
|---|---|---|---|---|---|---|---|---|
| Train | 70,138 | 98.97% | 0.9900 | 0.9895 | 0.9896 | 0.9897 | 1.0000 | 99.99% |
| Validation | 8,904 | 98.24% | 0.9833 | 0.9821 | 0.9825 | 0.9823 | 0.9999 | 99.94% |
| **Test** | **8,799** | **98.35%** | **0.9839** | **0.9833** | **0.9834** | **0.9835** | **0.9999** | **99.97%** |

Lowest test F1: Tomato Target Spot 0.904, Tomato Early blight 0.931, Tomato Spider mites 0.946,
Corn Cercospora leaf spot 0.948, Tomato Late blight 0.949. Per-class numbers:
[`artifacts/reports/resnet50/classification_report.txt`](artifacts/reports/resnet50/classification_report.txt).

**Supported crops (14):** Apple, Blueberry, Cherry, Corn (maize), Grape, Orange, Peach,
Bell pepper, Potato, Raspberry, Soybean, Squash, Strawberry and Tomato.

**80% confidence threshold (EfficientNetB0):**
- **Test set:** it flags 3.0% of images, catches 71% of the model's mistakes, and the remaining predictions are 99.54% accurate. ResNet50: 2.1% flagged, 58% of mistakes caught, 99.29% accurate.
- **5,430 held-out validation images (measured locally):** it flags 3.3%, catches 65% of mistakes, and the remaining predictions are 99.37% accurate.

**Rejecting non-leaf images.** Softmax confidence cannot catch pictures that are not leaves: the
model must pick one of its 38 classes, and it is often confident on faces or noise. So the check
works on features instead:

- The image's feature vector (after global average pooling) is compared with 11,400 real training
  leaves (300 per class).
- The score is the mean cosine similarity to the 5 most similar leaves.
- Below the model's threshold the image is rejected: **0.44** for EfficientNetB0, **0.60** for ResNet50.

EfficientNetB0: of 10,703 held-out real leaves none were rejected (lowest score 0.468), and all 33
non-leaf test images were rejected (highest score 0.417). ResNet50: no real leaf rejected in the
Kaggle check (lowest 0.693) and all 33 non-leaf images rejected (highest 0.537). Details:
[`scripts/build_leaf_reference.py`](scripts/build_leaf_reference.py),
[`artifacts/reports/efficientnetb0/leaf_scope_check_app.json`](artifacts/reports/efficientnetb0/leaf_scope_check_app.json).

## 7. Dataset

The models use the Kaggle
[New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset),
an augmented version of PlantVillage. Its provided `train/` and `valid/` folders share
flipped or rotated copies of the same leaves, and all 33 images in its `test/` folder are
copies of `valid/` images. So:

1. The `train/` and `valid/` folders were pooled (87,867 images), and 26 exact duplicates removed.
2. Images were grouped by physical leaf: same leaf name or UUID in the file name, same MD5, same
   perceptual hash, or a perceptual hash at most 4 bits different (the same leaf photographed
   twice). This gave 45,181 groups.
3. Whole groups were split 80/10/10 per class (seed 42): 70,138 / 8,904 / 8,799 images.

A leakage check in the notebook found **0** shared files, leaf groups, leaf names, UUIDs, identical
or near-identical images between the splits. See [`data/splits_kaggle/`](data/splits_kaggle/)
(`leakage_check.json`). The local dataset analysis is in
[`notebooks/01_dataset_analysis.ipynb`](notebooks/01_dataset_analysis.ipynb) and
[`notebooks/02_data_preprocessing.ipynb`](notebooks/02_data_preprocessing.ipynb).

**Setup.** The dataset is **not needed to run the app**. It is needed for most tests, the test scripts
and training:

1. Download the dataset from Kaggle (link above) and unzip it.
2. Copy its `train/`, `valid/` and `test/` folders into `Dataset/raw/` in the project folder:
   ```
   Dataset/raw/train/<class folders>
   Dataset/raw/valid/<class folders>
   Dataset/raw/test/<33 sample images>
   ```
3. Do not rename the class folders.

`Dataset/` is git-ignored. The test scripts find held-out images by matching file contents (MD5)
against [`data/splits_kaggle/`](data/splits_kaggle/), so any complete copy of the Kaggle dataset works.

## 8. Installation on Windows

Tested on Windows 11 with **Python 3.11.9**, Git 2.52 and Git LFS 3.7.

1. Install [Python 3.11](https://www.python.org/downloads/) (tick "Add python.exe to PATH"),
   [Git](https://git-scm.com/download/win) and [Git LFS](https://git-lfs.com/).
2. **Clone the project.** The model files are stored with Git LFS, so set up Git LFS first.
   In PowerShell:
   ```powershell
   git lfs install
   git clone https://github.com/JemisKevadiya/Smart-Crop-Care-System-.git
   cd Smart-Crop-Care-System-
   ```

## 9. Virtual Environment and Dependencies

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1          # in Command Prompt: .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` lists the packages without fixed versions. The app was tested with:

| Package | Version | Package | Version |
|---|---|---|---|
| tensorflow | 2.21.0 | keras | 3.15.1 |
| streamlit | 1.64.0 | numpy | 2.4.6 |
| pandas | 3.0.6 | pillow | 12.3.0 |
| langchain-groq | 1.1.3 | langchain-core | 1.6.7 |
| groq | 0.37.1 | requests | 2.34.2 |
| ImageHash | 4.3.2 | scikit-learn | 1.9.1 |

If a newer release causes problems, install these versions. `requirements-docker.txt` pins the
runtime ones.

## 10. Environment Variables

Settings are read from the environment or from `.env` in the project folder. `.env` is
git-ignored: never commit it. Key values are never printed or stored by the app.

Copy the template, then put your own keys in `.env`:

```powershell
Copy-Item .env.example .env
```

```
OPENWEATHER_API_KEY=your_openweather_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | For the assistant | Agriculture Assistant (free key at console.groq.com). Without it, the app shows that the assistant is unavailable. |
| `OPENWEATHER_API_KEY` | Optional | Current weather from OpenWeather. Without it, Open-Meteo is used. |
| `GROQ_MODEL` | Optional | A different Groq model (default `openai/gpt-oss-120b`). |
| `GROQ_FALLBACK_MODEL` | Optional | Backup model used when the main one is rate-limited (default `openai/gpt-oss-20b`; `none` turns it off). |
| `NEWS_FEEDS` | Optional | Replaces the default news feeds: `Publisher\|feed URL\|all` entries separated by `;`. |
| `DISEASE_MODEL` | Optional | `efficientnetb0` (default) or `resnet50`. If the chosen model's files are missing, ResNet50 is used. DenseNet121 cannot be selected. |

Live farmer news needs no API key. Check the setup without showing the keys:

```powershell
python scripts/check_config.py            # add --online to also test the keys and news feeds
```

## 11. Model Artifact Setup

The trained models are in the repository; nothing has to be trained to run the app.

| File | Size | Storage |
|---|---|---|
| `models/efficientnetb0/best_efficientnetb0.keras` | 46 MB | Git LFS |
| `models/resnet50/best_resnet50.keras` | 221 MB | Git LFS |
| `models/densenet121/best_densenet121.keras` | 50 MB | Git LFS (comparison only) |
| `models/<model>/class_names.npy`, `class_names.json` | small | Git |
| `models/<model>/leaf_reference.npz` | 21–38 MB | Git |

After cloning, make sure the real model files were downloaded, not only LFS pointers:

```powershell
git lfs pull
git lfs ls-files -s           # sizes should be 46 MB, 221 MB and 50 MB
python scripts/check_config.py
```

`check_config.py` should report `Disease model: models\efficientnetb0\best_efficientnetb0.keras (46 MB)`
and 38 classes. A file of only about 130 bytes is an LFS pointer: run `git lfs pull` again.

## 12. Running the App

`app.py` is the entry point. With the virtual environment active:

```powershell
streamlit run app.py
```

Then open http://localhost:8501. In the test run the app was ready about 5 seconds after starting.

**Using the app:**
1. **Weather** page: search for your city or village (or enter coordinates) to set your location.
2. **Disease detection** page: upload a clear photo of a single leaf. The dashboard shows the
   disease and confidence, the fertilizer and treatment advice, the weather for your location,
   farmer news, a summary and the assistant.
3. Ask the assistant a question, or use the suggested questions. The answer language can be
   English, Hindi or Gujarati.
4. **Farmer news** and **Agriculture Assistant** pages give the full news list and a full-page chat.

The first upload takes a few seconds longer while the model loads.

### Run with Docker

The image contains only what the app needs to run: both trained models, the advice table,
the app code and CPU-only TensorFlow (`requirements-docker.txt`). It contains no dataset,
notebooks, tests or API keys. The keys are passed in when the container starts.

```powershell
docker build -t smart-crop-care .
docker run --rm -p 8501:8501 --env-file .env smart-crop-care
```

Or, with Docker Compose:

```powershell
docker compose up --build
```

Then open http://localhost:8501. If a local copy of the app is already using port 8501, use
`-p 8502:8501` and open http://localhost:8502. The container used about 550 MB of memory after a
test with the EfficientNetB0 model (measured on 30 October 2026).

## 13. Testing and Test Status

```powershell
python -m pytest tests                     # 178 tests
$env:DISEASE_MODEL="resnet50"; python -m pytest tests; Remove-Item Env:DISEASE_MODEL
python scripts/test_prediction.py          # model checks on held-out images
python scripts/test_integration.py         # image -> advice on 380 held-out images
python scripts/check_config.py --online    # configuration and live services
```

Most tests and both scripts read images from `Dataset/raw/` (see [Dataset](#7-dataset)). The
weather, news and chatbot unit tests use mocked services, so they run offline. The two scripts write
their results to `artifacts/reports/<active model>/`.

**Results on 30 October 2026:**

| Test | Result |
|---|---|
| pytest, EfficientNetB0 | 178 passed |
| pytest, ResNet50 | 178 passed |
| `scripts/test_prediction.py` | All checks passed; validation accuracy 98.27% (5,430 images); one image per class 37/38 |
| `scripts/test_integration.py` | 380 held-out images: 98.42%; 9/9 error-handling checks passed |
| Real-browser end-to-end, local app | 12/12 stages passed |
| Real-browser end-to-end, Docker | 12/12 stages passed |
| Real failure cases (wrong keys, no network, missing model) | Handled with clear messages; detection still works offline |
| Not run | `scripts/build_leaf_reference.py`: it overwrites the model's leaf reference file |
| Not tested | IP-based location; real Hindi and Gujarati answers |

Full details: [`reports/module_test_results.md`](reports/module_test_results.md),
[`reports/end_to_end_test.md`](reports/end_to_end_test.md),
[`reports/model_validation.md`](reports/model_validation.md).

## 14. Troubleshooting

| Problem | What to do |
|---|---|
| `Activate.ps1 cannot be loaded because running scripts is disabled` | Use Command Prompt and `.venv\Scripts\activate.bat`, or allow scripts for your user: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| "The disease detection model could not be loaded" | The model file is missing or is an LFS pointer. Run `git lfs pull`, then `python scripts/check_config.py`. |
| Assistant says "GROQ_API_KEY is not set" or "Groq rejected the API key" | Add or correct `GROQ_API_KEY` in `.env`, then restart the app. |
| Weather note says Open-Meteo is used | `OPENWEATHER_API_KEY` is missing or not active yet (new keys can take a few hours). The app keeps working with Open-Meteo. |
| "Live farmer news is temporarily unavailable" | No internet, or all feeds are down. Try **Refresh news** later. |
| "Port 8501 is already in use" or a Docker "ports are not available" error | Another copy of the app is running. Stop it, or start this one on another port: `streamlit run app.py --server.port 8502` or `docker run -p 8502:8501 ...` |
| "This image does not look like a leaf…" for a real leaf | Use a clear photo of one leaf of a supported crop that fills most of the picture. |
| Low-confidence warning | Take a closer, well-lit photo of a single leaf. |
| TensorFlow prints `oneDNN` or GPU messages at start | These are information messages, not errors. On native Windows TensorFlow runs on the CPU. |
| Tests fail with "file not found" for images | Download the dataset into `Dataset/raw/` (see [Dataset](#7-dataset)). |

## 15. Known Limitations

- **Image quality and conditions.** The model was trained on PlantVillage-style photos of
  single leaves on plain backgrounds. Blurry photos, field photos with several leaves,
  soil or shadows, and early or mixed infections may be less accurate. Accuracy on field photos
  has not been measured.
- **Only 14 crops and 38 classes are recognized.** A healthy leaf of an unsupported plant,
  or a disease outside the 38 classes, cannot be identified correctly. The scope check
  rejects clear non-leaf images, but a leaf of an unsupported plant may look similar
  enough to pass.
- **The prediction can be wrong** even at high confidence. Tomato diseases are the weakest
  classes.
- **External services:**
  - Weather depends on OpenWeather and Open-Meteo being reachable.
  - News depends on the publishers' feeds being online and keeping their format.
  - The chatbot depends on Groq and its free-tier rate limits.
  - IP-based location is approximate, can be off by hundreds of kilometres, and was not tested.
- **News topics** are assigned by keywords and may sometimes be imprecise.
- **The chatbot can make mistakes.** It is an LLM and can misread or overstate things,
  even with the rules in its prompt. Hindi and Gujarati answers were not checked with the live
  service.
- **The advice has not been reviewed by an agronomist.** The fertilizer and treatment
  advice is general guidance. Products, registration and doses differ by region. Verify
  every recommendation with local agricultural guidance or an extension officer, and
  follow the product label.
- **Sessions:**
  - **Not permanent:** they are kept only in server memory for up to 12 hours and are lost when the app or container restarts. There are no user accounts.
  - **Shared by link:** the session id is in the page address, so anyone who opens a copied link sees that session (leaf photo, results and chat), and two tabs opened from one link overwrite each other.
- **Network access.** Streamlit listens on all network interfaces by default, so other computers on
  the same network can open the app while it runs. Docker publishes the port the same way.
- **Unpinned dependencies.** `requirements.txt` has no fixed versions, so a future install may get
  newer packages than the tested ones (see [section 9](#9-virtual-environment-and-dependencies)).

## 16. Future Improvements

None of the following are implemented yet:

- More crops and more disease classes, including crops important in India such as rice,
  wheat and cotton.
- Training on real field photos, so the model works better outside the lab setting.
- A mobile application, with offline detection on the phone.
- Soil sensor and IoT integration (soil moisture, nutrients) for better fertilizer and
  irrigation advice.
- Multimodal disease detection that combines the image with symptoms the farmer
  describes and with weather.
- Regional agricultural advisories (state agriculture department and IMD agro-advisories)
  and local-language news.
- User accounts with crop history saved over time, which would also replace the shareable
  session link.

## 17. Contributors

- **Jemis Kevadiya** ([@JemisKevadiya](https://github.com/JemisKevadiya)): repository owner and
  author of the commits, named in the [LICENSE](LICENSE).

Other team members are not recorded in the repository.

## Appendix A. Module Details

### Fertilizer and Treatment

Advice is **looked up, not generated**. The predicted class name (for example
`Tomato___Late_blight`) is the key into
[`data/fertilizer/fertilizer_recommendations.csv`](data/fertilizer/fertilizer_recommendations.csv),
which has one row for each of the 38 model classes. Its columns are `disease`, `crop`,
`category`, `fertilizer`, `treatment`, `eco_friendly_treatment`, `chemical_treatment`
and `notes`.

- `CropCareAnalyzer` (`src/integration/analyzer.py`) calls `get_recommendation()`
  (`src/fertilizer/recommendation.py`) with the prediction. It only does this for confident
  predictions of supported leaves.
- `get_recommendation()` also accepts readable names ("Tomato Early blight"). A disease
  without its crop is accepted only when the name matches exactly one crop.
- `get_recommendation()` never crashes on bad input. It returns a status (`found`,
  `unknown_disease`, `missing_recommendation` or `data_error`) and a message. Missing
  advice is shown as "Not available."
- The result is saved in the Crop Context, so the summary and the assistant quote the same
  advice the user sees.

The advice is general guidance from common plant-protection practice. Product
registration and application rates vary by country, and the app says so next to the
advice.

### Weather

Weather gives **environmental context for crop-care decisions**, such as when to spray,
irrigate or work in the field. It does **not** change or confirm the disease prediction.
The app and the assistant both state this.

- **Location** (`src/weather/location.py`): search by city or village (Open-Meteo
  geocoding), enter coordinates, or use an approximate location from the IP address
  (ipinfo.io). The chosen location is saved in the Crop Context and shared by all pages.
- **Current weather** (`openweather.py`): temperature, feels-like temperature, humidity,
  wind, condition and rain, from OpenWeather. If the key is missing or rejected, or the
  service is down, `weather_service.py` gets current conditions from Open-Meteo instead
  and shows why.
- **7-day forecast** (`openmeteo.py`): from Open-Meteo, with these fields per day:
  - condition;
  - max and min temperature;
  - rain amount (mm) and chance of rain;
  - mean humidity and maximum wind;
  - reference evapotranspiration (ET₀), which shows irrigation need.

  The Weather page shows these as a table and as temperature and rain charts.
- Current weather and the forecast are fetched separately, so one failing service does not
  hide the other. Results are cached for 10 minutes. Network and API errors show a message
  instead of crashing.

### Live Farmer News

- **Sources:** the public RSS feeds of three Indian publishers, read with `requests` (no
  scraping, no API key):
  - The Economic Times (Agriculture section);
  - The Hindu BusinessLine (Agri-business section);
  - Krishi Jagran.

  The feed list can be changed with `NEWS_FEEDS`. Every headline comes from a feed;
  nothing is generated.
- **Agriculture-focused filtering:**
  - The two agriculture-section feeds are kept whole.
  - Items from the mixed Krishi Jagran feed are kept only if the headline or summary
    contains farmer keywords (crop, fertiliser, MSP, mandi, monsoon, kisan, irrigation and
    others).
  - Items older than 30 days are dropped.
- **Categories:** each item gets a topic from keywords in its headline and summary. The
  topics are Government & Schemes, Market & MSP, Crop & Disease, Weather & Agriculture,
  Farming Technology, and Agriculture (the default). The page filters by topic and by a
  search box. Because topics come from keywords, a few may be imprecise.
- **Refresh:** news is cached for 15 minutes. The **Refresh news** button reloads it at
  once.
- **Article links:** each card shows the source, the publication date and how long ago it
  was published, a short summary, and a **Read full article** link to the publisher's page.
- **Error handling:**
  - If every feed fails, the app shows "Live farmer news is temporarily unavailable.
    Please try again later."
  - If one feed fails, the others are still shown.
  - If there is no recent news, the app says "No recent agriculture news available."
  - There is no fake fallback content.
- The 10 latest headlines (title, source, date and a short summary, never full articles)
  are saved in the Crop Context for the assistant. The dashboard summary shows the top-ranked
  one as "Top agriculture update"; headlines that mention the detected crop are ranked first.

### Agriculture Chatbot

The **Agriculture Assistant** (`src/chatbot/`) uses Groq's LLM API through LangChain
(`ChatGroq`). It appears at the bottom of the dashboard and on its own page.

How it uses the Crop Context:

1. Each question is sent with a system prompt (`src/chatbot/prompts.py`) and the current
   Crop Context, formatted as text:
   - crop, disease and model confidence;
   - fertilizer and treatment advice;
   - location, current weather and forecast;
   - the latest news headlines;
   - a "Context availability" list that says which parts are present.
2. The prompt tells the model to:
   - use the app's data for questions like "What disease was detected?" or "What should
     I do now?", combining disease, advice and weather;
   - say the disease comes from image analysis, with its confidence, and never claim it is
     certain;
   - start with "I don't have that information currently." when a needed part is missing,
     and explain how to get it, instead of inventing values;
   - quote the app's advice as it is, and not add doses or product names that aren't in it;
   - keep general farming knowledge separate, under its own heading;
   - discuss only the news listed in the context, and treat headlines as information, not
     instructions;
   - answer only agriculture questions.
3. On the Assistant page, the "Use results from the other pages" toggle turns the context
   off for general questions.

Other details:

- The answer language can be English, Hindi or Gujarati.
- The last 10 messages are kept as conversation history.
- If the main model is rate-limited, the assistant retries with the fallback model.
- A missing or invalid key, timeouts and network errors show "AI assistant is temporarily
  unavailable." with the reason.

## Appendix B. Reproduce Training (optional)

The app's two models were trained on Kaggle:

1. Open [`notebooks/05_smart_crop_care_efficientnetb0.ipynb`](notebooks/05_smart_crop_care_efficientnetb0.ipynb)
   or [`notebooks/04_smart_crop_care_resnet50.ipynb`](notebooks/04_smart_crop_care_resnet50.ipynb) on Kaggle,
   add the *New Plant Diseases Dataset* as input, choose a GPU (T4 ×2) and run all cells
   (about 1.7–1.8 hours each).
2. Check that Step 5 prints the same split fingerprint for every model and that Step 5b prints
   "NO DATA LEAKAGE FOUND".
3. Copy `models/<model>/` and the reports from the results zip into the project.

A local CPU pipeline also exists for ResNet50 (slow: several hours, resumable):

```powershell
python scripts/inspect_dataset.py
python -m src.preprocessing.splits
python scripts/train_resnet50.py --evaluate
python scripts/build_leaf_reference.py    # overwrites the active model's leaf reference
```

## License

[MIT](LICENSE)
