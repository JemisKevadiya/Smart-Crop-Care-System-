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

- **Plant disease detection**: a ResNet50 image model, 38 classes across 14 crops
- **Fertilizer recommendation**: from a curated table for each disease
- **Treatment recommendation**: general, eco-friendly and chemical options
- **Weather information**: current conditions and a 7-day forecast
- **Live farmer news**: from the RSS feeds of Indian agriculture publishers
- **Agriculture chatbot**: Groq LLM through LangChain, using the app's results

## 2. Key Features

All features below are implemented and tested (see [Tests](#tests)).

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

**Docker**
- A ready-to-run container image (see [Run with Docker](#run-with-docker)).

## 3. System Architecture

```
Farmer
  ↓  uploads a leaf photo, chooses a location, asks questions
Streamlit app  (app.py + app_pages/)
  ↓
Disease Detection        src/disease_detection  (validation → ResNet50 → scope check)
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
modules.

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
| Deep learning | TensorFlow / Keras, ResNet50 (ImageNet weights, transfer learning) |
| Data and images | NumPy, pandas, Pillow, ImageHash, scikit-learn (metrics, splits) |
| Plots and notebooks | Matplotlib, Jupyter (nbformat, nbconvert, ipykernel) |
| Web app | Streamlit (multipage navigation, `st.cache_data` / `st.cache_resource`) |
| Weather | OpenWeather (current conditions); Open-Meteo (forecast, geocoding, fallback current); ipinfo.io (approximate location) |
| News | RSS feeds of The Economic Times, The Hindu BusinessLine and Krishi Jagran, read with `requests` and `xml.etree` |
| Chatbot | Groq LLM API through LangChain (`langchain-groq`); default model `openai/gpt-oss-120b`, fallback `openai/gpt-oss-20b` |
| Testing | pytest, Streamlit AppTest |
| Deployment | Docker, Docker Compose |
| Model storage | Git LFS |

## 5. Project Structure

```
app.py                       Streamlit entry point: navigation, sidebar, shared styles
app_pages/
  home.py                    overview and location choice
  disease_detection.py       upload + combined dashboard
  weather.py                 current weather and 7-day forecast
  news.py                    live farmer news
  assistant.py               Agriculture Assistant chat
src/
  disease_detection/         image validation, preprocessing, model loading,
                             prediction, non-leaf scope check
  fertilizer/                recommendation table loader and get_recommendation()
  integration/               CropCareAnalyzer: image -> disease -> advice
  context/                   CropContext and per-session storage (survives refresh)
  weather/                   location search, OpenWeather, Open-Meteo, combined report
  news/                      RSS reading, filtering and topic categories
  chatbot/                   Groq/LangChain model, system prompt, AgricultureAssistant
  ui/                        shared Streamlit parts: location picker, news panel, chat panel
  preprocessing/             dataset config, leaf-grouped splits, tf.data training pipeline
  model/                     ResNet50 definition, two-stage training, evaluation
  env.py                     reads settings from the environment or .env
data/
  fertilizer/                fertilizer_recommendations.csv (archive/ holds the old table)
  splits/                    train/val/test lists and class names
models/resnet50/             best_resnet50.keras (Git LFS), class_names.npy, leaf_reference.npz
notebooks/                   01 dataset analysis, 02 preprocessing, 03 ResNet50 training
scripts/                     dataset inspection, training, evaluation, leaf reference,
                             config check, prediction and integration tests
artifacts/
  plots/                     dataset, training curves, confusion matrix
  reports/                   metrics, classification report, test results
assets/hero_leaves.jpg       banner image
tests/                       pytest suite, including Streamlit AppTest UI tests
.streamlit/config.toml       Streamlit theme and server settings
.env.example                 template for API keys
Dockerfile, docker-compose.yml, .dockerignore, requirements-docker.txt
requirements.txt
LICENSE
```

`Dataset/raw/` (the Kaggle images) is not in the repository. It is needed only for
training and for most tests, not for running the app.

## 6. Installation

Requires Python 3.11. It was tested with TensorFlow 2.21 and Streamlit 1.64 on Windows.

1. **Clone the project.** The model file is stored with Git LFS, so install Git LFS first:
   ```bash
   git lfs install
   git clone https://github.com/JemisKevadiya/Smart-Crop-Care-System-.git
   cd Smart-Crop-Care-System-
   ```
2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
   ```
3. **Install the requirements:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure `.env`.** Copy `.env.example` to `.env` and add your keys (see
   [Environment Variables](#7-environment-variables)). To check the setup, run:
   ```bash
   python scripts/check_config.py            # add --online to test the keys
   ```
5. **Run Streamlit:**
   ```bash
   streamlit run app.py
   ```
   Then open http://localhost:8501.

### Run with Docker

The image contains only what the app needs to run: the trained model, the advice table,
the app code and CPU-only TensorFlow (`requirements-docker.txt`). It contains no dataset,
notebooks, tests or API keys. The keys are passed in when the container starts.

```bash
docker build -t smart-crop-care .
docker run --rm -p 8501:8501 --env-file .env smart-crop-care
```

Or, with Docker Compose:

```bash
docker compose up --build
```

Then open http://localhost:8501. The container uses about 600–700 MB of memory once the
model has loaded.

## 7. Environment Variables

Settings are read from the environment or from `.env` in the project folder. `.env` is
git-ignored: never commit it. Key values are never printed or stored by the app.

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

Live farmer news needs no API key.

## 8. Disease Detection

```
Image
  → Validation      format (JPEG/PNG/BMP/WEBP), size ≤ 15 MB, not empty,
                    not corrupted, not blank
  → Preprocessing   centre square crop, resize to 224×224,
                    resnet50.preprocess_input
  → ResNet50        softmax over 38 classes → disease + confidence (top 3 kept)
  → Scope check     features compared with real training leaves;
                    too different → "not a supported leaf"
  → Confidence      ≥ 80%: advice shown;
                    < 80%: low confidence, advice withheld, clearer photo requested
```

The code is in `src/disease_detection/` (`preprocessing.py`, `model_loader.py`,
`predictor.py`, `scope_check.py`).

**Model** (`src/model/resnet50.py`): ResNet50 with ImageNet weights (`include_top=False`),
followed by global average pooling, dropout 0.3, Dense 256 (ReLU) and a softmax over the
38 classes. It was trained on CPU in two stages:

1. Backbone frozen; the head trained for 5 epochs (Adam, learning rate 1e-3).
2. The last residual stage (`conv5`) fine-tuned for 5 epochs (Adam, learning rate 1e-5),
   with BatchNorm frozen.

Training used random flips, rotation, zoom, translation, brightness and contrast
changes, plus ModelCheckpoint, EarlyStopping and ReduceLROnPlateau.

**Supported crops (14):** Apple, Blueberry, Cherry, Corn (maize), Grape, Orange, Peach,
Bell pepper, Potato, Raspberry, Soybean, Squash, Strawberry and Tomato.

**Results** on a held-out test set that shares no leaf with the training data:

| Split | Images | Accuracy | Macro F1 | Top-3 accuracy |
|---|---|---|---|---|
| Train | 42,114 | 98.61% | 0.9861 | 99.97% |
| Validation | 5,304 | 97.72% | 0.9771 | 99.83% |
| **Test** | **5,291** | **97.62%** | **0.9761** | **99.92%** |

The weakest classes are tomato diseases (Target Spot F1 0.82, Early blight F1 0.90).
Per-class numbers are in
[`artifacts/reports/classification_report.txt`](artifacts/reports/classification_report.txt),
and the plots are in [`artifacts/plots/`](artifacts/plots/).

**80% confidence threshold.** This was chosen on the validation set. It flags 3.2% of
images, catches 65% of the model's mistakes, and the remaining predictions are 99.2%
accurate.

**Rejecting non-leaf images.** Softmax confidence cannot catch pictures that are not
leaves: the model must pick one of its 38 classes, and it is often close to 100% confident
on faces or noise. So the check works on features instead:

- Each image's 2048-value ResNet50 feature vector is compared with 2,280 real training
  leaves (60 per class).
- The score is the mean cosine similarity to the 5 most similar leaves.
- Below **0.60**, the image is rejected.

On 2,280 real leaves from the validation and test sets, none were rejected (lowest score
0.64). On 33 non-leaf images, all 33
were rejected (highest score 0.53). Details:
[`scripts/build_leaf_reference.py`](scripts/build_leaf_reference.py) and
[`artifacts/reports/leaf_scope_check.json`](artifacts/reports/leaf_scope_check.json).

**Dataset and leakage.** The model uses the Kaggle
[New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset),
an augmented version of PlantVillage. Its provided `train/` and `valid/` folders share
flipped or rotated copies of about 19% of source leaves. All 33 images in its `test/`
folder are copies of `valid/` images.

So the folders were handled as follows:

1. The `train/` and `valid/` folders were pooled.
2. Exact duplicates were removed, leaving 52,709 images.
3. Images were grouped by physical leaf.
4. Whole groups were split 80/10/10 per class.

No file, near-duplicate or source leaf appears in more than one split. See
[`notebooks/01_dataset_analysis.ipynb`](notebooks/01_dataset_analysis.ipynb),
[`notebooks/02_data_preprocessing.ipynb`](notebooks/02_data_preprocessing.ipynb) and
[`data/splits/`](data/splits/).

## 9. Fertilizer and Treatment

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

## 10. Weather

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

## 11. Live Farmer News

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
  are saved in the Crop Context for the assistant.

## 12. Agriculture Chatbot

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

## 13. Limitations

- **Image quality and conditions.** The model was trained on PlantVillage-style photos of
  single leaves on plain backgrounds. Blurry photos, field photos with several leaves,
  soil or shadows, and early or mixed infections may be less accurate.
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
  - IP-based location is approximate and can be off by hundreds of kilometres.
- **News topics** are assigned by keywords and may sometimes be imprecise.
- **The chatbot can make mistakes.** It is an LLM and can misread or overstate things,
  even with the rules in its prompt.
- **The advice has not been reviewed by an agronomist.** The fertilizer and treatment
  advice is general guidance. Products, registration and doses differ by region. Verify
  every recommendation with local agricultural guidance or an extension officer, and
  follow the product label.
- **Saved sessions are not permanent.** They are kept only in server memory for up to 12
  hours and are lost when the app or container restarts. There are no user accounts.

## 14. Future Improvements

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
- User accounts with crop history saved over time.

## Tests

```bash
python -m pytest tests
python scripts/test_prediction.py      # prediction checks on real images
python scripts/test_integration.py     # end-to-end checks on 380 test images
```

Most tests read images from `Dataset/raw/`, so download the dataset first. The weather,
news and chatbot tests mock the external services, so they run offline.

## Reproduce training (optional)

1. Download the Kaggle dataset and copy its `train/`, `valid/` and `test/` folders into
   `Dataset/raw/`.
2. Inspect the data and build the leaf-grouped splits:
   ```bash
   python scripts/inspect_dataset.py
   python -m src.preprocessing.splits
   ```
3. Train both stages and evaluate. This takes several hours on CPU and resumes if it is
   interrupted:
   ```bash
   python scripts/train_resnet50.py --evaluate
   ```
4. Rebuild the reference leaves used to reject non-leaf images:
   ```bash
   python scripts/build_leaf_reference.py
   ```

## License

[MIT](LICENSE)
