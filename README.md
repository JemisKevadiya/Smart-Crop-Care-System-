# Smart Crop Care System

Detects plant diseases from a photo of a single leaf and suggests fertilizer and
treatment for the detected condition. A ResNet50 model (transfer learning) classifies
the leaf into one of **38 crop/disease classes across 14 crops**, and a curated table
maps each class to fertilizer, cultural, eco-friendly and chemical treatment advice.
A Streamlit web app ties it together.

| Feature | Status |
|---|---|
| Plant disease detection | Done |
| Fertilizer and treatment recommendation | Done |
| Weather | Upcoming |
| Chatbot | Upcoming |

## Results

Measured on a held-out test set of 5,291 images that share no leaf with the training data
(see [Dataset and leakage](#dataset-and-leakage)):

| Split | Images | Accuracy | Macro F1 | Top-3 accuracy |
|---|---|---|---|---|
| Train | 42,114 | 98.61% | 0.9861 | 99.97% |
| Validation | 5,304 | 97.72% | 0.9771 | 99.83% |
| **Test** | **5,291** | **97.62%** | **0.9761** | **99.92%** |

The weakest classes are tomato diseases (Target Spot F1 0.82, Early blight F1 0.90).
Full per-class numbers: [`artifacts/reports/classification_report.txt`](artifacts/reports/classification_report.txt);
plots: [`artifacts/plots/`](artifacts/plots/).

## How it works

```
Leaf image
  -> validation (format, size, corrupted/blank check)
  -> centre square crop, resize to 224x224, resnet50.preprocess_input
  -> ResNet50 classifier -> disease + confidence
  -> scope check: features far from real training leaves -> "not a supported leaf"
  -> confidence >= 80%: fertilizer and treatment lookup
     confidence <  80%: advice withheld, user asked for a clearer photo
```

**Model** (`src/model/resnet50.py`): ResNet50 with ImageNet weights (`include_top=False`)
followed by global average pooling, dropout 0.3, Dense 256 (ReLU) and a softmax over the
classes. Trained in two stages:

1. Backbone frozen, head trained for 5 epochs (Adam, lr 1e-3).
2. The last residual stage (`conv5`) fine-tuned for 5 epochs (Adam, lr 1e-5), BatchNorm frozen.

Training uses random flips, rotation, zoom, translation, brightness and contrast, plus
ModelCheckpoint, EarlyStopping and ReduceLROnPlateau. It was trained on CPU.

**Low-confidence threshold (80%)** was chosen on the validation set: it flags 3.2% of
images, catches 65% of the model's mistakes, and the remaining predictions are 99.2% accurate.

**Rejecting non-leaf images.** Softmax confidence cannot detect pictures that are not
leaves: the model must pick one of its 38 classes, and it is often ~100% confident on faces
or noise. So each image's ResNet50 features (2048-value pooling output) are compared with
2,280 real training leaves (60 per class); the score is the mean cosine similarity to the
5 most similar ones. Below **0.60** the image is reported as "not a supported leaf" and no
disease or advice is shown ([`scripts/build_leaf_reference.py`](scripts/build_leaf_reference.py),
[`artifacts/reports/leaf_scope_check.json`](artifacts/reports/leaf_scope_check.json)):

| Images | Result |
|---|---|
| Real leaves (1,900 validation + 380 test) | 0 rejected (lowest score 0.64) |
| Non-leaf images (faces, scenes, plots, noise, patterns; 33) | 33 rejected (highest score 0.53), although their mean softmax confidence was 86.7% |

## Dataset and leakage

The model uses the Kaggle
[New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
(an augmented version of PlantVillage). Inspection
([`notebooks/01_dataset_analysis.ipynb`](notebooks/01_dataset_analysis.ipynb)) found that:

- the provided `train/` and `valid/` folders contain flipped/rotated copies of the same
  leaf on both sides (about 19% of source leaves), which inflates validation accuracy;
- all 33 images in the provided `test/` folder are byte-identical copies of `valid/` images.

So the train and valid folders were pooled, exact duplicates removed (52,709 images left),
images grouped by physical leaf, and whole groups split 80/10/10 per class
([`notebooks/02_data_preprocessing.ipynb`](notebooks/02_data_preprocessing.ipynb)).
No file, near-duplicate or source leaf appears in more than one split. The split lists
are in [`data/splits/`](data/splits/).

## Project structure

```
app.py                         Streamlit UI (calls the modules below, no model code)
src/
  preprocessing/               dataset splitting and the tf.data training pipeline
  model/                       ResNet50 definition, two-stage training, evaluation
  disease_detection/           image validation, model loading, prediction
  fertilizer/                  recommendation table loader and get_recommendation()
  integration/                 CropCareAnalyzer: image -> disease -> advice
data/
  splits/                      train/val/test lists (paths into Dataset/raw)
  fertilizer/                  fertilizer_recommendations.csv
models/resnet50/               best_resnet50.keras (Git LFS), class_names.npy
notebooks/                     01 analysis, 02 preprocessing, 03 training
scripts/                       dataset inspection, training, evaluation, end-to-end tests
artifacts/                     plots and reports (metrics, classification report)
tests/                         pytest suite, including Streamlit AppTest UI tests
```

## Setup

Requires Python 3.11 (tested with TensorFlow 2.21 and Streamlit 1.64 on Windows).
The model file is stored with **Git LFS**, so install it before cloning:

```bash
git lfs install
git clone https://github.com/JemisKevadiya/Smart-Crop-Care-System-.git
cd Smart-Crop-Care-System-

python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

Run it from the activated `.venv`. The app needs only the model and the files in `data/`;
the raw dataset is not required.

### API keys

Copy `.env.example` to `.env` and add your keys. `.env` is git-ignored; never commit it.

- `GROQ_API_KEY` - Agriculture Assistant (free key at console.groq.com).
- `OPENWEATHER_API_KEY` - current weather (optional; without it Open-Meteo is used).

Live farmer news reads publishers' RSS feeds and needs no key. Check the setup with
`python scripts/check_config.py` (add `--online` to test the keys); key values are never
printed.

## Run with Docker

The image contains only what inference needs: the trained model, the advice table,
the app code and CPU-only TensorFlow (`requirements-docker.txt`). It does not contain
the dataset, notebooks, tests or any API keys; keys are passed when the container starts.

```bash
docker build -t smart-crop-care .
docker run --rm -p 8501:8501 --env-file .env smart-crop-care
```

or, with Docker Compose:

```bash
docker compose up --build
```

Then open http://localhost:8501.

## Reproduce training (optional)

1. Download the Kaggle dataset and copy its `train/`, `valid/` and `test/` folders into
   `Dataset/raw/`.
2. Inspect the data and build the leaf-grouped splits:
   ```bash
   python scripts/inspect_dataset.py
   python -m src.preprocessing.splits
   ```
3. Train both stages and evaluate (several hours on CPU; resumes if interrupted):
   ```bash
   python scripts/train_resnet50.py --evaluate
   ```
4. Rebuild the reference leaves used to reject non-leaf images:
   ```bash
   python scripts/build_leaf_reference.py
   ```

## Tests

```bash
python -m pytest tests
python scripts/test_prediction.py      # prediction checks on real images
python scripts/test_integration.py     # end-to-end checks on 380 test images
```

Most tests read images from `Dataset/raw/`, so the dataset must be downloaded first.

## Limitations

- Trained on PlantVillage-style photos of single leaves on plain backgrounds; field
  photos may be less accurate.
- Only the 14 crops and 38 classes in the dataset are recognized.
- Non-leaf images (faces, scenes, noise) are rejected by the scope check, but a leaf of
  an unsupported plant may still look similar enough to pass.
- The fertilizer and treatment advice is general guidance based on common
  plant-protection practice and has not been reviewed by an agronomist. Product
  registration and doses vary by region: confirm locally and follow the label.

## License

[MIT](LICENSE)
