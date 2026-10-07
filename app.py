"""Smart Crop Care - Streamlit UI (entry point).

UI only: image checks, model inference and advice lookup live in src/integration
(CropCareAnalyzer); weather data comes from src/weather and news from src/news. This file holds the
shared banner and sidebar; the pages are in app_pages/.

    streamlit run app.py
"""

import base64
import importlib.util
import json

import streamlit as st

st.set_page_config(page_title="Smart Crop Care", page_icon=":material/eco:", layout="centered")

if importlib.util.find_spec("tensorflow") is None:
    st.error(
        "TensorFlow is not installed in the Python environment running this app. "
        "Start the app from the project's virtual environment instead:\n\n"
        "`.venv\\Scripts\\streamlit run app.py`",
        icon=":material/error:")
    st.stop()

from src.disease_detection.model_loader import load_class_names  # noqa: E402
from src.disease_detection.predictor import parse_class_name  # noqa: E402
from src.disease_detection.predictor import DEFAULT_CONFIDENCE_THRESHOLD  # noqa: E402
from src.fertilizer import FertilizerDataError, load_recommendations  # noqa: E402
from src.model.resnet50 import TRAIN  # noqa: E402
from src.preprocessing import config  # noqa: E402
from tensorflow import __version__ as tf_version  # noqa: E402

METRICS_PATH = config.ROOT / "artifacts" / "reports" / "model_metrics.json"
SPLIT_REPORT_PATH = config.SPLITS_DIR / "split_report.json"
HERO_IMAGE = config.ROOT / "assets" / "hero_leaves.jpg"  # scripts/make_hero_image.py


@st.cache_data
def hero_html():
    """Banner with the leaf background; falls back to plain green if the image is missing."""
    try:
        encoded = base64.b64encode(HERO_IMAGE.read_bytes()).decode()
        background = f"url('data:image/jpeg;base64,{encoded}') center / cover"
    except OSError:
        background = "#1b5e20"
    return f"""
<style>
.scc-hero {{
    background: linear-gradient(rgba(0, 20, 8, 0.45), rgba(0, 20, 8, 0.55)), {background};
    border-radius: 1rem;
    padding: 3.5rem 1.5rem;
    text-align: center;
    color: #ffffff;
}}
.scc-hero h1 {{
    color: #ffffff;
    font-size: 2.6rem;
    font-weight: 800;
    margin: 0 0 0.5rem 0;
    padding: 0;
    text-shadow: 0 2px 12px rgba(0, 0, 0, 0.6);
}}
.scc-hero p {{
    color: #e8f5e9;
    font-size: 1.1rem;
    margin: 0;
    text-shadow: 0 1px 8px rgba(0, 0, 0, 0.6);
}}
</style>
<div class="scc-hero">
    <h1>Smart Crop Care</h1>
    <p>AI-powered plant disease detection, treatment advice, local weather, farmer news and a farming assistant</p>
</div>
"""


@st.cache_data
def supported_crops():
    crops = {}
    for name in load_class_names():
        crop, disease, healthy = parse_class_name(name)
        if not healthy:
            crops.setdefault(crop, []).append(disease)
        else:
            crops.setdefault(crop, [])
    return crops


def _read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


@st.cache_data
def project_facts():
    """Numbers for the About section, read from saved reports and code settings."""
    metrics = _read_json(METRICS_PATH) or {}
    splits_report = _read_json(SPLIT_REPORT_PATH) or {}
    try:
        advice_entries = len(load_recommendations())
    except FertilizerDataError:
        advice_entries = None
    return {
        "evaluation": metrics.get("evaluation", {}),
        "epochs": metrics.get("training_epochs", {}),
        "weakest": list(metrics.get("test_lowest_f1_classes", {}))[:2],
        "splits": splits_report,
        "num_classes": len(load_class_names()),
        "num_crops": len(supported_crops()),
        "threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        "train_cfg": TRAIN,
        "advice_entries": advice_entries,
    }


# --- Sidebar ---------------------------------------------------------------------
with st.sidebar:
    st.header("Features", icon=":material/apps:")
    st.markdown(":material/eco: **Disease detection & advice** :green-badge[Active]")
    st.markdown(":material/partly_cloudy_day: **Weather** :green-badge[Active]")
    st.markdown(":material/newspaper: **Live farmer news** :green-badge[Active]")
    st.markdown(":material/smart_toy: **Agriculture Assistant** :green-badge[Active]")

    facts = project_facts()
    test = facts["evaluation"].get("test")
    sizes = facts["splits"].get("split_sizes", {})
    cfg = facts["train_cfg"]

    st.header("About this project", icon=":material/info:")
    st.caption("Smart Crop Care identifies plant diseases from a photo of a single leaf "
               "and suggests fertilizer and treatment for the detected condition.")
    with st.container(horizontal=True):
        if test:
            st.metric("Test accuracy", f"{test['accuracy']:.1%}",
                      help=f"Measured on {test['images']:,} held-out images never used "
                           "in training.")
        st.metric("Crops", facts["num_crops"])
        st.metric("Classes", facts["num_classes"])

    with st.expander("How it works", icon=":material/route:"):
        st.markdown(
            "1. **Upload** a photo of one leaf.\n"
            "2. **Check** - the file is validated (format, size, not blank or corrupted).\n"
            "3. **Prepare** - the image is cropped to a square, resized to 224 x 224 and "
            "normalized for ResNet50.\n"
            f"4. **Predict** - the model picks one of {facts['num_classes']} crop/disease "
            "classes and reports its confidence.\n"
            "5. **Scope check** - the image's features are compared with real training "
            "leaves; pictures that are not leaves of a supported crop are rejected.\n"
            f"6. **Advise** - if confidence is at least {facts['threshold']:.0%}, fertilizer "
            "and treatment advice is shown; otherwise the app asks for a clearer photo.")

    with st.expander("Model", icon=":material/neurology:"):
        st.markdown(
            "**ResNet50** pretrained on ImageNet, with a new classification head:\n"
            f"global average pooling, dropout {cfg['dropout']}, dense {cfg['dense_units']} "
            f"(ReLU) and a {facts['num_classes']}-class softmax.")
        st.markdown(
            "**Two-stage transfer learning**\n"
            f"- Stage 1: backbone frozen, head trained for "
            f"{facts['epochs'].get('stage1', '?')} epochs (Adam, lr {cfg['stage1_lr']:g}).\n"
            f"- Stage 2: last ResNet block (conv5) fine-tuned for "
            f"{facts['epochs'].get('stage2', '?')} epochs (Adam, lr {cfg['stage2_lr']:g}), "
            "BatchNorm kept frozen.\n"
            "- Training-only augmentation: flips, rotation, zoom, shift, brightness, "
            "contrast.")

    with st.expander("Dataset", icon=":material/dataset:"):
        st.markdown(
            "**New Plant Diseases Dataset** (an augmented version of PlantVillage): "
            f"{facts['splits'].get('images_after_dedup', 0):,} unique 256 x 256 leaf "
            f"images across {facts['num_classes']} classes.")
        if sizes:
            st.markdown(
                "The original train/valid folders contain flipped and rotated copies of "
                "the same leaf on both sides, which would inflate accuracy. Images were "
                "therefore grouped by physical leaf and re-split so that no leaf appears "
                "in more than one set:\n"
                f"- Train: {sizes['train']:,}\n"
                f"- Validation: {sizes['val']:,}\n"
                f"- Test: {sizes['test']:,}")

    if test:
        with st.expander("Results", icon=":material/monitoring:"):
            st.markdown(
                f"On the **test set** ({test['images']:,} images):\n"
                f"- Accuracy: **{test['accuracy']:.1%}**\n"
                f"- Precision / recall / F1 (macro): {test['precision_macro']:.3f} / "
                f"{test['recall_macro']:.3f} / {test['f1_macro']:.3f}\n"
                f"- Correct class in top 3: {test['top3_accuracy']:.1%}")
            val = facts["evaluation"].get("val")
            if val:
                st.markdown(f"Validation accuracy: {val['accuracy']:.1%}")
            if facts["weakest"]:
                weakest = ", ".join(" - ".join(parse_class_name(c)[:2])
                                    for c in facts["weakest"])
                st.caption(f"Hardest classes: {weakest}.")

    with st.expander("Supported crops", icon=":material/grass:"):
        for crop, diseases in supported_crops().items():
            st.markdown(f"**{crop}**: {', '.join(diseases) if diseases else 'healthy only'}")

    with st.expander("Fertilizer & treatment advice", icon=":material/compost:"):
        entries = facts["advice_entries"]
        st.markdown(
            (f"A table of {entries} entries, one per class, " if entries else "One entry per "
             "class, ") +
            "covering fertilizer, cultural treatment, eco-friendly and chemical options.")
        st.caption("General guidance based on common plant-protection practice; it has not "
                   "been reviewed by an agronomist. Product registration and doses vary by "
                   "region, so confirm locally and follow the label.")

    with st.expander("Limitations", icon=":material/warning:"):
        st.markdown(
            "- Trained on PlantVillage-style photos of single leaves on plain "
            "backgrounds; field photos may be less accurate.\n"
            "- Only the listed crops and diseases are recognized.\n"
            "- Non-leaf images (faces, objects, scenes) are rejected by a feature "
            "similarity check, but a leaf of an unsupported plant may still pass.\n"
            "- Not a substitute for an expert diagnosis.")

    with st.expander("Built with", icon=":material/build:"):
        st.markdown(f"- Python, TensorFlow {tf_version} / Keras\n"
                    f"- Streamlit {st.__version__}\n"
                    "- Pillow, NumPy, pandas, scikit-learn")

# --- Pages -------------------------------------------------------------------------
page = st.navigation(
    [
        st.Page("app_pages/home.py", title="Home", icon=":material/home:", default=True),
        st.Page("app_pages/disease_detection.py", title="Disease detection",
                icon=":material/eco:"),
        st.Page("app_pages/weather.py", title="Weather", icon=":material/partly_cloudy_day:"),
        st.Page("app_pages/news.py", title="Farmer news", icon=":material/newspaper:"),
        st.Page("app_pages/assistant.py", title="Agriculture Assistant",
                icon=":material/smart_toy:"),
    ],
    position="top",
)
st.html(hero_html())
page.run()
