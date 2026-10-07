# Smart Crop Care - Streamlit app with the trained ResNet50 model (CPU inference).
#
#   docker build -t smart-crop-care .
#   docker run --rm -p 8501:8501 --env-file .env smart-crop-care
#
# API keys are never part of the image: pass them at run time (--env-file .env or
# -e GROQ_API_KEY=... -e OPENWEATHER_API_KEY=...).

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Dependencies first, so code changes do not reinstall them.
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

# Unprivileged user to run the app.
RUN useradd --create-home --uid 1000 appuser

# Only what inference and the app need at runtime (no dataset, notebooks or tests).
COPY --chown=appuser models/resnet50/best_resnet50.keras \
     models/resnet50/class_names.npy \
     models/resnet50/leaf_reference.npz  models/resnet50/
COPY --chown=appuser data/fertilizer/fertilizer_recommendations.csv  data/fertilizer/
COPY --chown=appuser data/splits/split_report.json  data/splits/
COPY --chown=appuser artifacts/reports/model_metrics.json  artifacts/reports/
COPY --chown=appuser assets/hero_leaves.jpg  assets/
COPY --chown=appuser .streamlit/config.toml  .streamlit/
COPY --chown=appuser src/ src/
COPY --chown=appuser app_pages/ app_pages/
COPY --chown=appuser app.py .

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=5)"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
