import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s")

PROJECT_NAME = "smart-crop-care"

list_of_files = [
    "app.py",
    "requirements.txt",
    "setup.py",
    ".env",
    "Dockerfile",
    "docker-compose.yml",
    "params.yaml",
    "dvc.yaml",
    "dvc.lock",
    "config/config.yaml",
    "data/raw/.gitkeep",
    "data/interim/.gitkeep",
    "data/processed/.gitkeep",
    "models/.gitkeep",
    "logs/.gitkeep",
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_validation.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/components/model_evaluation.py",
    "src/pipeline/__init__.py",
    "src/pipeline/training_pipeline.py",
    "src/pipeline/prediction_pipeline.py",
    "src/disease_detection/__init__.py",
    "src/disease_detection/preprocessing.py",
    "src/disease_detection/predictor.py",
    "src/disease_detection/class_mapping.py",
    "src/recommendations/__init__.py",
    "src/recommendations/disease_recommendation.py",
    "src/recommendations/treatment.py",
    "src/recommendations/prevention.py",
    "src/recommendations/fertilizer.py",
    "src/weather/__init__.py",
    "src/weather/weather_service.py",
    "src/weather/weather_advisor.py",
    "src/chatbot/__init__.py",
    "src/chatbot/chatbot_service.py",
    "src/chatbot/prompts.py",
    "src/utils/__init__.py",
    "src/utils/logger.py",
    "src/utils/exception.py",
    "src/utils/common.py",
    "notebooks/01_eda.ipynb",
    "notebooks/02_training.ipynb",
    "notebooks/03_evaluation.ipynb",
    "tests/__init__.py",
    "tests/test_data.py",
    "tests/test_model.py",
    "tests/test_recommendation.py",
    "tests/test_prediction.py",
    ".github/workflows/ci.yml",
]


def create_project_structure(files):
    for filepath in files:
        path = Path(filepath)
        filedir = path.parent

        if filedir != Path("."):
            os.makedirs(filedir, exist_ok=True)
            logging.info(f"Creating directory: {filedir}")

        if not path.exists() or path.stat().st_size == 0:
            path.touch()
            logging.info(f"Creating empty file: {path}")
        else:
            logging.info(f"{path} already exists, skipping")


if __name__ == "__main__":
    create_project_structure(list_of_files)
