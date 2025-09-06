import os
from pathlib import Path

# AWS credentials - Use environment variables for security
ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID', '')
SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', '')

# Validate that credentials are provided
if not ACCESS_KEY or not SECRET_KEY:
    raise ValueError(
        "AWS credentials not found. Please set the following environment variables:\n"
        "- AWS_ACCESS_KEY_ID\n"
        "- AWS_SECRET_ACCESS_KEY"
    )

SRC_PATH = str(Path(__file__).parent.parent / "src")
DATASET_ROOT_PATH = str(Path(__file__).parent.parent / "dataset")
MODEL_ROOT_PATH = str(Path(__file__).parent.parent / "model")
os.makedirs(DATASET_ROOT_PATH, exist_ok=True)

DATASET_TRAIN = str(Path(DATASET_ROOT_PATH) / "movies_review_train_aai.csv")
DATASET_TRAIN_URL = "https://drive.google.com/uc?id=1nSeixkiFj1zmK5-Eo6gA3Ak4doJguzNm"
DATASET_TEST = str(Path(DATASET_ROOT_PATH) / "movies_review_test_aai.csv")
DATASET_TEST_URL = "https://drive.google.com/uc?id=18Fx4HPofqXsIzQZIYoaroa4o1VSMx9-H"

DATASET_ROOT_PATH = str(Path(__file__).parent.parent / "dataset")
CITIES = str(Path(__file__).parent.parent / "src/auxiliar_files/Lista_Municípios_com_IBGE_Brasil_Versao_CSV.csv")
os.makedirs(DATASET_ROOT_PATH, exist_ok=True)