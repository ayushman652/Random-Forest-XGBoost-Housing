from pathlib import Path


# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR.parent / "datasets" / "housing"
OUTPUT_DIR = PROJECT_DIR / "outputs"


# Dataset
DATASET_PATH = DATASET_DIR / "California_housing.csv"
TARGET_COLUMN = "Target"


# Data splitting
TEST_SIZE = 0.2
RANDOM_STATE = 42


# Random Forest
RF_N_ESTIMATORS = 100


# XGBoost
XGB_N_ESTIMATORS = 100