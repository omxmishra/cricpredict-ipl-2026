import pandas as pd
import pickle
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR /
    "data" /
    "processed" /
    "final_model_df_v2.csv"
)

MODELS_DIR = (
    BASE_DIR /
    "models"
)


def load_processed_data():
    """
    Load processed match-level dataset.
    """

    return pd.read_csv(DATA_PATH)


def load_model(model_filename):
    """
    Load a saved ML model.

    Example:
    load_model("logistic_regression.pkl")
    """

    model_path = MODELS_DIR / model_filename

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model