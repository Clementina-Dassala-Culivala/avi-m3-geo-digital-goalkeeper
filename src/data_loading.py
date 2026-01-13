import pandas as pd
from pathlib import Path

# raiz do projeto (independente do local de execução)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw"


def load_datasets():
    """
    Carrega os datasets de andebol a partir da pasta data/raw.
    Retorna um dicionário com os DataFrames.
    """
    print(">>> DATA_RAW_PATH:", DATA_RAW_PATH)

    datasets = {}

    datasets["X_train"] = pd.read_csv(DATA_RAW_PATH / "handball_X_train.csv")
    datasets["X_test"] = pd.read_csv(DATA_RAW_PATH / "handball_X_test.csv")
    datasets["y_train"] = pd.read_csv(DATA_RAW_PATH / "handball_y_train.csv")
    datasets["y_test"] = pd.read_csv(DATA_RAW_PATH / "handball_y_test.csv")

    return datasets
