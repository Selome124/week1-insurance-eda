import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Reads a CSV file and returns a DataFrame."""
    return pd.read_csv(path)
