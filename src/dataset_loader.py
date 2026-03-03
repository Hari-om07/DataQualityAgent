import pandas as pd
import os


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Universal dataset loader.
    Supports any CSV with automatic cleaning.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    # Try common encodings
    encodings = ["utf-8", "latin1", "iso-8859-1"]

    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            break
        except Exception:
            continue

    if df is None:
        raise ValueError("Unable to read dataset with common encodings")

    # Normalize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    # Remove empty rows
    df = df.dropna(how="all")

    # Reset index
    df = df.reset_index(drop=True)

    return df


if __name__ == "__main__":
    # Example test
    test_path = "data/clean_adult.csv"
    df = load_dataset(test_path)
    print("Loaded dataset shape:", df.shape)
    print("Columns:", list(df.columns))