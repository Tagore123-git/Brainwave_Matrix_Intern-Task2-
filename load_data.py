# load_data.py
"""
Load the Sentiment140 CSV and return a DataFrame that **includes `date`.**
"""

import pandas as pd

def load_sentiment140_data(csv_path: str) -> pd.DataFrame:
    # Sentiment140 column layout
    columns = ['target', 'id', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(csv_path, encoding='latin-1', names=columns)

    # Map target 0→0 (negative) and 4→1 (positive)
    df['label'] = df['target'].map({0: 0, 4: 1})

    # Keep date + cleaned text + label for later processing
    return df[['date', 'text', 'label']]
