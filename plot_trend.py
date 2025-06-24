# plot_trend.py
"""
Plot yearly sentiment trend from the Sentiment140 dataset.
Run:
    python plot_trend.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from load_data import load_sentiment140_data
from preprocess import clean_dataframe

# ---------- 1. Load the CSV (change path if stored elsewhere) ----------
CSV_FILE = "training.1600000.processed.noemoticon.csv"
df = load_sentiment140_data(CSV_FILE)

# ---------- 2. Parse the date column ----------
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])           # keep rows with valid dates

# ---------- 3. Clean the tweet texts ----------
df = clean_dataframe(df)

# ---------- 4. Aggregate by YEAR ----------
df['year'] = df['date'].dt.year
trend = df.groupby(['year', 'label']).size().unstack(fill_value=0)

# ---------- 5. Plot ----------
trend.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='coolwarm')
plt.title("Sentiment Trend by Year (Sentiment140)")
plt.xlabel("Year")
plt.ylabel("Tweet Count")
plt.legend(['Negative', 'Positive'])
plt.tight_layout()
plt.show()
