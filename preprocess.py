# preprocess.py
"""
Tweet-cleaning utilities for training and inference.
Provides:
    preprocess_text(text)  → cleaned string
    clean_tweet(text)      → alias to the same function
    clean_dataframe(df)    → adds 'clean_text' column
"""

import re, string, nltk, spacy, pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ------------------------------------------------------------------ #
# 1. Ensure NLTK resources are present
# ------------------------------------------------------------------ #
for pkg in ["punkt", "stopwords", "wordnet"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}") if pkg == "punkt" else nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

# ------------------------------------------------------------------ #
# 2. Globals
# ------------------------------------------------------------------ #
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])  # light-weight

# ------------------------------------------------------------------ #
# 3. Cleaning helpers
# ------------------------------------------------------------------ #
def preprocess_text(text: str) -> str:
    """
    Lower-case, strip URLs/mentions/hashtags, remove punctuation, stop-words,
    lemmatise. Returns a single whitespace-normalised string.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = word_tokenize(text)
    cleaned = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok.isalpha() and tok not in stop_words
    ]
    return " ".join(cleaned)

# Alias so older code that imports clean_tweet still works
clean_tweet = preprocess_text

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'clean_text' column using preprocess_text()."""
    df = df.copy()
    df["clean_text"] = df["text"].apply(preprocess_text)
    return df
