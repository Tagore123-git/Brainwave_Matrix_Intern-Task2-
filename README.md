# 💬 Twitter Sentiment Analyzer — Flask + ML + Modern UI

A full-stack sentiment analysis web application that analyzes emotional tone in tweets using **Machine Learning**, **Flask**, and an **investor-ready web interface** styled with **Tailwind CSS**.

> Built to showcase both technical capabilities and visual polish that impress users and investors.

---

![Interface Screenshot](https://i.imgur.com/8H8qk9Z.png) <!-- Replace this with your own screenshot later -->

---

## 🚀 Features

- 📌 Analyze live tweet or typed text sentiment (Positive/Negative)
- 🧠 Preprocessing with **NLTK** (stopwords, lemmatization, cleaning)
- 📊 Trained on **Sentiment140** dataset (1.6M+ tweets)
- 🧮 ML model: TF-IDF + Logistic Regression
- 🌐 API with **Flask** and elegant **Tailwind CSS UI**
- 📈 Trend plotting capability
- 🔐 Production-ready structure (modular, reusable)

---

## 📁 Project Structure
Twitter-API/
├── app.py # Flask web app
├── preprocess.py # Text cleaning & preprocessing
├── train_model.py # Train ML model
├── predict_tweet.py # Predict sentiment from CLI
├── plot_trend.py # Plot sentiment trend
├── load_data.py # Load dataset
├── tfidf_vectorizer.pkl # Saved TF-IDF model
├── sentiment_model.pkl # Trained sentiment model
├── templates/
│ └── index.html # Tailwind-styled frontend
├── training.1600000.processed.noemoticon.csv # Dataset
└── .venv/ # Virtual environment
