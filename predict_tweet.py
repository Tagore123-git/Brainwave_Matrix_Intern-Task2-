import joblib
from preprocess import clean_tweet

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

while True:
    tweet = input("Enter a tweet to analyze (or type 'exit' to quit): ")
    if tweet.lower() == 'exit':
        break
    cleaned = clean_tweet(tweet)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)[0]
    print("Prediction:", "Positive" if prediction == 1 else "Negative")
