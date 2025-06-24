from flask import Flask, render_template, request
import joblib
from preprocess import preprocess_text   # <— unified import

app = Flask(__name__)

# ------------------------------------------------------------------ #
# Load trained artefacts
# ------------------------------------------------------------------ #
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ------------------------------------------------------------------ #
# Routes
# ------------------------------------------------------------------ #
@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    text      = ""

    if request.method == "POST":
        text = request.form.get("tweet", "")
        if text.strip():
            cleaned = preprocess_text(text)
            vect    = vectorizer.transform([cleaned])
            pred    = model.predict(vect)[0]
            sentiment = "😊 Positive" if pred == 1 else "😞 Negative"

    return render_template("index.html", sentiment=sentiment, text=text)

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    app.run(debug=True)
