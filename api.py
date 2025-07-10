from flask import Flask, request, jsonify
import joblib

model = joblib.load("sentiment_svm_model.pkl")

app = Flask(__name__)


sentiment_map = {
    "positive": "Positive",    #maping sentiment label
    "negative": "Negative",
    "neutral": "Neutral"
}

@app.route("/")
def home():
    return "Sentiment Analysis API is up"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        prediction = model.predict([text])[0]
        label = sentiment_map.get(str(prediction).lower(), "Unknown")

        return jsonify({
            "input_text": text,
            "predicted_sentiment": label
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5002,debug=True)
