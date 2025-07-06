from flask import Flask, request, jsonify
import joblib

# Load the trained model
model = joblib.load("sentiment_svm_model.pkl")

# Create Flask app
app = Flask(__name__)

# Sentiment label map (in case labels need formatting)
sentiment_map = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral": "Neutral"
}

@app.route("/")
def home():
    return "Sentiment Analysis API is up!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Predict
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
