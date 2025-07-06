from flask import Flask, request, jsonify, render_template
from kafka import KafkaProducer, KafkaConsumer
import json
import threading

app = Flask(__name__)
messages = []

# Kafka Producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Route: Home / Dashboard
@app.route('/')
def index():
    return render_template('dashboard.html')

# Route: Submit Feedback
@app.route('/submit', methods=['POST'])
def submit():
    text = request.form.get('feedback')
    print(f"[Flask] Submitted feedback: {text}")  # ✅ Debug line
    producer.send('feedback_topic', {"text": text})
    producer.flush()
    return jsonify({"status": "sent"})

# Route: Get Processed Data
@app.route('/data')
def data():
    return jsonify(messages[-100:])

# Kafka Consumer in Background
def consume_feedback():
    consumer = KafkaConsumer(
        "feedback_with_sentiment",
        bootstrap_servers='localhost:9092',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    for msg in consumer:
        print(f"[Flask] Received prediction: {msg.value}")  # ✅ Debug line
        messages.append(msg.value)

# Start Consumer Thread
threading.Thread(target=consume_feedback, daemon=True).start()

if __name__ == '__main__':
    app.run(port=5000, debug=True)
