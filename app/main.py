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

# Dashboard (renders your HTML)
@app.route('/')
def index():
    return render_template('dashboard.html')

# Submit Feedback (from frontend form)
@app.route('/submit', methods=['POST'])
def submit():
    text = request.form.get('feedback')
    print(f"[Flask] Submitted feedback: {text}")  
    
    producer.send('feedback_topic', {"text": text})
    producer.flush()
    return jsonify({"status": "sent"})

# Send only the latest feedback to frontend
@app.route('/data')
def data():
    return jsonify([messages[-1]] if messages else [])

# Kafka Consumer Thread (receives sentiment-predicted messages)
def consume_feedback():
    consumer = KafkaConsumer(
        "feedback_with_sentiment",
        bootstrap_servers='localhost:9092',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    for msg in consumer:
        print(f"[Flask] Received prediction: {msg.value}")
        messages.append(msg.value)

# Start background consumer thread
threading.Thread(target=consume_feedback, daemon=True).start()

if __name__ == '__main__':
    app.run(port=5000, debug=True)
