from kafka import KafkaConsumer, KafkaProducer
import json
import joblib
import re

model = joblib.load("sentiment_nb_model.pkl")

def clean_text(text):
    return re.sub(r"[^a-zA-Z\s]", "", text.lower().strip())

consumer = KafkaConsumer(
    'feedback_topic',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

for msg in consumer:
    data = msg.value
    cleaned = clean_text(data["text"])
    prediction = model.predict([cleaned])[0]
    data["prediction"] = prediction
    producer.send("feedback_with_sentiment", data)
    producer.flush()
