import pandas as pd
import re
import string
import joblib
import nltk
import mlflow
import mlflow.sklearn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("data.csv")

df.dropna(subset=["text", "sentiment"], inplace=True)# Drop rows with missing text or sentiment

# Ensure column names are consistent
df.columns = [col.strip().lower() for col in df.columns]

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Apply cleaning
df["text_clean"] = df["text"].apply(clean_text)

# Features and labels
X = df["text_clean"]
y = df["sentiment"]

# Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Pipeline: TF-IDF + SVM
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=25000,
        stop_words='english'
    )),
    ("clf", LinearSVC()) 
])

# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("sentiment_analysis_svm")
mlflow.sklearn.autolog()

# Training & Logging
with mlflow.start_run():
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_artifact("data.csv")

    # Save model
    joblib.dump(pipeline, "sentiment_svm_model.pkl")
