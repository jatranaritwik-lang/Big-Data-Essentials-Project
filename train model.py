import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "news.csv"  # CSV must contain: title,text,label
# label: 0 = real news, 1 = fake news

def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    X = df["title"].fillna("") + " " + df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=20000,
            ngram_range=(1, 2),
            lowercase=True
        )),
        ("classifier", LogisticRegression(max_iter=1000, C=1.0))
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("\nClassification Report:\n", classification_report(y_test, predictions))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))

    joblib.dump(model, "fake_news_model.joblib")
    print("\nModel saved as fake_news_model.joblib")

if __name__ == "__main__":
    main()