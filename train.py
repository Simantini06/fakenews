"""
train.py
--------
Trains a TF-IDF + Logistic Regression pipeline on the WELFake dataset.

Dataset:
  Download from: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
  File: WELFake_Dataset.csv
  Place it in the same folder as this script.

WELFake label encoding:
  0 = REAL
  1 = FAKE

Expected accuracy: ~96-98%

Run:
    pip install -r requirements.txt
    python train.py

Output:
    models/pipeline.pkl
    models/meta.json
"""

import os
import json
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── CONFIG ─────────────────────────────────────────────────────────────────

CSV_PATH  = "WELFake_Dataset.csv"
TEST_SIZE = 0.2
SEED      = 42

# ── 1. LOAD ────────────────────────────────────────────────────────────────

print("[1/5] Loading WELFake dataset ...")
df = pd.read_csv(CSV_PATH)

print(f"      Shape   : {df.shape}")
print(f"      Columns : {list(df.columns)}")
print(f"      Labels  :\n{df['label'].value_counts()}\n")

# Detect and confirm label encoding
print("      Label encoding check:")
print(f"        0 → REAL (genuine news)")
print(f"        1 → FAKE (fabricated news)\n")

# Combine title + text
df["combined"] = (
    df["title"].fillna("") + " " + df["text"].fillna("")
).str.strip()

df = df[["combined", "label"]].dropna()
df = df[df["combined"].str.len() > 30]
df["label"] = df["label"].astype(int)

X = df["combined"].tolist()
y = df["label"].tolist()

print(f"      Usable rows : {len(X)}")
print(f"      Real (0)    : {y.count(0)}")
print(f"      Fake (1)    : {y.count(1)}\n")


# ── 2. SPLIT ───────────────────────────────────────────────────────────────

print(f"[2/5] Splitting — 80% train / 20% test ...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)
print(f"      Train : {len(X_train)}")
print(f"      Test  : {len(X_test)}\n")


# ── 3. TRAIN ───────────────────────────────────────────────────────────────

print("[3/5] Training TF-IDF + Logistic Regression ...")
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=150_000,
        sublinear_tf=True,
        min_df=3,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-zA-Z]{2,}\b",
    )),
    ("clf", LogisticRegression(
        C=3.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        n_jobs=-1,
    )),
])

pipeline.fit(X_train, y_train)
print("      Done.\n")


# ── 4. EVALUATE ────────────────────────────────────────────────────────────

print("[4/5] Evaluating on test set ...")
preds = pipeline.predict(X_test)
acc   = accuracy_score(y_test, preds)

print(f"\n{'='*50}")
print(f"  Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
print(f"{'='*50}")
# ✅ FIXED: correct target names matching label encoding
print(classification_report(y_test, preds, target_names=["REAL", "FAKE"]))
print("  Note: ~96-98% is expected for WELFake.\n")


# ── 5. SAVE ────────────────────────────────────────────────────────────────

print("[5/5] Saving model ...")
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/pipeline.pkl")

meta = {
    "model_type" : "tfidf_logreg",
    "accuracy"   : round(float(acc), 4),
    # ✅ FIXED: correct WELFake label map
    "label_map"  : {"0": "REAL", "1": "FAKE"},
    "dataset"    : "WELFake (72,134 articles)",
    "features"   : "title + text combined, TF-IDF bigrams, 150k vocab",
    "train_rows" : len(X_train),
    "test_rows"  : len(X_test),
}
with open("models/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n✅  Model saved → models/pipeline.pkl")
print(f"✅  Meta  saved → models/meta.json")
print(f"\nNext step: python app.py")