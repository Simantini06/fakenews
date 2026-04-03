"""
train.py
========
Trains the Fake News Detection ensemble:
  1. Logistic Regression  (TF-IDF features)
  2. LSTM                 (Keras / TensorFlow)
  3. BERT                 (HuggingFace Transformers)

Dataset: Kaggle Fake News Dataset
  - True.csv  (real news)
  - Fake.csv  (fake news)

Usage:
  python train.py --data_dir ./data --output_dir ./models
"""

import os
import re
import argparse
import pickle
import numpy as np
import pandas as pd

# ── NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)

# ── Sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ── Keras / TF
import tensorflow as tf
try:
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except (ImportError, ModuleNotFoundError):
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# ── HuggingFace
from transformers import (
    BertTokenizerFast,
    TFBertForSequenceClassification,
    create_optimizer,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MAX_LEN        = 256
LSTM_VOCAB     = 30_000
LSTM_MAXLEN    = 300
LSTM_EMBED_DIM = 128
LSTM_EPOCHS    = 5
BERT_EPOCHS    = 3
BATCH_SIZE     = 32
TEST_SIZE      = 0.2
RANDOM_STATE   = 42

stop_words  = set(stopwords.words('english'))
lemmatizer  = WordNetLemmatizer()


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────
def load_data(data_dir: str) -> pd.DataFrame:
    true_path = os.path.join(data_dir, 'True.csv')
    fake_path = os.path.join(data_dir, 'Fake.csv')

    if not os.path.exists(true_path) or not os.path.exists(fake_path):
        raise FileNotFoundError(
            f"Expected True.csv and Fake.csv in '{data_dir}'.\n"
            "Download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset"
        )

    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)

    df_true['label'] = 1   # REAL
    df_fake['label'] = 0   # FAKE

    df = pd.concat([df_true, df_fake], ignore_index=True)

    # Combine title + text for richer signal
    df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    df = df[['content', 'label']].dropna()
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"[Data] Loaded {len(df)} samples  |  Real: {df.label.sum()}  Fake: {(df.label==0).sum()}")
    return df


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)          # URLs
    text = re.sub(r'[^a-z\s]', '', text)                # non-alpha
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(tokens)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("[Preprocess] Cleaning text ...")
    df['clean'] = df['content'].apply(clean_text)
    return df


# ─────────────────────────────────────────────
# 3. LOGISTIC REGRESSION  (TF-IDF)
# ─────────────────────────────────────────────
def train_lr(X_train, X_test, y_train, y_test, output_dir):
    print("\n[LR] Fitting TF-IDF + Logistic Regression ...")

    tfidf = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2))
    Xtr   = tfidf.fit_transform(X_train)
    Xte   = tfidf.transform(X_test)

    lr = LogisticRegression(max_iter=1000, C=5.0, solver='lbfgs', n_jobs=-1)
    lr.fit(Xtr, y_train)

    preds = lr.predict(Xte)
    acc   = accuracy_score(y_test, preds)
    print(f"[LR] Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=['FAKE', 'REAL']))

    # Save
    with open(os.path.join(output_dir, 'tfidf.pkl'), 'wb') as f:
        pickle.dump(tfidf, f)
    with open(os.path.join(output_dir, 'lr_model.pkl'), 'wb') as f:
        pickle.dump(lr, f)

    return tfidf, lr


# ─────────────────────────────────────────────
# 4. LSTM
# ─────────────────────────────────────────────
def train_lstm(X_train, X_test, y_train, y_test, output_dir):
    print("\n[LSTM] Tokenizing ...")

    tokenizer = Tokenizer(num_words=LSTM_VOCAB, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)

    Xtr = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=LSTM_MAXLEN, truncating='post')
    Xte = pad_sequences(tokenizer.texts_to_sequences(X_test),  maxlen=LSTM_MAXLEN, truncating='post')

    model = Sequential([
        Embedding(LSTM_VOCAB, LSTM_EMBED_DIM, input_length=LSTM_MAXLEN),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    model.fit(
        Xtr, np.array(y_train),
        validation_split=0.1,
        epochs=LSTM_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es],
        verbose=1,
    )

    loss, acc = model.evaluate(Xte, np.array(y_test), verbose=0)
    print(f"[LSTM] Test Accuracy: {acc:.4f}")

    # Save
    model.save(os.path.join(output_dir, 'lstm_model.h5'))
    with open(os.path.join(output_dir, 'lstm_tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)

    return tokenizer, model


# ─────────────────────────────────────────────
# 5. BERT
# ─────────────────────────────────────────────
def encode_bert(texts, tokenizer, max_len=MAX_LEN):
    enc = tokenizer(
        list(texts),
        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_tensors='tf',
    )
    return {'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask']}


def train_bert(X_train, X_test, y_train, y_test, output_dir):
    print("\n[BERT] Loading tokenizer & model ...")
    bert_name = 'bert-base-uncased'
    bert_tok  = BertTokenizerFast.from_pretrained(bert_name)
    bert_model = TFBertForSequenceClassification.from_pretrained(bert_name, num_labels=2)

    print("[BERT] Encoding train/test ...")
    train_enc = encode_bert(X_train, bert_tok)
    test_enc  = encode_bert(X_test,  bert_tok)

    train_ds = tf.data.Dataset.from_tensor_slices((train_enc, list(y_train))).batch(BATCH_SIZE)
    test_ds  = tf.data.Dataset.from_tensor_slices((test_enc,  list(y_test))).batch(BATCH_SIZE)

    steps       = len(X_train) // BATCH_SIZE * BERT_EPOCHS
    warmup_steps = int(steps * 0.1)
    optimizer, schedule = create_optimizer(
        init_lr=2e-5, num_warmup_steps=warmup_steps, num_train_steps=steps
    )

    bert_model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    bert_model.fit(train_ds, epochs=BERT_EPOCHS, validation_data=test_ds, verbose=1)

    _, acc = bert_model.evaluate(test_ds, verbose=0)
    print(f"[BERT] Test Accuracy: {acc:.4f}")

    bert_dir = os.path.join(output_dir, 'bert_model')
    bert_model.save_pretrained(bert_dir)
    bert_tok.save_pretrained(bert_dir)
    print(f"[BERT] Saved to {bert_dir}")

    return bert_tok, bert_model


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   default='./data',   help='Folder with True.csv and Fake.csv')
    parser.add_argument('--output_dir', default='./models', help='Folder to save trained models')
    parser.add_argument('--skip_bert',  action='store_true', help='Skip BERT (faster for testing)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_data(args.data_dir)
    df = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean'], df['label'],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df['label'],
    )
    print(f"[Split] Train: {len(X_train)}  Test: {len(X_test)}")

    # Save split for app to reference
    split_info = {'test_size': TEST_SIZE, 'train_samples': len(X_train), 'test_samples': len(X_test)}
    with open(os.path.join(args.output_dir, 'split_info.pkl'), 'wb') as f:
        pickle.dump(split_info, f)

    train_lr(X_train, X_test, y_train, y_test, args.output_dir)
    train_lstm(X_train, X_test, y_train, y_test, args.output_dir)

    if not args.skip_bert:
        train_bert(X_train, X_test, y_train, y_test, args.output_dir)
    else:
        print("\n[BERT] Skipped (--skip_bert flag set)")

    print("\n✅ All models trained and saved to:", args.output_dir)


if __name__ == '__main__':
    main()