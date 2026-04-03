"""
app.py
======
Flask API that loads trained models and serves predictions.
Ensemble: Logistic Regression + LSTM + BERT (weighted average)

Endpoints:
  POST /predict   { "text": "..." }
  GET  /health

Run:
  python app.py
"""

import os
import re
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)

# ── Lazy-loaded globals
_tfidf       = None
_lr          = None
_lstm_tok    = None
_lstm_model  = None
_bert_tok    = None
_bert_model  = None

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

MODEL_DIR  = os.environ.get('MODEL_DIR', './models')
MAX_LEN    = 256
LSTM_MAXLEN = 300

# Ensemble weights  (LR, LSTM, BERT)
WEIGHTS = (0.25, 0.30, 0.45)


# ─────────────────────────────────────────────
# TEXT CLEANING  (same as train.py)
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(tokens)


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
def load_models():
    global _tfidf, _lr, _lstm_tok, _lstm_model, _bert_tok, _bert_model

    if _lr is None:
        tfidf_path = os.path.join(MODEL_DIR, 'tfidf.pkl')
        lr_path    = os.path.join(MODEL_DIR, 'lr_model.pkl')
        if not os.path.exists(tfidf_path):
            raise FileNotFoundError(f"Model not found: {tfidf_path}. Run train.py first.")
        with open(tfidf_path, 'rb') as f:
            _tfidf = pickle.load(f)
        with open(lr_path, 'rb') as f:
            _lr = pickle.load(f)
        print("[Load] LR + TF-IDF loaded")

    if _lstm_model is None:
        from tensorflow.keras.models import load_model
        lstm_path = os.path.join(MODEL_DIR, 'lstm_model.h5')
        lstm_tok_path = os.path.join(MODEL_DIR, 'lstm_tokenizer.pkl')
        if os.path.exists(lstm_path):
            _lstm_model = load_model(lstm_path)
            with open(lstm_tok_path, 'rb') as f:
                _lstm_tok = pickle.load(f)
            print("[Load] LSTM loaded")
        else:
            print("[Load] LSTM model not found — skipping")

    if _bert_model is None:
        bert_dir = os.path.join(MODEL_DIR, 'bert_model')
        if os.path.exists(bert_dir):
            from transformers import BertTokenizerFast, TFBertForSequenceClassification
            _bert_tok   = BertTokenizerFast.from_pretrained(bert_dir)
            _bert_model = TFBertForSequenceClassification.from_pretrained(bert_dir)
            print("[Load] BERT loaded")
        else:
            print("[Load] BERT model not found — skipping")


# ─────────────────────────────────────────────
# PREDICTION HELPERS
# ─────────────────────────────────────────────
def predict_lr(clean: str) -> float:
    """Returns probability of REAL (class 1)."""
    vec = _tfidf.transform([clean])
    return float(_lr.predict_proba(vec)[0][1])


def predict_lstm(clean: str) -> float:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq = _lstm_tok.texts_to_sequences([clean])
    pad = pad_sequences(seq, maxlen=LSTM_MAXLEN, truncating='post')
    prob = float(_lstm_model.predict(pad, verbose=0)[0][0])
    return prob


def predict_bert(raw: str) -> float:
    import tensorflow as tf
    enc = _bert_tok(
        raw,
        truncation=True,
        padding='max_length',
        max_length=MAX_LEN,
        return_tensors='tf',
    )
    logits = _bert_model(enc).logits
    probs  = tf.nn.softmax(logits, axis=-1).numpy()[0]
    return float(probs[1])  # probability of REAL


def ensemble_predict(raw_text: str) -> dict:
    clean = clean_text(raw_text)

    scores      = {}
    probs_real  = []
    active_weights = []

    # LR
    lr_prob = predict_lr(clean)
    scores['lr']   = lr_prob
    probs_real.append(lr_prob)
    active_weights.append(WEIGHTS[0])

    # LSTM
    if _lstm_model:
        lstm_prob = predict_lstm(clean)
        scores['lstm'] = lstm_prob
        probs_real.append(lstm_prob)
        active_weights.append(WEIGHTS[1])

    # BERT
    if _bert_model:
        bert_prob = predict_bert(raw_text)  # raw text for BERT
        scores['bert'] = bert_prob
        probs_real.append(bert_prob)
        active_weights.append(WEIGHTS[2])

    # Weighted ensemble
    total_w  = sum(active_weights)
    norm_w   = [w / total_w for w in active_weights]
    prob_real = sum(p * w for p, w in zip(probs_real, norm_w))
    prob_fake = 1 - prob_real

    confidence = max(prob_real, prob_fake) * 100

    if prob_real >= 0.60:
        verdict = 'REAL'
    elif prob_fake >= 0.60:
        verdict = 'FAKE'
    else:
        verdict = 'UNCERTAIN'

    # Build linguistic indicators
    indicators = build_indicators(raw_text, verdict)

    return {
        'verdict':    verdict,
        'confidence': round(confidence, 1),
        'prob_real':  round(prob_real * 100, 1),
        'prob_fake':  round(prob_fake * 100, 1),
        'model_scores': {k: round(v * 100, 1) for k, v in scores.items()},
        'indicators': indicators,
        'analysis':   build_analysis(verdict, confidence, scores),
    }


def build_indicators(text: str, verdict: str) -> list:
    indicators = []
    t = text.lower()

    sensational = ['breaking', 'shocking', 'unbelievable', 'you won\'t believe', 'secret', 'exposed', 'bombshell']
    emotional   = ['outrage', 'outrageous', 'disgusting', 'horrifying', 'terrifying', 'panic']
    credible    = ['according to', 'study shows', 'researchers found', 'report says', 'data shows', 'percent', 'confirmed']
    conspiracy  = ['deep state', 'cover up', 'they don\'t want you', 'mainstream media', 'fake news', 'agenda']
    source_words= ['reuters', 'ap news', 'bbc', 'cnn', 'nyt', 'new york times', 'washington post']

    if any(w in t for w in sensational):
        indicators.append({'label': 'Sensationalist language', 'type': 'warn'})
    if any(w in t for w in emotional):
        indicators.append({'label': 'Emotional manipulation', 'type': 'warn'})
    if any(w in t for w in conspiracy):
        indicators.append({'label': 'Conspiracy framing', 'type': 'warn'})
    if any(w in t for w in credible):
        indicators.append({'label': 'Credible attribution', 'type': 'ok'})
    if any(w in t for w in source_words):
        indicators.append({'label': 'Known news source cited', 'type': 'ok'})
    if len(text.split()) > 100:
        indicators.append({'label': 'Substantial article length', 'type': 'neutral'})
    if text[0].isupper() and '.' in text:
        indicators.append({'label': 'Proper sentence structure', 'type': 'ok' if verdict == 'REAL' else 'neutral'})
    if '!' in text or text == text.upper():
        indicators.append({'label': 'Excessive punctuation / caps', 'type': 'warn'})

    return indicators[:5]


def build_analysis(verdict: str, confidence: float, scores: dict) -> str:
    models_used = ', '.join([m.upper() for m in scores.keys()])
    conf_str    = f"{confidence:.0f}%"

    if verdict == 'FAKE':
        return (
            f"The ensemble ({models_used}) classified this content as FAKE with {conf_str} confidence. "
            "Linguistic signals include emotionally charged or sensationalist language patterns, "
            "unverified claims, and structural markers commonly found in the Kaggle Fake News corpus."
        )
    elif verdict == 'REAL':
        return (
            f"The ensemble ({models_used}) classified this content as REAL with {conf_str} confidence. "
            "The text exhibits neutral tone, factual framing, and attribution patterns consistent "
            "with credible journalism found in the True.csv training data."
        )
    else:
        return (
            f"The ensemble ({models_used}) could not reach a confident verdict ({conf_str} peak confidence). "
            "The text contains mixed signals — some credibility markers alongside uncertain or ambiguous framing. "
            "Manual fact-checking is recommended."
        )


# ─────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_dir': MODEL_DIR})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = (data.get('text') or '').strip()

    if not text:
        return jsonify({'error': 'No text provided'}), 400
    if len(text) < 10:
        return jsonify({'error': 'Text too short for analysis'}), 400

    try:
        load_models()
        result = ensemble_predict(text)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


if __name__ == '__main__':
    print("Starting Fake News Detector API ...")
    print(f"Model directory: {MODEL_DIR}")
    app.run(host='0.0.0.0', port=5000, debug=False)