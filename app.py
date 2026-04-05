"""
app.py
------
Flask backend for the Fake News Detector.
Uses a 3-method cascade for URL extraction:
  1. trafilatura  (best, handles most modern news sites)
  2. newspaper3k  (fallback)
  3. requests + BeautifulSoup (last resort)

Endpoints:
  GET  /            — health check + model info
  GET  /model-info  — loaded model metadata
  POST /predict     — JSON: {"text": "..."} OR {"url": "https://..."}
"""

import os
import time
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import FakeNewsPredictor

app = Flask(__name__)
CORS(app)

print("[STARTUP] Loading model...")
predictor = FakeNewsPredictor()
print("[STARTUP] Flask ready.\n")


# ── Shared headers (mimic a real browser) ─────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


# ── Method 1: trafilatura ─────────────────────────────────────────────────
'''
def try_trafilatura(url: str) -> str:
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return ""
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
        return (text or "").strip()
    except Exception as e:
        print(f"[trafilatura] failed: {e}")
        return ""
'''
def try_trafilatura(url: str) -> str:
    try:
        import trafilatura

        resp = requests.get(url, headers=HEADERS, timeout=12)
        resp.raise_for_status()

        html = resp.text

        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )

        return (text or "").strip()

    except Exception as e:
        print(f"[trafilatura] failed: {e}")
        return ""


# ── Method 2: newspaper3k ─────────────────────────────────────────────────

def try_newspaper(url: str) -> str:
    try:
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()
        title = article.title or ""
        body  = article.text  or ""
        return (title + " " + body).strip()
    except Exception as e:
        print(f"[newspaper3k] failed: {e}")
        return ""


# ── Method 3: requests + BeautifulSoup ───────────────────────────────────

def try_bs4(url: str) -> str:
    try:
        from bs4 import BeautifulSoup
        resp = requests.get(url, headers=HEADERS, timeout=12)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer",
                          "header", "aside", "form", "noscript"]):
            tag.decompose()

        article = soup.find("article")
        container = article if article else soup.find("body")
        if not container:
            return ""

        title_tag = soup.find("h1")
        title = title_tag.get_text(" ", strip=True) if title_tag else ""
        paragraphs = container.find_all("p")
        body = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
        return (title + " " + body).strip()
    except Exception as e:
        print(f"[bs4] failed: {e}")
        return ""


# ── Master extractor ──────────────────────────────────────────────────────

def extract_from_url(url: str):
    print(f"[extract] Trying URL: {url}")

    text = try_trafilatura(url)
    if len(text) > 20:
        print(f"[extract] trafilatura succeeded ({len(text)} chars)")
        return text, "trafilatura"

    text = try_newspaper(url)
    if len(text) > 20:
        print(f"[extract] newspaper3k succeeded ({len(text)} chars)")
        return text, "newspaper3k"

    text = try_bs4(url)
    if len(text) > 20:
        print(f"[extract] bs4 succeeded ({len(text)} chars)")
        return text, "bs4"

    print("[extract] All methods failed.")
    return "", ""


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def health():
    return jsonify({"status": "ok", "model": predictor.info()})


@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify(predictor.info())


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": 'Send JSON: {"text": "..."}'}), 400

    url    = (data.get("url")  or "").strip()
    text   = (data.get("text") or "").strip()
    source = "text"
    extractor_used = None

    if url:
        source = "url"
        extracted, extractor_used = extract_from_url(url)

        if not extracted:
            return jsonify({
                "error": (
                    "Could not extract text from this URL. "
                    "The site may be paywalled or blocking scrapers. "
                    "Please paste the article text directly instead."
                )
            }), 422

        text = extracted

    if not text:
        return jsonify({"error": "Provide either 'text' or 'url' in the request body."}), 400

    if len(text) < 10:
        return jsonify({"error": "Text is too short. Provide at least a full headline."}), 422

    t0     = time.perf_counter()
    result = predictor.predict(text)
    ms     = int((time.perf_counter() - t0) * 1000)

    return jsonify({
        **result,
        "source":     source,
        "extractor":  extractor_used,
        "char_count": len(text),
        "elapsed_ms": ms,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")