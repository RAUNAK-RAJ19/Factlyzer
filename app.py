import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
import time
from scipy.sparse import hstack


# --------------------------------
# Load model ONCE using joblib
# --------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    return model, vectorizer

model, vectorizer = load_model()


# --------------------------------
# Text Cleaning
# --------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------
# Custom Features
# --------------------------------
clickbait_words = [
    "breaking", "shocking", "unbelievable",
    "you wont believe", "must read", "exclusive",
    "top", "amazing", "incredible", "surprising",
    "jaw dropping", "mind blowing", "omg",
    "stunning", "secret", "secrets", "mind-blowing"
]

def custom_features(text):
    t = text.lower()
    return [
        len(t.split()),
        text.count("!"),
        text.count("?"),
        sum(w in t for w in clickbait_words)
    ]


# --------------------------------
# Prediction Logic
# --------------------------------
def analyze_text(user_text):
    cleaned = clean_text(user_text)

    tfidf_vec = vectorizer.transform([cleaned])
    custom_vec = np.array([custom_features(user_text)])
    final_vec = hstack([tfidf_vec, custom_vec])

    prob = model.predict_proba(final_vec)[0][1]
    score = int(prob * 100)

    if score >= 80:
        level = "High Trust"
        message = "The article is Highly Trustworthy"
    elif score >= 50:
        level = "Medium Trust"
        message = "The article is Moderately Trustworthy"
    else:
        level = "Low Trust"
        message = "The article is Not Trustworthy"

    return score, level, message


# --------------------------------
# Streamlit Page Config
# --------------------------------
st.set_page_config(page_title="Factlyzer", page_icon="🛡️", layout="centered")

st.markdown("""
<style>
.fade { animation: fadeIn 2.3s; }
@keyframes fadeIn { from {opacity:0;} to {opacity:1;} }
.score { font-size: 50px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='fade'>🛡️ Factlyzer</h1>", unsafe_allow_html=True)
st.write("Article Trustworthiness Analyzer.")

text = st.text_area("Enter the article:", height=200)


# --------------------------------
# Button Action
# --------------------------------
if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing credibility..."):
            time.sleep(1)
            score, level, message = analyze_text(text)

        # Animated score
        placeholder = st.empty()
        for i in range(score + 1):
            placeholder.markdown(
                f"<div class='score'>{i}/100</div>",
                unsafe_allow_html=True
            )
            time.sleep(0.02)

        # Trust message
        if level == "High Trust":
            st.success(message)
        elif level == "Medium Trust":
            st.warning(message)
        else:
            st.error(message)
