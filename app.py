import streamlit as st
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import random
import numpy as np
import re
from better_profanity import profanity

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Load GPT-2 generator with caching
@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="distilgpt2")

generator = load_generator()

# App title
st.title("📰 Fake News Generator & Detector")
st.write("Enter a prompt to generate and detect whether the news is REAL or FAKE.")

# Load mini sample dataset and train the detector
@st.cache_data
def load_data():
    data = {
        'text': [
            'Breaking news: Government announces free healthcare for all.',
            'Aliens landed in New York last night, witnesses claim.',
            'Stock markets crash due to unforeseen economic crisis.',
            'Scientists discover unicorns living in Amazon rainforest.',
            'Local elections postponed amid fraud allegations.',
            'Chocolate cures cancer, new study reveals.',
        ],
        'label': [1, 0, 1, 0, 1, 0]  # 1 = Real, 0 = Fake
    }
    return pd.DataFrame(data)

df = load_data()
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
model = LogisticRegression()
model.fit(X, df['label'])

# User input
profanity.load_censor_words()
prompt = st.text_input("✍️ Enter a news prompt:", "Breaking News:")

# Input Validation
input_valid = True

if prompt.strip() == "":
    st.warning("⚠️ Please enter a valid news prompt.")
    input_valid = False
elif len(prompt.strip()) < 10:
    st.warning("⚠️ Prompt is too short. Try to write a full headline or sentence.")
    input_valid = False
elif not any(char.isalpha() for char in prompt):
    st.warning("⚠️ Prompt must contain readable text.")
    input_valid = False
elif profanity.contains_profanity(prompt):
    st.error("⚠️ Inappropriate language detected. Please rephrase.")
    input_valid = False

def is_nonsense(text):
    total_chars = len(text)
    alpha_chars = sum(c.isalpha() for c in text)
    return alpha_chars / total_chars < 0.3  # Too few letters → likely gibberish

...

if input_valid and st.button("Generate and Detect"):
    with st.spinner("Generating fake news article..."):
        result = generator(prompt, max_length=100, do_sample=False)[0]['generated_text']

    st.subheader("📰 Generated News Article:")
    st.write(result)

    if is_nonsense(result):
        st.error("⚠️ Generated article does not contain meaningful content.")
    else:
        text_vector = vectorizer.transform([result])
        prediction = model.predict(text_vector)[0]
        confidence = model.predict_proba(text_vector)[0].max()

        label = "REAL" if prediction == 1 else "FAKE"
        st.subheader("🔍 Detection Result:")
        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {confidence:.2f}")

        label = "REAL" if prediction == 1 else "FAKE"
        st.subheader("🔍 Detection Result:")
        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {confidence:.2f}")
