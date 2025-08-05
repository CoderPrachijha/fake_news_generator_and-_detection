import streamlit as st
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import random
import numpy as np

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)


@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="distilgpt2")

generator = load_generator()



# Title
st.title("📰 Fake News Generator & Detector")
st.write("Enter a prompt to detect a news , whether it's REAL or FAKE.")
# Sample dataset for training detector (used only for demo purpose)
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

# Train detector model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
model = LogisticRegression()
model.fit(X, df['label'])

# User input
from better_profanity import profanity

prompt = st.text_input("✍️ Enter a news prompt:", "Breaking News:")

# ✅ Load the profanity filter once
profanity.load_censor_words()

# ✅ Filter the input before generating/detecting
if prompt.strip() == "":
    st.warning("⚠️ Please enter a valid news prompt.")
elif len(prompt.strip()) < 10:
    st.warning("⚠️ Prompt is too short. Try to write a full headline or sentence.")
elif not any(char.isalpha() for char in prompt):
    st.warning("⚠️ Prompt must contain readable text.")
elif profanity.contains_profanity(prompt):
    st.error("⚠️ Inappropriate language detected. Please rephrase.")
else:
    if st.button("Generate and Detect"):
        with st.spinner("Generating fake news article..."):
            result = generator(prompt, max_length=100, do_sample=False)[0]['generated_text']

        st.subheader("📰 Generated News Article:")
        st.write(result)

        text_vector = vectorizer.transform([result])
        prediction = model.predict(text_vector)[0]
        confidence = model.predict_proba(text_vector)[0].max()

        label = "REAL" if prediction == 1 else "FAKE"
        st.subheader("🔍 Detection Result:")
        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {confidence:.2f}")


        


    
