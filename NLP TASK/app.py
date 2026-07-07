import streamlit as st
import pickle
import re

# ---------------------------
# Load Saved Model & Vectorizer
# ---------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ---------------------------
# Text Cleaning Function
# ---------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="😊")

st.title("😊 Twitter Sentiment Analysis")
st.write("Enter a tweet below to predict its sentiment.")

tweet = st.text_area("Enter Tweet")

if st.button("Predict Sentiment"):

    if tweet.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        cleaned = clean_text(tweet)

        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0]

        if prediction == 1:
            st.success("😊 Positive Tweet")
        else:
            st.error("😞 Negative Tweet")

        st.subheader("Prediction Confidence")

        st.write(f"Positive : {probability[1]*100:.2f}%")
        st.write(f"Negative : {probability[0]*100:.2f}%")

        st.progress(float(max(probability)))