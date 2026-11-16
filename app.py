
import streamlit as st
import joblib
import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Load saved files
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_map = joblib.load("label_map.pkl")

# Stopwords
stop_word = set(stopwords.words("english"))

# Text preprocessing
def preprocess_text(txt):

    # Lowercase
    txt = txt.lower()

    # Remove punctuation
    txt = txt.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove numbers
    txt = "".join(
        i for i in txt if not i.isdigit()
    )

    # Remove non-ASCII characters
    txt = "".join(
        i for i in txt if i.isascii()
    )

    # Remove stopwords
    word_tokens = word_tokenize(txt)

    filtered = [
        w for w in word_tokens
        if w not in stop_word
    ]

    return " ".join(filtered)


# Streamlit page configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="centered"
)

# UI
st.title("😊 NLP Sentiment Analysis")

st.write(
    "Enter your text below and let Machine Learning "
    "predict its emotion."
)

st.divider()

# Text input
user_text = st.text_area(
    "Enter your text",
    placeholder="Example: I am feeling very happy today...",
    height=150
)

# Prediction button
if st.button("Predict Sentiment", use_container_width=True):

    if user_text.strip() == "":
        st.warning("Please enter some text first.")

    else:
        # Preprocessing
        cleaned_text = preprocess_text(user_text)

        if not cleaned_text.strip():
            st.warning("Please enter meaningful text.")
        else:
            # Vectorization
            text_vector = vectorizer.transform(
                [cleaned_text]
            )

            # Prediction
            prediction = model.predict(text_vector)[0]

            # Convert numeric label to original emotion
            emotion = label_map.get(
                int(prediction),
                str(prediction)
            )

            st.divider()

            st.subheader("Prediction Result")

            st.success(
                f"Predicted Emotion: {emotion.upper()}"
            )

            # Prediction probabilities
            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    text_vector
                )[0]

                confidence = max(probabilities) * 100

                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )

                st.progress(float(max(probabilities)))

st.divider()

st.caption("Built with Python, Scikit-learn and Streamlit")
