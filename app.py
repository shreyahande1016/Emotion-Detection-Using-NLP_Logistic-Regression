
import streamlit as st
import joblib
import re
import string
import pandas as pd

# --------------------------------
# 1. Load trained model and files
# --------------------------------


# Load trained model and files

model = joblib.load("sentiment_model.pkl")

tfidf = joblib.load("tfidf_vectorizer.pkl")

emotion_mapping = joblib.load("emotion_mapping.pkl")

stop_words = joblib.load("stop_words.pkl")

# Reverse mapping: number -> emotion

reverse_mapping = {
    value: key for key, value in emotion_mapping.items()
}


# Reverse mapping: number -> emotion
reverse_mapping = {
    value: key for key, value in emotion_mapping.items()
}

# --------------------------------
# 2. Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Emotion Detection",
    page_icon="💭",
    layout="wide"
)

# --------------------------------
# 3. Custom CSS
# --------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #101827;
    color: white;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #8B9CF7;
}

.subtitle {
    text-align: center;
    color: #B8C2D8;
    font-size: 17px;
}

.result-box {
    background-color: #202D43;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #465675;
}

.result-emotion {
    font-size: 32px;
    font-weight: bold;
    color: #A5B4FC;
}

.stButton button {
    width: 100%;
    background-color: #6366F1;
    color: white;
    border-radius: 10px;
    height: 45px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------
# 4. Text Preprocessing
# --------------------------------

def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # Remove numbers
    text = re.sub(r'\d', '', text)

    # Remove non-ASCII characters and emojis
    text = ''.join(
        char for char in text if char.isascii()
    )

    # Remove stopwords
    words = text.split()

    cleaned_words = [
        word for word in words
        if word not in stop_words
    ]

    return ' '.join(cleaned_words)

# --------------------------------
# 5. Header
# --------------------------------

st.markdown(
    '<div class="main-title">💭 Emotion Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">NLP Based Emotion Classification using Logistic Regression</div>',
    unsafe_allow_html=True
)

st.write("")
st.write("")

st.info(
    "Enter a sentence or express your feelings. "
    "Our Machine Learning model will identify the emotion."
)

# --------------------------------
# 6. Emotion Information
# --------------------------------

emotion_icons = {
    "sadness": "😢",
    "anger": "😠",
    "love": "❤️",
    "surprise": "😲",
    "fear": "😨",
    "joy": "😊"
}

# --------------------------------
# 7. User Input
# --------------------------------

st.subheader("✍️ Enter Your Text")

user_text = st.text_area(
    "Write your sentence below:",
    placeholder="Example: I am feeling very happy today...",
    height=150
)

# Example sentences
st.write("**Try an example:**")

examples = [
    "I am feeling very happy today",
    "I feel sad and lonely",
    "I am extremely angry",
    "I love spending time with my family",
    "I am scared about tomorrow",
    "Wow this is amazing"
]

selected_example = st.selectbox(
    "Choose an example sentence",
    ["Select an example"] + examples
)

if selected_example != "Select an example":
    user_text = selected_example

# --------------------------------
# 8. Prediction
# --------------------------------

if st.button("🔍 Analyze Emotion"):

    if not user_text.strip():

        st.warning("Please enter some text first.")

    else:

        # Preprocess input
        cleaned_text = preprocess_text(user_text)

        if not cleaned_text.strip():
            st.warning("Please enter meaningful text.")
        else:

            # Convert text into TF-IDF features
            text_vector = tfidf.transform([cleaned_text])

            # Predict emotion
            prediction = model.predict(text_vector)[0]

            emotion = reverse_mapping[prediction]

            icon = emotion_icons.get(
                str(emotion).lower(), "💭"
            )

            # Prediction probabilities
            probabilities = model.predict_proba(
                text_vector
            )[0]

            st.write("")
            st.divider()

            # Main result
            st.subheader("🎯 Prediction Result")

            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-emotion">
                        {icon} {str(emotion).upper()}
                    </div>
                    <p>Detected Emotion</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Confidence
            confidence = max(probabilities) * 100

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(float(confidence / 100))

            # Probability chart
            st.write("")
            st.subheader("📊 Emotion Probability Distribution")

            emotion_names = [
                reverse_mapping[class_id]
                for class_id in model.classes_
            ]

            probability_df = pd.DataFrame({
                "Emotion": emotion_names,
                "Probability": probabilities * 100
            })

            probability_df = probability_df.sort_values(
                by="Probability",
                ascending=False
            )

            st.bar_chart(
                probability_df.set_index("Emotion")
            )

            st.write("**Detailed Probabilities**")

            st.dataframe(
                probability_df.style.format({
                    "Probability": "{:.2f}%"
                }),
                use_container_width=True,
                hide_index=True
            )

# --------------------------------
# 9. Footer
# --------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center;color:#9CA3AF;">
        Built using Python | NLP | TF-IDF |
        Logistic Regression | Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
