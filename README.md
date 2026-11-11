# 💭 Emotion Detection using NLP and Logistic Regression

## Project Overview

This project is a Machine Learning-based Emotion Detection System that identifies the emotion expressed in a given text or sentence.

It uses Natural Language Processing (NLP) techniques and a Logistic Regression classifier to classify text into six different emotions.

The application provides an interactive web interface developed using Streamlit, where users can enter a sentence and instantly view the predicted emotion.

## 🎯 Objectives

- Detect emotions from text using Machine Learning.
- Apply NLP techniques for text preprocessing.
- Convert text into numerical features using TF-IDF.
- Train a Logistic Regression classification model.
- Build an interactive web application using Streamlit.

## 😊 Emotions Detected

The model classifies text into six categories:

1. Sadness 😢
2. Anger 😠
3. Love ❤️
4. Surprise 😲
5. Fear 😨
6. Joy 😊

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Streamlit
- Joblib
- Matplotlib and Seaborn (for exploratory data analysis)

## ⚙️ Project Workflow

1. Dataset loading and exploration.
2. Checking missing values.
3. Emotion label encoding.
4. Text preprocessing:
   - Lowercase conversion
   - Punctuation removal
   - Number removal
   - Emoji and non-ASCII character removal
   - Stopword removal
5. Splitting data into training and testing sets.
6. Feature extraction using TF-IDF.
7. Training the Logistic Regression model.
8. Evaluating model performance.
9. Saving the trained model and vectorizer using Joblib.
10. Building the Streamlit user interface.

## 📊 Model Performance

| Model | Accuracy |
|---|---:|
| Multinomial Naive Bayes (Bag of Words) | 76.81% |
| Multinomial Naive Bayes (TF-IDF) | 66.09% |
| Logistic Regression (TF-IDF) | **86.28%** |

The Logistic Regression model achieved the highest accuracy among the three approaches tested.

## 📁 Project Structure

```text
Emotion-Detection/
│
├── app.py
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── emotion_mapping.pkl
├── stop_words.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder

```bash
cd Emotion-Detection-Using-Logistic-Regression
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

Windows:

```powershell
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser at:

`http://localhost:8501`

## 💻 Application Features

- Text input for emotion detection.
- Six-class emotion classification.
- Prediction confidence display.
- Emotion probability distribution chart.
- Interactive Streamlit interface.

## 📌 Future Enhancements

- Add voice-based emotion detection.
- Support multiple language inputs.
- Add CSV file upload for bulk emotion detection.
- Improve model accuracy using advanced NLP techniques.
- Deploy the application online.

## 👩‍💻 Developed Using

Python | NLP | Machine Learning | Logistic Regression | TF-IDF | Streamlit

**Note:** The reported accuracy is based on the project's test dataset and does not guarantee the same performance on all real-world text.