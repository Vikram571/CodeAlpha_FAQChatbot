import streamlit as st
import pandas as pd
import nltk

# Download required NLTK files
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="FAQ Chatbot")

st.title("🤖 FAQ Chatbot")

try:
    data = pd.read_csv("faq.csv")

    questions = data['Question']
    answers = data['Answer']

    def preprocess(text):
        text = text.lower()

        words = word_tokenize(str(text))

        stop_words = set(stopwords.words('english'))

        filtered = [
            word for word in words
            if word.isalnum() and word not in stop_words
        ]

        return " ".join(filtered)

    processed_questions = questions.apply(preprocess)

    vectorizer = TfidfVectorizer()

    question_vectors = vectorizer.fit_transform(
        processed_questions
    )

    user_question = st.text_input(
        "Ask your question"
    )

    if st.button("Send"):

        if user_question.strip():

            processed_input = preprocess(
                user_question
            )

            user_vector = vectorizer.transform(
                [processed_input]
            )

            similarity = cosine_similarity(
                user_vector,
                question_vectors
            )

            best_match = similarity.argmax()

            response = answers.iloc[
                best_match
            ]

            st.success(response)

        else:
            st.warning(
                "Please enter a question"
            )

except Exception as e:
    st.error(f"Error: {e}")