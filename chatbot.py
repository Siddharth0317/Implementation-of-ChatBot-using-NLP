import os
import json
import datetime
import csv
import random
import re
import nltk
import joblib
import streamlit as st
import openai
from dotenv import load_dotenv
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer, util

load_dotenv()
nltk.download('punkt', quiet=True)
stemmer = PorterStemmer()
openai.api_key = os.getenv("OPENAI_API_KEY")

BASE_DIR = os.path.dirname(__file__)
INTENT_PATH = os.path.join(BASE_DIR, 'intents.json')
VEC_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')
CLF_PATH = os.path.join(BASE_DIR, 'classifier.pkl')
LOG_PATH = os.path.join(BASE_DIR, 'chat_log.csv')
FULL_LOG_PATH = os.path.join(BASE_DIR, 'full_log.json')

with open(INTENT_PATH, 'r') as f:
    intents = json.load(f)

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)

def train_model():
    tags, patterns = [], []
    for intent in intents["intents"]:
        for pattern in intent['patterns']:
            tags.append(intent['tag'])
            patterns.append(preprocess(pattern))

    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(patterns)
    y = tags

    clf = LogisticRegression(max_iter=10000)
    clf.fit(x, y)

    joblib.dump(vectorizer, VEC_PATH)
    joblib.dump(clf, CLF_PATH)
    return vectorizer, clf

if os.path.exists(VEC_PATH) and os.path.exists(CLF_PATH):
    vectorizer = joblib.load(VEC_PATH)
    clf = joblib.load(CLF_PATH)
else:
    vectorizer, clf = train_model()

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
pattern_map = []
for intent in intents["intents"]:
    for pattern in intent['patterns']:
        pattern_map.append((pattern, intent['tag']))
pattern_texts = [p[0] for p in pattern_map]
pattern_embeddings = sbert_model.encode(pattern_texts, convert_to_tensor=True)

def get_intent_tag(user_input):
    input_processed = preprocess(user_input)
    input_vector = vectorizer.transform([input_processed])
    probs = clf.predict_proba(input_vector)[0]
    max_prob = max(probs)
    tag = clf.predict(input_vector)[0] if max_prob >= 0.4 else None
    return tag, max_prob

def fallback_with_gpt(text):
    if not openai.api_key:
        return "Sorry, I couldn't understand that and no backup model is available."
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": text}]
    )
    return response.choices[0].message.content

def chatbot_response(user_input):
    tag, confidence = get_intent_tag(user_input)
    if tag:
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"]), tag, confidence
        # If tag is found but not in intents, fallback
        return fallback_with_gpt(user_input), "fallback", confidence
    else:
        return fallback_with_gpt(user_input), "fallback", confidence

st.title("🤖 Advanced NLP Chatbot")
menu = ["Home", "History", "About"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.markdown("### Start chatting below")

    if st.button("Tell me a joke"): user_input = "Tell me a joke"
    elif st.button("Help me"): user_input = "I need help"
    else: user_input = st.text_input("You:")

    if user_input:
        response, tag, confidence = chatbot_response(user_input)
        st.markdown(f"**Bot:** {response}")
        st.markdown(f"*Predicted intent: `{tag}` | Confidence: `{round(confidence*100, 2) if tag != 'fallback' else 'N/A'}%`*")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_PATH, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([user_input, response, timestamp])
        with open(FULL_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"user": user_input, "bot": response, "tag": tag, "time": timestamp}) + "\n")

elif choice == "History":
    st.header("📜 Conversation History")
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 3:
                    st.text(f"User: {row[0]}\nBot: {row[1]}\nTime: {row[2]}")
                    st.markdown("---")
    else:
        st.info("No history found yet.")

elif choice == "About":
    st.markdown("""
    ###  About This Chatbot
    This is an advanced chatbot using:
    - Intent classification (TF-IDF + Logistic Regression)
    - Semantic matching (BERT)
    - GPT fallback for unknown queries
    - Streamlit UI
    - CSV + JSON logging
    - Reusable pre-trained models
    """)
