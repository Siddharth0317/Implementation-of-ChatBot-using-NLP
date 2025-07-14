# 🤖 AI Chatbot with NLP, BERT, GPT-3.5 Fallback & Streamlit UI

An advanced conversational chatbot combining traditional machine learning (TF-IDF + Logistic Regression), semantic understanding via BERT (`sentence-transformers`), and fallback to OpenAI GPT-3.5 for unmatched queries. Features a modern, interactive Streamlit web interface.

---

## 🚀 Features

- ✅ Intent classification using Scikit-learn
- ✅ Semantic matching with BERT (MiniLM model)
- ✅ GPT fallback using OpenAI API
- ✅ Pre-trained model caching via joblib
- ✅ Streamlit UI with quick action buttons
- ✅ Secure `.env` support using `python-dotenv`
- ✅ Chat history logging (CSV and JSON)
- ✅ Ready to deploy on Streamlit Cloud or Render

---

## 📁 Folder Structure

```
chatbot_project/
├── chatbot_advanced.py
├── intents.json
├── vectorizer.pkl
├── classifier.pkl
├── chat_log.csv
├── full_log.json
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repo & create virtual environment

```bash
git clone https://github.com/your-username/ai-chatbot-streamlit.git
cd ai-chatbot-streamlit
python -m venv chatbot_env
source chatbot_env/bin/activate  # or chatbot_env\Scripts\activate on Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your environment

Rename `.env.example` to `.env` and add your OpenAI API key:

```bash
cp .env.example .env
```

---

## ▶️ Run Locally

```bash
streamlit run chatbot_advanced.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push your project to GitHub
2. Visit [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Create a new app:
   - Repo: `your-username/ai-chatbot-streamlit`
   - File: `chatbot_advanced.py`
4. Under **Secrets**, add:

```ini
OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

5. Click **Deploy**

---

## 🔐 .env Example

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> Keep your `.env` private and don’t push it to GitHub.

---

## 🏷️ GitHub Topics

Add these to your repository:

```
python, chatbot, nlp, machine-learning, ai, streamlit, openai, gpt, transformers, sklearn, bert, sentence-transformers, intelligent-agent
```

---

## 🙌 Credits

Built using:

- 🧠 `nltk`, `scikit-learn`, `sentence-transformers`
- 🤖 `openai` API (GPT-3.5 Turbo)
- 🎨 `streamlit` for frontend
- 🔐 `dotenv` for secure environment config

---

Made with 💡 by [Siddharth Rahane]
