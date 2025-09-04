from datetime import datetime
import streamlit as st
import base64

st.set_page_config(page_title="Elevvo NLP Internship — Project Showcase", page_icon="🧠", layout="wide")

# ---------- Theme & Small CSS tweaks ----------
def inject_css():
    with open("nav_logo-5vkEKoZL.svg", "rb") as f:
        svg_bytes = f.read()
        b64_svg = base64.b64encode(svg_bytes).decode()

    st.markdown(
        f"""
        <style>
            html, body, [class*="css"] {{
                font-family: 'Segoe UI', sans-serif;
                background-color: #0f172a;
                color: #f1f5f9;
                font-size: 20px;
            }}
            .app-title-wrapper {{
                display: flex;
                align-items: center;
                justify-content: flex-start;
                margin-bottom: 2rem;
            }}
            .app-logo {{
                height: 160px;
                max-width: none;
                filter: brightness(0) invert(1);
            }}
            .app-subtitle {{
                color: #94a3b8;
                font-size: 1.5rem;
                margin-bottom: 2.5rem;
            }}
            .card {{
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 32px;
                padding: 40px;
                margin-bottom: 40px;
                transition: all 0.3s ease;
                font-size: 1.3rem;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                height: 100%;
            }}
            .card:hover {{
                box-shadow: 0 12px 40px rgba(255, 255, 255, 0.07);
                transform: translateY(-8px) scale(1.02);
            }}
            .card-title {{
                background: linear-gradient(135deg, #1e3a8a, #2563eb);
                padding: 24px 36px;
                border-radius: 20px;
                color: #fff;
                font-weight: 900;
                font-size: 1.8rem;
                margin-bottom: 20px;
                box-shadow: inset 0 0 0 1px rgba(255,255,255,0.05);
                transition: transform 0.3s ease;
            }}
            .card-title:hover {{
                transform: scale(1.03);
            }}
            .pill {{
                display: inline-block;
                padding: 10px 20px;
                font-size: 16px;
                font-weight: 700;
                border-radius: 24px;
                margin: 8px 10px 12px 0;
                border: 1px solid #334155;
                background: #1e293b;
                color: #e2e8f0;
                transition: transform 0.3s ease;
            }}
            .pill:hover {{
                transform: scale(1.1);
            }}
            .pill-blue {{
                background-color: #1d4ed8;
                border-color: #1e40af;
                color: #ffffff;
            }}
            .pill-violet {{
                background-color: #7c3aed;
                border-color: #6d28d9;
                color: #ffffff;
            }}
            .pill-rose {{
                background-color: #e11d48;
                border-color: #be123c;
                color: #ffffff;
            }}
            .link-button {{
                display: inline-block;
                font-size: 18px;
                font-weight: 800;
                padding: 16px 28px;
                border-radius: 14px;
                text-align: center;
                text-decoration: none;
                margin: 14px 16px 0 0;
                border: 2px solid #f59e0b;
                background: #fbbf24;
                color: #000000 !important;
                transition: all 0.25s ease, transform 0.3s ease;
                text-decoration: none;
            }}
            .link-button:hover {{
                background: #f59e0b;
                border-color: #d97706;
                color: #000000 !important;
                box-shadow: 0 0 0 5px rgba(251, 191, 36, 0.4);
                transform: scale(1.05);
                text-decoration: none;
            }}
            .link-button:disabled {{
                opacity: 0.5;
                pointer-events: none;
            }}
            .link-button-wrapper {{
                margin-top: auto;
            }}

            /* Streamlit-native buttons (e.g. view mode) */
            .stButton>button {{
                transition: transform 0.3s ease;
            }}
            .stButton>button:hover {{
                transform: scale(1.05);
            }}
        </style>
        <div class="app-title-wrapper">
            <img class="app-logo" src="data:image/svg+xml;base64,{b64_svg}" alt="Logo">
        </div>
        """,
        unsafe_allow_html=True
    )


inject_css()

# ---------- Project Data ----------
PROJECTS = [
    {
        "id": "sentiment-analysis",
        "title": "💬 IMDb Sentiment Analyzer",
        "description": "A Streamlit web app for sentiment analysis on IMDb reviews using Logistic Regression and Naive Bayes models.",
        "topics": ["TF‑IDF", "LogReg", "Keras", "scikit-learn", "naive-bayes"],
        "live": "https://sentimentapp77.streamlit.app/",
        "repo": "https://github.com/mohamedtaha77/sentiment_app?tab=readme-ov-file",
        "notebook": "https://github.com/mohamedtaha77/sentiment_app/blob/main/Elevvo_NLP_Internship_Task1.ipynb",
    },
    {
        "id": "news-classification",
        "title": "🗞️ AG News Category Classifier",
        "description": "A Streamlit web app for classifying news articles using Logistic Regression and Neural Network models trained on the AG News dataset.",
        "topics": ["AG News", "TF‑IDF", "Neural", "logistic-regression"],
        "live": "https://news-category-classification-77.streamlit.app/",
        "repo": "https://github.com/mohamedtaha77/News-Category-Classification-App",
        "notebook": "https://github.com/mohamedtaha77/News-Category-Classification-App/blob/main/Elevvo_NLP_Internship_Task2.ipynb",
    },
    {
        "id": "fake-news",
        "title": "📰 Fake News Detection App",
        "description": "A Streamlit-based Fake News Detection app using Logistic Regression, SVM, and LSTM models. Built on a custom Kaggle dataset with full preprocessing, visualization, and evaluation pipelines.",
        "topics": ["DistilBERT", "Transformers", "PyTorch", "HuggingFace"],
        "live": "https://fake-news-detection-app-77.streamlit.app/",
        "repo": "https://github.com/mohamedtaha77/Fake-News-Detection-App",
        "notebook": "https://github.com/mohamedtaha77/Fake-News-Detection-App/blob/main/Elevvo_NLP_Internship_Task3.ipynb",
    },
    {
        "id": "ner",
        "title": "Named Entity Recognition (NER)",
        "description": "A Streamlit-based Named Entity Recognition (NER) app for news text using rule-based and spaCy-based pipelines. Built and evaluated on the CoNLL-2003 dataset with displaCy visualization.",
        "topics": ["NER", "entity-extraction", "spaCy", "Transformers"],
        "live": "https://named-entity-recognition-app-77.streamlit.app/",
        "repo": "https://github.com/mohamedtaha77/Named-Entity-Recognition-App",
        "notebook": "https://github.com/mohamedtaha77/Named-Entity-Recognition-App/blob/main/Elevvo_NLP_Internship_Task4.ipynb",
    },
    {
        "id": "topic-modeling",
        "title": "🧠 Topic Modeling on News Articles",
        "description": "An interactive topic modeling app built with Streamlit. It discovers hidden themes in news articles",
        "topics": ["LDA", "scikit-learn", "NMF", "Genism", "pydavis"],
        "live": "https://topic-modeling-app-77.streamlit.app/",
        "repo": "https://github.com/mohamedtaha77/Topic-Modeling-App",
        "notebook": "https://github.com/mohamedtaha77/Topic-Modeling-App/blob/main/Elevvo_NLP_Internship_Task5.ipynb",
    },
    {
        "id": "qa",
        "title": "🧠 QA with Transformers",
        "description": "interactive Question Answering (QA) App built with Streamlit. It extracts answers from a given paragraph using Transformer-based models.",
        "topics": ["QA", "BERT", "DistilBERT", "SQuAD‑style"],
        "live": "https://question-answering-with-transformers-77.streamlit.app/",
        "repo": "https://github.com/mohamedtaha77/Question-Answering-with-Transformers-App",
        "notebook": "https://github.com/mohamedtaha77/Question-Answering-with-Transformers-App/tree/master/notebooks",
    },
]

# ---------- Utility Functions ----------
def topic_pill(topic: str) -> str:
    t = topic.lower()
    if any(k in t for k in ["bert", "transformer", "huggingface", "qa"]):
        return f'<span class="pill pill-blue">{topic}</span>'
    if any(k in t for k in ["lda", "bertopic", "unsupervised", "hdbscan", "umap"]):
        return f'<span class="pill pill-violet">{topic}</span>'
    if any(k in t for k in ["tf", "logreg", "classical"]):
        return f'<span class="pill pill-rose">{topic}</span>'
    return f'<span class="pill">{topic}</span>'

def link_button(url: str, label: str):
    if not url or url.startswith("<"):
        st.markdown(f'<a class="link-button" disabled>{label}</a>', unsafe_allow_html=True)
    else:
        st.markdown(f'<a class="link-button" href="{url}" target="_blank">{label}</a>', unsafe_allow_html=True)

# ---------- Header ----------
st.markdown(
    '<div class="app-subtitle">Natural Language Processing Internship</div>',
    unsafe_allow_html=True,
)

# ---------- View Mode Toggle ----------
st.markdown("#### View Mode")
if "view_mode" not in st.session_state:
    st.session_state["view_mode"] = "Grid"

col1, col2 = st.columns(2)
with col1:
    if st.button("🔳 Grid", use_container_width=True):
        st.session_state["view_mode"] = "Grid"
with col2:
    if st.button("📅 Timeline", use_container_width=True):
        st.session_state["view_mode"] = "Timeline"

view = st.session_state["view_mode"]

# ---------- Card Renderer ----------
def render_card(project: dict):
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="card-title">{project["title"]}</div>', unsafe_allow_html=True)
        st.markdown(project["description"])
        pills = " ".join([topic_pill(t) for t in project["topics"]])
        st.markdown(pills, unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="link-button-wrapper">
                <a class="link-button" href="{project['live']}" target="_blank">🚀 Live App</a>
                <a class="link-button" href="{project['repo']}" target="_blank">
                    <img src="https://github.githubassets.com/favicons/favicon-dark.png" width="18" style="vertical-align:middle; margin-right:6px;">
                    GitHub Repo
                </a>
                <a class="link-button" href="{project['notebook']}" target="_blank">📓 Notebook</a>
            </div>
            """, unsafe_allow_html=True
        )



        st.markdown("</div>", unsafe_allow_html=True)

# ---------- Display ----------
visible = PROJECTS
if view == "Grid":
    rows, row = [], []
    for i, p in enumerate(visible):
        row.append(p)
        if len(row) == 2:  # 2 cards per row instead of 3
            rows.append(row)
            row = []
    if row: rows.append(row)
    for row in rows:
        cols = st.columns(len(row))
        for col, proj in zip(cols, row):
            with col:
                render_card(proj)
    if not visible:
        st.info("No projects found.")
else:  # Timeline
    for p in visible:
        render_card(p)

# ---------- Footer ----------
st.markdown("---")
st.caption(f"© {datetime.now().year} Elevvo NLP Internship - Mohammed Taha. Built with ❤️.")
