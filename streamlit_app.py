# streamlit_app.py
import streamlit as st
from app.retriever import query_index, load_index
from app.llm import generate_answer, get_generator
from datetime import datetime
import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    return conn

conn = init_db()

def save_message(role, content):
    ts = datetime.utcnow().isoformat()
    cur = conn.cursor()
    cur.execute("INSERT INTO chats (role, content, timestamp) VALUES (?, ?, ?)", (role, content, ts))
    conn.commit()

def get_history(limit=50):
    cur = conn.cursor()
    cur.execute("SELECT role, content, timestamp FROM chats ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    return list(reversed(rows))

def st_session_state_safe(text):
    return str(text).replace("<", "&lt;").replace(">", "&gt;")

st.set_page_config(page_title="Construction Chat", page_icon="🏗️", layout="wide")

st.markdown(
    """
    <style>
    .user-bubble {
        background: #0ea5e9;
        color: #fff;
        padding: 10px 15px;
        border-radius: 12px;
        margin: 6px 0;
        display: inline-block;
        text-align: right;
        max-width: 70%;
    }
    .bot-bubble {
        background: #1e293b;
        color: #f1f5f9;
        padding: 10px 15px;
        border-radius: 12px;
        margin: 6px 0;
        display: inline-block;
        text-align: left;
        max-width: 70%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([3,1])

with col1:
    st.markdown("## 🏗 Construction Chat")
    st.markdown("Ask anything construction-related — materials, methods, codes, estimations, safety, or design guidance.")

    if "messages" not in st.session_state:
        st.session_state.messages = get_history(limit=200) or [
            ("system","Hello! I am a construction specialist. Ask me anything about construction.")
        ]

    with st.container():
        for role, content, *_ in st.session_state.messages:
            if role == "user":
                st.markdown(
                    f"<div style='text-align:right'><div class='user-bubble'>{st_session_state_safe(content)}</div></div>",
                    unsafe_allow_html=True
                )
            elif role == "bot":
                st.markdown(
                    f"<div style='text-align:left'><div class='bot-bubble'>{content}</div></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(f"<div>{content}</div>", unsafe_allow_html=True)

    user_input = st.text_area("Your question", key="input", height=120, placeholder="E.g., 'How do I calculate concrete volume for a footing?'")
    submit = st.button("Send")

with col2:
    st.markdown("### ⚙️ Settings")
    gen_model = st.selectbox("Generator model", ["distilgpt2", "gpt2"], index=0)
    top_k = st.slider("Retriever top_k", 1, 5, 3)
    temp = st.slider("Temperature", 0.0, 1.0, 0.2)
    max_tokens = st.slider("Max tokens", 50, 500, 200)

    st.markdown("---")
    st.markdown("### 📦 Index status")
    idx, meta = load_index()
    if idx is None:
        st.warning("No vector index found. Run `python scripts/build_index.py` to index docs in data/docs/")
    else:
        st.success(f"Index loaded — {len(meta)} chunks")

    if st.button("🗑 Clear chat history"):
        cur = conn.cursor()
        cur.execute("DELETE FROM chats")
        conn.commit()
        st.session_state.messages = []
        st.experimental_rerun()

if submit and user_input.strip():
    save_message("user", user_input)
    st.session_state.messages.append(("user", user_input, datetime.utcnow().isoformat()))

    retrieved = query_index(user_input, top_k=top_k)
    context = "\n\n".join([f"Source: {r.get('source','unknown')}\n{r.get('text_snippet','')}" for r in retrieved])

    system_preamble = (
        "You are an expert construction assistant. Use ONLY the provided context from construction documents. "
        "If the context does not contain the answer, reply concisely and state you couldn't find a definitive answer. "
        "Always mention sources when available.\n\n"
    )
    full_prompt = f"{system_preamble}CONTEXT:\n{context}\n\nQUESTION: {user_input}\n\nAnswer:"

    try:
        get_generator(gen_model)
        answer = generate_answer(full_prompt, max_new_tokens=max_tokens, temperature=temp)
    except Exception as e:
        answer = "❗ Model generation failed: " + str(e)

    save_message("bot", answer)
    st.session_state.messages.append(("bot", answer, datetime.utcnow().isoformat()))
    st.experimental_rerun()
