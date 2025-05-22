import streamlit as st
import pandas as pd
import openai
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import numpy as np

# Set the page title
st.set_page_config(page_title="omniSense Assistant", page_icon="💬")
st.title("💬 omniSense Chat")

# --- API Key Input ---
user_api_key = st.text_input("🔑 Enter your OpenAI API Key:", type="password")
if not user_api_key:
    st.warning("⚠️ Please enter your OpenAI API key to continue.")
    st.stop()
openai.api_key = user_api_key

# --- Load Data ---
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# --- Chunk data ---
def chunk_dataframe(df, chunk_size=1000):
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        text = "\n".join([", ".join(map(str, row)) for row in chunk.values])
        chunks.append(text)
    return chunks

# --- Embed chunks with OpenAI ---
def embed_texts(texts):
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=texts
    )
    return [e.embedding for e in response.data]

# --- Store embeddings with FAISS ---
def build_vector_store(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index

# --- Search similar chunk ---
def find_most_relevant_chunk(question, chunk_texts, index, embeddings):
    q_embed = embed_texts([question])[0]
    D, I = index.search(np.array([q_embed]).astype("float32"), k=1)
    return chunk_texts[I[0][0]]

# --- Classify question type ---
def classify_question_type(question):
    prompt = f"""
You are a smart assistant. Classify this as 'Quantitative' or 'Qualitative':
"{question}"
Answer only one word.
"""
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# --- Generate Python expression ---
def ask_gpt_for_python_expression(question, table_structure):
    prompt = f"""
You are a data analyst. Given the table:

{table_structure}

Write a pandas expression to answer:
{question}

Only return the expression (e.g., df['Amount'].sum())
"""
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# --- Smart Response Polishing ---
def ask_SmartResponse(user_question, result):
    prompt = f"""
User Question: "{user_question}"
Answer: {result}

Respond naturally as a helpful data assistant. Use full English sentences and bullet points where needed.
"""
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# --- Qualitative Answer with chunk context ---
def ask_openai_with_context(question, chunk):
    prompt = f"""
Data snippet:
{chunk}

Based on this, answer:
{question}
"""
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# --- Session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Upload CSV ---
uploaded_file = st.file_uploader("📎 Upload your CSV data", type="csv")
if uploaded_file:
    df = load_data(uploaded_file)

    # Define table structure (optional, used for code generation)
    table_structure = "\n".join([f"- {col} ({str(df[col].dtype)})" for col in df.columns])

    # Chunk and embed once
    if "chunk_texts" not in st.session_state:
        with st.spinner("🔍 Processing data..."):
            st.session_state.chunk_texts = chunk_dataframe(df)
            st.session_state.embeddings = embed_texts(st.session_state.chunk_texts)
            st.session_state.vector_index = build_vector_store(st.session_state.embeddings)

    # Show chat history
    for entry in st.session_state.chat_history:
        st.markdown(f"**You:** {entry['question']}")
        st.markdown(f"**omniSense:** {entry['answer']}")

    # --- Chat ---
    user_question = st.chat_input("Ask anything...")
    if user_question:
        st.write("You:", user_question)
        with st.spinner("Thinking..."):
            time.sleep(1)
            try:
                q_type = classify_question_type(user_question)
            except Exception as e:
                st.error(f"❌ Error classifying: {e}")
                st.stop()

            if q_type.lower() == "quantitative":
                try:
                    expr = ask_gpt_for_python_expression(user_question, table_structure)
                    result = eval(expr, {"df": df, "pd": pd})
                    response = ask_SmartResponse(user_question, result)
                except Exception as e:
                    response = f"❌ Error evaluating: {e}"
            else:
                try:
                    best_chunk = find_most_relevant_chunk(
                        user_question,
                        st.session_state.chunk_texts,
                        st.session_state.vector_index,
                        st.session_state.embeddings
                    )
                    answer = ask_openai_with_context(user_question, best_chunk)
                    response = ask_SmartResponse(user_question, answer)
                except Exception as e:
                    response = f"❌ Error generating response: {e}"

        st.session_state.chat_history.append({
            "question": user_question,
            "answer": response
        })
        st.rerun()
else:
    st.info("📥 Upload a CSV file to start chatting with your data.")
