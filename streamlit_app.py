import streamlit as st
import pandas as pd
import openai

# Replace with your actual OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- Load CSV ---
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# --- Prepare data context ---
def format_data_context(df):
    context = ""
    sample = df.head(5).fillna("N/A")
    for _, row in sample.iterrows():
        context += "\n" + "\n".join([f"{col}: {row[col]}" for col in df.columns]) + "\n"
    return context

# --- Classify question ---
def classify_question_type(question):
    prompt = f"""
You are a smart assistant that classifies questions as either 'Quantitative' or 'Qualitative'.

A quantitative question asks for numbers, counts, averages, percentages, etc.
A qualitative question asks for reasons, descriptions, categories, or opinions.

Question: "{question}"
Answer with only one word: Quantitative or Qualitative.
"""
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# --- Generate Python expression ---
def ask_gpt_for_python_expression(question, table_structure):
    prompt = f"""
You are a Python data analyst assistant. Based on the table structure below, and the DataFrame called `df`:

{table_structure}

Write a Python expression (no print, no comments) using pandas that answers this question:
{question}

Only return the expression (e.g., df['Amount'].sum()).
"""
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# --- Qualitative response ---
def ask_openai(question, context):
    prompt = f"""
You are a data analysis assistant. Here is the data context:

{context}

Now, based on this data, answer the following question:
{question}
"""
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# --- Streamlit Interface ---
st.set_page_config(page_title="omniSense Assistant", page_icon="💬")
st.title("💬 omniSense Chatbot")

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload CSV
uploaded_file = st.file_uploader("📎 Upload your CSV data", type="csv")
if uploaded_file:
    df = load_data(uploaded_file)
    #st.bar_chart(df, x="Category", y="Amount")
    context = format_data_context(df)
    table_structure = """
Table Name: VerizonData

Columns:
- Unit (text)
- Category (text)
- Month (text or date)
- Amount (numeric)
"""

    # Show chat history
    for entry in st.session_state.chat_history:
        st.markdown(f"**You:** {entry['question']}")
        st.markdown(f"**omniSense:** {entry['answer']}")

    # User input
    with st.form("chat_form", clear_on_submit=True):
        user_question = st.text_input("Ask a question about your data:", key="user_input") #st.text_area("Ask a question about your data:", height=100)
        submitted = st.form_submit_button("Submit")

    if submitted and user_question.strip():
        question_type = classify_question_type(user_question)

        if question_type.lower() == "quantitative":
            try:
                python_expr = ask_gpt_for_python_expression(user_question, table_structure)
                result = eval(python_expr, {"df": df, "pd": pd})
                response = str(result)
            except Exception as e:
                response = f"❌ Error evaluating expression: {e}"
        else:
            try:
                response = ask_openai(user_question, context)
            except Exception as e:
                response = f"❌ Error generating response: {e}"

        # Append to chat history
        st.session_state.chat_history.append({
            "question": user_question,
            "answer": response
        })

        # Rerun to show updated chat
        st.rerun()
else:
    st.info("📥 Please upload a CSV file to begin.")
