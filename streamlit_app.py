import streamlit as st
import pandas as pd
import openai

#---Set the page title---
st.set_page_config(page_title="omniSense Assistant", page_icon="💬")
st.title("💬 omniSense Chat")


# --- API Key Input ---
user_api_key = st.text_input("🔑 Enter your OpenAI API Key:", type="password")

if not user_api_key:
    st.warning("⚠️ Please enter your OpenAI API key to continue.")
    st.stop()

# Set the API key
openai.api_key = user_api_key

# --- Load CSV ---
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# --- Format Data Context ---
def format_data_context(df):
    context = ""
    sample = df.head(5).fillna("N/A")
    for _, row in sample.iterrows():
        context += "\n" + "\n".join([f"{col}: {row[col]}" for col in df.columns]) + "\n"
    return context

# --- Classify Question ---
def classify_question_type(question):
    prompt = f"""
You are a smart assistant that classifies questions as either 'Quantitative' or 'Qualitative'.

A quantitative question asks for total numbers, counts, averages, percentages, year, month etc.
A qualitative question asks for reasons, descriptions, categories, sales, amount, unit, or opinions.

Question: "{question}"
Answer with only one word: Quantitative or Qualitative.
"""
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# --- Generate Python Expression ---
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

# --- Qualitative Answer Generator ---
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

# --- Smart response Generator ---
def ask_SmartResponse(user_question, result):
    polish_prompt = f"""
        The user asked: "{user_question}"
        The answer is: {result}

        Please respond in a natural and intelligent tone, like a data assistant, using complete English sentences.
        """

    polished_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": polish_prompt}]
        )

    return polished_response.choices[0].message.content.strip()
    

# --- Session state for chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Upload CSV ---
uploaded_file = st.file_uploader("📎 Upload your CSV data", type="csv")
if uploaded_file:
    df = load_data(uploaded_file)
    context = format_data_context(df)

    table_structure = """
Table Name: VerizonData

Columns:
- Unit (text)
- Category (text)
- Month (text or date)
- Amount (numeric)
"""

    # Show previous chat history
    for entry in st.session_state.chat_history:
        st.markdown(f"**You:** {entry['question']}")
        st.markdown(f"**omniSense:** {entry['answer']}")

    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        user_question = st.text_input("Ask a question about your data:", key="user_input")
        submitted = st.form_submit_button("Submit")

    if submitted and user_question.strip():
        try:
            question_type = classify_question_type(user_question)
        except Exception as e:
            st.error(f"❌ Error classifying question: {e}")
            st.stop()

        if question_type.lower() == "quantitative":
            try:
                python_expr = ask_gpt_for_python_expression(user_question, table_structure)
                result = eval(python_expr, {"df": df, "pd": pd})
                #response = str(result)
                response=ask_SmartResponse(user_question,result)
            except Exception as e:
                response = f"❌ Error evaluating expression: {e}"
        else:
            try:
                response = ask_openai(user_question, context)
                response=ask_SmartResponse(user_question,response)
            except Exception as e:
                response = f"❌ Error generating response: {e}"

        # Store in chat history
        st.session_state.chat_history.append({
            "question": user_question,
            "answer": response
        })

        # Refresh UI
        st.rerun()
else:
    st.info("📥 Please upload a CSV file to begin.")
