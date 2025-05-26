import streamlit as st
import pandas as pd
import openai
import time

#---Set the page title---
st.set_page_config(page_title="omniSense Assistant", page_icon="💬")
st.title("💬 omniSense ChatBot")


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

A quantitative question asks for total, numbers, counts, averages, percentages, sum, group by, unique list of categories, unique list of unit, categories list, unit list, top, max, min, from date, to date, year, month, month names, date etc and all query type questions.
A qualitative question asks for reasons, descriptions, categories, sales, amount, unit, month or opinions.

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
- System Role: "You are an AI assistant that converts natural language questions into SQL queries for a financial transactions database. The database has a single table with the following schema:" 
- Schema Definition: 
- unit (TEXT): Represents the location or outlet. Examples: 'Austin', 'Seattle'. 
- category (TEXT): Describes the type of transaction. Examples: 'sales', 'expenses'. 
- date (DATE): The date of the transaction. 
- amount (DECIMAL): The monetary value. 
- Instructions: 
- "When a user asks for a location or unit or city or site, map it to the unit column." examples: Basking Ridge, Cary, Irving, etc. 
- "When a user asks for a type of activity, category, expense, income, map it to the category column." examples: Food cost, sales, catering sales, etc. 
- "For month and year requests, use MONTH(date) and YEAR(date) functions." 
- "Use SUM(amount), Average(amount), min(amount), max(amount), etc. for calculations unless specified otherwise." 
- "Ensure all string comparisons are exact (e.g., unit = 'Austin')." 
- "Only generate a SQL query. Do not add any explanatory text." 
- User Question Placeholder: "User Question: {user_question}" 
- Expected Output: "SQL Query:" 
Putting it Together (Example Prompt Structure): 
You are an AI assistant that converts natural language questions into SQL queries for a financial transactions database. The database has a single table. 
Examples of user_question that I will give you, and the sql query output that you will return: 
User Question: What were the total sales for Austin in April 2025? 
SQL Query: SELECT SUM(amount) FROM your_table WHERE unit = 'Austin' AND category = 'sales' AND MONTH(date) = 4 AND YEAR(date) = 2025; 
User Question: Show me the expenses for Seattle last month. 
SQL Query: SELECT SUM(amount) FROM your_table WHERE unit = 'Seattle' AND category = 'expenses' AND date >= DATEADD(month, DATEDIFF(month, 0, GETDATE()) - 1, 0) AND date < DATEADD(month, DATEDIFF(month, 0, GETDATE()), 0); 
Few Table values entries: 
Columns: Unit, category, date, amount 
Values: 
Basking ridge, food cost, 04/28/2025, 35654.23 
Irving, sales, 02/27/2024, 4399870 
Cary, food cost, 01/28/2025, 2333.45 
Basking ridge, catering sales, 11/28/2024, 2333.44 
Irving, insurance, 09/28/2024, 88943
With this info, the user_question is "{question}", generate the sql query for the same.
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
    The core answer or result is: {result}

    Please respond in a natural, helpful, and intelligent tone, like a helpful data assistant.
    Focus on directly answering the user's question based on the provided result.
    Use complete English sentences.
    If the result is a long list, you can summarize it or mention a few key items naturally.
    If the result is a numerical value, clearly state what it represents.
    If the result is an error message, gracefully explain that the operation could not be completed and suggest a rephrase.

    Examples for numerical results:
    - User: "What is the total amount?" Result: "1234.56"
      Response: "The total amount across all records is $1,234.56."
    - User: "What was the travel expense in March?" Result: "150.00"
      Response: "The travel expense in March was $150.00."

    Examples for list results:
    - User: "List unique categories?" Result: "['Food', 'Rent', 'Utilities']"
      Response: "The unique categories found in the dataset are Food, Rent, and Utilities."
    - User: "Show me the units with amounts over 500." Result: "['Unit_A', 'Unit_C']"
      Response: "Units with amounts over $500 include Unit_A and Unit_C."

    Examples for error results:
    - User: "Calculate X divided by zero." Result: "An error occurred: Division by zero"
      Response: "I encountered an issue while processing that request: Division by zero. Please try rephrasing your question."
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
    # 🔧 Ensure Date column is datetime
     if 'Month' in df.columns:
         df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    context = format_data_context(df)

    table_structure = """
Table Name: VerizonData

Columns:
- Unit (text)
- Category (text)
- Month (text or date)
- Year (text)
- TimeFrame (text)
- Amount (numeric)
"""

    # Show previous chat history
    for entry in st.session_state.chat_history:
        st.markdown(f"**You:** {entry['question']}")
        st.markdown(f"**omniSense:** {entry['answer']}")

    # Chat input
    #with st.form("chat_form", clear_on_submit=True):
        #user_question = st.chat_input("Ask anything") #st.text_input("Ask a question about your data:", key="user_input")
        #submitted = st.form_submit_button("Submit")
   
    #if submitted and user_question.strip():
    # Chat input (at bottom)
    user_question = st.chat_input("Ask anything...")
    if user_question:
      st.write("You:", user_question)
      with st.spinner("..."):
        time.sleep(2)  # Simulate a delay
        
        try:
            question_type = classify_question_type(user_question)
        except Exception as e:
            st.error(f"❌ Error classifying question: {e}")
            st.stop()

        if question_type.lower() == "quantitative":
            try:
                python_expr = ask_gpt_for_python_expression(user_question, table_structure)
                result = eval(python_expr, {"df": df, "pd": pd})
                #response = str(python_expr)
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
