import streamlit as st
import pandas as pd
import openai
import time
import io # Added for df.info() debugging
import pyodbc
import os
import requests
# --- DATABASE CONFIG ---
server = 'bizlyzer.database.windows.net,1433;'  # e.g., 'localhost\\SQLEXPRESS' or '192.168.1.10'
database = 'BizlyzerBeta;'
username = 'BizlyzerDBA;'
password = 'B1zlyz3rDBA;'
API_KEY = os.getenv("OMNI_API_KEY")
secrets = st.secrets["database"]
#OPENAI_APIKEY=f"{secrets['keyvalue']}" #os.getenv("OPENAI_API_KEY")
# --- DATABASE CONNECTION FUNCTION ---
def get_connection():
    try:
        #st.write("Attempting to connect to database...")
        secrets = st.secrets["database"]

        # Corrected f-string for printing the driver
        st.write(f"Using driver: {secrets['driver']}")
        # Corrected f-string for the connection string
        # Each parameter needs to be separated by a semicolon within the string
        conn_str = (
            f"DRIVER={secrets['driver']};"
            f"SERVER={secrets['server']};"
            f"DATABASE={secrets['database']};"
            f"UID={secrets['username']};"
            f"PWD={secrets['password']};"
            "TrustServerCertificate=yes;" # This is specific to SQL Server
        )

        conn = pyodbc.connect(conn_str)
        st.success("Successfully connected to the database!")
        return conn
    except Exception as e:
        st.error(f"Error connecting to the database: {e}")
        st.info("Please check your database credentials in Streamlit Cloud secrets, "
                "database firewall rules, and ensure the correct ODBC driver "
                "is installed via packages.txt.")
        return None

# --- FETCH DATA BASED ON USER QUERY ---
# def run_query(user_query):
#     conn = get_connection()
#     if conn:
#         st.info("✅ Connected to database")
#         try:
#             df = pd.read_sql(user_query, conn)
#             #st.success("✅ Data fetched successfully!")
#             #st.dataframe(df)  # Show the data
#             return df
#         except Exception as e:
#             #st.error(f"❌ Query error: {e}")
#             return "Query error: {e}"
#         finally:
#             conn.close()
#     else:
#         #st.error("❌ Failed to connect to the database.")
#         return "Failed to connect to the database."

# --- FETCH DATA BASED ON USER QUERY API ---
def run_query(user_query):
   
    url = f"https://omniservicesapi.azurewebsites.net/api/v1/Data/GetData"
    
   # This will go into the POST body, not the URL
    payload = {
                "databaseIdentifier": "bizlyzer",
                "query": user_query  # Don't wrap in curly braces again
    }

                
    headers = {
                "accept": "text/plain",  # Use "application/json" if API returns JSON
                "X-API-KEY": "bdudu4@dkndf45d",
                "Content-Type": "application/json"
    }
   
    try:
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            try:
                data = response.json()  # If API returns JSON
                df = pd.DataFrame(data)                
                #st.success("✅ Data fetched successfully!")
                return df
            except ValueError:
                return response.text  # If response is plain text
        else:
            return f"API call failed with status code {response.status_code}: {response.text}"
    except Exception as e:
        return f"API request error: {e}"

# --- Set the page title ---
st.set_page_config(page_title="omniSense Assistant", page_icon="💬")
st.title("💬 omniSense ChatBot")

# --- API Key Input ---
user_api_key = st.text_input("🔑 Enter your OpenAI API Key:", type="password") #f"{secrets['keyvalue']}"

if not user_api_key:
    st.warning("⚠️ Please enter your OpenAI API key to continue.")
    st.stop()

# Set the API key
openai.api_key = user_api_key


# --- Load CSV ---
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# --- Format Data Context for Qualitative Questions ---
def format_data_context(df):
    context = ""
    # Take a smaller sample for context to avoid excessive token usage
    sample = df.head(5).fillna("N/A")
    context += "Here are the first 5 rows of your data:\n"
    context += sample.to_string() + "\n"
    context += f"The columns are: {', '.join(df.columns)}\n"
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
def ask_gpt_for_python_expression(user_question):
    prompt = f"""
System Role:
You are an AI assistant that converts natural language questions into **SQL Server SELECT queries** for a financial transactions database. The database has a single table.

Schema Definition:
- Table Name: Z_Verizonomnisense
- unit (TEXT): Represents the location, unit, food hall, site or outlet. Examples: 'Austin', 'Seattle'.
- category (TEXT): Describes the type of transaction. Examples: 'sales', 'expenses', 'food cost', etc.
- date (DATE): The date of the transaction.
- amount (DECIMAL(18,0)): The monetary value.

Instructions:
- 🔒 **Only generate SELECT queries. Do NOT generate INSERT, UPDATE, DELETE, TRUNCATE, DROP, ALTER, or any DDL/DML/maintenance queries.**
- 🔎 Always generate queries from a read-only perspective for data **retrieval** only.
- 📅 Use `MONTH(date)` and `YEAR(date)` functions for month/year filtering.
- 🧮 For aggregations, use SQL Server-compatible functions. **Always provide a descriptive alias for aggregated columns** (e.g., `SUM(amount) AS TotalAmount`, `COUNT(*) AS TransactionCount`, `AVG(amount) AS AverageAmount`, `MAX(amount) AS MaxValue`, `MIN(amount) AS MinValue`,`MONTH(date) AS Monthname`,`Year(date) AS Yearname`).
- 🗃 Use `GROUP BY`, `ORDER BY`, `HAVING`, and `WHERE` clauses as needed.
- 🧵 Use `TOP N` (e.g., `SELECT TOP 1 ...`) instead of `LIMIT`.
- 🔤 Match strings exactly — e.g., `unit = 'Austin'`, not `LIKE '%Austin%'`.
- ❌ Do not change or auto-correct user input (unit names, categories, dates, etc.) — use values exactly as provided.
- 🧠 Do not assume corrections — if a user types "Alphretta" instead of "Alpharetta", use it exactly as "Alphretta".
- 📜 Do not add explanation or commentary. Output only the SQL query, no markdown or code block.
- 🧱 Always assume the database is SQL Server (T-SQL syntax).
- 🧑‍💼 Do not attempt to infer missing columns or structure — only use the provided schema.
- We are provide the unit and Category list use below both list  
Category List:
Food Cost
Product Costs
Paper Cost
Profit Before Fee and Admin
Management Fee
Sales
Volume Allowances
Labor Cost
Profit/Loss Per Contract
Supv/Clerk Wages
Direct Fringe
Direct Wages
Travel
Cafe Sales

Unit List:
Lake Mary
Basking Ridge
Alphretta
Silver Spr
Irving Tele
Cary

Examples:

User Question: What were the total sales for Austin in April 2025?  
SQL Query: SELECT SUM(amount) FROM Z_Verizonomnisense WHERE unit = 'Austin' AND category = 'sales' AND MONTH(date) = 4 AND YEAR(date) = 2025;

User Question: Show me the top category by total amount.  
SQL Query: SELECT TOP 1 category, SUM(amount) AS total_amount FROM Z_Verizonomnisense GROUP BY category ORDER BY total_amount DESC;

User Question: How many transactions did Irving have in 2024?  
SQL Query: SELECT COUNT(*) FROM Z_Verizonomnisense WHERE unit = 'Irving' AND YEAR(date) = 2024;

User Question: Give me the average expense in Basking Ridge for November 2023.  
SQL Query: SELECT AVG(amount) FROM Z_Verizonomnisense WHERE unit = 'Basking Ridge' AND category = 'expenses' AND MONTH(date) = 11 AND YEAR(date) = 2023;

User Question: What’s the maximum sales recorded in Alphretta?  
SQL Query: SELECT MAX(amount) FROM Z_Verizonomnisense WHERE unit = 'Alphretta' AND category = 'sales';

User Question: Which category has the lowest total amount?  
SQL Query: SELECT TOP 1 category, SUM(amount) AS total_amount FROM Z_Verizonomnisense GROUP BY category ORDER BY total_amount ASC;

User Question: Give me all data for Cary in 2025.  
SQL Query: SELECT * FROM Z_Verizonomnisense WHERE unit = 'Cary' AND YEAR(date) = 2025;

User Question Placeholder:
User Question: {user_question}  
Expected Output: SQL Query:
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
    # Ensure result is a string, especially if it's a number or a list
    result_str = str(result)

    polish_prompt = f"""
    The user asked: "{user_question}"
    The core answer or result is: {result_str}

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

# --- Remove CSV upload logic ---
# Assume `run_query()` directly queries your SQL database (e.g., Z_Verizonomnisense)

# Show previous chat history
for entry in st.session_state.chat_history:
    st.markdown(f"**You:** {entry['question']}")
    st.markdown(f"**omniSense:** {entry['answer']}")

# Chat input (at bottom)
user_question = st.chat_input("Ask anything...")
if user_question:
    st.write("You:", user_question)
    with st.spinner("Processing..."):
        time.sleep(1)  # Simulate a short delay

        try:
            question_type = classify_question_type(user_question)
        except Exception as e:
            st.error(f"❌ Error classifying question: {e}")
            st.stop()  # Stop execution if classification fails

        if question_type.lower() == "quantitative":
            try:
                python_expr = ask_gpt_for_python_expression(user_question)

                # --- Clean the LLM's output ---
                if python_expr.startswith("SQL Query:"):
                    python_expr = python_expr.replace("SQL Query:", "").strip()
                else:
                    python_expr = python_expr.strip()
                
                # --- Run SQL query from expression ---
                result_df = run_query(python_expr)                
                if result_df is not None and not result_df.empty:
                   
                    if result_df.shape == (1, 1):
                        result_value = result_df.iloc[0, 0]
                        
                        response = ask_SmartResponse(user_question, result_value)
                    else:
                        response = ask_SmartResponse(user_question, result_df)
                else:
                    # Case 1: Query ran successfully but returned no rows.
                    # This is where you want your "no data" smart answer.
                    # Prompt for ask_SmartResponse: "No data was found for your specific question.
                    # Please consider rephrasing or checking details."
                    response = f"I couldn't find any information for your specific question.  " \
                    f"Perhaps try rephrasing it or checking for typos."
                    # response = ask_SmartResponse(
                    #     user_question,
                    #     "I couldn't find any information for your specific question. "
                    #     "Perhaps try rephrasing it or checking for typos."
                    # )

            except Exception as e:
                # Case 2: An error occurred during query generation or execution.
                # This provides error details to the user, including the problematic expression.
                response = f"I'm sorry, I couldn't generate a response for that question right now. " \
                f"Could you please try asking something else?"
                # response = ask_SmartResponse(
                #     user_question,
                #     f"I couldn't process that request due to an error. "
                #     f"The attempted expression was: `{python_expr}`. "
                #     f"Please check your table or column names, or try a different question."
                # )

        else:  # Qualitative
            try:
                # Minimal context or static schema assumption (e.g., 'unit', 'category', 'date', 'amount')
                context = "The table contains data about transactions with columns like unit, category, date, and amount."
                raw_response = ask_openai(user_question, context)
                response = ask_SmartResponse(user_question, raw_response)
            except Exception as e:
                #response = f"❌ Error generating qualitative response:"
                response = f"I'm sorry, I couldn't generate a response for that question right now. " \
                f"Could you please try asking something else?"

        # Store in chat history
        st.session_state.chat_history.append({
            "question": user_question,
            "answer": response
        })

        st.rerun()
        
# # --- Upload CSV ---
# uploaded_file = st.file_uploader("📎 Upload your CSV data", type="csv")
# if uploaded_file:
#     df = load_data(uploaded_file)

#     # --- PRIMARY FIX: Standardize all column names to lowercase FIRST ---
#     # This ensures consistency with the AI's generated queries ('unit', 'category', 'date', 'amount')
#     df.columns = df.columns.str.lower()

#     # 🔧 Ensure 'date' column (now lowercase) is datetime
#     # Check for 'date' (lowercase) after standardizing column names
#     if 'date' in df.columns:
#         df['date'] = pd.to_datetime(df['date'], errors='coerce')
#         # Drop rows where 'date' conversion failed (results in NaT)
#         df.dropna(subset=['date'], inplace=True)
#     else:
#         st.warning("⚠️ 'date' column not found or not parsable. Date-based queries may not work correctly.")

#     # Convert 'amount' to numeric, coercing errors to NaN
#     if 'amount' in df.columns:
#         df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
#         # Drop rows where amount is not numeric (results in NaN)
#         df.dropna(subset=['amount'], inplace=True)
#     else:
#         st.warning("⚠️ 'amount' column not found or not parsable. Quantitative queries may not work correctly.")

#     # --- Debugging Prints (Optional, remove after testing) ---
#     # st.write("--- Debugging DataFrame Info ---")
#     # buffer = io.StringIO()
#     # df.info(buf=buffer)
#     # st.text(buffer.getvalue())
#     # st.write("--- Debugging DataFrame Head ---")
#     # st.dataframe(df.head())
#     # st.write(f"Unique 'unit' values: {df['unit'].unique()}")
#     # st.write(f"Unique 'category' values: {df['category'].unique()}")
#     # st.write(f"Years in 'date' column: {df['date'].dt.year.unique() if 'date' in df.columns else 'N/A'}")
#     # --- End Debugging Prints ---

#     # Prepare context for qualitative questions
#     context = format_data_context(df)

#     # Show previous chat history
#     for entry in st.session_state.chat_history:
#         st.markdown(f"**You:** {entry['question']}")
#         st.markdown(f"**omniSense:** {entry['answer']}")

#     # Chat input (at bottom)
#     user_question = st.chat_input("Ask anything...")
#     if user_question:
#         st.write("You:", user_question)
#         with st.spinner("Processing..."):
#             time.sleep(1) # Simulate a short delay

#             try:
#                 question_type = classify_question_type(user_question)
#             except Exception as e:
#                 st.error(f"❌ Error classifying question: {e}")
#                 st.stop() # Stop execution if classification fails

#             if question_type.lower() == "quantitative":
#                 try:
#                     python_expr = ask_gpt_for_python_expression(user_question)

#                     # --- IMPORTANT FIX: Clean the LLM's output ---
#                     # Remove any "Pandas Expression:" prefix if the LLM includes it
#                     if python_expr.startswith("SQL Query:"):
#                         python_expr = python_expr.replace("SQL Query:", "").strip()
#                     else:
#                         python_expr = python_expr.strip()
                    
#                       # Run the query
#                     result_df = run_query(python_expr)
#                     if result_df is not None and not result_df.empty:
#                     # Try to extract a single value if this is an aggregate (e.g., SUM)
#                         if result_df.shape == (1, 1):
#                             result_value = result_df.iloc[0, 0]
#                             response = ask_SmartResponse(user_question, result_value)
#                         else:
#                             response = ask_SmartResponse(user_question, result_df)
#                     else:
#                         response = ask_SmartResponse(user_question, "No data returned for the query.")

#                 except Exception as e:
#                     # Provide a more informative error message to the user for quantitative queries
#                     response = f"❌ I couldn't process that quantitative request due to an error: `{e}`. " \
#                                f"The attempted expression was: `{python_expr}`. " \
#                                "Please try rephrasing your question or ensure your data columns match the expected format (unit, category, date, amount)."
#             else: # Qualitative question
#                 try:
#                     raw_response = ask_openai(user_question, context)
#                     response = ask_SmartResponse(user_question, raw_response)
#                 except Exception as e:
#                     response = f"❌ Error generating qualitative response: {e}"

#             # Store in chat history
#             st.session_state.chat_history.append({
#                 "question": user_question,
#                 "answer": response
#             })

#             # Refresh UI to show new chat history
#             st.rerun()
# else:
#     st.info("📥 Please upload a CSV file to begin.")
