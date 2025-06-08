#My Python Project to check LLMS Capabilities
import streamlit as st
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatTogether
import os

st.set_page_config(page_title="CSV Chatbot with Together.ai", layout="wide")
st.title("🤖 CSV Chatbot using Together.ai")
st.markdown("Upload a CSV file and ask questions using a cloud-based open LLM.")

# API Key input
api_key = st.text_input("Enter your Together.ai API Key", type="password")

# Model selection
model_name = st.selectbox("Select a model", [
    "togethercomputer/llama-3-8b-instruct",
    "togethercomputer/mistral-7b-instruct",
    "togethercomputer/gemma-7b-it",
    "togethercomputer/phi-2"
])

# File upload
uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if api_key and uploaded_file:
    os.environ["TOGETHER_API_KEY"] = api_key

    # Read CSV
    df = pd.read_csv(uploaded_file)
    st.success("CSV loaded successfully!")
    st.dataframe(df.head())

    # Setup LangChain + Together
    llm = ChatTogether(model=model_name, temperature=0.2)

    # Agent
    agent = create_pandas_dataframe_agent(llm, df, verbose=True)

    # Question input
    st.markdown("### 💬 Ask a question about your CSV")
    user_query = st.text_input("Your question")

    if st.button("Ask"):
        if user_query.strip():
            with st.spinner("Thinking..."):
                try:
                    response = agent.run(user_query)
                    st.success("Answer:")
                    st.write(response)
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a question.")

# End of Execution master
# Another Section Below 

