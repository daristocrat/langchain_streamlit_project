from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# Create the Groq model
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

st.header("Research Tool")

user_input = st.text_input("Enter your prompt")

if st.button("Summarize"):
    if user_input.strip() == "":
        st.warning("Please enter a prompt")
    else:
        result = model.invoke(user_input)
        st.write(result.content)

