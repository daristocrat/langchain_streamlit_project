import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Initialize model
model = ChatGroq(model="llama-3.1-8b-instant")

# Prompt 1: Detailed report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

# Prompt 2: Summary
template2 = PromptTemplate(
    template="Write a 5 line summary of the following text:\n{text}",
    input_variables=["text"]
)

# Output parser
parser = StrOutputParser()

# Chain
chain = template1 | model | parser | template2 | model | parser

# ---------------- STREAMLIT UI ---------------- #

st.set_page_config(page_title="LangChain Report & Summary", layout="centered")

st.title(" AI Report & Summary Generator")
st.write("Generate a detailed report and then summarize it using Groq + LangChain")

topic = st.text_input("Enter a topic:", placeholder="e.g. Wave theory of light")

if st.button("Generate Summary"):
    if topic.strip() == "":
        st.warning("Please enter a topic.")
    else:
        with st.spinner("Generating..."):
            result = chain.invoke({"topic": topic})
        st.success("Done!")
        st.subheader("📝 Final 5-Line Summary")
        st.write(result)
