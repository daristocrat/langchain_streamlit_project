import os
import re
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------
# Load Environment
# ---------------------------
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="YouTube RAG", layout="wide")
st.title("YouTube RAG Chatbot")

# ---------------------------
# Extract Video ID
# ---------------------------
def extract_video_id(url):
    import re

    patterns = [
        r"(?:v=)([a-zA-Z0-9_-]{11})",          # watch?v=
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",  # youtu.be/
        r"(?:shorts/)([a-zA-Z0-9_-]{11})",     # shorts/
        r"(?:embed/)([a-zA-Z0-9_-]{11})"       # embed/
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    raise ValueError("Invalid YouTube URL")

# ---------------------------
# Build RAG Pipeline
# ---------------------------
@st.cache_resource
def build_rag(youtube_url):
    video_id = extract_video_id(youtube_url)

    ytt_api = YouTubeTranscriptApi()

# Fetch available transcripts
    transcript_list = ytt_api.list(video_id)

# Automatically get first available transcript
    transcript = transcript_list.find_transcript(
    [t.language_code for t in transcript_list]
)

    fetched = transcript.fetch()

    full_text = " ".join([entry.text for entry in fetched])

    documents = [Document(page_content=full_text)]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(docs, embedding)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        groq_api_key=groq_key
    )

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question ONLY using the context below.

        Context:
        {context}

        Question:
        {question}
        """
    )

    chain = (
        {"context": retriever, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# ---------------------------
# Streamlit UI
# ---------------------------
youtube_url = st.text_input("Enter YouTube URL")

if youtube_url:
    try:
        st.success("Transcript loaded successfully!")
        rag_chain = build_rag(youtube_url)

        query = st.text_input("Ask a question about the video")

        if query:
            with st.spinner("Thinking..."):
                response = rag_chain.invoke(query)

            st.subheader("Answer")
            st.write(response)

    except Exception as e:
        st.error(f"Error: {str(e)}")
