# ---------------------------
# OPTIONAL: Suppress warnings
# ---------------------------
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------
# IMPORTS
# ---------------------------
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------------------
# CREATE DOCUMENTS
# ---------------------------
documents = [
    Document(
        page_content="Virat Kohli is one of the most consistent batsmen in IPL history.",
        metadata={"team": "Royal Challengers Bangalore"}
    ),
    Document(
        page_content="Rohit Sharma has led Mumbai Indians to five IPL titles.",
        metadata={"team": "Mumbai Indians"}
    ),
    Document(
        page_content="MS Dhoni is known as Captain Cool and has led CSK to multiple titles.",
        metadata={"team": "Chennai Super Kings"}
    ),
]

# ---------------------------
# LOAD LOCAL EMBEDDING MODEL
# ---------------------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},  # use "cuda" if you have GPU
)

# ---------------------------
# CREATE VECTOR STORE
# ---------------------------
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embedding,
    persist_directory="hf_chroma_db",
    collection_name="ipl_collection"
)

vector_store.persist()

print("✅ Hugging Face embedding system created successfully!")

query = "Who is called Captain Cool?"
results = vector_store.similarity_search(query, k=2)

for r in results:
    print("Result:", r.page_content)

