import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # Hide C++ logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"     # Disable oneDNN logs

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Now import TensorFlow
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import tensorflow_hub as hub       

import tensorflow_hub as hub
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings

class USEEmbeddings(Embeddings):
    def __init__(self):
        self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    def embed_documents(self, texts):
        return self.model(texts).numpy().tolist()

    def embed_query(self, text):
        return self.model([text]).numpy()[0].tolist()

documents = [
    Document(page_content="Sign language recognition is fascinating."),
    Document(page_content="Neural networks help interpret gestures.")
]

embedding = USEEmbeddings()

vector_store = Chroma.from_documents(
    documents,
    embedding,
    persist_directory="use_db"
)

print("Working successfully!")
