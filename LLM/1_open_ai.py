from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

assert os.getenv("OPENAI_API_KEY"), "API key not loaded!"

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

response = llm.invoke("What is the capital of India?")
print(response.content)
