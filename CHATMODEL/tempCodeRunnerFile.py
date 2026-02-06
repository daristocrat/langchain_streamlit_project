from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

model = HuggingFaceEndpoint(
    repo_id="google/flan-t5-base",
    task="text2text-generation",
    max_new_tokens=64,
)

print(model.invoke("What is the capital of India?"))
