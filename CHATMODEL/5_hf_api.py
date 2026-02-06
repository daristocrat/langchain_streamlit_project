from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# Base endpoint (no task= needed here)
model = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",   # reliable for text-generation
    task="text-generation",
    max_new_tokens=128,
)
print(model.invoke("What is the capital of India?"))