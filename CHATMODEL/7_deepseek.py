from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize DeepSeek LLM
llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat"
)

# Test prompt
response = llm.invoke("Explain what DeepSeek is in one sentence")

print("✅ DeepSeek Response:\n")
print(response.content)
