from langchain_openai import ChatOpenAi
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAi(model="gpt-4o-mini" , temperature=0)

result=model.invoke("what is the history of capital of india")
print(result)