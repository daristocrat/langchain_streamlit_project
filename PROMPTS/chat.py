from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model=ChatGroq( model="llama-3.1-8b-instant")

chat_history=[
    SystemMessage(content='you are my virtual assistant')
]

while True:
    user_input=input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input=='exit':
        break

    result=model.invoke(user_input)

    print('AI: ', result.content)
    chat_history.append(result)
print(chat_history)