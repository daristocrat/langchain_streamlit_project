from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model=ChatGroq(model="llama-3.1-8b-instant")

prompt =PromptTemplate(
    template="write a 5 line summery on the following {text}",
    input_variables="text"
)

parsers=StrOutputParser()

loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()

print(type(docs))

print (len(docs))

print(docs[0])

chain = prompt|model|parsers

result=chain.invoke({'text':docs[0].page_content})

print(result)