from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant" ,
                 temperature=1.5)

prompt1 = PromptTemplate(
    template="generate 100 lines of  intresting facts about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="summerize the {text}",
    input_variables=['text']
)

parsers = StrOutputParser()

chain= prompt1|model|parsers|prompt2|model|parsers

print(chain.invoke({'topic':'is jesus real'}))
