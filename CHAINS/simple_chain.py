from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant" ,
                 temperature=1.5)

prompt = PromptTemplate(
    template="generate 5 intresting facts about {topic}",
    input_variables=['topic']
)

parsers = StrOutputParser()

chain = prompt | model | parsers

result = chain.invoke({'topic' : 'cricket'})

print(result)

chain.get_graph().print_ascii()
