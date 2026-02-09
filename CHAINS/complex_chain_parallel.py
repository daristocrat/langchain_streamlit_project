from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant" ,
                 temperature=1.5)

prompt1 = PromptTemplate(
    template="generate a detailed report on the following topic \n {topic}" , 
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="generate a series of 5 question and answer on the following {topic}",
    input_variables=['topic']
)

prompt3 = PromptTemplate(
    template="generate a combined text with both the notes and quiz on the topic \n notes->{notes} quiz->{quiz}",
    input_variables=['notes','quiz']
)

parsers = StrOutputParser()

parallel_chain = RunnableParallel(
    notes=prompt1|model|parsers,
    quiz=prompt2|model|parsers
)

merge_chain = prompt3|model|parsers

chain = parallel_chain|merge_chain

result=chain.invoke({'topic':'ford'})

print(result)

chain.get_graph().print_ascii()
