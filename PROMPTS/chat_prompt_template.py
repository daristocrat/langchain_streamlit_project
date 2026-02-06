from langchain_core.prompts import ChatPromptTemplate

prompt_template=ChatPromptTemplate([
    ('system','you are a helpful {domain} agent'),
    ('human','explain about {topic}')
])

result=prompt_template.invoke({'domain':'cricket','topic':'lbw'})
print(result)