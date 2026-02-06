from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

os.environ["HF_HOME"] = "D:/huggingface_cache"

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    device=-1,  
    pipeline_kwargs={
        "temperature": 1.5,
        "max_new_tokens": 100,
        "do_sample": True
    }
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Write a poem on cricket")
print(result.content)
