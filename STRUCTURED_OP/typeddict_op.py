from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

class Review(TypedDict):
    summary: Annotated[str, "Summarize the text in about 30 words."]
    sentiment: Literal["positive", "negative", "neutral"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke(
    """Artificial intelligence is increasingly integrated into education systems to personalize learning, automate assessments, and provide real-time feedback. However, concerns persist regarding data privacy, algorithmic bias, over-reliance on automated tools, and the potential erosion of critical thinking skills. Limited teacher training and unequal access to AI-driven resources further widen the digital divide, raising questions about whether efficiency gains truly outweigh long-term educational and ethical costs.""")

print(result)
