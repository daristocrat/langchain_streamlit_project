import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize model
model = ChatGroq(model="llama-3.1-8b-instant")

# Prompt template
prompt = PromptTemplate(
    template="""
You are an expert Instagram Reels music curator.

User details:
- Mood: {mood}
- Reel vibe: {vibe}
- Content type: {content}
- Language preference: {language}

Task:
1. Suggest 5 Instagram Reel-friendly songs.
2. For each song, briefly explain why it matches the vibe.
3. Prefer trending, catchy, or emotionally resonant tracks.
4. Keep the response concise and fun.
""",
    input_variables=["mood", "vibe", "content", "language"]
)

parser = StrOutputParser()
chain = prompt | model | parser

# ---------------- STREAMLIT UI ---------------- #

st.set_page_config(page_title="Insta Song Suggestor", layout="centered")

st.title("personalized AI Instagram Song Suggestor")
st.write("Find the perfect song for your next Reel in seconds!")

# User Inputs
mood = st.selectbox(
    "What's your mood right now?",
    ["Happy", "Chill", "Romantic", "Energetic", "Sad", "Confident", "Aesthetic"]
)

vibe = st.selectbox(
    "What vibe are you going for?",
    ["Trending", "Soft", "Dark", "Fun", "Motivational", "Luxury", "Emotional"]
)

content = st.selectbox(
    "What's your Reel about?",
    ["Travel", "Workout", "Dance", "Fashion", "Love", "Friends", "Daily Life", "Glow Up"]
)

language = st.selectbox(
    "Preferred song language?",
    ["Any", "English", "Hindi", "Punjabi", "Tamil", "Bangali"]
)

if st.button("Suggest Songs"):
    with st.spinner("Curating the perfect playlist for your Reel... "):
        result = chain.invoke({
            "mood": mood,
            "vibe": vibe,
            "content": content,
            "language": language
        })

    st.subheader("Song Suggestions for You")
    st.write(result)

st.markdown("---")

