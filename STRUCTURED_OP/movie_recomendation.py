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
You are a movie recommendation expert.

User preferences:
- Mood: {mood}
- Genres: {genres}
- Language preference: {language}
- Watching with: {company}

Based on these preferences:
1. Recommend 3 movies.
2. For each movie, explain in 1–2 lines why it fits.
3. Keep the tone friendly and conversational.
""",
    input_variables=["mood", "genres", "language", "company"]
)

parser = StrOutputParser()
chain = prompt | model | parser

# ----------------- STREAMLIT UI ----------------- #

st.set_page_config(page_title="AI Movie Recommender", layout="centered")

st.title("AI Movie Recommendation App")
st.write("Not sure what to watch today? Answer a few questions and let AI decide!")

# User inputs
mood = st.selectbox(
    "How are you feeling today?",
    ["Happy", "Sad", "Excited", "Romantic", "Stressed", "Adventurous", "Relaxed"]
)

genres = st.multiselect(
    "Which genres do you feel like watching?",
    ["Action", "Comedy", "Drama", "Romance", "Thriller", "Sci-Fi", "Horror", "Fantasy" ,"Hopeful"]
)

language = st.selectbox(
    "Preferred language?",
    ["Any", "English", "Hindi", "Korean", "Japanese", "Tamil", "Telugu"]
)

company = st.selectbox(
    "Who are you watching with?",
    ["Alone", "Friends", "Family", "Partner"]
)

if st.button("🎥 Recommend Movies"):
    if not genres:
        st.warning("Please select at least one genre.")
    else:
        with st.spinner("Finding the perfect movies for you... 🍿"):
            result = chain.invoke({
                "mood": mood,
                "genres": ", ".join(genres),
                "language": language,
                "company": company
            })

        st.subheader("✨ Your Movie Recommendations")
        st.write(result)

st.markdown("---")
st.caption("Built by THE KING IN THE NORTH")
