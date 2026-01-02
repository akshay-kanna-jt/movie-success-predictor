import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Load Data ---
@st.cache_data
def load_director_data():
    return pd.read_csv("movies_with_director_scores.csv")

@st.cache_resource
def load_model():
    model = joblib.load("movie_with_director_scores.pkl")
    le_genre = joblib.load("genre_encoder_v2.pkl")
    return model, le_genre

# --- Load all assets ---
directors = load_director_data()
model, le_genre = load_model()

# --- Streamlit App ---
st.set_page_config(page_title="🎬 Movie Success Predictor", layout="centered")
st.title("🎯 Unified Movie Success Predictor (with Director Intelligence)")

st.markdown("""
This unified model predicts a movie’s **future IMDb rating and box-office success**
based on:
- 🎬 Director’s past performance  
- 🎭 Genre  
- ⏱️ Runtime  
- 📅 Release Year  
- 🗳️ Expected Votes
""")

# --- Input Fields ---
director_name = st.text_input("Director Name", "Christopher Nolan")
genre = st.text_input("Movie Genre", "Action")
language = st.text_input("Language Code", "en")
year = st.number_input("Release Year", min_value=1930, max_value=2025, value=2024)
runtime = st.number_input("Runtime (minutes)", min_value=30, max_value=400, value=150)
votes = st.number_input("Expected IMDb Votes", min_value=0, value=10000)

# --- Helper to fetch director score ---
def get_director_score(name):
    row = directors[directors["primaryName"].str.lower() == name.lower()]
    if not row.empty:
        return float(row["director_score"].iloc[0])
    else:
        st.warning("⚠️ Director not found — using average director score.")
        return directors["director_score"].mean()

# --- Verdict logic ---
def movie_verdict(rating):
    if rating >= 8.5:
        return "🔥 Blockbuster", "This movie is expected to dominate the box office!"
    elif rating >= 7.5:
        return "🏆 Super Hit", "Strong performance expected with high audience appeal."
    elif rating >= 6.5:
        return "💥 Hit", "Likely to perform well in most regions."
    elif rating >= 5.0:
        return "⚖️ Average", "Might have a mixed response from the audience."
    else:
        return "💔 Flop", "Predicted to underperform at the box office."

# --- Predict Button ---
if st.button("Predict Success 🎬"):
    director_score = get_director_score(director_name)

    try:
        genre_encoded = le_genre.transform([genre])[0]
    except:
        genre_encoded = 0

    X_new = np.array([[director_score, year, runtime, votes, genre_encoded]])
    rating = model.predict(X_new)[0]
    verdict, message = movie_verdict(rating)
    success_rate = min(100, round((rating / 10) * 100, 1))

    # --- Display Results ---
    st.subheader(f"🎥 Predicted IMDb Rating: **{rating:.2f} / 10**")
    st.progress(success_rate / 100)
    st.markdown(f"### {verdict}")
    st.success(message)
