import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load saved artifacts
le_genre = joblib.load("genre_encoder.pkl")
directors = pd.read_csv("movies_with_director_scores.csv")

st.set_page_config(page_title="🎬 Movie Success Predictor", layout="centered")
st.title("🎬 Movie Success Predictor — Director Intelligence Edition")

st.markdown("""
This tool predicts a movie’s **potential IMDb rating** based on:
- Director's past success 🎯  
- Runtime and genre 🎭  
- Expected votes and release year 📅  
""")

# --- Input Fields ---
director_name = st.text_input("Director Name", "Christopher Nolan")
year = st.number_input("Release Year", min_value=1930, max_value=2025, value=2025)
runtime = st.number_input("Runtime (minutes)", min_value=30, max_value=400, value=120)
genre = st.text_input("Genre", "Action")
votes = st.number_input("Expected IMDb Votes", min_value=0, value=5000)

# Helper function to fetch director score
def get_director_score(name):
    row = directors[directors["primaryName"].str.lower() == name.lower()]
    if not row.empty:
        return float(row["director_score"].iloc[0])
    else:
        st.warning("⚠️ Director not found — using median director score.")
        return directors["director_score"].median()

if st.button("Predict 🎯"):
    director_score = get_director_score(director_name)

    try:
        genre_encoded = le_genre.transform([genre])[0]
    except:
        if "Unknown" in le_genre.classes_:
            genre_encoded = le_genre.transform(["Unknown"])[0]
        else:
            genre_encoded = le_genre.transform([le_genre.classes_[0]])[0]

    X_new = np.array([[director_score, year, runtime, votes, genre_encoded]])
    rating = model.predict(X_new)[0]
    success_rate = min(100, round((rating / 10) * 100, 1))  # success percentage

    st.success(f"🎥 Predicted IMDb Rating: **{rating:.2f}**")
    st.progress(success_rate / 100)
    st.info(f"Estimated Success Probability: **{success_rate}%**")
