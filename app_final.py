import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(
    page_title="🎬 Movie Success Predictor",
    page_icon="🎥",
    layout="wide",
)

# --- Cached Loaders ---
@st.cache_data
def load_data():
    df = pd.read_csv("indian_movies_1930_2025.csv")
    df = df.dropna(subset=["averageRating"])
    return df

@st.cache_resource
def load_models():
    model_basic = joblib.load("movie_rating_model.pkl")
    le_genre = joblib.load("genre_encoder.pkl")
    le_lang = joblib.load("lang_encoder.pkl")
    return model_basic, le_genre, le_lang

# --- Load Data and Models ---
df = load_data()
model_basic, le_genre, le_lang = load_models()

# --- Sidebar Navigation ---
st.sidebar.title("🎞️ Navigation")
page = st.sidebar.radio("Go to:", [
    "📊 Explore Data",
    "🎯 Predict IMDb Rating",
    "📈 Insights"
])

# --- PAGE 1: Explore Dataset ---
if page == "📊 Explore Data":
    st.title("📊 Explore Movie Dataset")
    st.write(f"Total Movies: **{len(df):,}** (1930–2025)")

    st.dataframe(df.head(20))

    st.subheader("⭐ Rating Distribution")
    fig, ax = plt.subplots()
    df["averageRating"].astype(float).plot(kind="hist", bins=20, ax=ax, color="#1f77b4")
    ax.set_xlabel("IMDb Rating")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.subheader("🎭 Top 10 Genres by Movie Count")
    top_genres = df["genres"].value_counts().head(10)
    st.bar_chart(top_genres)

    st.subheader("📅 Movie Releases Over Time")
    year_counts = df["startYear"].value_counts().sort_index()
    st.line_chart(year_counts)

# --- PAGE 2: Predict IMDb Rating ---
elif page == "🎯 Predict IMDb Rating":
    st.title("🎯 Movie Success Predictor (Based on Past Data)")

    st.markdown("""
    Predict your movie’s **future IMDb rating** and **success verdict**
    using patterns learned from thousands of Indian films.
    """)

    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Release Year", min_value=1930, max_value=2025, value=2024)
        runtime = st.number_input("Runtime (minutes)", min_value=30, max_value=300, value=120)
        votes = st.number_input("Expected IMDb Votes", min_value=10, max_value=500000, value=5000)
    with col2:
        genres_list = sorted(df["genres"].dropna().unique().tolist())
        genre = st.selectbox("Genre", genres_list)
        languages_list = sorted(df["language"].dropna().unique().tolist()) if "language" in df.columns else ["hi", "ta", "te", "ml", "kn", "bn"]
        language = st.selectbox("Language Code", languages_list)

    st.markdown("---")

    # --- Verdict Function ---
    def get_movie_verdict(rating):
        if rating >= 8.5:
            return "🔥 Blockbuster", "This movie will dominate the box office!"
        elif rating >= 7.5:
            return "🏆 Super Hit", "Strong performance and critical success expected."
        elif rating >= 6.5:
            return "💥 Hit", "Likely to perform well with good audience reception."
        elif rating >= 5.0:
            return "⚖️ Average", "Might get mixed reviews."
        else:
            return "💔 Flop", "Predicted to underperform commercially."

    # --- Prediction Logic ---
    if st.button("🎬 Predict Success"):
        try:
            genre_encoded = le_genre.transform([genre])[0]
            lang_encoded = le_lang.transform([language])[0]
            X_new = np.array([[year, runtime, genre_encoded, lang_encoded, votes]])
            rating = model_basic.predict(X_new)[0]
            verdict, message = get_movie_verdict(rating)

            success_rate = min(100, round((rating / 10) * 100, 1))

            st.subheader(f"🎥 Predicted IMDb Rating: **{rating:.2f} / 10**")
            st.progress(success_rate / 100)
            st.markdown(f"### {verdict}")
            st.success(message)

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
            st.info("Try using a genre or language available in the dataset.")

# --- PAGE 3: Insights ---
elif page == "📈 Insights":
    st.title("📈 Movie Insights")

    st.subheader("🏆 Top 10 Highest Rated Indian Movies")
    top_movies = df.sort_values(by="averageRating", ascending=False).head(10)[
        ["primaryTitle", "averageRating", "startYear", "genres"]
    ]
    st.table(top_movies)

    if "language" in df.columns:
        st.subheader("🌐 Average Rating by Language")
        lang_mean = df.groupby("language")["averageRating"].mean().sort_values(ascending=False)
        st.bar_chart(lang_mean)

    st.subheader("🎬 Average Rating by Genre")
    genre_mean = df.groupby("genres")["averageRating"].mean().sort_values(ascending=False).head(15)
    st.bar_chart(genre_mean)


