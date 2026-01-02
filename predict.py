import joblib
import numpy as np
import pandas as pd

# Load model and encoders
model = joblib.load("movie_rating_model.pkl")
le_genre = joblib.load("genre_encoder.pkl")
le_lang = joblib.load("lang_encoder.pkl")

# Load dataset to get movie titles
basics = pd.read_csv("data/title.basics.tsv.gz", sep='\t', nrows=1000000, low_memory=False)
ratings = pd.read_csv("data/title.ratings.tsv.gz", sep='\t', nrows=1000000, low_memory=False)
movies = basics.merge(ratings, on="tconst", how="inner")

# ---- User input ----
year = int(input("Enter release year (e.g., 2023): "))
runtime = int(input("Enter runtime in minutes (e.g., 120): "))
genre = input("Enter genre (e.g., Action): ")
language = input("Enter language code (e.g., en): ")
votes = int(input("Enter number of votes (e.g., 5000): "))

# ---- Encode text inputs ----
try:
    genre_encoded = le_genre.transform([genre])[0]
    lang_encoded = le_lang.transform([language])[0]

    # Prepare input array
    X_new = np.array([[year, runtime, genre_encoded, lang_encoded, votes]])

    # Predict rating
    predicted_rating = model.predict(X_new)[0]

    # Find a sample movie title with same genre & language
    sample = movies[
        (movies['genres'].str.contains(genre)) &
        (movies['original_language'] == language)
    ]
    if not sample.empty:
        title = sample['primaryTitle'].sample(1).values[0]
    else:
        title = "No matching movie found"

    print(f"\n🎬 Sample Movie: {title}")
    print(f"🎯 Predicted IMDb Rating: {predicted_rating:.2f}")

except:
    print("\n❌ Genre or language not found in training data. Try another input.")
