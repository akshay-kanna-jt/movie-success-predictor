import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error

print("⏳ Loading IMDb datasets (only limited rows for speed)...")
basics = pd.read_csv("data/title.basics.tsv.gz", sep='\t', nrows=1000000, low_memory=False)
ratings = pd.read_csv("data/title.ratings.tsv.gz", sep='\t', nrows=1000000, low_memory=False)
akas = pd.read_csv("data/title.akas.tsv.gz", sep='\t', nrows=1000000, low_memory=False)
crew = pd.read_csv("data/title.crew.tsv.gz", sep='\t', nrows=1000000, low_memory=False)

# ---- Merge on tconst ----
movies = basics.merge(ratings, on="tconst", how="inner")
movies = movies.merge(
    akas[['titleId', 'region', 'language']],
    left_on='tconst', right_on='titleId', how='left'
)
movies = movies.merge(crew, on='tconst', how='left')

# ---- Keep only useful columns ----
movies = movies[['primaryTitle', 'startYear', 'runtimeMinutes', 'genres',
                 'region', 'language', 'averageRating', 'numVotes']]

# ---- Clean data ----
movies.replace("\\N", pd.NA, inplace=True)   # Replace "\N" with NaN
movies.dropna(subset=['averageRating', 'genres', 'language', 'runtimeMinutes', 'numVotes'], inplace=True)

# Convert to correct data types
movies['startYear'] = pd.to_numeric(movies['startYear'], errors='coerce')
movies['runtimeMinutes'] = pd.to_numeric(movies['runtimeMinutes'], errors='coerce')
movies['numVotes'] = pd.to_numeric(movies['numVotes'], errors='coerce')

# Drop any remaining NaNs
movies.dropna(subset=['startYear', 'runtimeMinutes', 'numVotes'], inplace=True)

# Filter years
movies = movies[movies['startYear'].between(1930, 2025)]

# ---- Encode text columns ----
le_genre = LabelEncoder()
le_lang = LabelEncoder()
movies['genres_encoded'] = le_genre.fit_transform(movies['genres'])
movies['lang_encoded'] = le_lang.fit_transform(movies['language'])

# ---- Select features and target ----
X = movies[['startYear', 'runtimeMinutes', 'genres_encoded', 'lang_encoded', 'numVotes']]
y = movies['averageRating']

# ---- Split & Train ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---- Evaluate ----
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n✅ Model Trained Successfully!")
print(f"🎯 R² Score: {r2:.3f}")
print(f"📊 Mean Absolute Error: {mae:.3f}")

# ---- Show sample predictions ----
sample_results = pd.DataFrame({
    'Movie': movies['primaryTitle'].iloc[:10].values,
    'Actual Rating': y_test.iloc[:10].values,
    'Predicted Rating': y_pred[:10]
})
print("\n🎬 Sample Predictions:")
print(sample_results)



import joblib

# Save the trained model
joblib.dump(model, "movie_rating_model.pkl")

# Save encoders for genre and language
joblib.dump(le_genre, "genre_encoder.pkl")
joblib.dump(le_lang, "lang_encoder.pkl")

print("\n💾 Model and encoders saved successfully!")
