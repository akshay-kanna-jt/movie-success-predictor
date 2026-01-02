import pandas as pd

print("🎬 Step 1 & 2 — Loading IMDb movie and crew data...")

# --- Load core IMDb datasets ---
basics = pd.read_csv("data/title.basics.tsv.gz", sep='\t', low_memory=False)
ratings = pd.read_csv("data/title.ratings.tsv.gz", sep='\t', low_memory=False)
crew = pd.read_csv("data/title.crew.tsv.gz", sep='\t', low_memory=False)

# ✅ Keep only movies
movies = basics[basics['titleType'] == 'movie']

# ✅ Convert and filter years
movies = movies[movies['startYear'].apply(lambda x: str(x).isdigit())]
movies['startYear'] = movies['startYear'].astype(int)
movies = movies[movies['startYear'].between(1980, 2025)]

# ✅ Merge movies + ratings
movies = movies.merge(ratings, on="tconst", how="inner")

print(f"✅ Movies after merge: {len(movies):,}")

# ✅ Merge with crew info (director, writer IDs)
movies_crew = movies.merge(crew, on="tconst", how="left")

print(f"✅ Movies with crew info: {len(movies_crew):,}")

# ✅ Keep only useful columns
movies_crew = movies_crew[[
    'tconst', 'primaryTitle', 'startYear', 'runtimeMinutes',
    'genres', 'averageRating', 'numVotes', 'directors', 'writers'
]]

# ✅ Save intermediate dataset
movies_crew.to_csv("movies_with_crew.csv", index=False)

print("💾 Saved: movies_with_crew.csv")
print("🎞️ Sample rows:")
print(movies_crew.head(10))
