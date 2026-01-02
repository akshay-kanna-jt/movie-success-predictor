import pandas as pd

# ---- Load datasets ----
basics = pd.read_csv("data/title.basics.tsv.gz", sep='\t', nrows=500000, low_memory=False)
ratings = pd.read_csv("data/title.ratings.tsv.gz", sep='\t', nrows=500000, low_memory=False)
akas = pd.read_csv("data/title.akas.tsv.gz", sep='\t', nrows=500000, low_memory=False)

# ---- Merge datasets ----
movies = basics.merge(ratings, on="tconst", how="inner")
movies = movies.merge(
    akas[['titleId', 'region', 'language']],
    left_on='tconst', right_on='titleId', how='left'
)

# ---- Keep useful columns ----
movies = movies[['primaryTitle', 'startYear', 'runtimeMinutes', 'genres', 'region', 'language', 'averageRating', 'numVotes']]

# ---- Clean and filter by year 1930–2025 ----
movies = movies[movies['startYear'].apply(lambda x: str(x).isdigit())]
movies['startYear'] = movies['startYear'].astype(int)
movies = movies[movies['startYear'].between(1930, 2025)]

# ---- Remove duplicates by title ----
movies = movies.drop_duplicates(subset=['primaryTitle'], keep='first').reset_index(drop=True)

# ---- Filter Indian movies by language codes ----
indian_langs = ['en','hi','ta','te','ml','kn','bn','mr','pa','gu','or']
indian_movies = movies[movies['language'].isin(indian_langs)].reset_index(drop=True)

print("Total unique movies from 1930–2025:", len(movies))
print("Total Indian movies (based on language):", len(indian_movies))
print(indian_movies.head(50))
