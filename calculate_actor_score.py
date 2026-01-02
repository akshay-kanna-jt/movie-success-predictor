import pandas as pd

print("🎭 Calculating Actor Success Scores (Memory-Safe Mode)...")

# Load the movies dataset (already contains director scores)
movies = pd.read_csv("movies_with_director_scores.csv")

# --- Step 1: Load actor data in chunks ---
print("📂 Reading title.principals.tsv.gz in chunks...")
actors_list = []
chunksize = 250000  # Adjust this based on your RAM (lower = safer)

reader = pd.read_csv("data/title.principals.tsv.gz", sep='\t', low_memory=False, chunksize=chunksize)
for i, chunk in enumerate(reader):
    # Filter only actors and actresses
    filtered = chunk[chunk['category'].isin(['actor', 'actress'])]
    actors_list.append(filtered[['tconst', 'nconst']])
    if (i + 1) % 20 == 0:
        print(f"Processed {((i + 1) * chunksize):,} rows...")

actors_df = pd.concat(actors_list, ignore_index=True)
print(f"✅ Total actor entries: {len(actors_df):,}")

# --- Step 2: Load actor names in chunks (memory-safe) ---
print("📂 Reading name.basics.tsv.gz in chunks...")
names_list = []
reader2 = pd.read_csv("data/name.basics.tsv.gz", sep='\t', low_memory=False, chunksize=chunksize)
for i, chunk in enumerate(reader2):
    names_list.append(chunk[['nconst', 'primaryName']])
    if (i + 1) % 20 == 0:
        print(f"Processed {(i + 1) * chunksize:,} rows from names...")

names_df = pd.concat(names_list, ignore_index=True)

# --- Step 3: Merge movies with actors ---
movies_actors = movies.merge(actors_df, on='tconst', how='inner')
movies_actors = movies_actors.merge(names_df, on='nconst', how='left')
print(f"✅ Merged movies with actor names: {len(movies_actors):,} records")

# --- Step 4: Compute average rating per actor ---
actor_scores = (
    movies_actors.groupby('primaryName')['averageRating']
    .mean()
    .reset_index()
    .rename(columns={'averageRating': 'actor_score'})
)
print(f"✅ Actor scores calculated: {len(actor_scores):,}")

# --- Step 5: Merge back to movie data ---
movies_actors = movies_actors.merge(actor_scores, on='primaryName', how='left')
actor_avg_per_movie = (
    movies_actors.groupby('tconst')['actor_score']
    .mean()
    .reset_index()
)

# --- Step 6: Final merge with director scores ---
final_movies = movies.merge(actor_avg_per_movie, on='tconst', how='left')

# --- Step 7: Save results ---
final_movies.to_csv("movies_with_director_actor_scores.csv", index=False)

print("💾 Saved: movies_with_director_actor_scores.csv")
print("🎞️ Sample rows:")
print(final_movies.head(10))
