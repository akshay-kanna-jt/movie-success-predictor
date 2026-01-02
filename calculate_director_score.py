import pandas as pd

print("🎭 Step 4 — Calculating Actor Success Scores (Memory-Safe Mode)...")

# --- Step 1: Load movie dataset ---
movies = pd.read_csv("movies_with_director_scores.csv")
print(f"✅ Movies loaded: {len(movies):,}")

# --- Step 2: Stream title.principals.tsv.gz ---
print("📂 Reading title.principals.tsv.gz in chunks...")
actors_list = []
chunksize = 250000
reader = pd.read_csv("data/title.principals.tsv.gz", sep='\t', low_memory=False, chunksize=chunksize)

for i, chunk in enumerate(reader):
    filtered = chunk[chunk['category'].isin(['actor', 'actress'])]
    actors_list.append(filtered[['tconst', 'nconst']])
    if (i + 1) % 20 == 0:
        print(f"Processed {((i + 1) * chunksize):,} rows...")

actors_df = pd.concat(actors_list, ignore_index=True)
print(f"✅ Total actor records: {len(actors_df):,}")

# --- Step 3: Stream name.basics.tsv.gz safely ---
print("📂 Reading name.basics.tsv.gz in chunks...")
name_chunks = []
reader2 = pd.read_csv("data/name.basics.tsv.gz", sep='\t', low_memory=False, chunksize=chunksize)

for i, chunk in enumerate(reader2):
    if 'nconst' in chunk.columns and 'primaryName' in chunk.columns:
        name_chunks.append(chunk[['nconst', 'primaryName']])
    else:
        print("⚠️ Skipping chunk due to missing columns")
    if (i + 1) % 20 == 0:
        print(f"Processed {(i + 1) * chunksize:,} rows from names...")

names_df = pd.concat(name_chunks, ignore_index=True)
names_df.columns = names_df.columns.str.strip()  # remove hidden spaces
print(f"✅ Total names loaded: {len(names_df):,}")
print(f"🧩 Columns in names_df: {list(names_df.columns)}")

# --- Step 4: Merge movies with actors ---
movies_actors = movies.merge(actors_df, on='tconst', how='inner')
print(f"🔗 Merged with actor IDs: {len(movies_actors):,}")

# --- Step 5: Merge actor IDs with names ---
movies_actors = movies_actors.merge(names_df, on='nconst', how='left')

# Safety check
if 'primaryName' not in movies_actors.columns:
    print("❌ 'primaryName' column missing — forcing rename or drop detection...")
    possible_cols = [c for c in movies_actors.columns if 'name' in c.lower()]
    print("🧭 Possible name columns found:", possible_cols)
    if possible_cols:
        movies_actors.rename(columns={possible_cols[0]: 'primaryName'}, inplace=True)

print(f"🎬 Added actor names: {movies_actors['primaryName'].notna().sum():,} valid names found")

# --- Step 6: Compute actor success scores ---
actor_scores = (
    movies_actors.groupby('primaryName', dropna=True)['averageRating']
    .mean()
    .reset_index()
    .rename(columns={'averageRating': 'actor_score'})
)
print(f"✅ Actor scores calculated: {len(actor_scores):,}")

# --- Step 7: Merge actor scores back to movie dataset ---
movies_actors = movies_actors.merge(actor_scores, on='primaryName', how='left')

# Average actor score per movie
actor_avg_per_movie = (
    movies_actors.groupby('tconst')['actor_score']
    .mean()
    .reset_index()
)

# --- Step 8: Merge with director score dataset ---
final_movies = movies.merge(actor_avg_per_movie, on='tconst', how='left')

# --- Step 9: Save results ---
final_movies.to_csv("movies_with_director_actor_scores.csv", index=False)

print("💾 Saved: movies_with_director_actor_scores.csv")
print("🎞️ Sample rows:")
print(final_movies[['primaryTitle', 'startYear', 'averageRating', 'director_score', 'actor_score']].head(10))
