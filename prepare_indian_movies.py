import pandas as pd
import numpy as np
import re

print("⏳ Loading IMDb datasets safely...")

# --- Helper: detect language via Unicode script blocks ---
def detect_indian_lang_from_text(text: str):
    if not isinstance(text, str):
        return np.nan
    # Normalize once
    s = text.strip()
    # Unicode blocks:
    # Devanagari (Hindi/Marathi): 0900–097F
    if re.search(r'[\u0900-\u097F]', s): return 'hi'  # treat Devanagari as Hindi for this project
    # Gurmukhi (Punjabi): 0A00–0A7F
    if re.search(r'[\u0A00-\u0A7F]', s): return 'pa'
    # Gujarati: 0A80–0AFF
    if re.search(r'[\u0A80-\u0AFF]', s): return 'gu'
    # Oriya/Odia: 0B00–0B7F
    if re.search(r'[\u0B00-\u0B7F]', s): return 'or'
    # Tamil: 0B80–0BFF
    if re.search(r'[\u0B80-\u0BFF]', s): return 'ta'
    # Telugu: 0C00–0C7F
    if re.search(r'[\u0C00-\u0C7F]', s): return 'te'
    # Kannada: 0C80–0CFF
    if re.search(r'[\u0C80-\u0CFF]', s): return 'kn'
    # Malayalam: 0D00–0D7F
    if re.search(r'[\u0D00-\u0D7F]', s): return 'ml'
    # Bengali: 0980–09FF
    if re.search(r'[\u0980-\u09FF]', s): return 'bn'
    return np.nan

INDIAN_LANGS = {'hi','ta','te','ml','kn','bn','mr','pa','gu','or'}

# --- Load basics + ratings ---
basics = pd.read_csv("data/title.basics.tsv.gz", sep="\t", low_memory=False)
ratings = pd.read_csv("data/title.ratings.tsv.gz", sep="\t", low_memory=False)

# Filter: movies, year range
movies = basics.merge(ratings, on="tconst", how="inner")
movies = movies[movies['titleType'] == 'movie']
movies = movies[movies['startYear'].apply(lambda x: str(x).isdigit())]
movies['startYear'] = movies['startYear'].astype(int)
movies = movies[movies['startYear'].between(1930, 2025)]

# Precompute language guess from primaryTitle script (useful fallback)
movies['lang_from_primary'] = movies['primaryTitle'].apply(detect_indian_lang_from_text)

# --- Stream akas to collect Indian region rows and detect language from akas.title ---
akas_iter = pd.read_csv("data/title.akas.tsv.gz", sep="\t", chunksize=250_000, low_memory=False)
akas_rows = []

processed = 0
for chunk in akas_iter:
    processed += len(chunk)
    # Keep Indian region rows (titles that appeared in India)
    sub = chunk[chunk['region'] == 'IN'][['titleId','title','language','isOriginalTitle']]
    # Normalize types
    if 'isOriginalTitle' in sub.columns:
        sub['isOriginalTitle'] = sub['isOriginalTitle'].astype(str)
    else:
        sub['isOriginalTitle'] = '0'
    # Detect language from akas.title script if missing
    sub['lang_from_title'] = sub['title'].apply(detect_indian_lang_from_text)
    # Prefer provided language if in our set, else detected script
    sub['lang_final'] = np.where(
        sub['language'].isin(INDIAN_LANGS), sub['language'], sub['lang_from_title']
    )
    # Keep only rows where we have an Indian language identified
    sub = sub[sub['lang_final'].isin(INDIAN_LANGS)]
    akas_rows.append(sub[['titleId','lang_final','isOriginalTitle']])
    if processed % 1_000_000 == 0:
        print(f"Processed {processed:,} akas rows...")

if not akas_rows:
    print("⚠️ No Indian-language rows detected in akas; falling back to primaryTitle script only.")
    # Fallback: keep only movies where primaryTitle is in an Indian script
    final = movies[movies['lang_from_primary'].isin(INDIAN_LANGS)].copy()
    final['language'] = final['lang_from_primary']
else:
    akas_filtered = pd.concat(akas_rows, ignore_index=True)
    # Prefer original title entry when available
    akas_filtered['isOriginalTitle'] = akas_filtered['isOriginalTitle'].astype(str)
    akas_filtered.sort_values(by=['titleId','isOriginalTitle'], ascending=[True, False], inplace=True)
    # One row per titleId (best language guess)
    akas_best = akas_filtered.drop_duplicates(subset=['titleId'], keep='first')

    # Merge with movies
    final = movies.merge(akas_best, left_on='tconst', right_on='titleId', how='inner')
    # If still missing, fall back to primaryTitle script detection
    final['language'] = final['lang_final'].fillna(final['lang_from_primary'])
    # Keep only rows with a language detected
    final = final[final['language'].isin(INDIAN_LANGS)].copy()

# Deduplicate by movie title (primaryTitle)
final = final.drop_duplicates(subset=['primaryTitle']).reset_index(drop=True)

# Select output columns
cols = ['primaryTitle','startYear','runtimeMinutes','genres','averageRating','numVotes','language']
existing = [c for c in cols if c in final.columns]
final = final[existing + ([c for c in ['region'] if c in final.columns])]

# Optional filter to drop obvious non-Indian categories if you want (comment out if not needed)
# non_indian_keywords = ['Western','Documentary','Animation']  # keep Horror/Musical if you want
# final = final[~final['genres'].fillna('').str.contains('|'.join(non_indian_keywords), case=False)]

final.to_csv("indian_movies_1930_2025_with_lang.csv", index=False)

print(f"✅ Done! Saved {len(final)} Indian-language movies (1930–2025).")
print("🔤 Language counts:\n", final['language'].value_counts(dropna=False).to_string())
print(final.head(20))
