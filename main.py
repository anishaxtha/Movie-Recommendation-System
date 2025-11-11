import streamlit as st
import pandas as pd
import ast
import requests
from dotenv import load_dotenv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Streamlit Page Config ---
st.set_page_config(page_title="Movie Recommender 🎬", layout="centered")

# --- Custom Styling ---
st.markdown("""
    <style>
    body { background-color: #0E1117; color: white; }
    .movie-card {
        background-color: #1E1E1E;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #FF4B4B;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load TMDB API Key ---
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# --- Poster Fetch Function ---
def get_poster_from_tmdb(title):
    """Fetch poster using TMDB API"""
    try:
        if not TMDB_API_KEY:
            print("TMDB API key not found.")
            return "https://via.placeholder.com/300x450?text=No+Image"

        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
        res = requests.get(url).json()
        if res.get("results"):
            poster_path = res["results"][0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception as e:
        print("Poster fetch error:", e)
    return "https://via.placeholder.com/300x450?text=No+Image"

# --- Load Movie Data ---
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    credits = pd.read_csv("credits.csv")
    credits = credits.rename(columns={"movie_id": "id"})
    df = movies.merge(credits, on="id")

    df["overview"] = df["overview"].fillna("")
    df["genres"] = df["genres"].apply(lambda x: [d["name"] for d in ast.literal_eval(x)] if pd.notna(x) else [])
    df["cast"] = df["cast"].apply(lambda x: [d["name"] for d in ast.literal_eval(x)][:5] if pd.notna(x) else [])
    df["crew"] = df["crew"].apply(
        lambda x: [d["name"] for d in ast.literal_eval(x) if d["job"] in ["Director", "Producer", "Writer"]]
        if pd.notna(x) else []
    )
    return df

# --- Build Similarity Matrix ---
@st.cache_data
def build_similarity(df):
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3)
    matrix = tfidf.fit_transform(df["overview"])
    similarity = cosine_similarity(matrix)
    indices = pd.Series(df.index, index=df["original_title"]).drop_duplicates()
    return similarity, indices

# --- Recommendation Function ---
def recommend(movie, similarity, indices, df, n=5):
    idx = indices.get(movie)
    if idx is None:
        return None
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_ids = [i[0] for i in scores]
    return df.iloc[movie_ids]

# --- UI ---
st.title("🎬 English Movie Recommender System")
st.write("Type your favorite movie and discover similar ones!")

# Load data and similarity model
df = load_data()
similarity, indices = build_similarity(df)

# Search input
movie = st.text_input("🎥 Enter movie name")

# --- Show Movie Details ---
if movie:
    if movie not in indices.index:
        st.error("Movie not found. Try typing the exact title (case-sensitive).")
    else:
        selected = df[df["original_title"] == movie].iloc[0]
        st.markdown("---")

        col1, col2 = st.columns([1, 2])
        with col1:
            poster = get_poster_from_tmdb(selected["original_title"])
            st.image(poster, width=240)
        with col2:
            st.subheader(selected["original_title"])
            st.write(f"⭐ **Rating:** {selected['vote_average']:.1f} | 📅 **Year:** {selected['release_date'][:4] if pd.notna(selected['release_date']) else 'N/A'}")
            st.write(f"🎭 **Genres:** {', '.join(selected['genres']) or 'N/A'}")
            st.write(f"👥 **Cast:** {', '.join(selected['cast']) or 'N/A'}")
            st.write(f"🎬 **Crew:** {', '.join(selected['crew']) or 'N/A'}")
            st.write("📖 **Overview:**")
            st.write(selected["overview"] or "No description available.")

        # --- Recommended Movies ---
        st.markdown("---")
        st.subheader("🔥 Recommended Movies")
        recs = recommend(movie, similarity, indices, df, n=5)
        if recs is not None:
            for _, row in recs.iterrows():
                with st.container():
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        poster = get_poster_from_tmdb(row["original_title"])
                        st.image(poster, width=150)
                    with col2:
                        st.markdown(f"**{row['original_title']}**")
                        st.write(f"🎭 {', '.join(row['genres'])}")
                        st.write(f"⭐ {row['vote_average']:.1f}")
                        overview = row['overview'][:200] + "..." if len(row['overview']) > 200 else row['overview']
                        st.write(overview)
