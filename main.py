import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {background-color: #0E1117;}
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
    }
    .movie-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #FF4B4B;
    }
    </style>
""", unsafe_allow_html=True)

# Load and process data
@st.cache_data
def load_movies():
    movies = pd.read_csv('movies.csv')
    credits = pd.read_csv('credits.csv')
    
    credits = credits.rename(columns={"movie_id": "id"})
    df = movies.merge(credits, on='id')
    df = df.drop(columns=['homepage', 'title_x', 'title_y', 'status', 'production_countries'])
    df['overview'] = df['overview'].fillna('')
    
    return df

@st.cache_data
def get_recommendations_engine(df):
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=3)
    matrix = tfidf.fit_transform(df['overview'])
    similarity = cosine_similarity(matrix, matrix)
    indices = pd.Series(df.index, index=df['original_title']).drop_duplicates()
    return similarity, indices

def recommend(movie, similarity, indices, df, n=10):
    idx = indices[movie]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_ids = [i[0] for i in scores]
    return df.iloc[movie_ids], [s[1] for s in scores]

# Main App
st.title("🎬 Movie Recommendation System")
st.markdown("### Discover your next favorite movie!")

# Load data
df = load_movies()
similarity, indices = get_recommendations_engine(df)

# Search section
st.markdown("---")
col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    movie = st.selectbox("🔍 Search for a movie", sorted(indices.index), label_visibility="collapsed", placeholder="Type or select a movie...")

with col2:
    n_recs = st.number_input("How many?", 5, 20, 10, label_visibility="collapsed")

with col3:
    search = st.button("🎯 Find Movies")

# Results
if search or movie:
    st.markdown("---")
    st.markdown(f"## 🎥 Because you liked **{movie}**")
    st.markdown("")
    
    recs, scores = recommend(movie, similarity, indices, df, n_recs)
    
    for i, (idx, row) in enumerate(recs.iterrows(), 1):
        with st.container():
            st.markdown(f"""
                <div class="movie-card">
                    <h3>{i}. {row['original_title']}</h3>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([5, 1, 1])
            
            with col1:
                overview = row['overview'][:250] + "..." if len(row['overview']) > 250 else row['overview']
                st.write(overview or "No description available")
            
            with col2:
                st.metric("⭐ Rating", f"{row['vote_average']:.1f}")
            
            with col3:
                year = str(row['release_date']).split('-')[0] if pd.notna(row['release_date']) else "N/A"
                st.metric("📅 Year", year)
            
            st.progress(scores[i-1], text=f"Match: {scores[i-1]*100:.0f}%")
            st.markdown("")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/movie.png")
    st.title("About")
    st.info(f"🎬 **{len(df):,}** movies in database")
    st.markdown("---")
    st.markdown("""
    ### How it works
    1. Select your favorite movie
    2. Choose number of recommendations
    3. Click 'Find Movies'
    4. Enjoy similar movies!
    """)
    st.markdown("---")
    # st.success("Made with ❤️ using Streamlit")