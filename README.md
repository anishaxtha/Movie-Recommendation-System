# 🎬 Movie Recommendation System

A sleek and interactive **Movie Recommendation System** built with **Streamlit**, **Python**, and **TMDB API**.  
Discover movies similar to your favorites with detailed information including posters, genres, cast, crew, and ratings.

---

## 🌟 Features

- **Search your favorite movie:** Type a movie name to get recommendations.
- **Movie details:** Display poster, genres, cast, crew, release year, rating, and overview.
- **Recommendations:** Shows top similar movies based on movie overview text.
- **TMDB Integration:** Fetches real movie posters dynamically from **The Movie Database (TMDB) API**.
- **Beautiful UI:** Custom dark theme with movie cards for a modern look.

---

## 🛠️ Technologies Used

- Python 3.x
- Streamlit – For interactive web app UI
- Pandas – For data manipulation
- Scikit-learn – For TF-IDF and cosine similarity calculations
- TMDB API – For movie posters and metadata

---

## 📂 Project Structure

MovieRecommendationSystem/
├── main.py # Main Streamlit app
├── movies.csv # Movies dataset
├── credits.csv # Movie credits dataset
├── .env # API key for TMDB (ignored in Git)
├── requirements.txt # Project dependencies
└── README.md # Project documentation

---

## ⚡ Installation & Setup

1. **Clone the repository**

```bash
git clone https://github.com/anishaxtha/Movie-Recommendation-System
cd MovieRecommendationSystem/


```

2. Create a virtual environment (optional but recommended)

```
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows
```

3. Install dependencies

`pip install -r requirements.txt`

4. Set up TMDB API key

- Create a .env file in the project root:
  `TMDB_API_KEY=your_tmdb_api_key_here`

5. Run the app
   `streamlit run main.py`
