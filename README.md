# Movie Recommendation System 🎬

This project is a **Content-Based Movie Recommendation System** built using Python, Pandas, Scikit-Learn, and Natural Language Processing (NLP).  
It recommends movies similar to a given movie by analyzing metadata such as **genres, keywords, cast, crew, and overview**.

---

## 📌 Features
- Uses **content-based filtering** with movie metadata.
- Preprocessing includes:
  - Extracting genres, keywords, cast (top 3), and director.
  - Removing spaces from multi-word tokens.
  - Combining all features into a single **tags column**.
  - Text preprocessing using **Stemming** and **Lowercasing**.
- Vectorization using **CountVectorizer (Bag of Words model)**.
- Computes **cosine similarity** between movies for recommendations.
- Saves preprocessed data and similarity matrix using `pickle`.

---

## 🛠️ Technologies Used
- Python 🐍
- Pandas, NumPy
- NLTK (PorterStemmer)
- Scikit-Learn (CountVectorizer, cosine_similarity)
- Pickle (for model/data storage)

---

## 🚀 How It Works
1. Load movie and credits dataset (`tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`).
2. Merge datasets and select useful features.
3. Preprocess data (clean, tokenize, stem, and normalize text).
4. Create a new `tags` feature containing all movie metadata in one string.
5. Apply `CountVectorizer` to convert text into feature vectors.
6. Compute similarity scores using **cosine similarity**.
7. Recommend top 5 movies similar to the input movie.

---

## 📂 Files
- `movies.pkl` → Preprocessed movie dataset
- `moviedict.pkl` → Dictionary version of the dataset
- `similarity.pkl` → Cosine similarity matrix

---

## 📖 Usage
```python
import pickle

# Load saved files
movies = pickle.load(open('movies.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))

# Recommendation function
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    for i in movies_list:
        print(movies.iloc[i[0]].title)

# Example
recommend('Avatar')
```

---

## 🎯 Example Output
```
Spectre
John Carter
Battle: Los Angeles
Mars Needs Moms
Titan A.E.
```

---

## 📌 Future Improvements
- Add **hybrid recommendation** (content + collaborative filtering).
- Deploy with **Flask/Django** and integrate a **web UI**.
- Improve NLP preprocessing with **lemmatization** and **TF-IDF**.

---


