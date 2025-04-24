from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the processed data
movies = pd.read_csv('processed_movies.csv')

# Vectorizing the movie tags
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):
    try:
        idx = movies[movies['title'].str.lower() == movie.lower()].index[0]
    except IndexError:
        return []
    distances = similarity[idx]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return movies['title'].iloc[[i[0] for i in movie_list]].tolist()

# Web routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        movie = request.form['movie']
        recs = recommend(movie)
        return render_template('index.html', movie=movie, recommendations=recs)
    return render_template('index.html', recommendations=[])

if __name__ == '__main__':
    app.run(debug=True)
