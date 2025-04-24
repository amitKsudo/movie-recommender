import pandas as pd
import ast
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def clean_data():
    movies = pd.read_csv('movies.csv')
    credits = pd.read_csv('credits.csv')

    # Merge on title
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'genres', 'overview', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)

    # Convert stringified lists to python lists
    def convert(obj):
        return [i['name'] for i in ast.literal_eval(obj)]

    def convert_cast(obj):
        return [i['name'] for i in ast.literal_eval(obj)[:3]]

    def get_director(obj):
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                return [i['name']]
        return []

    # Apply functions
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert_cast)
    movies['crew'] = movies['crew'].apply(get_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    # Remove spaces
    for col in ['genres', 'keywords', 'cast', 'crew']:
        movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])

    # Create final tags column
    movies['tags'] = movies['overview'] + movies['keywords'] + movies['genres'] + movies['cast'] + movies['crew']
    new_df = movies[['movie_id', 'title', 'tags']]

    # Convert list to string
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

    # Stemming
    def stem(text):
        return " ".join([ps.stem(word) for word in text.split()])

    new_df['tags'] = new_df['tags'].apply(stem)

    # Save cleaned data
    new_df.to_csv('processed_movies.csv', index=False)
    print("✅ Processed and saved to 'processed_movies.csv'.")

if __name__ == "__main__":
    clean_data()
