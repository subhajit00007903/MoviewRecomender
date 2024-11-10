import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__)

# Initialize the stemmer
ps = PorterStemmer()

# Load datasets
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = pd.read_csv('tmdb_5000_movies.csv')

# Merge datasets on title
merged = movies.merge(credits, on='title')

# Select relevant columns
merged = merged[['genres', 'movie_id', 'title', 'keywords', 'original_language', 'cast', 'crew', 'overview']]

# Drop rows with missing values
merged.dropna(inplace=True)

# Function to convert stringified list to list of names
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

# Apply the conversion function to genres and keywords
merged['genres'] = merged['genres'].apply(convert)
merged['keywords'] = merged['keywords'].apply(convert)

# Function to keep only the first three cast members
def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

# Apply the conversion function to cast
merged['cast'] = merged['cast'].apply(convert3)

# Function to fetch the director's name
def fetch_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return [i['name']]
    return []

# Apply the fetch_director function to crew
merged['crew'] = merged['crew'].apply(fetch_director)

# Split the overview text into a list of words
merged['overview'] = merged['overview'].apply(lambda x: x.split())

# Remove spaces in the genres, keywords, crew, and overview
merged['genres'] = merged['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
merged['keywords'] = merged['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
merged['crew'] = merged['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
merged['overview'] = merged['overview'].apply(lambda x: [i.replace(" ", "") for i in x])

# Create a tags column by joining all elements into a single string per row
merged['tags'] = merged['overview'] + merged['genres'] + merged['keywords'] + merged['crew'] + merged['cast']
merged['tags'] = merged['tags'].apply(lambda x: " ".join(x))

# Select new dataframe with relevant columns
new_df = merged[['movie_id', 'title', 'tags']]

# Apply lowercase transformation to the 'tags' column using .loc to avoid the warning
new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: x.lower())

# Stem the tags column using the PorterStemmer and .loc to avoid the warning
def stem(text):
    y = [ps.stem(i) for i in text.split()]
    return " ".join(y)

new_df.loc[:, 'tags'] = new_df['tags'].apply(stem)

# Apply CountVectorizer to the 'tags' column
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Calculate cosine similarity between vectors
similarity = cosine_similarity(vectors)

# Function to recommend movies based on cosine similarity
def recommend(movie_title):
    if movie_title not in new_df['title'].values:
        return []

    movie_index = new_df[new_df['title'] == movie_title].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommendations = []
    for i in movies_list:
        recommendations.append(new_df.iloc[i[0]].title)
    
    return recommendations

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        movie_title = request.form.get('movie_title')
        recommendations = recommend(movie_title)
    
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
#commited 2nd time