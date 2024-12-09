from flask import Flask, render_template, request
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from tqdm import tqdm  # For progress bar during embedding generation
import torch  # Import torch for GPU/CPU compatibility

app = Flask(__name__)

# Load or Generate Embeddings
def load_or_generate_embeddings(file_path='movies_with_embeddings.json'):
    if os.path.exists(file_path):
        print("Loading precomputed embeddings...")
        movies_df = pd.read_json(file_path, lines=True)
    else:
        print("Generating embeddings from scratch...")
        # Load the dataset
        movies_df = pd.read_json('cleaned_movies.json', lines=True)
        movies_df = movies_df.dropna(subset=['overview'])  # Clean data
        
        # Generate embeddings in batches
        movies_df['embeddings'] = generate_embeddings_in_batches(movies_df['overview'])
        
        # Save embeddings to a file
        movies_df.to_json(file_path, orient='records', lines=True)
        print(f"Embeddings saved to '{file_path}'.")
    
    return movies_df

# Generate embeddings in batches
def generate_embeddings_in_batches(data, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(data), batch_size), desc="Generating embeddings"):
        batch = data[i:i+batch_size].tolist()  # Get batch of descriptions
        batch_embeddings = embedding_model.encode(batch)
        embeddings.extend(batch_embeddings)
    return embeddings

# Initialize SentenceTransformer with GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Load the movies DataFrame
movies_df = load_or_generate_embeddings()

# Search for movies based on query
def search_movies(query, movies_df, top_n=5):
    query_embedding = embedding_model.encode(query)
    similarities = cosine_similarity([query_embedding], np.vstack(movies_df['embeddings']))
    top_indices = np.argsort(similarities[0])[-top_n:][::-1]
    return movies_df.iloc[top_indices][['names', 'overview', 'genre', 'score']].to_dict(orient='records')

# Flask Routes
@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    if request.method == 'POST':
        query = request.form.get('query')
        if query:
            recommendations = search_movies(query, movies_df)
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
