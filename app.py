from flask import Flask, render_template, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialisation de l'application Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

# Charger les données et le modèle
movies_df = pd.read_json('cleaned_movies.json', lines=True)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Générer les embeddings (vous pouvez les sauvegarder au préalable pour éviter de recalculer à chaque fois)
movies_df['embeddings'] = movies_df['overview'].apply(lambda x: embedding_model.encode(x) if isinstance(x, str) else None)

# Fonction de recherche de films
def search_movies(query, movies_df, top_n=5):
    query_embedding = embedding_model.encode(query)
    similarities = cosine_similarity([query_embedding], np.vstack(movies_df['embeddings']))
    top_indices = np.argsort(similarities[0])[-top_n:][::-1]
    return movies_df.iloc[top_indices][['names', 'overview', 'genre', 'score']]

# Route pour la page d'accueil avec le formulaire
@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = None
    if request.method == 'POST':
        # Récupérer la requête de l'utilisateur
        user_query = request.form.get('query')
        if user_query:
            # Effectuer une recherche
            results = search_movies(user_query, movies_df)
            recommendations = results.to_dict(orient='records')  # Convertir en liste de dictionnaires
    return render_template('index.html', recommendations=recommendations)

