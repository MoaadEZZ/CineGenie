import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Vérification de l'existence du fichier JSON
if not os.path.exists('cleaned_movies.json'):
    raise FileNotFoundError("Le fichier 'cleaned_movies.json' est introuvable. Assurez-vous qu'il est dans le même dossier que ce script.")

# Charger le fichier JSON
movies_df = pd.read_json('cleaned_movies.json', lines=True)

# Supprimer les films sans description
movies_df = movies_df.dropna(subset=['overview'])

# Afficher un aperçu des données
print("Aperçu des données :")
print(movies_df.head())

# Charger le modèle d'embedding
print("Chargement du modèle d'embedding...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Générer les embeddings pour les descriptions des films
print("Génération des embeddings pour les descriptions de films...")
movies_df['embeddings'] = movies_df['overview'].apply(lambda x: embedding_model.encode(x) if isinstance(x, str) else None)
print("Embeddings générés pour chaque description.")

# Charger le modèle GPT et le tokenizer
print("Chargement du modèle GPT...")
gpt_model_name = "gpt2"
gpt_model = GPT2LMHeadModel.from_pretrained(gpt_model_name)
gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

# Fonction pour rechercher des films
def search_movies(query, movies_df, top_n=5):
    if not query.strip():
        print("Aucune requête fournie. Renvoi de films aléatoires.")
        return movies_df.sample(n=top_n)[['names', 'overview', 'genre', 'score']]

    # Créer l'embedding pour la requête utilisateur
    query_embedding = embedding_model.encode(query)

    # Calculer la similarité cosinus avec les embeddings des films
    similarities = cosine_similarity([query_embedding], np.vstack(movies_df['embeddings']))

    # Obtenir les indices des films les plus similaires
    top_indices = np.argsort(similarities[0])[-top_n:][::-1]

    # Retourner les films correspondants
    return movies_df.iloc[top_indices][['names', 'overview', 'genre', 'score']]

# Fonction pour générer une réponse avec GPT
def generate_response_with_gpt(movies, query):
    # Préparer une liste des recommandations
    recommendations = "\n".join(
        [f"{row['names']} - {row['genre']} (Score: {row['score']})" for _, row in movies.iterrows()]
    )

    # Prompt pour GPT
    prompt = f"""L'utilisateur a demandé : "{query}".
Voici des recommandations de films basées sur sa demande :
{recommendations}

Rédigez une réponse engageante pour l'utilisateur expliquant pourquoi ces films pourraient être intéressants.
"""

    # Encodage du prompt pour GPT
    inputs = gpt_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # Générer du texte avec GPT
    outputs = gpt_model.generate(
        inputs.input_ids,
        max_length=200,
        num_return_sequences=1,
        temperature=0.7,
        pad_token_id=gpt_tokenizer.pad_token_id,
    )

    # Décoder la sortie générée
    generated_text = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extraire uniquement la partie générée après le prompt
    return generated_text[len(prompt):].strip()

# Pipeline complet pour la recommandation
def recommend_movies(query, movies_df, top_n=5):
    # Étape 1 : Recherche des films les plus pertinents
    recommended_movies = search_movies(query, movies_df, top_n=top_n)

    # Étape 2 : Génération de réponse avec GPT
    response = generate_response_with_gpt(recommended_movies, query)

    return response, recommended_movies

# Query de l'utilisateur
query = "Je cherche un film de science-fiction avec une histoire captivante."

# Appeler la fonction de recommandation
response, recommended_movies = recommend_movies(query, movies_df)

# Fonction pour afficher les recommandations
def display_recommendations(recommended_movies):
    print("🎬 Recommandations de films :\n")
    for index, movie in recommended_movies.iterrows():
        print(f"📽️ **Titre** : {movie['names']}")
        print(f"   🏷️ Genre : {movie['genre']}")
        print(f"   ⭐ Score : {movie['score']}/100")
        print(f"   📝 Description : {movie['overview']}")
        print("-" * 50)

# Afficher les résultats des recommandations
display_recommendations(recommended_movies)

# Afficher la réponse générée par GPT
print("\n🤖 Réponse générée par GPT :")
print(response)
