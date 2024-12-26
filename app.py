import pandas as pd
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import torch
from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS 

# Download NLTK data files (only needed for first-time setup)
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Global variables for model, embeddings, and dataset
model = None
title_embeddings = None
data = None

# Preprocessing function
def preprocess_text(text):
    """
    Cleans and preprocesses the input text:
    - Converts to lowercase.
    - Tokenizes the text.
    - Removes non-alphabetic tokens and stopwords.
    - Lemmatizes tokens to their base forms.
    """
    if not isinstance(text, str):
        return ''
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def load_resources():
    """
    Loads the model, embeddings, and dataset once during app startup.
    """
    global model, title_embeddings, data
    # Load the pre-trained Sentence-BERT model
    model = SentenceTransformer(r'saved_model/sbert_model')

    # Load precomputed title embeddings
    title_embeddings = torch.load(r'saved_model/title_embeddings_v2.pt')

    # Load the cleaned dataset
    data = pd.read_csv('updated500k_arXivDataset.csv')

# Search function
def search_titles(user_query, top_k=5):
    """
    Searches for the most similar titles to the user's query.
    """
    # Check for invalid input
    if not user_query or not isinstance(user_query, str):
        return [("Invalid query. Please provide a valid string.", 0)]

    # Preprocess the query
    processed_query = preprocess_text(user_query)

    # Check if preprocessing results in an empty string
    if not processed_query:
        return [("The query doesn't contain meaningful content after preprocessing.", 0)]

    # Generate embedding for the query
    query_embedding = model.encode(processed_query, convert_to_tensor=True)

    # Compute similarity scores
    similarity_scores = util.cos_sim(query_embedding, title_embeddings)

    # Combine original titles, processed titles, and scores
    combined_results = list(zip(data['title'], data['processed_title'], similarity_scores.squeeze().tolist()))

    # Sort results based on similarity scores
    sorted_results = sorted(combined_results, key=lambda x: x[2], reverse=True)[:top_k]

    # Return only the original title and similarity score
    return [(original_title, score) for original_title, processed_title, score in sorted_results]


# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Renders the search page and handles user queries.
    """
    if request.method == "POST":
        user_query = request.form.get("query")
        results = search_titles(user_query, top_k=5)
        return render_template("index.html", query=user_query, results=results)
    return render_template("index.html", query=None, results=None)

@app.route('/search', methods=['POST'])
def search():
    """
    API Endpoint to search titles based on user query.
    """
    query = request.json.get('query', '')
    results = search_titles(query, top_k=5)
    return jsonify({'results': [{'title': title, 'score': score} for title, score in results]})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    load_resources()  # Load resources (model, embeddings, dataset) once at startup
    app.run(host="0.0.0.0", port=port)
