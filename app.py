import pickle
import yaml

import numpy as np
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from movie_simalirity import MovieEmbeddings

app = Flask(__name__)
with open('genre_clf_predict_config.yaml', 'r') as f:
    genre_config = yaml.safe_load(f)
# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(genre_config['model']['tokenizer'])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained(genre_config['model']['model'])
with open(genre_config['model']['label_encoder'], 'rb') as f:
    label_encoder = pickle.load(f)

RESULT_THRESHOLD = genre_config['model']['result_threshold']
with open('similarity_config.yaml', 'r') as f:
    similarity_config = yaml.safe_load(f)


def post_process(raw_probabilities: np.ndarray) -> str:
    """
    Post-processes raw probabilities to output the top n labels and their probabilities.

    Args:
        raw_probabilities (np.ndarray): The raw probabilities from the model.

    Returns:
        str: The top n labels and their corresponding probabilities.
    """
    top_n_indices = raw_probabilities.argsort()[-genre_config['model']['top_n']:][::-1][:genre_config['model']['top_n']]
    top_n_probs = raw_probabilities[top_n_indices]

    top3_label_names = [label_encoder.classes_[i] for i in top_n_indices]
    result = list(zip(top3_label_names, top_n_probs))
    output = []
    for current_prediction in result:
        if current_prediction[1] >= RESULT_THRESHOLD:
            curr_probability = "%.2f" % (current_prediction[1] * 100)
            output.append(f"{current_prediction[0]} ({curr_probability}%)")
    prediction_output = ', '.join(output)
    return prediction_output


def perform_prediction(text_input: str) -> str:
    """
    Performs prediction on the input text.

    Args:
        text_input (str): The input text.

    Returns:
        str: The predicted labels and their corresponding probabilities.
    """
    inputs = tokenizer(text_input, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.sigmoid(logits).numpy()[0]
    prediction = post_process(probabilities)
    return prediction


@app.route('/predict_genre', methods=['POST'])
def predict_genre():
    """
    Flask endpoint for predicting the genre of a movie.

    Returns:
        str: The predicted genres and their corresponding probabilities.
    """
    text = request.json.get('text')
    prediction = perform_prediction(text)
    return jsonify({'prediction': prediction})


@app.route('/suggest', methods=['POST'])
def suggest_movies():
    """
    Flask endpoint for suggesting similar movies.

    Returns:
        str: Titles of similar movies.
    """
    text = request.json.get('text')
    movie_embeddings = MovieEmbeddings(similarity_config)
    similar_movies = movie_embeddings.get_similar_titles(text, similarity_config['model']['top_n'])
    return jsonify({'prediction': similar_movies})


@app.route('/health', methods=['GET'])
def health_check():
    """
    Flask endpoint for health check.

    Returns:
        str: A simple message to indicate the application is healthy.
    """
    return "Application is healthy", 200


if __name__ == '__main__':
    app.run(host=genre_config['server']['host'], port=genre_config['server']['port'])
