import pandas as pd
from flask import Flask, jsonify, request
import pickle
import spacy
import numpy as np


# Load Search Model
knn_search_file = open('knn_240leaf.pickle', 'rb')
knn_search = pickle.load(knn_search_file)

# Load Spacy Model
nlp = spacy.load('en_core_web_md')

# Import Data
lines = pd.read_csv('simpsons_script_lines.csv', low_memory=False)
lines = lines.dropna()

print('done')

# function to get a list of lemmas from a string
def get_lemmas(text):
    if(isinstance(text, float)): #return NaN for NaN values
        return np.nan
    else:
        lemmas = []
        doc = nlp(text.lower())
    
        for token in doc:
            if (token.is_stop == False and token.is_punct == False) and token.pos_ != "-PRON-":
                lemmas.append(token.lemma_)
        return lemmas
    
# Get Vectors Function
def get_vectors_of_string(inp_str):
    lemma_list = get_lemmas(inp_str)
    joined = " ".join(lemma_list)
    return nlp(joined).vector

# Find Quotes Function
def find_quotes(inp_str):
    vect = get_vectors_of_string(inp_str)
    closest_quotes = knn_search.kneighbors([vect])
    indices = closest_quotes[1][0].tolist()
    results = [(lines['raw_character_text'].iloc[i], lines['spoken_words'].iloc[i]) for i in indices]
    return results


# Instantiate App
app = Flask(__name__)


# Predictions API
@app.route('/search', methods=['POST'])
def search():
    # Get input
    data = request.form

     # Parse & Transform Data
    search_request = str(data)

    # Make Predictions
    search_results = find_quotes(search_request)

    # Send output back to Browser
    output = {'one': search_results[0],
              'two': search_results[1],
              'three': search_results[2],
              'four': search_results[3],
              'five': search_results[4]}
    
    # Return a json
    return jsonify(results = output)


# Run App
if __name__ == '__main__':
    app.run(debug = True)
