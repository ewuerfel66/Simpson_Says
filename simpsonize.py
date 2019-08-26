import pandas as pd
from flask import Flask, jsonify, request
import pickle


# Pickle thing in
my_model = pickle.load(open('.pkl', 'rb'))


# Instantiate App
app = Flask(__name__)


# Predictions API
@app.route('/search', methods=['POST'])
def predict():
    # Get input
    data = request.get_json(force=True)

     # Parse & Transform Data
    predict_request = []

    predict_request = np.array(predict_request).reshape(1, -1)

    # Make Predictions
    y_pred = my_model.predict(predict_request)

    # Send output back to Browser
    output = {'one': y_pred[0],
              'two': y_pred[1],
              'three': y_pred[2],
              'four': y_pred[3],
              'five': y_pred[4]}
    return jsonify(results = output)


# Run App
if __name__ == '__main__':
    app.run(debug = True)