from flask import Flask, request, jsonify
from extractUrl import FeatureExtraction
import numpy as np
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load models
model_gbc = pickle.load(open("model_gbc.pkl", "rb"))
model_tree = pickle.load(open("model_tree.pkl", "rb"))
model_forest = pickle.load(open("model_forest.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    url = data.get('url')

    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    try:
        features = FeatureExtraction(url).getFeaturesList()
        x = np.array(features).reshape(1, -1)
        
        pred_gbc = model_gbc.predict(x)[0]
        pred_tree = model_tree.predict(x)[0]
        pred_forest = model_forest.predict(x)[0]

        votes = [int(pred_gbc), int(pred_tree), int(pred_forest)]
        final_prediction = 1 if votes.count(1) > votes.count(-1) else -1

        return jsonify({
            'url': url,
            'prediction': final_prediction,
            'votes': votes
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
