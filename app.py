from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from random import randrange

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from frontend

# Load your pre-trained model and vectorizer
model = joblib.load("model/fake_news_model.pkl")         # Your ML model
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")         # Your text vectorizer (e.g., TfidfVectorizer)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text.strip():
            return jsonify({'error': 'Empty text'}), 400

        # Preprocess and vectorize
        features = vectorizer.transform([text])
        prediction = model.predict(features)[0]
        
        # Invert the prediction to fix the label mismatch
        prediction = 1 - prediction
        
        return jsonify({'label': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/random', methods=['GET'])
def random():
    try:
        data = pd.read_csv("random_dataset.csv")
        index = randrange(0, len(data)-1, 1)
        return jsonify({
            'text': data.loc[index].text,
            'label': int(data.loc[index].label) if 'label' in data.columns else 0
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)