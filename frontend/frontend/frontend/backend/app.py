from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('description', '')
    if not text.strip():
        return jsonify({'error': 'Description is empty'}), 400
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    return jsonify({'RO': prediction})

if __name__ == '__main__':
    app.run(debug=True)
