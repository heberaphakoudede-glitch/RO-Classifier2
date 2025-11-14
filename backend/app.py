from flask import Flask, request, jsonify
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

MODEL_FILE = "model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"

# Vérifier si les fichiers existent, sinon entraîner le modèle
if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
    print("⚠ Fichiers .pkl manquants. Entraînement du modèle...")
    data = pd.read_csv("incidents.csv")
    X = data["description"]
    y = data["rule"]

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_vec, y)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_FILE, "wb") as f:
        pickle.dump(vectorizer, f)

    print("✅ Modèle entraîné et sauvegardé.")
else:
    print("✅ Fichiers .pkl trouvés. Chargement du modèle...")

# Charger le modèle et le vectorizer
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)
with open(VECTORIZER_FILE, "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    description = data.get("description")
    if not description:
        return jsonify({"error": "Description manquante"}), 400

    X_vec = vectorizer.transform([description])
    prediction = model.predict(X_vec)[0]
    return jsonify({"predicted_rule": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
