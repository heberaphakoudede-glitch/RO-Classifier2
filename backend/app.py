from flask import Flask, request, jsonify
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from flask_cors import CORS  # ✅ Ajout pour CORS

app = Flask(__name__)
CORS(app)  # ✅ Active CORS pour toutes les routes

MODEL_FILE = "model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"
DATA_FILE = "incidents.csv"
CLEAN_FILE = "incidents_clean.csv"

# ✅ Fonction pour corriger le format et réentraîner
def retrain_model():
    if not os.path.exists(DATA_FILE):
        return "❌ Fichier incidents.csv introuvable.", False

    # Charger et nettoyer les données
    df = pd.read_csv(DATA_FILE)
    df = df.iloc[:, :2]  # garder les deux premières colonnes
    df.columns = ['description', 'rule']
    df.dropna(subset=['description', 'rule'], inplace=True)
    df.drop_duplicates(inplace=True)
    df['description'] = df['description'].astype(str).str.strip()
    df['rule'] = df['rule'].astype(str).str.strip().str.upper()

    # Sauvegarder le fichier nettoyé
    df.to_csv(CLEAN_FILE, index=False)

    # Réentraîner le modèle
    X = df['description']
    y = df['rule']
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)
    model = LogisticRegression()
    model.fit(X_vec, y)

    # Sauvegarder le modèle et le vectorizer
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_FILE, "wb") as f:
        pickle.dump(vectorizer, f)

    return f"✅ Réentraînement terminé avec {len(df)} lignes.", True

# ✅ Vérifier si les fichiers existent, sinon entraîner le modèle
if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
    print("⚠ Fichiers .pkl manquants. Entraînement du modèle...")
    retrain_model()
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

# ✅ Nouvelle route pour réentraîner
@app.route("/retrain", methods=["POST"])
def retrain():
    message, status = retrain_model()
    if status:
        return jsonify({"message": message})
    else:
        return jsonify({"error": message}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
