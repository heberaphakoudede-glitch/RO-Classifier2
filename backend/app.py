from flask import Flask, request, jsonify
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_FILE = "model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"
MLB_FILE = "mlb.pkl"
DATA_FILE = "incidents.csv"
CLEAN_FILE = "incidents_clean.csv"

# ✅ Fonction pour corriger le format et réentraîner
def retrain_model():
    if not os.path.exists(DATA_FILE):
        return "❌ Fichier incidents.csv introuvable.", False

    # Charger et nettoyer les données
    df = pd.read_csv(DATA_FILE, error_bad_lines=False, quoting=3)
    df = df.iloc[:, :2]
    df.columns = ['description', 'rule']
    df.dropna(subset=['description', 'rule'], inplace=True)
    df.drop_duplicates(inplace=True)
    df['description'] = df['description'].astype(str).str.strip()
    df['rule'] = df['rule'].astype(str).str.strip()
    df['rule'] = df['rule'].apply(lambda x: [r.strip().upper() for r in str(x).split(',')])

    df.to_csv(CLEAN_FILE, index=False)

    # Réentraîner le modèle multi-label
    X = df['description']
    y = df['rule']

    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.multiclass import OneVsRestClassifier

    mlb = MultiLabelBinarizer()
    y_encoded = mlb.fit_transform(y)

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    model.fit(X_vec, y_encoded)

    # Sauvegarde
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_FILE, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MLB_FILE, "wb") as f:
        pickle.dump(mlb, f)

    return f"✅ Réentraînement multi-label terminé avec {len(df)} lignes.", True

# ✅ Vérifier si les fichiers existent, sinon entraîner le modèle
if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE) or not os.path.exists(MLB_FILE):
    print("⚠ Fichiers .pkl manquants. Entraînement du modèle...")
    retrain_model()
else:
    print("✅ Fichiers .pkl trouvés. Chargement du modèle...")

# Charger le modèle, vectorizer et encodeur
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)
with open(VECTORIZER_FILE, "rb") as f:
    vectorizer = pickle.load(f)
with open(MLB_FILE, "rb") as f:
    mlb = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    description = data.get("description")
    if not description:
        return jsonify({"error": "Description manquante"}), 400

    X_vec = vectorizer.transform([description])
    y_pred = model.predict(X_vec)
    rules = mlb.inverse_transform(y_pred)[0]  # Liste des règles
    return jsonify({"predicted_rules": rules})

@app.route("/retrain", methods=["POST"])
def retrain():
    message, status = retrain_model()
    if status:
        return jsonify({"message": message})
    else:
        return jsonify({"error": message}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
