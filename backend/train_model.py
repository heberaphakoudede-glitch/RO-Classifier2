
# backend/train_model.py

import os
import sys
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Chemins sûrs (écrire dans le dossier backend)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECT_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
SEED_CSV = os.path.join(BASE_DIR, "incidents.csv")

# Colonnes attendues (on gère plusieurs noms possibles comme dans app.py)
DESC_CANDS = ["Description", "description", "texte", "text", "Texte"]
RO_CANDS   = ["RO", "ro", "label", "Règles d'or attribué"]

def pick_col(df, candidates):
    norm_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in norm_map:
            return norm_map[key]
    return None

def train_from_csv():
    print("🔄 Chargement des données depuis incidents.csv ...")
    if not os.path.exists(SEED_CSV):
        print(f"❌ Fichier seed introuvable : {SEED_CSV}")
        sys.exit(1)

    # Lecture robuste (UTF-8 par défaut)
    try:
        df = pd.read_csv(SEED_CSV)
    except UnicodeDecodeError:
        df = pd.read_csv(SEED_CSV, encoding="latin-1")

    # Nettoyage colonnes
    df.columns = [str(c).strip() for c in df.columns]

    desc_col = pick_col(df, DESC_CANDS)
    ro_col   = pick_col(df, RO_CANDS)

    if not desc_col or not ro_col:
        print("❌ Colonnes introuvables. Requis (au moins) : Description & RO/label")
        print(f"Colonnes présentes : {list(df.columns)}")
        sys.exit(1)

    texts  = df[desc_col].astype(str).fillna("")
    labels = df[ro_col].astype(str).fillna("")

    if len(texts) == 0:
        print("❌ Aucune ligne dans incidents.csv")
        sys.exit(1)

    print(f"📚 Lignes d'entraînement : {len(texts)}")
    print("⚙️ Entraînement TF-IDF + LogisticRegression ...")

    vect = TfidfVectorizer()
    X = vect.fit_transform(texts)

    model = LogisticRegression(max_iter=2000)
    model.fit(X, labels)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vect, VECT_PATH)

    print("✅ Fichiers créés :")
    print("   -", MODEL_PATH)
    print("   -", VECT_PATH)
    print("🎉 Entraînement terminé.")

if __name__ == "__main__":
    train_from_csv()
