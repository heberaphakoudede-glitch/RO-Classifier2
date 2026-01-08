
import joblib
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from app import DataRow  # seulement DataRow (pas Base, pas Flask)

# Paths du modèle identiques à app.py
MODEL_PATH = "model.pkl"
VECT_PATH = "vectorizer.pkl"

# Connexion à SQLite
DATABASE_URL = "sqlite:///tadila.db"
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)

def train_from_db():
    print("🔄 Chargement des données d'entraînement depuis la base...")

    try:
        with SessionLocal() as session:
            rows = session.query(DataRow).filter(DataRow.kind == "training").all()

            if not rows:
                print("❌ Aucune donnée d'entraînement trouvée dans la base.")
                return

            texts = [r.description or "" for r in rows]
            labels = [r.ro_label or "" for r in rows]

    except SQLAlchemyError as e:
        print("❌ Erreur SQL :", e)
        return

    print(f"📚 Nombre de lignes d'entraînement : {len(texts)}")

    print("⚙️ Entraînement du modèle TADILA...")
    vect = TfidfVectorizer()
    X = vect.fit_transform(texts)

    model = LogisticRegression(max_iter=2000)
    model.fit(X, labels)

    # Sauvegardes
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vect, VECT_PATH)

    print("✅ Modèle & vecteur enregistrés dans :")
    print("   -", MODEL_PATH)
    print("   -", VECT_PATH)
    print("🎉 Entraînement terminé avec succès.")

if __name__ == "__main__":
    train_from_db()
