
import joblib
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app import Base, DataRow, TfidfVectorizer, LogisticRegression, MODEL_PATH, VECT_PATH

DATABASE_URL = "sqlite:///tadila.db"
engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine)

def train_from_db():
    with SessionLocal() as s:
        rows = s.query(DataRow).filter(DataRow.kind=="training").all()
        texts = [r.description or "" for r in rows]
        labels = [r.ro_label or "" for r in rows]
    vect = TfidfVectorizer()
    X = vect.fit_transform(texts)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, labels)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vect, VECT_PATH)
    print("Model & vectorizer saved.")

if __name__ == "__main__":
    train_from_db()
``
