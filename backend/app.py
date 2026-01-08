
import os
import io
import datetime as dt
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import joblib

# --------- SQLAlchemy ----------
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.exc import SQLAlchemyError

# --------- ML ---------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --------- PPT / Excel ----------
from pptx import Presentation
from pptx.util import Inches, Pt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openpyxl import load_workbook, Workbook
from openpyxl.drawing.image import Image as XLImage
from PIL import Image

# -----------------------------------------------------------
# FLASK APP
# -----------------------------------------------------------
app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# -----------------------------------------------------------
# DATABASE
# -----------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///tadila.db")
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class FileRecord(Base):
    __tablename__ = "files"
    id = Column(Integer, primary_key=True)
    kind = Column(String(20))      # 'training' | 'processed'
    filename = Column(String(255))
    stored_name = Column(String(255))
    rows = Column(Integer, default=0)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    lines = relationship("DataRow", back_populates="file", cascade="all, delete-orphan")


class DataRow(Base):
    __tablename__ = "rows"
    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey("files.id"))
    date = Column(String(100))
    sacs_id = Column(String(100))
    description = Column(Text)
    ro_label = Column(String(100))
    kind = Column(String(20))      # training | processed

    file = relationship("FileRecord", back_populates="lines")


Base.metadata.create_all(engine)

# -----------------------------------------------------------
# MODEL + VECTORIZER
# -----------------------------------------------------------
MODEL_PATH = "model.pkl"
VECT_PATH = "vectorizer.pkl"
SEED_CSV = "incidents.csv"

model = None
vectorizer = None

REQUIRED_COLS = ["Date", "SACS ID N°", "Description", "Règles d'or attribué"]

# -----------------------------------------------------------
# RULES (no images inside code — only paths)
# -----------------------------------------------------------
RULES = {
    f"RO{i}": {
        "fr": "...",    # tu pourras remplir les définitions FR
        "en": "...",    # et EN plus tard
        "icon": f"/static/icons/RO{i}.png"
    }
    for i in range(1, 13)
}

# -----------------------------------------------------------
# HELPERS
# -----------------------------------------------------------
def normalize_cols(df):
    df.columns = [c.strip() for c in df.columns]
    return df


def load_or_initialize_model():
    global model, vectorizer

    # If model exists → load it
    if os.path.exists(MODEL_PATH) and os.path.exists(VECT_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECT_PATH)
        return

    # Else → try training from DB
    with SessionLocal() as s:
        rows = s.query(DataRow).filter(DataRow.kind == "training").all()
        if rows:
            texts = [r.description or "" for r in rows]
            labels = [r.ro_label for r in rows]
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(texts)
            model = LogisticRegression(max_iter=2000)
            model.fit(X, labels)

            joblib.dump(model, MODEL_PATH)
            joblib.dump(vectorizer, VECT_PATH)
            return

    # Else → fallback to seed CSV
    if os.path.exists(SEED_CSV):
        df = pd.read_csv(SEED_CSV)
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df["Description"].astype(str))
        model = LogisticRegression(max_iter=2000)
        model.fit(X, df["RO"].astype(str))

        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECT_PATH)
        return

    # Last backup (empty model)
    vectorizer = TfidfVectorizer()
    model = LogisticRegression(max_iter=2000)


load_or_initialize_model()

# -----------------------------------------------------------
# ENDPOINT : GET TEMPLATE
# -----------------------------------------------------------
@app.get("/template")
def download_template():
    output = io.BytesIO()
    wb = Workbook()
    ws = wb.active
    ws.append(REQUIRED_COLS)
    wb.save(output)
    output.seek(0)

    return send_file(
        output,
        download_name="template_TADILA.xlsx",
        as_attachment=True
    )

# -----------------------------------------------------------
# ENDPOINT : LIST RULES
# -----------------------------------------------------------
@app.get("/rules")
def get_rules():
    return jsonify(RULES)

# -----------------------------------------------------------
# ENDPOINT : UPLOAD TRAINING FILE
# -----------------------------------------------------------
@app.post("/upload-training")
def upload_training():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "Aucun fichier fourni"}), 400

    df = pd.read_excel(file)
    df = normalize_cols(df)

    # validate columns
    for col in REQUIRED_COLS:
        if col not in df.columns:
            return jsonify({"error": f"Colonne manquante : {col}"}), 400

    # save metadata
    with SessionLocal() as s:
        fr = FileRecord(
            kind="training",
            filename=file.filename,
            stored_name=file.filename,
            rows=len(df)
        )
        s.add(fr)
        s.commit()

        # insert rows
        for _, row in df.iterrows():
            dr = DataRow(
                file_id=fr.id,
                kind="training",
                date=str(row["Date"]),
                sacs_id=str(row["SACS ID N°"]),
                description=str(row["Description"]),
                ro_label=str(row["Règles d'or attribué"])
            )
            s.add(dr)
        s.commit()

    # re-train model
    with SessionLocal() as s:
        rows = s.query(DataRow).filter(DataRow.kind == "training").all()
        texts = [r.description for r in rows]
        labels = [r.ro_label for r in rows]

    global model, vectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression(max_iter=2000)
    model.fit(X, labels)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECT_PATH)

    return jsonify({"message": "Importation successfully"}), 200

# -----------------------------------------------------------
# ENDPOINT : PROCESS RAW DATA
# -----------------------------------------------------------
@app.post("/process-data")
def process_data():
    global model, vectorizer

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "Aucun fichier fourni"}), 400

    df = pd.read_excel(file)
    df = normalize_cols(df)

    if "Description" not in df.columns:
        return jsonify({"error": "La colonne 'Description' est obligatoire"}), 400

    # predict
    texts = df["Description"].astype(str)
    X = vectorizer.transform(texts)
    preds = model.predict(X)

    # add predictions
    df["Règles d'or attribué"] = preds

    # record file
    with SessionLocal() as s:
        fr = FileRecord(
            kind="processed",
            filename=file.filename,
            stored_name=file.filename,
            rows=len(df)
        )
        s.add(fr)
        s.commit()

        for i, row in df.iterrows():
            dr = DataRow(
                file_id=fr.id,
                kind="processed",
                date=str(row.get("Date", "")),
                sacs_id=str(row.get("SACS ID N°", "")),
                description=str(row.get("Description", "")),
                ro_label=str(row["Règles d'or attribué"])
            )
            s.add(dr)
        s.commit()

    # return dataframe to frontend
    return jsonify(df.to_dict(orient="records"))

# -----------------------------------------------------------
# ENDPOINT : LIST FILES
# -----------------------------------------------------------
@app.get("/files")
def list_files():
    kind = request.args.get("type", None)
    with SessionLocal() as s:
        q = s.query(FileRecord)
        if kind:
            q = q.filter(FileRecord.kind == kind)
        res = [
            {
                "id": f.id,
                "filename": f.filename,
                "kind": f.kind,
                "rows": f.rows,
                "created_at": f.created_at.isoformat()
            }
            for f in q.order_by(FileRecord.created_at.desc()).all()
        ]
    return jsonify(res)

# -----------------------------------------------------------
# ENDPOINT : DELETE FILE
# -----------------------------------------------------------
@app.delete("/files/<int:id>")
def delete_file(id):
    with SessionLocal() as s:
        fr = s.get(FileRecord, id)
        if not fr:
            return jsonify({"error": "Fichier introuvable"}), 404

        s.delete(fr)
        s.commit()

    return jsonify({"message": "Supprimé avec succès"})

# -----------------------------------------------------------
# ENDPOINT : GET STATS
# -----------------------------------------------------------
@app.get("/stats")
def stats():
    with SessionLocal() as s:
        training = s.query(DataRow).filter(DataRow.kind == "training").count()
        processed = s.query(DataRow).filter(DataRow.kind == "processed").count()
    return jsonify({"training_rows": training, "processed_rows": processed})

# -----------------------------------------------------------
# ENDPOINT : EXPORT TO EXCEL WITH ICONS
# -----------------------------------------------------------
@app.get("/export/excel/<int:file_id>")
def export_excel(file_id):
    with SessionLocal() as s:
        fr = s.get(FileRecord, file_id)
        if not fr:
            return jsonify({"error": "Fichier introuvable"}), 404

        rows = fr.lines

    # create workbook
    wb = Workbook()
    ws = wb.active

    ws.append(["Date", "SACS ID N°", "Description", "Règle d'or", "Icône"])

    for r in rows:
        ws.append([r.date, r.sacs_id, r.description, r.ro_label, ""])

    # Insert icons
    for idx, r in enumerate(rows, start=2):
        icon_path = RULES.get(r.ro_label, {}).get("icon")
        if icon_path:
            full_path = "." + icon_path
            if os.path.exists(full_path):
                img = XLImage(full_path)
                img.width = 30
                img.height = 30
                ws.add_image(img, f"E{idx}")

    # Save to memory
    output = io.BytesIO()
    wb.save(output)
