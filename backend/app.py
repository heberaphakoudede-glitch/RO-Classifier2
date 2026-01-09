
import os
import io
import uuid
import datetime as dt

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

import pandas as pd
import joblib

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.exc import SQLAlchemyError

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from pptx import Presentation
from pptx.util import Inches, Pt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

# -----------------------------------------------------------
# FLASK (TADILA)
# -----------------------------------------------------------
app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app, resources={r"/*": {"origins": "*"}})

TOOL_NAME = "TADILA"
TOOL_SIGLE = "TDL"

# -----------------------------------------------------------
# STORAGE / DB
# -----------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///tadila.db")
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class FileRecord(Base):
    __tablename__ = "files"
    id = Column(Integer, primary_key=True)
    kind = Column(String(20))  # "training" | "processed"
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
    ro_label = Column(String(100))      # label réel (pour training) OU label attendu si connu
    predicted_ro = Column(String(100))  # prédiction TADILA (pour processed)
    kind = Column(String(20))           # "training" | "processed"

    file = relationship("FileRecord", back_populates="lines")

Base.metadata.create_all(engine)

# -----------------------------------------------------------
# MODELE (PKL / CSV)
# -----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECT_PATH  = os.path.join(BASE_DIR, "vectorizer.pkl")
SEED_CSV   = os.path.join(BASE_DIR, "incidents.csv")

model = None
vectorizer = None

REQUIRED_COLS = ["Date", "SACS ID N°", "Description", "Règles d'or attribué"]

# -----------------------------------------------------------
# REGLES D'OR + ICONES (placer vos PNG dans /static/icons/)
# -----------------------------------------------------------
RULES = {
    "RO1":  {"fr":"Situations à haut risque","en":"High-risk situations","icon":"/static/icons/RO1.png"},
    "RO2":  {"fr":"Circulation","en":"Traffic","icon":"/static/icons/RO2.png"},
    "RO3":  {"fr":"Gestes & Outils","en":"Body mechanics & tools","icon":"/static/icons/RO3.png"},
    "RO4":  {"fr":"EPI","en":"PPE","icon":"/static/icons/RO4.png"},
    "RO5":  {"fr":"Permis de travail","en":"Work permits","icon":"/static/icons/RO5.png"},
    "RO6":  {"fr":"Opérations de levage","en":"Lifting operations","icon":"/static/icons/RO6.png"},
    "RO7":  {"fr":"Systèmes sous énergie","en":"Powered systems","icon":"/static/icons/RO7.png"},
    "RO8":  {"fr":"Espaces confinés","en":"Confined spaces","icon":"/static/icons/RO8.png"},
    "RO9":  {"fr":"Excavation","en":"Excavation","icon":"/static/icons/RO9.png"},
    "RO10": {"fr":"Travail en hauteur","en":"Work at height","icon":"/static/icons/RO10.png"},
    "RO11": {"fr":"Travaux par point chaud","en":"Hot work","icon":"/static/icons/RO11.png"},
    "RO12": {"fr":"Ligne de tir","en":"Line of fire","icon":"/static/icons/RO12.png"},
}

# -----------------------------------------------------------
# HELPERS
# -----------------------------------------------------------
def normalize_cols(df):
    df.columns = [str(c).strip() for c in df.columns]
    return df

def pick_col(df, candidates):
    norm_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in norm_map:
            return norm_map[key]
    return None

def find_required_cols(df):
    df = normalize_cols(df)
    date_col = pick_col(df, ["date", "Date"])
    sacs_col = pick_col(df, ["sacs id n°", "sacs id", "SACS ID N°"])
    desc_col = pick_col(df, ["description", "texte", "text", "Description"])
    ro_col   = pick_col(df, ["règles d'or attribué", "ro", "label", "Règles d'or attribué"])
    return date_col, sacs_col, desc_col, ro_col

def ensure_model_ready_or_error():
    if model is None or vectorizer is None or not hasattr(vectorizer, "vocabulary_"):
        raise RuntimeError("Le modèle n'est pas encore entraîné. Importez d'abord un fichier d'entraînement.")

def save_model(mdl, vect):
    global model, vectorizer
    model, vectorizer = mdl, vect
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECT_PATH)

def train_from_rows(texts, labels):
    vect = TfidfVectorizer()
    X = vect.fit_transform(texts)
    mdl = LogisticRegression(max_iter=2000).fit(X, labels)
    save_model(mdl, vect)

# -----------------------------------------------------------
# INIT MODEL
# -----------------------------------------------------------
def load_or_initialize_model():
    global model, vectorizer

    # 1) PKL déjà présents
    if os.path.exists(MODEL_PATH) and os.path.exists(VECT_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECT_PATH)
        return

    # 2) Données d'entraînement en DB
    with SessionLocal() as s:
        rows = s.query(DataRow).filter(DataRow.kind == "training").all()
        if rows:
            texts = [r.description or "" for r in rows]
            labels = [r.ro_label or "" for r in rows]
            train_from_rows(texts, labels)
            return

    # 3) Seed CSV
    if os.path.exists(SEED_CSV):
        df = pd.read_csv(SEED_CSV)
        df = normalize_cols(df)
        _, _, desc_col, ro_col = find_required_cols(df)
        if desc_col and ro_col:
            texts = df[desc_col].astype(str).fillna("")
            labels = df[ro_col].astype(str).fillna("")
            train_from_rows(texts, labels)
            return

    # 4) Fallback (vide)
    vectorizer = TfidfVectorizer()
    model = LogisticRegression(max_iter=2000)

load_or_initialize_model()

# -----------------------------------------------------------
# ROUTES DE BASE
# -----------------------------------------------------------
@app.get("/")
def home():
    return jsonify({"tool": TOOL_NAME, "sigle": TOOL_SIGLE, "status": "ok", "message": f"{TOOL_NAME} API running"})

@app.get("/rules")
def get_rules():
    return jsonify(RULES)

@app.get("/template")
def download_template():
    # Modèle Excel (colonnes obligatoires)
    output = io.BytesIO()
    wb = Workbook()
    ws = wb.active
    ws.title = "Modèle TADILA"
    ws.append(["Date", "SACS ID N°", "Description", "Règles d'or attribué"])
    wb.save(output)
    output.seek(0)
    return send_file(output, download_name="template_TADILA.xlsx", as_attachment=True)

# -----------------------------------------------------------
# UPLOAD ENTRAINEMENT
# -----------------------------------------------------------
@app.post("/upload-training")
def upload_training():
    """
    Import Excel (xlsx) avec colonnes obligatoires.
    Stocke en DB (kind='training'), réentraîne, met à jour les PKL et le modèle en mémoire.
    """
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "Aucun fichier fourni"}), 400

    try:
        df = pd.read_excel(file)  # nécessite openpyxl
    except Exception as e:
        return jsonify({"error": f"Lecture Excel impossible: {e}"}), 400

    df = normalize_cols(df)
    date_col, sacs_col, desc_col, ro_col = find_required_cols(df)
    if not (date_col and sacs_col and desc_col and ro_col):
        return jsonify({
            "error": "Colonnes requises manquantes",
            "obligatoires": REQUIRED_COLS,
            "trouvees": list(df.columns)
        }), 400

    # Enregistrer le fichier (métadonnées)
    stored_name = f"training_{uuid.uuid4().hex}.xlsx"
    inserted = 0
    try:
        with SessionLocal() as s:
            f = FileRecord(kind="training", filename=getattr(file, "filename", "upload.xlsx"), stored_name=stored_name)
            s.add(f)
            s.flush()
            for _, row in df.iterrows():
                s.add(DataRow(
                    file_id=f.id,
                    date=str(row.get(date_col, "")),
                    sacs_id=str(row.get(sacs_col, "")),
                    description=str(row.get(desc_col, "")),
                    ro_label=str(row.get(ro_col, "")),
                    kind="training"
                ))
                inserted += 1
            f.rows = inserted
            s.commit()
    except SQLAlchemyError as e:
        return jsonify({"error": f"Insertion DB échouée: {e}"}), 500

    # Réentraîner
    try:
        with SessionLocal() as s:
            rows = s.query(DataRow).filter(DataRow.kind == "training").all()
            texts  = [r.description or "" for r in rows]
            labels = [r.ro_label or "" for r in rows]
        train_from_rows(texts, labels)
    except Exception as e:
        return jsonify({"error": f"Réentraînement échoué: {e}"}), 500

    return jsonify({"status": "ok", "message": "Importation successfully", "inserted_rows": inserted})

# -----------------------------------------------------------
# TRAITEMENT (analyse d'un Excel erroné)
# -----------------------------------------------------------
@app.post("/process-data")
def process_data():
    """
    Import d'un Excel 'source' à analyser (prédire la RO).
    Retourne file_id (job), et stocke en DB (kind='processed').
    """
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "Aucun fichier fourni"}), 400

    try:
        df = pd.read_excel(file)
    except Exception as e:
        return jsonify({"error": f"Lecture Excel impossible: {e}"}), 400

    df = normalize_cols(df)
    date_col, sacs_col, desc_col, ro_col = find_required_cols(df)
    if not (date_col and sacs_col and desc_col):
        return jsonify({
            "error": "Colonnes requises manquantes",
            "obligatoires_min": ["Date", "SACS ID N°", "Description"],
            "trouvees": list(df.columns)
        }), 400

    try:
        ensure_model_ready_or_error()
        X = vectorizer.transform(df[desc_col].astype(str).fillna(""))
        preds = model.predict(X)
    except Exception as e:
        return jsonify({"error": f"Analyse impossible: {e}"}), 500

    # Sauvegarde en DB
    stored_name = f"processed_{uuid.uuid4().hex}.xlsx"
    inserted = 0
    file_id = None
    try:
        with SessionLocal() as s:
            f = FileRecord(kind="processed", filename=getattr(file, "filename", "source.xlsx"), stored_name=stored_name)
            s.add(f)
            s.flush()
            file_id = f.id
            for i, row in df.iterrows():
                s.add(DataRow(
                    file_id=f.id,
                    date=str(row.get(date_col, "")),
                    sacs_id=str(row.get(sacs_col, "")),
                    description=str(row.get(desc_col, "")),
                    ro_label=str(row.get(ro_col, "")) if ro_col in df.columns else "",
                    predicted_ro=str(preds[i]),
                    kind="processed"
                ))
                inserted += 1
            f.rows = inserted
            s.commit()
    except SQLAlchemyError as e:
        return jsonify({"error": f"Sauvegarde DB échouée: {e}"}), 500

    return jsonify({"status": "ok", "file_id": file_id, "inserted_rows": inserted})

# -----------------------------------------------------------
# EXPORT EXCEL (avec icônes)
# -----------------------------------------------------------
@app.get("/export/excel/<int:file_id>")
def export_excel(file_id):
    """
    Export du fichier 'processed' vers Excel avec icônes RO insérées.
    """
    with SessionLocal() as s:
        f = s.query(FileRecord).filter(FileRecord.id == file_id, FileRecord.kind == "processed").first()
        if not f:
            return jsonify({"error": "Fichier traité introuvable"}), 404
        rows = s.query(DataRow).filter(DataRow.file_id == file_id).all()

    wb = Workbook()
    ws = wb.active
    ws.title = "Analysé"

    headers = ["Date", "SACS ID N°", "Description", "RO (attendu)", "RO (prédit)", "Icône"]
    ws.append(headers)

    # style en-têtes
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="009543")  # vert Congo
    thin = Side(style="thin", color="BBBBBB")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for col in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=col)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = border

    # lignes + icônes
    r = 2
    for line in rows:
        ws.append([
            line.date or "",
            line.sacs_id or "",
            line.description or "",
            line.ro_label or "",
            line.predicted_ro or "",
            ""  # placeholder Icône
        ])
        ro = (line.predicted_ro or "").strip()
        icon_rel = RULES.get(ro, {}).get("icon")
        if icon_rel:
            icon_abs = os.path.join(BASE_DIR, icon_rel.lstrip("/"))
            if os.path.exists(icon_abs):
                img = XLImage(icon_abs)
                img.width, img.height = 20, 20  # petite icône
                ws.add_image(img, f"F{r}")      # colonne 6 (Icône)
        r += 1

    # ajuster largeur
    ws.column_dimensions["A"].width = 14
    ws.column_dimensions["B"].width = 16
