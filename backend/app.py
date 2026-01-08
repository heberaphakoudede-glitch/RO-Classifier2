
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
SEED_CSV = "incidents.csv"  # Seed CSV minimal (2 colonnes suffisent: Description + RO)

model = None
vectorizer = None

# Colonnes strictes pour les fichiers d'entraînement
REQUIRED_COLS = ["Date", "SACS ID N°", "Description", "Règles d'or attribué"]

# -----------------------------------------------------------
# RULES (icônes servies via /static/icons, pas d'images dans le code)
# -----------------------------------------------------------
RULES = {
    "RO1":  {"fr":"Situations à haut risque",      "en":"High-risk situations",      "icon":"/static/icons/RO1.png"},
    "RO2":  {"fr":"Circulation",                    "en":"Traffic",                   "icon":"/static/icons/RO2.png"},
    "RO3":  {"fr":"Gestes & Outils",                "en":"Body mechanics & tools",    "icon":"/static/icons/RO3.png"},
    "RO4":  {"fr":"EPI",                            "en":"PPE",                       "icon":"/static/icons/RO4.png"},
    "RO5":  {"fr":"Permis de travail",              "en":"Work permits",              "icon":"/static/icons/RO5.png"},
    "RO6":  {"fr":"Opérations de levage",           "en":"Lifting operations",        "icon":"/static/icons/RO6.png"},
    "RO7":  {"fr":"Systèmes sous énergie",          "en":"Powered systems",           "icon":"/static/icons/RO7.png"},
    "RO8":  {"fr":"Espaces confinés",               "en":"Confined spaces",           "icon":"/static/icons/RO8.png"},
    "RO9":  {"fr":"Excavation",                     "en":"Excavation",                "icon":"/static/icons/RO9.png"},
    "RO10": {"fr":"Travail en hauteur",             "en":"Work at height",            "icon":"/static/icons/RO10.png"},
    "RO11": {"fr":"Travaux par point chaud",        "en":"Hot work",                  "icon":"/static/icons/RO11.png"},
    "RO12": {"fr":"Ligne de tir",                   "en":"Line of fire",              "icon":"/static/icons/RO12.png"},
}

# -----------------------------------------------------------
# HELPERS
# -----------------------------------------------------------
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les noms de colonnes (trim)."""
    df.columns = [str(c).strip() for c in df.columns]
    return df

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Retourne le nom réel de la première colonne existante parmi candidates (insensible à la casse/espaces)."""
    norm_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in norm_map:
            return norm_map[key]
    return None

def ensure_model_ready_or_error():
    """Vérifie que le modèle est prêt; sinon lève une Exception pour renvoyer un message clair au frontend."""
    if model is None or vectorizer is None or not hasattr(vectorizer, "vocabulary_"):
        raise RuntimeError("Le modèle n'est pas encore entraîné. Importez d'abord un fichier d'entraînement.")

# -----------------------------------------------------------
# CHARGEMENT / INITIALISATION DU MODÈLE
# -----------------------------------------------------------
def load_or_initialize_model():
    global model, vectorizer

    # 1) Si un modèle existe déjà → on le charge
    if os.path.exists(MODEL_PATH) and os.path.exists(VECT_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECT_PATH)
        return

    # 2) Sinon → essayer d'entraîner à partir de la base (lignes 'training')
    with SessionLocal() as s:
        rows = s.query(DataRow).filter(DataRow.kind == "training").all()
        if rows:
            texts = [r.description or "" for r in rows]
            labels = [r.ro_label or "" for r in rows]
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(texts)
            model = LogisticRegression(max_iter=2000)
            model.fit(X, labels)

            joblib.dump(model, MODEL_PATH)
            joblib.dump(vectorizer, VECT_PATH)
            return

    # 3) Sinon → fallback seed CSV (tolérant à 2 colonnes: Description + RO)
    if os.path.exists(SEED_CSV):
        df = pd.read_csv(SEED_CSV)
        df = normalize_cols(df)

        desc_col = pick_col(df, ["Description", "description", "Desc", "desc", "Texte", "texte", "Text", "text"])
        ro_col   = pick_col(df, ["Règles d'or attribué", "RO", "ro", "Règle", "Regle", "Label", "label"])

        if desc_col and ro_col:
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(df[desc_col].astype(str))
            model = LogisticRegression(max_iter=2000)
            model.fit(X, df[ro_col].astype(str))

            joblib.dump(model, MODEL_PATH)
            joblib.dump(vectorizer, VECT_PATH)
            return
        # Pas de seed exploitable → on laisse le modèle vide, la première importation training l'entraînera.

    # 4) Dernier recours (modèle vide pour éviter crash; utilisable seulement après /upload-training)
    vectorizer = TfidfVectorizer()
    model = LogisticRegression(max_iter=2000)

# Charger au démarrage
load_or_initialize_model()

# -----------------------------------------------------------
# ENDPOINT : GET TEMPLATE (modèle Excel)
# -----------------------------------------------------------
@app.get("/template")
def download_template():
    output = io.BytesIO()
    wb = Workbook()
    ws = wb.active
    ws.title = "TADILA modèle"
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
# ENDPOINT : UPLOAD TRAINING FILE (strict 4 colonnes)
# -----------------------------------------------------------
@app.post("/upload-training")
def upload_training():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "Aucun fichier fourni"}), 400

    try:
        df = pd.read_excel(file)
    except Exception:
        # Peut-être CSV ?
        file.seek(0)
        df = pd.read_csv(file)

    df = normalize_cols(df)

    # Validation stricte
    for col in REQUIRED_COLS:
        if col not in df.columns:
            return jsonify({"error": f"Colonne manquante : {col}"}), 400

    # Enregistrer méta + lignes en base
    with SessionLocal() as s:
        fr = FileRecord(
            kind="training",
            filename=file.filename,
            stored_name=file.filename,
            rows=len(df)
        )
        s.add(fr)
        s.commit()

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

    # Réentraîner le modèle sur toutes les données training
    with SessionLocal() as s:
        rows = s.query(DataRow).filter(DataRow.kind == "training").all()
        texts = [r.description or "" for r in rows]
        labels = [r.ro_label or "" for r in rows]

    global model, vectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression(max_iter=2000)
    model.fit(X, labels)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECT_PATH)

    # Message demandé par toi
    return jsonify({"message": "Importation successfully"}), 200

# -----------------------------------------------------------
# ENDPOINT : PROCESS RAW DATA (analyse d'un fichier brut)
# -----------------------------------------------------------
@app.post("/process-data")
def process_data():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "Aucun fichier fourni"}), 400

    try:
        df = pd.read_excel(file)
    except Exception:
        file.seek(0)
        df = pd.read_csv(file)

    df = normalize_cols(df)

    if "Description" not in df.columns:
        return jsonify({"error": "La colonne 'Description' est obligatoire"}), 400

    # Vérifier que le modèle est prêt
    try:
        ensure_model_ready_or_error()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400

    # Prédictions
    texts = df["Description"].astype(str)
    X = vectorizer.transform(texts)
    preds = model.predict(X)

    # Ajouter la colonne prédite
    df["Règles d'or attribué"] = preds

    # Enregistrer en base (kind=processed)
    with SessionLocal() as s:
        fr = FileRecord(
            kind="processed",
            filename=file.filename,
            stored_name=file.filename,
            rows=len(df)
        )
        s.add(fr)
        s.commit()

        for _, row in df.iterrows():
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

        file_id = fr.id

    # Retourner les données + l'identifiant du fichier traité
    return jsonify({
        "file_id": file_id,
        "rows": df.to_dict(orient="records")
    })

# -----------------------------------------------------------
# ENDPOINT : LIST FILES
# -----------------------------------------------------------
@app.get("/files")
def list_files():
    kind = request.args.get("type", None)  # training | processed | None
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
# ENDPOINT : GET STATS (compteurs lignes)
# -----------------------------------------------------------
@app.get("/stats")
def stats():
    with SessionLocal() as s:
        training = s.query(DataRow).filter(DataRow.kind == "training").count()
        processed = s.query(DataRow).filter(DataRow.kind == "processed").count()
    return jsonify({"training_rows": training, "processed_rows": processed})

# -----------------------------------------------------------
# ENDPOINT : EXPORT EXCEL AVEC ICÔNES
# -----------------------------------------------------------
@app.get("/export/excel/<int:file_id>")
def export_excel(file_id):
    with SessionLocal() as s:
        fr = s.get(FileRecord, file_id)
        if not fr:
            return jsonify({"error": "Fichier introuvable"}), 404
        rows = fr.lines

    wb = Workbook()
    ws = wb.active
    ws.title = "TADILA Export"
    ws.append(["Date", "SACS ID N°", "Description", "Règle d'or", "Icône"])

    for r in rows:
        ws.append([r.date, r.sacs_id, r.description, r.ro_label, ""])

    # Ajouter les icônes dans la colonne E
    for idx, r in enumerate(rows, start=2):
        icon_path = RULES.get(r.ro_label, {}).get("icon")
        if icon_path:
            full_path = "." + icon_path  # ex: ./static/icons/RO1.png
            if os.path.exists(full_path):
                img = XLImage(full_path)
                img.width = 30
                img.height = 30
                ws.add_image(img, f"E{idx}")

    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    return send_file(output, download_name="TADILA_export.xlsx", as_attachment=True)

# -----------------------------------------------------------
# ENDPOINT : EXPORT PPT (résumé par Règle d'or)
# -----------------------------------------------------------
@app.get("/export/ppt/<int:file_id>")
def export_ppt(file_id):
    with SessionLocal() as s:
        fr = s.get(FileRecord, file_id)
        if not fr:
            return jsonify({"error": "Fichier introuvable"}), 404
        rows = fr.lines

    # Compter les occurrences par RO
    ro_counts = {}
    for r in rows:
        label = r.ro_label or "N/A"
        ro_counts[label] = ro_counts.get(label, 0) + 1

    # Générer une figure (bar chart)
    labels = list(ro_counts.keys())
    values = [ro_counts[k] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color="#009543")  # vert Congo
    ax.set_title("Occurrences par Règle d'or (TADILA)", fontsize=12)
    ax.set_xlabel("Règle d'or", fontsize=10)
    ax.set_ylabel("Occurrences", fontsize=10)
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png")
    plt.close(fig)
    img_buf.seek(0)

    # Créer une présentation PPT
    prs = Presentation()

    # Slide 1 : Titre
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "TADILA — Résumé d'analyse"
    slide.placeholders[1].text = f"Fichier: {fr.filename}\nEntrées: {fr.rows}\nDate: {fr.created_at.strftime('%Y-%m-%d %H:%M')}"

    # Slide 2 : Graphique
    slide2 = prs.slides.add_slide(prs.slide_layouts[5])  # title only
    slide2.shapes.title.text = "Occurrences par Règle d'or"
    pic = slide2.shapes.add_picture(img_buf, Inches(1.0), Inches(1.5), width=Inches(8.0))

    # Slide 3 : Tableau simple (texte)
    slide3 = prs.slides.add_slide(prs.slide_layouts[5])
    slide3.shapes.title.text = "Détail des occurrences"
    left, top, width, height = Inches(0.7), Inches(1.2), Inches(8.0), Inches(4.5)
    textbox = slide3.shapes.add_textbox(left, top, width, height)
    tf = textbox.text_frame
    tf.word_wrap = True
    tf.clear()
    para = tf.paragraphs[0]
    para.text = "RO\tCompteur"
    para.font.size = Pt(12)
    for k in labels:
        p = tf.add_paragraph()
        p.text = f"{k}\t{ro_counts[k]}"
        p.font.size = Pt(12)

    # Exporter en mémoire
    output = io.BytesIO()
    prs.save(output)
    output.seek(0)
    return send_file(output, download_name="TADILA_export.pptx", as_attachment=True)

# -----------------------------------------------------------
# RUN
# -----------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
