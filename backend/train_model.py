import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import pickle

# ✅ Étape 1 : Charger et corriger le fichier incidents.csv
print("🔍 Vérification et correction du fichier incidents.csv...")
df = pd.read_csv('incidents.csv', on_bad_lines='skip', quoting=3)

# Garder uniquement les deux colonnes
df = df.iloc[:, :2]
df.columns = ['description', 'rule']

# Supprimer les lignes vides et doublons
df.dropna(subset=['description', 'rule'], inplace=True)
df.drop_duplicates(inplace=True)

# Nettoyer les espaces
df['description'] = df['description'].astype(str).str.strip()
df['rule'] = df['rule'].astype(str).str.strip()

# ✅ Convertir les règles en liste (multi-label)
df['rule'] = df['rule'].apply(lambda x: [r.strip().upper() for r in str(x).split(',')])

# Sauvegarder le fichier corrigé
df.to_csv('incidents_clean.csv', index=False)
print(f"✅ Fichier corrigé avec {len(df)} lignes.")

# ✅ Étape 2 : Entraînement du modèle multi-label
print("⚙ Entraînement du modèle multi-label...")
X = df['description']
y = df['rule']

# Encoder les règles
mlb = MultiLabelBinarizer()
y_encoded = mlb.fit_transform(y)

# Vectorisation
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Modèle multi-label
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X_vec, y_encoded)

# ✅ Étape 3 : Sauvegarde du modèle, vectorizer et encodeur
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('mlb.pkl', 'wb') as f:
    pickle.dump(mlb, f)

print("✅ Modèle multi-label entraîné et sauvegardé avec succès.")
