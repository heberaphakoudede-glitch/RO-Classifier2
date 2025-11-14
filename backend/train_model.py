import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# ✅ Étape 1 : Charger et corriger le fichier incidents.csv
print("🔍 Vérification et correction du fichier incidents.csv...")
df = pd.read_csv('incidents.csv', error_bad_lines=False, quoting=3)

# Garder uniquement les deux colonnes
df = df.iloc[:, :2]
df.columns = ['description', 'rule']

# Supprimer les lignes vides et doublons
df.dropna(subset=['description', 'rule'], inplace=True)
df.drop_duplicates(inplace=True)

# Nettoyer les espaces et uniformiser les règles
df['description'] = df['description'].astype(str).str.strip()
df['rule'] = df['rule'].astype(str).str.strip().str.upper()

# Sauvegarder le fichier corrigé
df.to_csv('incidents_clean.csv', index=False)
print(f"✅ Fichier corrigé avec {len(df)} lignes.")

# ✅ Étape 2 : Entraînement du modèle
print("⚙ Entraînement du modèle...")
X = df['description']
y = df['rule']

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

# ✅ Étape 3 : Sauvegarde du modèle et du vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Modèle entraîné et sauvegardé avec succès.")
