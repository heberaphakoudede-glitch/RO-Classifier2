import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Charger les données
data = pd.read_csv('incidents.csv')

X = data['description']
y = data['rule']

# Vectorisation
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Entraînement du modèle
model = LogisticRegression()
model.fit(X_vec, y)

# Sauvegarde du modèle et du vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Modèle entraîné et sauvegardé.")
