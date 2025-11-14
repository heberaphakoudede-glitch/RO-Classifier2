async function analyze() {
    const desc = document.getElementById('description').value;
    const resultElement = document.getElementById('result');
    const loader = document.getElementById('loader');

    if (!desc.trim()) {
        resultElement.innerText = 'Veuillez entrer une description.';
        return;
    }

    loader.style.display = 'block'; // Affiche le loader
    resultElement.innerText = ''; // Efface le texte précédent

    try {
        const response = await fetch('https://ro-classifier2.onrender.com/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ description: desc })
        });

        const data = await response.json();
        resultElement.innerText = data.error
            ? 'Erreur : ' + data.error
            : '✅ Règle d’Or prédite : ' + data.predicted_rule;
    } catch (error) {
        resultElement.innerText = 'Impossible de contacter le serveur.';
        console.error(error);
    } finally {
        loader.style.display = 'none'; // Cache le loader
    }
}
