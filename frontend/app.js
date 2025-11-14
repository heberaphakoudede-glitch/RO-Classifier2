async function analyze() {
    const desc = document.getElementById('description').value;
    const resultElement = document.getElementById('result');

    if (!desc.trim()) {
        resultElement.innerText = 'Veuillez entrer une description.';
        return;
    }

    try {
        const response = await fetch('https://TON-BACKEND.onrender.com/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ description: desc })
        });

        if (!response.ok) {
            resultElement.innerText = 'Erreur serveur : ' + response.status;
            return;
        }

        const data = await response.json();
        resultElement.innerText = data.error
            ? 'Erreur : ' + data.error
            : 'Règle d’Or prédite : ' + data.predicted_rule;
    } catch (error) {
        resultElement.innerText = 'Impossible de contacter le serveur.';
        console.error(error);
    }
}
