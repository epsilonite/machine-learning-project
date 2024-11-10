function classifyImage() {
    const file = document.getElementById("file-input").files[0];
    if (!file) return document.getElementById('result').innerText = 'No file selected.';

    const formData = new FormData();
    formData.append('file', file);

    fetch('/classify', { method: 'POST', body: formData })
        .then(response => response.json())
        .then(data => {
            if (data.label && data.confidence !== undefined) {
                document.getElementById('result').innerHTML = `
                    Predicted Label: ${data.label} <br/>
                    Confidence: ${data.confidence.toFixed(2)} <br/>
                `;
            } else {
                throw new Error('Invalid data');
            }
        })
        .catch(error => document.getElementById('result').innerText = 'Error: ' + error.message);
}
