document.getElementById('uploadForm').addEventListener('submit', function(e) {
    e.preventDefault();
    let fileInput = document.getElementById('fileInput');
    let progressBar = document.getElementById('progressBar');
    let resultDiv = document.getElementById('result');

    if (!fileInput.files[0] || fileInput.files[0].type !== 'text/csv') {
        resultDiv.innerHTML = 'Please upload a CSV file.';
        return;
    }

    progressBar.style.display = 'block';
    resultDiv.innerHTML = '';

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        progressBar.style.display = 'none';
        resultDiv.innerHTML = data.message;
    })
    .catch(error => {
        progressBar.style.display = 'none';
        console.error('Error:', error);
        resultDiv.innerHTML = 'An error occurred while processing your file.';
    });
});
