from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import os

app = Flask(__name__)
uploads_dir = 'uploads'
os.makedirs(uploads_dir, exist_ok=True)  # Ensure uploads directory exists

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Received a request")
    if 'file' not in request.files:
        print("No file part")
        return jsonify({'message': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return jsonify({'message': 'No selected file'})

    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(uploads_dir, filename)
        file.save(file_path)

        report = detect_fraud(file_path)
        return jsonify({'message': report})
    else:
        print("Invalid file format")
        return jsonify({'message': 'Invalid file format'})

def detect_fraud(file_path):
    try:
        df = pd.read_csv(file_path)
        df['is_fraud'] = df.apply(lambda row: "Suspicious" if row['amount'] > 1000 else "Normal", axis=1)
        fraud_report = df[['transaction_id', 'is_fraud']].to_string(index=False)
        return fraud_report
    except Exception as e:
        print(f"Error during fraud detection: {e}")
        return f"An error occurred during fraud detection: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
