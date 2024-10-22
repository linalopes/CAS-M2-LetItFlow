from flask import Flask, request, jsonify
import torch
import librosa
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the trained model
model = torch.load('model.pth')
model.eval()

# Function to extract audio features
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc.T  # Returns the features as a string

@app.route('/classify', methods=['POST'])
def classify_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']
    filename = secure_filename(file.filename)
    file_path = f'./uploads/{filename}'
    file.save(file_path)

    # Extract features from audio
    features = extract_features(file_path)

    # Transform to tensor for model
    features_tensor = torch.tensor(features).unsqueeze(0).float()

    # Make the prediction with the model
    with torch.no_grad():
        output = model(features_tensor)
        _, predicted = torch.max(output, 1)
    
    # Map prediction to classes
    label_map = {0: 'SHORT', 1: 'MEDIUM', 2: 'LONG'}
    result = label_map[predicted.item()]

    return jsonify({'class': result})

if __name__ == '__main__':
    app.run(debug=True)
