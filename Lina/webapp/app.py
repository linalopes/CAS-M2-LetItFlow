from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import librosa
import numpy as np
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
CORS(app)

# Path to where audio files will be temporarily saved
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Make sure the uploads directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define the model architecture (same as the one trained)
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Define the model parameters
input_size = 13  # MFCC features
hidden_size = 128  # Hidden units
num_layers = 2  # LSTM layers
num_classes = 3  # short, medium, long

# Initialize the model
model = RNNClassifier(input_size, hidden_size, num_layers, num_classes)

# Load model weights
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Put the model into evaluation mode

# Function to extract audio features (MFCC)
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc.T  # Returns features as a string

# Route to classify audio
@app.route('/classify', methods=['POST'])
def classify_audio():
    if 'audio' not in request.files:
        print('No audio file provided')
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    print(f"File {filename} received and saved at {file_path}")

    # Extract audio features (MFCC)
    try:
        features = extract_features(file_path)

        # Transform features into tensor
        features_tensor = torch.tensor(features).unsqueeze(0).float()

        # Make the prediction with the model
        with torch.no_grad():
            output = model(features_tensor)
            _, predicted = torch.max(output, 1)

        # Map prediction to classes
        label_map = {0: 'SHORT', 1: 'MEDIUM', 2: 'LONG'}
        result = label_map[predicted.item()]

        print(f"Prediction: {result}")
        return jsonify({'class': result})

    except Exception as e:
        print(f"Error processing audio: {e}")
        return jsonify({'error': 'An error occurred during processing.'}), 500


if __name__ == '__main__':
    app.run(debug=True)
