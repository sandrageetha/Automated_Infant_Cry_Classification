from flask import Flask, request, render_template
import librosa
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load models
with open('model/knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)
with open('model/svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)
with open('model/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Load audio file
    audio_file = request.files['audio']
    audio, sr = librosa.load(audio_file, sr=None)
    
    # Extract features (e.g., MFCC)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    
    # Feature scaling
    scaler = StandardScaler()
    mfcc_scaled = scaler.fit_transform(mfcc_mean.reshape(1, -1))
    
    # Prediction
    prediction = model.predict(mfcc_scaled)
    
    return render_template('index.html', prediction=prediction[0])


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            features = extract_features(file)
            # Example prediction (customize based on your application logic)
            prediction = knn_model.predict([features])
            return render_template('index.html', prediction=prediction[0])
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)





if __name__ == '__main__':
    app.run(debug=True)


