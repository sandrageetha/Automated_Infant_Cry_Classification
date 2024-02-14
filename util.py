import librosa
import numpy as np

def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    aggregated_mfcc = np.mean(mfcc.T, axis=0)
    return aggregated_mfcc
