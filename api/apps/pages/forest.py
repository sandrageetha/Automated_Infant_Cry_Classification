import numpy as np
from flask import Flask, render_template, request
from api.ml_logic.preprpcessings import extract_mfcc
from .services import forms, get_audio_data, random_pics
from main import get_forest_model
import time
from api.core.constants import CLASS_LABELS

app = Flask(__name__)

def predict(audio_features, model):
    features_flatten = audio_features.flatten()
    repeat_times = 1200 // len(features_flatten) + 1
    extended_features = np.tile(features_flatten, repeat_times)[:1200]
    extended_features = np.expand_dims(extended_features, axis=0)
    return model.predict(extended_features)

@app.route("/")
def get_forest_page():
    return render_template("forest.html")

@app.route("/", methods=["POST"])
def forest_predict():
    start = time.perf_counter()

    form = forms.FileUploadForm(request)
    form.load_data()

    if not form.file_is_valid():
        return render_template("forest.html", errors=form.errors)

    audio_base64, audio_content_b, y_clean, _, spectrogram_base64 = get_audio_data(form.file)

    mfccs = extract_mfcc(audio_content_b)
    prediction = predict(mfccs, get_forest_model())
    prediction_label = CLASS_LABELS.get(prediction[0])

    print(time.perf_counter() - start)

    return render_template("forest.html", {
        "msg": "File Loaded",
        "audio_base64": audio_base64,
        "spectrogram": spectrogram_base64,
        "filename": form.file.filename,
        "prediction_label": prediction_label,
        "random_image": random_pics(prediction_label)
    })

if __name__ == "__main__":
    app.run(debug=True)
