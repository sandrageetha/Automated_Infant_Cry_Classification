import time
import numpy as np
from keras.utils import img_to_array
from flask import Flask, render_template, request
from api.ml_logic.preprpcessings import get_spectrogram
from .services import forms, get_audio_data, random_pics
from main import get_vgg16_model
from api.core.constants import CLASSES

app = Flask(__name__)

def get_spectrogram_array(y_clean):
    spectrogram = get_spectrogram(y_clean)
    spectrogram = spectrogram.resize((224, 224))
    spectrogram_array = img_to_array(spectrogram)
    return np.expand_dims(spectrogram_array, axis=0)

def predict_and_sort(predictions):
    prediction_percentages = {label: round(p * 100, 2) for label, p in zip(CLASSES, predictions)}
    return dict(sorted(prediction_percentages.items(), key=lambda item: item[1], reverse=True))

@app.route("/")
def get_vgg16_page():
    return render_template("vgg16.html")

@app.route("/", methods=["POST"])
def vgg16_predict():
    start = time.perf_counter()

    form = forms.FileUploadForm(request)
    form.load_data()

    if not form.file_is_valid():
        return render_template("vgg16.html", errors=form.errors)

    audio_base64, audio_content_b, y_clean, _, spectrogram_base64 = get_audio_data(form.file)

    prediction = get_vgg16_model().predict(get_spectrogram_array(y_clean))
    sorted_predictions = predict_and_sort(prediction[0])

    print(time.perf_counter() - start)

    return render_template("vgg16.html", {
        "msg": "File Loaded",
        "spectrogram": spectrogram_base64,
        "audio_base64": audio_base64,
        "filename": form.file.filename,
        "prediction_percentages": sorted_predictions,
        "random_image": random_pics("different_cry")
    })

if __name__ == "__main__":
    app.run(debug=True)
