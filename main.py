from flask import Flask, jsonify
from flask_wtf.csrf import CSRFProtect
from api.ml_logic.model import vgg16_model_load, forest_model_load
from api.apps.routers import apps_router
import warnings
from pydantic_settings import BaseSettings

app = Flask(__name__)
csrf = CSRFProtect(app)

warnings.filterwarnings("ignore", category=UserWarning)

ALLOWED_HOSTS = ["*"]
forest_model = None
vgg16_model = None

def get_vgg16_model():
    global vgg16_model
    if vgg16_model is not None:
        return vgg16_model
    else:
        raise Exception("The model is not loaded yet!")

def get_forest_model():
    global forest_model
    if forest_model is not None:
        return forest_model
    else:
        raise Exception("The model is not loaded yet!")

def startup():
    global vgg16_model
    global forest_model
    vgg16_path = "VGG16_Baby_prod.h5"
    forest_path = "Forest_5_98.pkl"
    vgg16_model = vgg16_model_load(vgg16_path)
    forest_model = forest_model_load(forest_path)

@app.route("/")
def home():
    return jsonify(message="Hello, this is your Flask application!")

if __name__ == "__main__":
    startup()  # Load models on startup
    app.run(host="0.0.0.0", port=5000, debug=True)
