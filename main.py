from flask import Flask, jsonify
import warnings

app = Flask(__name__)

# It's good practice to configure warnings at the start of your application
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize global variables for models
forest_model = None
vgg16_model = None

def get_vgg16_model():
    """Returns the loaded VGG16 model or raises an exception if not loaded."""
    if vgg16_model is not None:
        return vgg16_model
    else:
        raise Exception("VGG16 model is not loaded yet!")

def get_forest_model():
    """Returns the loaded Forest model or raises an exception if not loaded."""
    if forest_model is not None:
        return forest_model
    else:
        raise Exception("Forest model is not loaded yet!")

def startup():
    """Loads the models from files into global variables."""
    global vgg16_model, forest_model
    vgg16_path = "VGG16_Baby_prod.h5"
    forest_path = "Forest_5_98.pkl"
    # Assuming these functions are implemented correctly in the imported module
    from api.ml_logic.model import vgg16_model_load, forest_model_load
    vgg16_model = vgg16_model_load(vgg16_path)
    forest_model = forest_model_load(forest_path)

@app.route("/")
def home():
    """Defines a simple home route."""
    return jsonify(message="Hello, this is your Flask application!")

# Import routers after the app and functions are defined to avoid circular imports
from api.apps.routers import apps_router
# Register the blueprint
app.register_blueprint(apps_router)

if __name__ == "__main__":
    startup()  # Load models on startup
    app.run(host="0.0.0.0", port=5000, debug=True)
