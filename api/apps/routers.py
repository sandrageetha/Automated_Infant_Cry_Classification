from flask import Blueprint
from api.apps.pages.home_page import home_endpoint
from api.apps.pages.vgg16 import vgg16_endpoint
from api.apps.pages.forest import forest_endpoint
from api.apps.pages.about_page import about_endpoint

apps_blueprint = Blueprint("apps", __name__)

apps_blueprint.add_url_rule("/", "home_page", home_endpoint, methods=["GET"])
apps_blueprint.add_url_rule("/vgg16", "vgg16_page", vgg16_endpoint, methods=["GET"])
apps_blueprint.add_url_rule("/forest", "forest_page", forest_endpoint, methods=["GET"])
apps_blueprint.add_url_rule("/about", "about_page", about_endpoint, methods=["GET"])
