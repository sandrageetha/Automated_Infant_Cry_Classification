# routes.py
from flask import Blueprint, render_template

about_blueprint = Blueprint("about", __name__, template_folder="views")

@about_blueprint.route("/presentation")
def get_presentation_page():
    return render_template("presentation.html")

@about_blueprint.route("/about")
def get_about_page():
    return render_template("about.html")
