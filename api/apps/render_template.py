from flask import render_template as flask_render_template
from api.core.constants import TEMPLATE_DIR


def render_template(template_name: str, context: dict):
    return flask_render_template(template_name, **context)
