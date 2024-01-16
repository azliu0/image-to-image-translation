from apiflask import APIFlask
from flask_cors import CORS
from flask import Blueprint

cors = CORS()


def create_app():
    app = APIFlask(
        __name__,
        title="Image-to-Image Translation API",
    )

    app.config.from_pyfile("config.py")

    with app.app_context():
        cors.init_app(
            app, origins=app.config.get("ALLOWED_DOMAINS"), supports_credentials=True
        )

        from server.api import api

        app.register_blueprint(api)

    return app
