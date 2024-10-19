from apiflask import APIFlask
from flask import render_template
from flask_cors import CORS

cors = CORS()

STATIC_FOLDER = "../client/dist"

def check_modal():
    from pathlib import Path
    modal_config_path = Path.home() / '.modal.toml'
    return modal_config_path.exists()

def write_modal(token_id: str, token_secret: str):
    from pathlib import Path
    modal_config_path = Path.home() / '.modal.toml'
    modal_config_path.write_text(f"[default]\ntoken_id = \"{token_id}\"\ntoken_secret = \"{token_secret}\"\n")

def create_app():
    app = APIFlask(
        __name__,
        title="Image-to-Image Translation API",
        static_folder=STATIC_FOLDER,
        template_folder=STATIC_FOLDER,
        static_url_path="",
    )

    app.config.from_pyfile("config.py")

    if not check_modal():
        if not app.config.get("ENV") == "production":
            raise Exception("To run this app, you need to set up Modal. See https://modal.com/docs/guide")
        else:
            token_id = app.config.get("MODAL_TOKEN_ID", "")
            token_secret = app.config.get("MODAL_TOKEN_SECRET", "")
            assert token_id, "MODAL_TOKEN_ID is not set"
            assert token_secret, "MODAL_TOKEN_SECRET is not set"
            write_modal(token_id, token_secret)

    with app.app_context():
        cors.init_app(
            app,
            origins=app.config.get("ALLOWED_DOMAINS", []),
            supports_credentials=True,
        )

        from server.api import api

        app.register_blueprint(api)

        @app.errorhandler(404)
        def _default(_error):
            return render_template("index.html"), 200

    return app
