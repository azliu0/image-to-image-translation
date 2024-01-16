from server import create_app
import os

if __name__ == "__main__":
    app = create_app()
    port = app.config["FLASK_RUN_PORT"]
    debug = app.config["DEBUG"]
    app.run(host="127.0.0.1", port=port, debug=debug)
