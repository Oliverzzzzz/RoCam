import logging
import os

from common.utils import set_scheduler_other, ip4_addresses
from control_process.state_management import StateManagement
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS


logger = logging.getLogger(__name__)
logging.getLogger("werkzeug").setLevel(logging.WARN)

def run_api_gateway(state_management: StateManagement):
    set_scheduler_other()

    app = Flask(__name__)
    CORS(app)
    FRONTEND_DIR = "../frontend"

    logger.info(f"ipv4 addresses: {ip4_addresses()}")
    if not os.path.isdir(FRONTEND_DIR):
        logger.warning(f"{FRONTEND_DIR} does not exist.")

    @app.post("/api/status")
    def get_status():
        return jsonify(state_management.status())

    @app.post("/api/manual_move")
    def manual_move():
        data = request.get_json()
        direction = data.get("direction")
        state_management.manual_move(direction)
        return jsonify({})

    @app.post("/api/manual_move_to")
    def manual_move_to():
        data = request.get_json()
        tilt = data.get("tilt")
        pan = data.get("pan")
        state_management.manual_move_to(tilt, pan)
        return jsonify({})

    @app.post("/api/arm")
    def arm():
        state_management.arm()
        return jsonify({})

    @app.post("/api/disarm")
    def disarm():
        state_management.disarm()
        return jsonify({})

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_frontend(path):
        try:
            return send_from_directory(FRONTEND_DIR, path)
        except Exception:
            return send_from_directory(FRONTEND_DIR, "index.html")

    app.run(host="0.0.0.0", port=80, debug=False)
