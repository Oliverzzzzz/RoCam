import logging
import os

from common.utils import set_scheduler_other, ip4_addresses
from control_process.state_management import StateManagement
from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
from pathlib import Path


logger = logging.getLogger(__name__)
logging.getLogger("werkzeug").setLevel(logging.WARN)

def _json_body():
    return request.get_json(silent=True) or {}

def _empty():
    return jsonify({})

def _is_named_exception(exc: Exception, name: str) -> bool:
    return exc.__class__.__name__ == name

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

    # -------------------------
    # Recordings endpoints
    # -------------------------
    @app.post("/api/recordings/start")
    def recordings_start():
        try:
            payload = state_management.start_recording()
            return jsonify(payload), 200
        except Exception as e:
            msg = str(e) or "Recording error"
            if _is_named_exception(e, "recordingError") or "already" in msg.lower():
                return jsonify({"error": msg}), 409
            return jsonify({"error": msg}), 500

    @app.post("/api/recordings/stop")
    def recordings_stop():
        try:
            payload = state_management.stop_recording()
            return jsonify(payload), 200
        except Exception as e:
            msg = str(e) or "Recording error"
            # make stop symmetric with start
            if _is_named_exception(e, "recordingError") or "not" in msg.lower() or "no" in msg.lower():
                return jsonify({"error": msg}), 409
            return jsonify({"error": msg}), 500

    @app.get("/api/recordings")
    def recordings_list():
        recordings = state_management.recording_db.list_all_recordings()
        return jsonify({"recordings": recordings}), 200

    @app.get("/api/recordings/<recordingId>")
    def recordings_get(recordingId: str):
        try:
            rec = state_management.recording_db.get_recording(recordingId)
            return jsonify({"recording": rec}), 200
        except Exception as e:
            if _is_named_exception(e, "recordingNotFoundError"):
                return jsonify({"error": "Recording not found"}), 404
            return jsonify({"error": str(e) or "Error"}), 500

    @app.patch("/api/recordings/<recordingId>")
    def recordings_rename(recordingId: str):
        data = _json_body()

        # canonical key: "filename" (keep backward compatible keys)
        new_name = (
            data.get("filename")
            or data.get("newName")
            or data.get("name")
            or data.get("title")
        )

        if not isinstance(new_name, str) or not new_name.strip():
            return jsonify({"error": "Missing filename"}), 400

        try:
            updated = state_management.recording_db.rename_recording(recordingId, new_name.strip())
            # Prefer returning updated recording metadata
            # If rename_recording currently returns None, then fetch it:
            if updated is None:
                updated = state_management.recording_db.get_recording(recordingId)
            return jsonify({"recording": updated}), 200
        except Exception as e:
            if _is_named_exception(e, "recordingNotFoundError"):
                return jsonify({"error": "Recording not found"}), 404
            return jsonify({"error": str(e) or "Error"}), 500

    @app.delete("/api/recordings/<recordingId>")
    def recordings_delete(recordingId: str):
        try:
            state_management.recording_db.delete_recording(recordingId)
            return _empty(), 200
        except Exception as e:
            if _is_named_exception(e, "recordingNotFoundError"):
                return jsonify({"error": "Recording not found"}), 404
            return jsonify({"error": str(e) or "Error"}), 500

    @app.get("/api/recordings/<recordingId>/download")
    def recordings_download(recordingId: str):
        try:
            output_path = state_management.download_recording(recordingId)
            path = Path(output_path)

            if not path.exists():
                return jsonify({"error": "Recording output not found"}), 404

            return send_file(
                path,
                mimetype="video/mp4",
                as_attachment=True,
                download_name=path.name,
            )
        except Exception as e:
            if _is_named_exception(e, "recordingNotFoundError"):
                return jsonify({"error": "Recording not found"}), 404
            return jsonify({"error": str(e) or "Error"}), 500

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_frontend(path):
        try:
            return send_from_directory(FRONTEND_DIR, path)
        except Exception:
            return send_from_directory(FRONTEND_DIR, "index.html")

    app.run(host="0.0.0.0", port=80, debug=False)
