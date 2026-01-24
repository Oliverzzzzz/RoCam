import logging

from utils import ip4_addresses

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

flask_logger = logging.getLogger('werkzeug')
flask_logger.setLevel(logging.WARN)

logger.info("Starting backend.....")

import os
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from state_management import StateManagement
from pathlib import Path
from flask import send_file


state_management = StateManagement()
app = Flask(__name__)
CORS(app)
FRONTEND_DIR = "../frontend"

if not os.path.isdir(FRONTEND_DIR):
    logger.warning(f"FRONTEND_DIR does not exist.")

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
    
def _json_body():
    return request.get_json(silent=True) or {}

def _empty():
    return jsonify({})

def _is_named_exception(exc: Exception, name: str) -> bool:
    return exc.__class__.__name__ == name

# -------------------------
# MIS recording endpoints
# -------------------------
@app.post("/api/recordings/start")
def recordings_start():
    try:
        payload = state_management.start_recording()
        return jsonify(payload), 200
    except Exception as e:
        # MIS names this "recordingError" when already recording
        if _is_named_exception(e, "recordingError") or "already" in str(e).lower():
            return jsonify({"error": str(e) or "Already recording"}), 409
        raise


@app.post("/api/recordings/stop")
def recordings_stop():
    payload = state_management.stop_recording()
    return jsonify(payload), 200


@app.get("/api/recordings")
def recordings_list():
    # MIS: RecordingListResponse = { recordings: [RecordingInfo...] }
    recordings = state_management.recording_db.list_all_recordings()
    return jsonify({"recordings": recordings}), 200


@app.patch("/api/recordings/<recordingId>")
def recordings_rename(recordingId: str):
    data = _json_body()

    # Accept common keys; MIS calls it "new name"
    new_name = data.get("newName") or data.get("name") or data.get("title")

    if not isinstance(new_name, str) or not new_name.strip():
        return jsonify({"error": "Missing new name"}), 400

    try:
        state_management.recording_db.rename_recording(recordingId, new_name.strip())
        return _empty(), 200
    except Exception as e:
        # MIS: recordingNotFoundError -> 404
        if _is_named_exception(e, "recordingNotFoundError"):
            return jsonify({"error": "Recording not found"}), 404
        raise


@app.delete("/api/recordings/<recordingId>")
def recordings_delete(recordingId: str):
    try:
        state_management.recording_db.delete_recording(recordingId)
        return _empty(), 200
    except Exception as e:
        if _is_named_exception(e, "recordingNotFoundError"):
            return jsonify({"error": "Recording not found"}), 404
        raise


@app.get("/api/recordings/<recordingId>/download")
def recordings_download(recordingId: str):
    try:
        # MIS: state_management.download_recording returns output path to an mp4
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
        raise

logger.info(f"ipv4 addresses: {ip4_addresses()}")