import logging
import os
from datetime import datetime, timezone
from uuid import uuid4

from utils import ip4_addresses

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

flask_logger = logging.getLogger('werkzeug')
flask_logger.setLevel(logging.WARN)

logger.info("Starting backend.....")

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from state_management import StateManagement

state_management = StateManagement()
app = Flask(__name__)
CORS(app)
FRONTEND_DIR = "../frontend"

if not os.path.isdir(FRONTEND_DIR):
    logger.warning(f"FRONTEND_DIR does not exist.")

recordings = []
active_recording = None


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def recording_title(started_at):
    started_dt = datetime.fromisoformat(started_at)
    return started_dt.strftime("Recording %Y-%m-%d %H:%M")


@app.before_request
def log_request():
    logger.info("request %s %s", request.method, request.path)


@app.after_request
def log_response(response):
    logger.info("response %s %s -> %s", request.method, request.path, response.status_code)
    return response


@app.errorhandler(Exception)
def handle_exception(error):
    if isinstance(error, HTTPException):
        return jsonify({"error": error.description}), error.code

    logger.exception("Unhandled error")
    return jsonify({"error": "Internal server error"}), 500


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


@app.get("/api/recordings")
def list_recordings():
    sorted_recordings = sorted(
        recordings,
        key=lambda recording: recording["startedAt"],
        reverse=True,
    )
    return jsonify({"recordings": sorted_recordings})


@app.post("/api/recordings/start")
def start_recording():
    global active_recording

    if active_recording and active_recording.get("status") == "recording":
        return jsonify(
            {"error": "Already recording", "activeRecording": active_recording}
        ), 409

    started_at = now_iso()
    recording = {
        "id": uuid4().hex,
        "status": "recording",
        "startedAt": started_at,
        "title": recording_title(started_at),
    }
    active_recording = recording
    recordings.append(recording)
    return jsonify({"recording": recording})


@app.post("/api/recordings/stop")
def stop_recording():
    global active_recording

    if not active_recording:
        return jsonify({"error": "Not recording"}), 409

    ended_at = now_iso()
    started_at = datetime.fromisoformat(active_recording["startedAt"])
    ended_dt = datetime.fromisoformat(ended_at)
    duration_sec = int((ended_dt - started_at).total_seconds())

    active_recording.update(
        {
            "status": "stopped",
            "endedAt": ended_at,
            "durationSec": duration_sec,
        }
    )
    recording = active_recording
    active_recording = None
    return jsonify({"recording": recording})


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    try:
        return send_from_directory(FRONTEND_DIR, path)
    except Exception:
        return send_from_directory(FRONTEND_DIR, "index.html")

logger.info(f"ipv4 addresses: {ip4_addresses()}")
