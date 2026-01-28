import os
import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List


class RecordingError(Exception):
    pass

class RecordingNotFoundError(Exception):
    pass

@dataclass(frozen=True)
class RecordingInfo:
    id: str
    name: str
    start_time: Optional[str]          # None per MIS
    duration_seconds: Optional[int]    # None per MIS
    video_path: str
    log_path: str

class recording_database:
    def __init__(self, base_path: str) -> None:
        self.base_path = base_path

    def allocate_recording(self) -> RecordingInfo:
        base = Path(self.base_path)

        try:
            base.mkdir(parents=True, exist_ok=True)

            # Generate a fresh random recording id (avoid collisions with existing folders)
            rec_id = None
            for _ in range(10):
                candidate = uuid.uuid4().hex
                if not (base / candidate).exists():
                    rec_id = candidate
                    break

            # Name format: "Recording {YYYY-MM-DD HH:MM:SS}"
            name = f"Recording {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            rec_dir = base / rec_id
            rec_dir.mkdir(parents=True, exist_ok=False)

            meta_path = rec_dir / "meta.json"
            video_path = rec_dir / "video.avi"
            log_path = rec_dir / "log.txt"

            # Write meta.json with name
            meta_path.write_text(json.dumps({"name": name}), encoding="utf-8")

            # Create empty placeholders (optional but helpful)
            log_path.touch(exist_ok=False)
            video_path.touch(exist_ok=False)

            return RecordingInfo(
                id=rec_id,
                name=name,
                start_time=None,
                duration_seconds=None,
                video_path=str(video_path),
                log_path=str(log_path),
            )

        except Exception as e:
            # best-effort cleanup if partially created
            try:
                if "rec_dir" in locals() and rec_dir.exists():
                    for p in [rec_dir / "meta.json", rec_dir / "log.txt", rec_dir / "video.avi"]:
                        if p.exists():
                            p.unlink()
                    rec_dir.rmdir()
            except Exception:
                pass
            raise RecordingError(f"allocate_recording failed: {e}") from e
    
    def read_log_by_id(self, recording_id: str) -> Optional[List]:
     
        log_file = Path(self.base_path) / recording_id / "log.txt"

        if not log_file.exists():
            return None

        logs: List = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Parse JSON line into dict
                data_dict = json.loads(line)

                osd_obj = data_dict

                logs.append(osd_obj)

        return logs
    def get_recording_by_id(self, recording_id):
        recording_path = self._get_recording_path(recording_id)

        if not os.path.exists(recording_path):
            return None

        if not os.path.exists(self._get_recording_metadata_path(recording_id)):
            return None

        if not os.path.exists(self._get_recording_video_path(recording_id)):
            return None

        if not os.path.exists(self._get_recording_log_path(recording_id)):
            return None

        name = ""
        with open(self._get_recording_metadata_path(recording_id), "r") as f:
            metadata = json.load(f)
            name = metadata.get("name", "")
        with open(self._get_recording_log_path(recording_id), "r") as f:
            # assuming a list of json objects in the log file
            logs = json.load(f)

            first_log = logs[0] if logs else {}
            start_time = first_log.get("timestamp_ms", "")
            last_log = logs[-1] if logs else {}
            end_time = last_log.get("timestamp_ms", "")

            duration_seconds = (int(end_time) - int(start_time)) / 1000 if start_time and end_time else 0

        return RecordingInfo(
            id=recording_id,
            name=name,
            start_time=start_time,
            duration_seconds=duration_seconds,
            video_path=self._get_recording_video_path(recording_id),
            log_path=self._get_recording_log_path(recording_id),
        )


    def list_all_recordings(self):
        recordings = []
        recording_folders = os.listdir(self.base_path)

        for folder_name in recording_folders:
            recording_info = self.get_recording_by_id(folder_name)

            if recording_info:
                recordings.append(recording_info)

        return recordings

    def rename_recording(self, recording_id, new_name):
        recording_metadata_path = self._get_recording_metadata_path(recording_id)

        if not os.path.exists(recording_metadata_path):
            raise RecordingNotFoundError(f"Metadata for recording with ID {recording_id} not found.")

        with open(recording_metadata_path, "w") as f:
            metadata = json.loads(f.read())
            metadata["name"] = new_name
            f.write(json.dumps(metadata))

    def delete_recording(self, recording_id):
        recording_path = self._get_recording_path(recording_id)

        if not os.path.exists(recording_path):
            raise RecordingNotFoundError(f"Recording with ID {recording_id} not found.")

        shutil.rmtree(recording_path)

    # private
    def _get_recording_path(self, recording_id):
        return os.path.join(self.base_path, recording_id)

    def _get_recording_metadata_path(self, recording_id):
        return os.path.join(self.base_path, recording_id + "/meta.json")

    def _get_recording_video_path(self, recording_id):
        return os.path.join(self.base_path, recording_id + "/video.avi")

    def _get_recording_log_path(self, recording_id):
        return os.path.join(self.base_path, recording_id + "/log.txt")

