from multiprocessing.connection import Connection
import logging
import threading
from common.ipc import RecoverLiveVideo, create_rocam_ipc_server
import subprocess

from live_video_process.main import LIVE_VIDEO_SOCKET_PATH

logger = logging.getLogger(__name__)

# TODO: this is buggy, if both cv and live video process crashes the same time, they will likely not recover
class LiveVideoProcessManagement:
    def __init__(self):
        self._conn: Connection | None = None
        self._ipc_server = create_rocam_ipc_server(LIVE_VIDEO_SOCKET_PATH)

    # call this function after cv process crashed
    # will return after the cv process is up and fully running
    def _start_process_loop(self):
        while True:
            p = subprocess.Popen(
                ["python3", "src/main.py", "live-video"],
            )

            logger.info("Waiting for live video process to initialize.....")
            self._conn = self._ipc_server.accept()

            p.wait()

    def on_cv_process_start(self):
        if self._conn:
            self._conn.send(RecoverLiveVideo())
        else:
            threading.Thread(target=self._start_process_loop, daemon=True).start()
