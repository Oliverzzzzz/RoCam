from multiprocessing.connection import Connection
import logging
from cv_process.main import CV_SOCKET_PATH
import threading
from common.ipc import OSDData, PreviewData, RecordingInfo, StopRecording, create_rocam_ipc_server, CVData
import subprocess

logger = logging.getLogger(__name__)


class CVProcessManagement:
    # will return after the cv process is up and fully running
    def __init__(self, cvdata_callback, preview_callback, process_start_callback):
        self._conn: Connection | None = None
        self._cvdata_callback = cvdata_callback
        self._preview_callback = preview_callback
        self._process_start_callback = process_start_callback

        self._ipc_server = create_rocam_ipc_server(CV_SOCKET_PATH)
        threading.Thread(target=self._start_process_loop, daemon=True).start()

    # call this function after cv process crashed
    # will return after the cv process is up and fully running
    def _start_process_loop(self):
        while True:
            # this subprocess will automatically stop when the control process stops
            # because shmsink in the subprocess will stop the pipeline in the subprocess when shmsrc stops
            # thus, no clean up code is required for this subprocess
            # TODO: not the case, does not clean up when the control process quits before connect to shmsink
            p = subprocess.Popen(
                ["python3", "src/main.py", "cv"],
            )

            logger.info("Waiting for CV process to initialize.....")
            self._conn = self._ipc_server.accept()

            self._process_start_callback()
            threading.Thread(target=self._recv_loop, daemon=True).start()

            p.wait()

    def _recv_loop(self):
        while self._conn:
            try:
                data = self._conn.recv()
                if isinstance(data, CVData):
                    self._cvdata_callback(data)
                elif isinstance(data, PreviewData):
                    self._preview_callback(data)
            except EOFError:
                logger.info("CV process disconnected")
                break

    def send_osd_data(self, osd_data: OSDData):
        if self._conn:
            try:
                self._conn.send(osd_data)
            except:
                pass

    def start_recording(self, recording_info: RecordingInfo):
        if self._conn:
            self._conn.send(recording_info)

    def stop_recording(self):
        if self._conn:
            self._conn.send(StopRecording())