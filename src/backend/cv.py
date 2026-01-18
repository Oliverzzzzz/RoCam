import os
import time

os.environ["GST_DEBUG_DUMP_DOT_DIR"] = "./"

import threading
from cv_process import ipc
from cv_process.ipc import create_rocam_ipc_server, CVData
from utils import *
import subprocess
import atexit
import signal
import sys

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

sys.modules["ipc"] = ipc
logger = logging.getLogger(__name__)


class CVProcess:
    # will return after the cv process is up and fully running
    def __init__(self, detection_callback):
        self._p = None
        self._conn = None
        self._detection_callback = detection_callback

        self._ipc_server = create_rocam_ipc_server()
        self.start_process()

    # call this function after cv process crashed
    # will return after the cv process is up and fully running
    def start_process(self):
        # this subprocess will automatically stop when the control process stops
        # because shmsink in the subprocess will stop the pipeline in the subprocess when shmsrc stops
        # thus, no clean up code is required for this subprocess
        # TODO: not the case, does not clean up when the control process quits before connect to shmsink
        # self._p = subprocess.Popen(
        #     ["python3", os.path.join(os.path.dirname(__file__), "cv_process", "main.py")],
        #     cwd=os.path.join(os.path.dirname(__file__), "cv_process"),
        # )
        logger.warning("Manually start the CV process now!")

        logger.info("Waiting for CV process to initialize.....")
        self._conn = self._ipc_server.accept()

        threading.Thread(target=self._recv_loop, daemon=True).start()

    def _recv_loop(self):
        while True:
            try:
                data = self._conn.recv()
                if isinstance(data, CVData):
                    # rotate 90 degrees
                    self._detection_callback(data)
            except EOFError:
                logger.info("CV process disconnected")
                break

# should be the same as the values in cv_process/main.py
WIDTH = 1920
HEIGHT = 1080
SOCKET_PATH = '/tmp/rocam-video'

class VideoPipeline:
    def __init__(self, detection_callback):
        self._cv_process = CVProcess(detection_callback)

        threading.Thread(target=self._video_pipeline, daemon=True).start()

    def _video_pipeline(self):
        Gst.init(None)

        # shmsrc name=shmsrc socket-path={SOCKET_PATH} is-live=true do-timestamp=true !
        # videotestsrc name=shmsrc pattern=smpte !
        pipeline_desc = f"""
            shmsrc socket-path={SOCKET_PATH} is-live=true do-timestamp=true !
            video/x-raw,format=RGBA,width={WIDTH},height={HEIGHT},framerate=60/1 !
            fallbackswitch name=switch timeout=500000000
            
            videotestsrc is-live=true pattern=smpte !
            video/x-raw,format=RGBA,width={WIDTH},height={HEIGHT},framerate=60/1 !
            switch.
        
            switch. !
            tee name=t

            t. !
            queue leaky=1 max-size-buffers=30 !
            nvvideoconvert !
            nvjpegenc quality=70 !
            queue !
            avimux !
            filesink location=recording.avi

            t. !
            queue leaky=1 max-size-buffers=1 !
            glupload !
            glshader name=shader !
            gldownload !
            video/x-raw,format=RGBA !
            textoverlay name=osd valignment=top halignment=left font-desc="Sans, 12" draw-outline=0 draw-shadow=0 color=0xFFFF0000 !
            nvvideoconvert !
            nvdrmvideosink name=drm-sink sync=false set-mode=1
            
            t. !
            queue leaky=1 max-size-buffers=1 !
            nvvideoconvert dest-crop=0:0:{int(WIDTH / 4)}:{int(HEIGHT / 4)} !
            video/x-raw(memory:NVMM),width={int(WIDTH / 4)},height={int(HEIGHT / 4)} !
            videorate !
            video/x-raw(memory:NVMM),framerate=30/1 !
            nvjpegenc quality=70 !
            multipartmux boundary=spionisto !
            tcpclientsink port=5001
        """

        self._pipeline = Gst.parse_launch(pipeline_desc)

        shmsrc = self._pipeline.get_by_name("shmsrc")
        shmsrc_source_pad = shmsrc.get_static_pad("src")
        shmsrc_source_pad.add_probe(Gst.PadProbeType.BUFFER, lambda a, b, c: self._shmsrc_probe(a, b, c), 0)

        self._shader = self._pipeline.get_by_name("shader")
        self._shader.set_property('fragment', open("shader.frag").read())
        self._shader.set_property('uniforms',
                              Gst.Structure.new_from_string("uniforms, tx=(float)0.0, ty=(float)0.0, scale=(float)1.0"))

        self._osd = self._pipeline.get_by_name("osd")

        loop = GLib.MainLoop()
        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", lambda a, b, c: self._bus_call(a, b, c), loop)

        logger.info("Starting pipeline")
        self._pipeline.set_state(Gst.State.PLAYING)
        try:
            loop.run()
        except:
            pass

        logger.info("Pipeline stopped")
        self._pipeline.set_state(Gst.State.NULL)

    def _shmsrc_probe(self, pad, info, u_data):
        return Gst.PadProbeReturn.OK


    def _bus_call(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            logger.info("End-of-stream\n")
            loop.quit()
        elif t == Gst.MessageType.STATE_CHANGED:
            old, new, pending = message.parse_state_changed()
            if message.src == self._pipeline and new == Gst.State.PLAYING:
                logger.info("Pipeline is now PLAYING")
                Gst.debug_bin_to_dot_file(self._pipeline, Gst.DebugGraphDetails.ALL, "pipeline")
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            logger.warning("%s: %s\n" % (err, debug))
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            if "Control socket has closed" in debug:
                logger.warning("cv process crash detected")
            else:
                logger.error("%s: %s\n" % (err, debug))
                loop.quit()

        return True
