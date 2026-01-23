import os
import time
import logging

os.environ["GST_DEBUG_DUMP_DOT_DIR"] = "./"

import threading
from common.ipc import create_rocam_ipc_server, CVData
import subprocess
import atexit
import signal
import sys

import gi

gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

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
        self._p = subprocess.Popen(
            ["python3", "src/main.py", "cv"],
        )
        # logger.warning("Manually start the CV process now!")

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
        self._using_fallback = False
        self._recovery_in_progress = False
        self._selector = None
        self._shmsrc = None
        self._loop = None
        self._lock = threading.Lock()

        threading.Thread(target=self._video_pipeline, daemon=True).start()

    def _video_pipeline(self):
        Gst.init(None)

        pipeline_desc = f"""
            input-selector name=selector sync-streams=true sync-mode=1 !
            tee name=t

            shmsrc name=shmsrc socket-path={SOCKET_PATH} is-live=true do-timestamp=true !
            video/x-raw,format=RGBA,width=(int){WIDTH},height=(int){HEIGHT},framerate=(fraction)60/1 !
            queue name=shm-queue leaky=1 max-size-buffers=2 !
            selector.sink_0

            videotestsrc name=testsrc is-live=true pattern=smpte !
            video/x-raw,format=RGBA,width=(int){WIDTH},height=(int){HEIGHT},framerate=(fraction)60/1 !
            queue name=test-queue leaky=1 max-size-buffers=2 !
            selector.sink_1

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

        self._selector = self._pipeline.get_by_name("selector")
        self._shmsrc = self._pipeline.get_by_name("shmsrc")

        # Set initial active pad to shmsrc (sink_0)
        shm_pad = self._selector.get_static_pad("sink_0")
        self._selector.set_property("active-pad", shm_pad)

        # Block EOS from shmsrc at the selector's sink pad to prevent pipeline shutdown
        shm_sink_pad = self._selector.get_static_pad("sink_0")
        shm_sink_pad.add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM,
            self._block_eos_probe,
            None
        )

        self._shader = self._pipeline.get_by_name("shader")
        self._shader.set_property('fragment', open(os.path.join(os.path.dirname(__file__), "shader.frag")).read())
        self._shader.set_property('uniforms',
                                  Gst.Structure.new_from_string(
                                      "uniforms, tx=(float)0.0, ty=(float)0.0, scale=(float)1.0"))

        self._osd = self._pipeline.get_by_name("osd")

        self._loop = GLib.MainLoop()
        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._bus_call, self._loop)

        logger.info("Starting pipeline")
        self._pipeline.set_state(Gst.State.PLAYING)
        try:
            self._loop.run()
        except:
            pass

        logger.info("Pipeline stopped")
        self._pipeline.set_state(Gst.State.NULL)

    def _block_eos_probe(self, pad, info, user_data):
        """Block EOS events from shmsrc to prevent pipeline shutdown."""
        event = info.get_event()
        if event.type == Gst.EventType.EOS:
            logger.warning("Blocking EOS from shmsrc, switching to fallback and starting recovery")
            GLib.idle_add(self._switch_to_fallback)
            # Start recovery in a separate thread to not block the pipeline
            threading.Thread(target=self._recover_shmsrc, daemon=True).start()
            return Gst.PadProbeReturn.DROP
        return Gst.PadProbeReturn.OK

    def _recover_shmsrc(self):
        """Recovery process: restart CV process and reset shmsrc element."""
        with self._lock:
            if self._recovery_in_progress:
                logger.debug("Recovery already in progress, skipping")
                return
            self._recovery_in_progress = True

        logger.info("Starting CV process recovery...")

        try:
            # This blocks until CV process is ready
            self._cv_process.start_process()

            logger.info("CV process ready, resetting shmsrc element")

            # Reset shmsrc element on the main loop thread
            GLib.idle_add(self._reset_and_switch_to_shmsrc)
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            with self._lock:
                self._recovery_in_progress = False

    def _reset_and_switch_to_shmsrc(self):
        """Reset shmsrc element state and switch back to it."""
        # First, ensure we're on fallback so downstream isn't using shmsrc buffers
        fallback_pad = self._selector.get_static_pad("sink_1")
        self._selector.set_property("active-pad", fallback_pad)

        # Get the queue after shmsrc to flush it
        shm_queue = self._pipeline.get_by_name("shm-queue")

        # Set shmsrc and its queue to NULL to release all buffers
        shm_queue.set_state(Gst.State.NULL)
        self._shmsrc.set_state(Gst.State.NULL)

        # Send flush events through the shmsrc pad to clear any lingering buffers
        shm_pad = self._selector.get_static_pad("sink_0")
        shm_pad.send_event(Gst.Event.new_flush_start())
        shm_pad.send_event(Gst.Event.new_flush_stop(True))

        # Now bring them back up
        self._shmsrc.set_state(Gst.State.PLAYING)
        shm_queue.set_state(Gst.State.PLAYING)

        logger.info("shmsrc reset complete, switching back")

        with self._lock:
            self._using_fallback = False
            self._recovery_in_progress = False

        self._selector.set_property("active-pad", shm_pad)
        return False

    def _switch_to_fallback(self):
        with self._lock:
            if self._using_fallback:
                return False
            self._using_fallback = True

        logger.warning("Switching to videotestsrc fallback")
        fallback_pad = self._selector.get_static_pad("sink_1")
        self._selector.set_property("active-pad", fallback_pad)
        return False

    def switch_source(self, use_fallback: bool):
        """Public method to manually switch sources."""
        if use_fallback:
            GLib.idle_add(self._switch_to_fallback)
        else:
            # Start recovery process in background
            threading.Thread(target=self._recover_shmsrc, daemon=True).start()

    def _bus_call(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            logger.info("End-of-stream")
            loop.quit()
        elif t == Gst.MessageType.STATE_CHANGED:
            old, new, pending = message.parse_state_changed()
            if message.src == self._pipeline and new == Gst.State.PLAYING:
                logger.info("Pipeline is now PLAYING")
                Gst.debug_bin_to_dot_file(self._pipeline, Gst.DebugGraphDetails.ALL, "pipeline")
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            logger.warning(f"{err}: {debug}")
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            src_name = message.src.get_name() if message.src else "unknown"

            # Check if error is from shmsrc
            if src_name == "shmsrc" or "shmsrc" in src_name or "Control socket has closed" in (debug or ""):
                logger.warning(f"shmsrc error detected: {err}")
                GLib.idle_add(self._switch_to_fallback)
                # Start recovery in a separate thread
                threading.Thread(target=self._recover_shmsrc, daemon=True).start()
                # Don't quit the loop, continue with fallback
            else:
                logger.error(f"{err}: {debug}")
                loop.quit()

        return True