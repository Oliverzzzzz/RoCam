import os
import threading

import logging
from common.utils import run_pipeline_and_wait_for_start, save_gst_pipeline_png, set_scheduler_fifo
from common.ipc import (
    RecoverLiveVideo,
    create_rocam_ipc_client,
)
import gi

from cv_process.main import HEIGHT, VIDEO_SOCKET_PATH, WIDTH

gi.require_version("Gst", "1.0")
os.environ["GST_DEBUG_DUMP_DOT_DIR"] = "./"
from gi.repository import GLib, Gst  # pyright: ignore[reportMissingModuleSource]

logger = logging.getLogger("live_video_process")

LIVE_VIDEO_SOCKET_PATH = "/tmp/rocam-live-video"


class LiveVideoProcess:
    _using_fallback = False

    def __init__(self):
        Gst.init(None)

        pipeline_desc = f"""
            input-selector name=selector sync-streams=true sync-mode=1 !

            shmsrc name=shmsrc socket-path={VIDEO_SOCKET_PATH} is-live=true do-timestamp=true !
            video/x-raw,format=RGBA,width=(int){WIDTH},height=(int){HEIGHT},framerate=(fraction)60/1 !
            queue name=shm-queue leaky=1 max-size-buffers=2 !
            selector.sink_0

            videotestsrc name=testsrc is-live=true pattern=smpte !
            video/x-raw,format=RGBA,width=(int){WIDTH},height=(int){HEIGHT},framerate=(fraction)60/1 !
            queue name=test-queue leaky=1 max-size-buffers=2 !
            selector.sink_1
           
            queue leaky=1 max-size-buffers=2 !
            nvvideoconvert !
            nvdrmvideosink name=drm-sink sync=false set-mode=1
        """

        self._pipeline: Gst.Element = Gst.parse_launch(pipeline_desc)

        self._selector: Gst.Element = self._pipeline.get_by_name("selector")  # pyright: ignore[reportAttributeAccessIssue]
        self._shmsrc: Gst.Element = self._pipeline.get_by_name("shmsrc")  # pyright: ignore[reportAttributeAccessIssue]

        # Set initial active pad to shmsrc (sink_0)
        self._shm_pad = self._selector.get_static_pad("sink_0")
        assert self._shm_pad
        self._fallback_pad = self._selector.get_static_pad("sink_1")
        assert self._fallback_pad
        self._selector.set_property("active-pad", self._shm_pad)

        # Block EOS from shmsrc at the selector's sink pad to prevent pipeline shutdown
        self._shm_pad.add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM, self._block_eos_probe, None
        )

        self._loop = GLib.MainLoop()
        bus = self._pipeline.get_bus()
        assert bus
        bus.add_signal_watch()
        bus.connect("message", self._bus_call, self._loop)

        self.pipeline_thread = run_pipeline_and_wait_for_start(self._pipeline, bus, self._loop)

        save_gst_pipeline_png(self._pipeline, "live_video_process_pipeline")

        # Set up IPC
        self._ipc_client = create_rocam_ipc_client(LIVE_VIDEO_SOCKET_PATH)
        logger.info("Connected to IPC server.")
        threading.Thread(target=self._ipc_command_listener, daemon=True).start()

    def _block_eos_probe(self, pad, info, user_data):
        """Block EOS events from shmsrc to prevent pipeline shutdown."""
        event = info.get_event()
        if event.type == Gst.EventType.EOS:
            logger.warning(
                "Blocking EOS from shmsrc, switching to fallback and starting recovery"
            )
            GLib.idle_add(self._switch_to_fallback)
            return Gst.PadProbeReturn.DROP
        return Gst.PadProbeReturn.OK

    def _switch_to_shmsrc(self):
        """Reset shmsrc element state and switch back to it."""
        if not self._using_fallback:
            return False
        self._using_fallback = False

        # Get the queue after shmsrc to flush it
        shm_queue: Gst.Element = self._pipeline.get_by_name("shm-queue")  # pyright: ignore[reportAttributeAccessIssue]

        # Set shmsrc and its queue to NULL to release all buffers
        shm_queue.set_state(Gst.State.NULL)
        self._shmsrc.set_state(Gst.State.NULL)

        # Send flush events through the shmsrc pad to clear any lingering buffers
        self._shm_pad.send_event(Gst.Event.new_flush_start())  # pyright: ignore[reportOptionalMemberAccess]
        self._shm_pad.send_event(Gst.Event.new_flush_stop(True))  # pyright: ignore[reportOptionalMemberAccess]

        # Now bring them back up
        self._shmsrc.set_state(Gst.State.PLAYING)
        shm_queue.set_state(Gst.State.PLAYING)

        logger.info("shmsrc reset complete, switching back")

        self._selector.set_property("active-pad", self._shm_pad)
        return False

    def _switch_to_fallback(self):
        if self._using_fallback:
            return False
        self._using_fallback = True

        logger.warning("Switching to videotestsrc fallback")
        fallback_pad = self._selector.get_static_pad("sink_1")
        self._selector.set_property("active-pad", fallback_pad)
        return False

    def _bus_call(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            logger.info("End-of-stream")
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            logger.warning(f"{err}: {debug}")
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            src_name = message.src.get_name() if message.src else "unknown"

            # Check if error is from shmsrc
            if (
                src_name == "shmsrc"
                or "shmsrc" in src_name
                or "Control socket has closed" in (debug or "")
            ):
                logger.warning(f"shmsrc error detected: {err}")
                GLib.idle_add(self._switch_to_fallback)
                # Don't quit the loop, continue with fallback
            else:
                logger.error(f"{err}: {debug}")
                loop.quit()

        return True

    def _ipc_command_listener(self):
        while True:
            try:
                msg = self._ipc_client.recv()
                if isinstance(msg, RecoverLiveVideo):
                    GLib.idle_add(self._switch_to_shmsrc)
            except EOFError:
                logger.info("IPC connection closed")
                break

def run_live_video_process():
    set_scheduler_fifo(30)
    
    live_video_process = LiveVideoProcess()
    live_video_process.pipeline_thread.join()
