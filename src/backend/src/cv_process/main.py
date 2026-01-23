import os

from common.utils import set_scheduler_fifo

os.environ["GST_DEBUG_DUMP_DOT_DIR"] = "./"

import gi

from common.ipc import create_rocam_ipc_client, BoundingBox, CVData

gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
import time
import pyds

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("cv_process")

WIDTH = 1920
HEIGHT = 1080
SOCKET_PATH = '/tmp/rocam-video'
ipc_client = None

def bus_call(bus, message, loop):
    global pipeline
    global ipc_client

    t = message.type
    if t == Gst.MessageType.EOS:
        logger.info("End-of-stream\n")
        loop.quit()
    elif t == Gst.MessageType.STATE_CHANGED:
        old, new, pending = message.parse_state_changed()
        if message.src == pipeline and new == Gst.State.PLAYING:
            logger.info("Pipeline is now PLAYING")
            Gst.debug_bin_to_dot_file(pipeline, Gst.DebugGraphDetails.ALL, "pipeline")
            logger.info("Trying to connect to IPC server...")
            ipc_client = create_rocam_ipc_client()
            logger.info("Connected to IPC server.")
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        logger.warning("%s: %s\n" % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        logger.error("%s: %s\n" % (err, debug))
        loop.quit()
    return True


_fps_last_time = time.perf_counter()
_fps_time_list = [0.0]

def inference_stop_probe(pad, info, u_data):
    global _fps_last_time
    global _fps_time_list
    global ipc_client

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    pts_s = gst_buffer.pts  / 1e9  # presentation timestamp in seconds

    now = time.perf_counter()
    avg_fps = len(_fps_time_list) / (now - _fps_time_list[0])
    _fps_last_time = now
    _fps_time_list.append(now)
    logger.info(f"FPS: {avg_fps}")
    if len(_fps_time_list) > 60:
        _fps_time_list.pop(0)

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    bounding_box = None
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            bbox = obj_meta.detector_bbox_info.org_bbox_coords
            if not bounding_box or obj_meta.confidence > bounding_box.conf:
                bounding_box = BoundingBox(
                    conf=obj_meta.confidence,
                    left=bbox.left / WIDTH,
                    top=bbox.top / HEIGHT,
                    width=bbox.width / WIDTH,
                    height=bbox.height / HEIGHT
                )
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    if ipc_client:
        if bounding_box and bounding_box.conf > 0.2:
            ipc_client.send(CVData(
                pts_s=pts_s,
                fps=avg_fps,
                bounding_box=bounding_box.get_rotate_90_deg(),
            ))
        else:
            ipc_client.send(CVData(
                pts_s=pts_s,
                fps=avg_fps,
                bounding_box=None,
            ))

    return Gst.PadProbeReturn.OK


def main():
    global pipeline

    Gst.init(None)

    pipeline_desc = f"""
        nvarguscamerasrc sensor-id=0 !
        video/x-raw(memory:NVMM),width=(int){WIDTH},height=(int){HEIGHT},framerate=(fraction)120/1,format=(string)NV12 !
        videorate !
        video/x-raw(memory:NVMM),width=(int){WIDTH},height=(int){HEIGHT},framerate=(fraction)60/1,format=(string)NV12 !
        tee name=t

        t. !
        nvvideoconvert !
        mux.sink_0 nvstreammux name=mux width=1920 height=1080 live-source=1 batch-size=1 !
        nvinfer name=infer config-file-path={os.path.join(os.path.dirname(__file__), "pgie_config.txt")} !
        fakesink sync=false
        
        t. !
        queue max-size-buffers=1 leaky=1 !
        nvvideoconvert !
        video/x-raw,format=RGBA !
        shmsink wait-for-connection=1 socket-path={SOCKET_PATH} shm-size=20000000 buffer-time=50000000
    """

    pipeline = Gst.parse_launch(pipeline_desc)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    infer = pipeline.get_by_name("infer")
    infer_source_pad = infer.get_static_pad("src")
    infer_source_pad.add_probe(Gst.PadProbeType.BUFFER, inference_stop_probe, 0)

    logger.info("Starting pipeline")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)


def start_cv_process():
    set_scheduler_fifo(40)
    main()
