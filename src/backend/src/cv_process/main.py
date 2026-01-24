import os
import time
import pyds
import threading

import logging
from common.utils import save_gst_pipeline_png, set_scheduler_fifo
from common.ipc import create_rocam_ipc_client, BoundingBox, CVData, PreviewData, RecordingInfo, StopRecording

import gi

gi.require_version('Gst', '1.0')
os.environ["GST_DEBUG_DUMP_DOT_DIR"] = "./"
from gi.repository import GLib, Gst  # pyright: ignore[reportMissingModuleSource]

logger = logging.getLogger("cv_process")

WIDTH = 1920
HEIGHT = 1080
VIDEO_SOCKET_PATH = '/tmp/rocam-video'
CV_SOCKET_PATH = '/tmp/rocam-cv'
ipc_client = None
pipeline = None
recording_lock = threading.Lock()


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

    pts_ns = gst_buffer.pts  # presentation timestamp in nanoseconds

    now = time.perf_counter()
    avg_fps = len(_fps_time_list) / (now - _fps_time_list[0])
    _fps_last_time = now
    _fps_time_list.append(now)
    # logger.info(f"FPS: {avg_fps}")
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
        try:
            if bounding_box and bounding_box.conf > 0.2:
                ipc_client.send(CVData(
                    pts_ns=pts_ns,
                    fps=avg_fps,
                    bounding_box=bounding_box.get_rotate_90_deg(),
                ))
            else:
                ipc_client.send(CVData(
                    pts_ns=pts_ns,
                    fps=avg_fps,
                    bounding_box=None,
                ))
        except Exception as e:
            logger.error(f"Failed to send CVData: {e}")
            exit(1)

    return Gst.PadProbeReturn.OK


def preview_sink_callback(sink):
    global ipc_client

    sample = sink.emit("pull-sample")
    if not sample:
        return Gst.FlowReturn.ERROR

    buffer = sample.get_buffer()
    if not buffer:
        return Gst.FlowReturn.ERROR

    pts_ns = buffer.pts

    success, map_info = buffer.map(Gst.MapFlags.READ)
    if not success:
        return Gst.FlowReturn.ERROR

    try:
        if ipc_client:
            try:
                ipc_client.send(PreviewData(
                    pts_ns=pts_ns,
                    frame=bytes(map_info.data)
                ))
            except Exception as e:
                logger.error(f"Failed to send PreviewData: {e}")
                exit(1)
    finally:
        buffer.unmap(map_info)

    return Gst.FlowReturn.OK


def start_recording(video_path: str):
    global pipeline
    with recording_lock:
        if not pipeline:
            return
        
        recording_queue = pipeline.get_by_name("recording-queue")  # pyright: ignore[reportAttributeAccessIssue]
        old_sink = pipeline.get_by_name("recording-sink")  # pyright: ignore[reportAttributeAccessIssue]
        
        if not recording_queue or not old_sink:
            logger.error("Could not find recording-queue or recording-sink")
            return
        
        # Check if already recording (avimux exists means we're recording)
        if pipeline.get_by_name("avimux"):  # pyright: ignore[reportAttributeAccessIssue]
            logger.warning("Already recording")
            return
        
        logger.info(f"Starting recording to {video_path}")
        
        # Remove old fakesink
        recording_queue.unlink(old_sink)
        old_sink.set_state(Gst.State.NULL)
        pipeline.remove(old_sink)  # pyright: ignore[reportAttributeAccessIssue]
        
        # Create and add avimux and filesink
        new_avimux = Gst.ElementFactory.make("avimux", "avimux")
        new_sink = Gst.ElementFactory.make("filesink", "recording-sink")
        assert new_avimux and new_sink
        new_sink.set_property("location", video_path)
        
        pipeline.add(new_avimux)  # pyright: ignore[reportAttributeAccessIssue]
        pipeline.add(new_sink)  # pyright: ignore[reportAttributeAccessIssue]
        recording_queue.link(new_avimux)
        new_avimux.link(new_sink)
        new_avimux.sync_state_with_parent()
        new_sink.sync_state_with_parent()
        
        logger.info("Recording started")


def stop_recording():
    global pipeline
    with recording_lock:
        if not pipeline:
            return
        
        recording_queue = pipeline.get_by_name("recording-queue")  # pyright: ignore[reportAttributeAccessIssue]
        old_avimux = pipeline.get_by_name("avimux")  # pyright: ignore[reportAttributeAccessIssue]
        old_sink = pipeline.get_by_name("recording-sink")  # pyright: ignore[reportAttributeAccessIssue]
        
        if not recording_queue or not old_sink:
            logger.error("Could not find recording-queue or recording-sink")
            return
        
        # Check if not recording (no avimux means not recording)
        if not old_avimux:
            logger.warning("Not currently recording")
            return
        
        logger.info("Stopping recording")
        
        # Unlink and remove avimux + filesink - setting avimux to NULL finalizes the file
        recording_queue.unlink(old_avimux)
        old_avimux.unlink(old_sink)
        old_avimux.set_state(Gst.State.NULL)
        old_sink.set_state(Gst.State.NULL)
        pipeline.remove(old_avimux)  # pyright: ignore[reportAttributeAccessIssue]
        pipeline.remove(old_sink)  # pyright: ignore[reportAttributeAccessIssue]
        
        # Create and add just fakesink (no avimux needed when not recording)
        new_sink = Gst.ElementFactory.make("fakesink", "recording-sink")
        assert new_sink
        
        pipeline.add(new_sink)  # pyright: ignore[reportAttributeAccessIssue]
        recording_queue.link(new_sink)
        new_sink.sync_state_with_parent()
        
        logger.info("Recording stopped")


def ipc_command_listener():
    global ipc_client
    assert ipc_client
    while True:
        try:
            msg = ipc_client.recv()
            if isinstance(msg, RecordingInfo):
                start_recording(msg.video_path)
            elif isinstance(msg, StopRecording):
                stop_recording()
        except EOFError:
            logger.info("IPC connection closed")
            break


def main():
    global pipeline

    Gst.init(None)

    pipeline_desc = f"""
        nvarguscamerasrc sensor-id=0 !
        video/x-raw(memory:NVMM),width=(int){WIDTH},height=(int){HEIGHT},framerate=(fraction)120/1,format=(string)NV12 !
        videorate !
        video/x-raw(memory:NVMM),width=(int){WIDTH},height=(int){HEIGHT},framerate=(fraction)60/1,format=(string)NV12 !
        
        nvvideoconvert !
        mux.sink_0 nvstreammux name=mux width=1920 height=1080 live-source=1 batch-size=1 !
        nvinfer name=infer config-file-path={os.path.dirname(__file__)}/pgie_config.txt !
        
        tee name=t
        
        t. !
        queue max-size-buffers=1 leaky=1 !
        nvvideoconvert !
        video/x-raw,format=RGBA !
        glupload !
        glshader name=shader !
        gldownload !
        video/x-raw,format=RGBA !
        textoverlay name=osd valignment=top halignment=left font-desc="Sans, 12" draw-outline=0 draw-shadow=0 color=0xFFFF0000 !
        video/x-raw,format=RGBA !
        queue max-size-buffers=1 leaky=1 !
        shmsink wait-for-connection=1 socket-path={VIDEO_SOCKET_PATH} shm-size=20000000 buffer-time=50000000

        t. !
        queue leaky=1 max-size-buffers=30 !
        nvvideoconvert !
        nvjpegenc quality=70 !
        queue name=recording-queue !
        fakesink name=recording-sink

        t. !
        queue leaky=1 max-size-buffers=1 !
        nvvideoconvert dest-crop=0:0:{int(WIDTH / 4)}:{int(HEIGHT / 4)} !
        video/x-raw(memory:NVMM),width={int(WIDTH / 4)},height={int(HEIGHT / 4)} !
        videorate !
        video/x-raw(memory:NVMM),framerate=30/1 !
        nvjpegenc quality=70 !
        appsink name=preview-sink emit-signals=true
    """

    # pipeline to convert avi: gst-launch-1.0 filesrc location=a.avi ! avidemux ! nvjpegdec ! nvvidconv ! 'video/x-raw,format=I420' ! x264enc ! h264parse ! mp4mux ! filesink location=output.mp4

    pipeline = Gst.parse_launch(pipeline_desc)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    assert bus
    bus.add_signal_watch()

    def bus_call(bus, message, loop):
        global ipc_client

        t = message.type
        if t == Gst.MessageType.EOS:
            logger.info("End-of-stream\n")
            loop.quit()
        elif t == Gst.MessageType.STATE_CHANGED:
            old, new, pending = message.parse_state_changed()
            if message.src == pipeline and new == Gst.State.PLAYING and not ipc_client:
                assert pipeline
                save_gst_pipeline_png(pipeline, "cv_process_pipeline")
                logger.info("Trying to connect to IPC server...")
                ipc_client = create_rocam_ipc_client(CV_SOCKET_PATH)
                logger.info("Connected to IPC server.")
                # Start command listener thread
                threading.Thread(target=ipc_command_listener, daemon=True).start()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            logger.warning("%s: %s" % (err, debug))
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error("%s: %s" % (err, debug))
            loop.quit()
        return True

    bus.connect("message", bus_call, loop)

    infer = pipeline.get_by_name("infer")  # pyright: ignore[reportAttributeAccessIssue]
    infer_source_pad = infer.get_static_pad("src")
    infer_source_pad.add_probe(Gst.PadProbeType.BUFFER, inference_stop_probe, 0)

    preview_sink = pipeline.get_by_name("preview-sink")  # pyright: ignore[reportAttributeAccessIssue]
    preview_sink.connect("new-sample", preview_sink_callback)

    logger.info("Starting pipeline")
    pipeline.set_state(Gst.State.PLAYING)


    # def test_thread():
    #     time.sleep(1)
    #     start_recording("test.avi")
    #     time.sleep(5)
    #     stop_recording()

    # threading.Thread(target=test_thread, daemon=True).start()

    try:
        loop.run()
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)


def start_cv_process():
    set_scheduler_fifo(40)
    main()
