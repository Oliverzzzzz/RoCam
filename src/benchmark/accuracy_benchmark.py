#!/usr/bin/env python3
import os
os.environ.setdefault("GST_DEBUG_DUMP_DOT_DIR", "/tmp")

import argparse
import logging
import re
import sys
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

import pyds
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

logger = logging.getLogger("accuracy_benchmark")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

# Match your DeepStream pipeline streammux size (main.py uses 1920x1080)
WIDTH = 1920
HEIGHT = 1080

# ---------------- image IO ----------------
def _load_image_size(path: Path) -> Tuple[int, int]:
    try:
        import cv2  # type: ignore
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"cv2.imread failed: {path}")
        h, w = img.shape[:2]
        return w, h
    except Exception:
        from PIL import Image  # type: ignore
        with Image.open(path) as im:
            return im.size

def _read_image_bgr(path: Path):
    try:
        import cv2  # type: ignore
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"cv2.imread failed: {path}")
        return img
    except Exception:
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore
        with Image.open(path) as im:
            im = im.convert("RGB")
            arr = np.array(im)[:, :, ::-1].copy()
            return arr

def _write_jpg(path: Path, bgr, quality: int = 95):
    try:
        import cv2  # type: ignore
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    except Exception:
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore
        rgb = bgr[:, :, ::-1]
        im = Image.fromarray(np.asarray(rgb))
        path.parent.mkdir(parents=True, exist_ok=True)
        im.save(path, format="JPEG", quality=quality, subsampling=0)

# ---------------- YAML loader ----------------
def load_data_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
        return yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        out: Dict[str, Any] = {}
        txt = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        key = None
        for line in txt:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if ":" in s and not s.startswith("-"):
                k, v = s.split(":", 1)
                key = k.strip()
                v = v.strip()
                if v.startswith("[") and v.endswith("]"):
                    items = [x.strip().strip("'\"") for x in v[1:-1].split(",") if x.strip()]
                    out[key] = items
                elif v:
                    out[key] = v.strip().strip("'\"")
                else:
                    out[key] = None
            elif s.startswith("-") and key:
                out.setdefault(key, [])
                out[key].append(s[1:].strip().strip("'\""))
        return out

# ---------------- pgie config patch ----------------
# MINIMAL CHANGE: 强制写入 scaling-compute-hw=1 + 可选 model-engine-file
def write_temp_pgie_config(orig_cfg: Path, engine_path: Optional[Path]) -> Path:
    text = orig_cfg.read_text(encoding="utf-8", errors="ignore").splitlines()

    desired: Dict[str, str] = {"scaling-compute-hw": "1"}  # force GPU preproc to avoid VIC random crash
    if engine_path is not None:
        desired["model-engine-file"] = str(engine_path)

    out: List[str] = []
    in_prop = False
    seen = set()

    kv_re = re.compile(r"^\s*([^=]+?)\s*=\s*(.+?)\s*$")

    for line in text:
        s = line.strip()

        if s.startswith("[") and s.endswith("]"):
            in_prop = (s.lower() == "[property]")
            out.append(line)
            continue

        if in_prop:
            m = kv_re.match(line)
            if m:
                k = m.group(1).strip()
                if k in desired:
                    out.append(f"{k}={desired[k]}")
                    seen.add(k)
                    continue

        out.append(line)

    missing = [k for k in desired.keys() if k not in seen]
    if missing:
        final: List[str] = []
        inserted = False
        for line in out:
            final.append(line)
            if (not inserted) and line.strip().lower() == "[property]":
                for k in missing:
                    final.append(f"{k}={desired[k]}")
                inserted = True
        out = final

    tmp = orig_cfg.parent / f".tmp_pgie_config_{int(time.time())}.txt"
    tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
    return tmp

# ---------------- staging + transform ----------------
@dataclass
class StageInfo:
    orig_path: Path
    staged_path: Path
    orig_w: int
    orig_h: int
    sx: float
    sy: float
    pad_x: float
    pad_y: float

def compute_stage_transform(orig_w: int, orig_h: int, target_w: int, target_h: int, mode: str) -> Tuple[float, float, float, float]:
    if mode == "stretch":
        return target_w / orig_w, target_h / orig_h, 0.0, 0.0
    if mode == "letterbox":
        s = min(target_w / orig_w, target_h / orig_h)
        new_w = orig_w * s
        new_h = orig_h * s
        pad_x = (target_w - new_w) / 2.0
        pad_y = (target_h - new_h) / 2.0
        return s, s, pad_x, pad_y
    if mode == "none":
        if orig_w != target_w or orig_h != target_h:
            raise ValueError(f"stage-mode=none requires images already {target_w}x{target_h}, got {orig_w}x{orig_h}")
        return 1.0, 1.0, 0.0, 0.0
    raise ValueError(f"Unknown stage-mode: {mode}")

def stage_images(images_dir: Path, staging_dir: Path, mode: str, limit: Optional[int]) -> Tuple[List[StageInfo], Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    imgs = [p for p in sorted(images_dir.rglob("*")) if p.suffix.lower() in exts and p.is_file()]
    if not imgs:
        raise RuntimeError(f"No images found under {images_dir}")
    if limit is not None:
        imgs = imgs[:limit]

    staging_dir.mkdir(parents=True, exist_ok=True)

    staged_files = sorted(staging_dir.glob("*.jpg"))
    if len(staged_files) == len(imgs):
        ok = True
        for k in [0, len(staged_files)//2, len(staged_files)-1]:
            if k < 0 or k >= len(staged_files):
                continue
            try:
                w, h = _load_image_size(staged_files[k])
                if w != WIDTH or h != HEIGHT:
                    ok = False
                    break
            except Exception:
                ok = False
                break
        if ok:
            logger.info(f"staging reuse: {len(imgs)} already in {staging_dir} (mode={mode})")
            infos: List[StageInfo] = []
            for i, orig in enumerate(imgs):
                ow, oh = _load_image_size(orig)
                sx, sy, px, py = compute_stage_transform(ow, oh, WIDTH, HEIGHT, mode)
                infos.append(StageInfo(orig, staging_dir / f"{i:06d}.jpg", ow, oh, sx, sy, px, py))
            return infos, staging_dir / "%06d.jpg"
        else:
            logger.warning("staging dir exists but size check failed -> rebuilding staging...")

    for old in staging_dir.glob("*"):
        try:
            old.unlink()
        except Exception:
            pass

    infos: List[StageInfo] = []
    n = len(imgs)

    for i, orig in enumerate(imgs):
        if (i + 1) % 200 == 0:
            logger.info(f"staging {i+1}/{n}")

        bgr = _read_image_bgr(orig)
        oh, ow = bgr.shape[:2]
        sx, sy, px, py = compute_stage_transform(ow, oh, WIDTH, HEIGHT, mode)

        if mode == "none":
            out_bgr = bgr
        elif mode == "stretch":
            import cv2  # type: ignore
            out_bgr = cv2.resize(bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        elif mode == "letterbox":
            import cv2  # type: ignore
            new_w = int(round(ow * sx))
            new_h = int(round(oh * sy))
            resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            top = int(round(py))
            left = int(round(px))
            bottom = HEIGHT - top - new_h
            right = WIDTH - left - new_w
            out_bgr = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            raise ValueError(mode)

        staged = staging_dir / f"{i:06d}.jpg"
        _write_jpg(staged, out_bgr, quality=95)
        infos.append(StageInfo(orig, staged, ow, oh, sx, sy, px, py))

    logger.info(f"staging done: {len(infos)} -> {staging_dir} (mode={mode})")
    return infos, staging_dir / "%06d.jpg"

# ---------------- YOLO label -> staged pixel bbox ----------------
def parse_yolo_label_file(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    rows = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 5:
            continue
        c = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:5])
        rows.append((c, x, y, w, h))
    return rows

def yolo_to_staged_xywh(cls: int, x: float, y: float, w: float, h: float, st: StageInfo) -> Optional[Tuple[int, float, float, float, float]]:
    ow, oh = st.orig_w, st.orig_h
    normalized = (0.0 <= x <= 1.5 and 0.0 <= y <= 1.5 and 0.0 <= w <= 1.5 and 0.0 <= h <= 1.5)
    if normalized:
        cx = x * ow
        cy = y * oh
        bw = w * ow
        bh = h * oh
    else:
        cx, cy, bw, bh = x, y, w, h

    left = cx - bw / 2.0
    top = cy - bh / 2.0

    left_s = left * st.sx + st.pad_x
    top_s = top * st.sy + st.pad_y
    bw_s = bw * st.sx
    bh_s = bh * st.sy

    if bw_s <= 0 or bh_s <= 0:
        return None

    if left_s >= WIDTH or top_s >= HEIGHT:
        return None
    if left_s + bw_s <= 0 or top_s + bh_s <= 0:
        return None

    left_s = max(0.0, min(float(WIDTH - 1), left_s))
    top_s = max(0.0, min(float(HEIGHT - 1), top_s))
    bw_s = max(0.0, min(float(WIDTH) - left_s, bw_s))
    bh_s = max(0.0, min(float(HEIGHT) - top_s, bh_s))
    if bw_s < 1e-3 or bh_s < 1e-3:
        return None
    return (cls, left_s, top_s, bw_s, bh_s)

# ---------------- NMS / postprocess ----------------
def _iou_xywh(a: List[float], b: List[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1e-9)

def nms_xywh(dets: List[Dict[str, Any]], iou_thr: float) -> List[Dict[str, Any]]:
    if iou_thr <= 0:
        return dets
    dets = sorted(dets, key=lambda d: float(d["score"]), reverse=True)
    keep: List[Dict[str, Any]] = []
    for d in dets:
        ok = True
        for k in keep:
            if _iou_xywh(d["bbox"], k["bbox"]) >= iou_thr:
                ok = False
                break
        if ok:
            keep.append(d)
    return keep

def postprocess_by_image(
    dets: List[Dict[str, Any]],
    top1: bool,
    topk: int,
    nms_iou: float
) -> List[Dict[str, Any]]:
    by_img: Dict[int, List[Dict[str, Any]]] = {}
    for d in dets:
        by_img.setdefault(int(d["image_id"]), []).append(d)

    out: List[Dict[str, Any]] = []
    for img_id, lst in by_img.items():
        lst = sorted(lst, key=lambda d: float(d["score"]), reverse=True)
        if top1:
            lst = lst[:1]
        elif topk > 0:
            lst = lst[:topk]
        lst = nms_xywh(lst, nms_iou)
        out.extend(lst)
    return out

# ---------------- debug visualization ----------------
def draw_debug_overlays(
    stage_infos: List[StageInfo],
    preds: List[Dict[str, Any]],
    gt_coco: Dict[str, Any],
    out_dir: Path,
    max_n: int
):
    out_dir.mkdir(parents=True, exist_ok=True)
    by_img_pred: Dict[int, List[Dict[str, Any]]] = {}
    for d in preds:
        by_img_pred.setdefault(int(d["image_id"]), []).append(d)

    by_img_gt: Dict[int, List[List[float]]] = {}
    for ann in gt_coco.get("annotations", []):
        by_img_gt.setdefault(int(ann["image_id"]), []).append(list(map(float, ann["bbox"])))

    n = min(max_n, len(stage_infos))
    for i in range(n):
        img_path = stage_infos[i].staged_path
        try:
            bgr = _read_image_bgr(img_path)
        except Exception:
            continue

        try:
            import cv2  # type: ignore
            for bbox in by_img_gt.get(i, []):
                x, y, w, h = bbox
                cv2.rectangle(bgr, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            for d in by_img_pred.get(i, []):
                x, y, w, h = d["bbox"]
                s = float(d["score"])
                cv2.rectangle(bgr, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
                cv2.putText(bgr, f"{s:.2f}", (int(x), max(0, int(y) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(str(out_dir / f"{i:06d}.jpg"), bgr)
        except Exception:
            _write_jpg(out_dir / f"{i:06d}.jpg", bgr, quality=95)

# ---------------- GStreamer globals ----------------
pipeline = None
detections_raw: List[Dict[str, Any]] = []
frame_counter = 0
seq_counter = 0
args_global = None

# MINIMAL CHANGE: 用于判定“崩了但还在算 mAP”的情况
had_error = False
had_error_msg = ""

def bus_call(bus, message, loop):
    global had_error, had_error_msg
    t = message.type
    if t == Gst.MessageType.ERROR:
        err, dbg = message.parse_error()
        had_error = True
        had_error_msg = f"{err}\n{dbg}"
        logger.error(f"GStreamer ERROR: {had_error_msg}")
        loop.quit()
    elif t == Gst.MessageType.EOS:
        sys.stdout.write("End-of-stream\n")
        loop.quit()
    return True

def infer_probe(pad, info, u_data):
    global detections_raw, frame_counter, seq_counter
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        if args_global.image_id_mode == "seq":
            image_id = int(seq_counter)
        else:
            image_id = int(frame_meta.frame_num)

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            score = float(obj.confidence)
            if score >= float(args_global.conf):
                if args_global.bbox_source == "detector":
                    bb = obj.detector_bbox_info.org_bbox_coords
                    left = float(bb.left)
                    top = float(bb.top)
                    w = float(bb.width)
                    h = float(bb.height)
                else:
                    rp = obj.rect_params
                    left = float(rp.left)
                    top = float(rp.top)
                    w = float(rp.width)
                    h = float(rp.height)

                left = max(0.0, min(float(WIDTH - 1), left))
                top = max(0.0, min(float(HEIGHT - 1), top))
                w = max(0.0, min(float(WIDTH) - left, w))
                h = max(0.0, min(float(HEIGHT) - top, h))

                if w > 1e-3 and h > 1e-3:
                    cls_id = int(getattr(obj, "class_id", 0))
                    detections_raw.append({
                        "image_id": image_id,
                        "category_id": cls_id + 1,
                        "bbox": [left, top, w, h],
                        "score": score,
                    })

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        frame_counter += 1
        seq_counter += 1

        if frame_counter % 200 == 0:
            logger.info(f"infer progress: frames={frame_counter}")

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

def build_minimal_pipeline_desc(staged_pattern: Path, count: int, pgie_cfg: Path, fps: int) -> str:
    # MINIMAL CHANGE: nvvideoconvert compute-hw=1 (force GPU path)
    return f"""
        multifilesrc location={staged_pattern} start-index=0 stop-index={count-1} do-timestamp=true !
        image/jpeg,width={WIDTH},height={HEIGHT},framerate={fps}/1 !
        jpegparse !
        jpegdec !
        videoconvert !
        nvvideoconvert compute-hw=1 !
        video/x-raw(memory:NVMM),format=NV12,width={WIDTH},height={HEIGHT},framerate={fps}/1 !
        mux.sink_0

        nvstreammux name=mux width={WIDTH} height={HEIGHT} live-source=0 batch-size=1 !
        nvinfer name=infer config-file-path={pgie_cfg} !
        fakesink sync=false
    """

def main():
    global pipeline, detections_raw, frame_counter, seq_counter, args_global, had_error, had_error_msg

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pgie_config.txt", help="DeepStream pgie config path")
    ap.add_argument("--engine", default=None, help="Force model-engine-file to this engine plan")
    ap.add_argument("--data-yaml", required=True, help="YOLO data.yaml path")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--conf", type=float, default=0.25, help="pre-cluster-threshold aligned (e.g. 0.25)")
    ap.add_argument("--stage-mode", choices=["none", "stretch", "letterbox"], default="letterbox")
    ap.add_argument("--staging-dir", default="/tmp/rocam_bench_fixedjpg")

    ap.add_argument("--bbox-source", choices=["detector", "rect"], default="detector",
                    help="Use detector org bbox (DeepStream-aligned) or rect_params bbox")
    ap.add_argument("--image-id-mode", choices=["seq", "frame"], default="seq",
                    help="seq=0..N-1 in file order; frame=use frame_meta.frame_num")

    ap.add_argument("--top1", action="store_true", help="keep only top-1 det per image before NMS")
    ap.add_argument("--topk", type=int, default=300, help="keep top-k dets per image before NMS (DeepStream topk)")
    ap.add_argument("--nms-iou", type=float, default=0.45, help="NMS IoU threshold (DeepStream nms-iou-threshold)")
    ap.add_argument("--dump-json", default=None, help="dump final detections (after postprocess) as json")
    ap.add_argument("--dump-stage-csv", default=None, help="dump staging transform info as csv")
    ap.add_argument("--debug-vis", type=int, default=0, help="save N debug overlay images to /tmp/rocam_bench_debug")
    args = ap.parse_args()
    args_global = args

    # reset error status each run
    had_error = False
    had_error_msg = ""

    data_yaml = Path(args.data_yaml).expanduser().resolve()
    data = load_data_yaml(data_yaml)

    names = data.get("names", ["rocket"])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]
    if not isinstance(names, list):
        names = ["rocket"]
    logger.info(f"names(from yaml)={names}")

    root = data_yaml.parent
    split_key = args.split

    if args.images_dir:
        images_dir = Path(args.images_dir).expanduser().resolve()
    else:
        split_path = data.get(split_key, None)
        if split_path is None:
            images_dir = (root / "images" / split_key).resolve()
        else:
            images_dir = (root / str(split_path)).expanduser().resolve()
            if (images_dir / "images" / split_key).is_dir():
                images_dir = (images_dir / "images" / split_key).resolve()

    if args.labels_dir:
        labels_dir = Path(args.labels_dir).expanduser().resolve()
    else:
        if len(images_dir.parts) >= 2 and images_dir.parts[-2] == "images":
            labels_dir = Path(*images_dir.parts[:-2], "labels", images_dir.parts[-1]).resolve()
        else:
            labels_dir = (root / "labels" / split_key).resolve()

    logger.info(f"Images dir: {images_dir}")
    logger.info(f"Labels dir: {labels_dir}")

    staging_dir = Path(args.staging_dir).resolve()
    stage_infos, pattern = stage_images(images_dir, staging_dir, args.stage_mode, args.limit)
    count = len(stage_infos)
    logger.info(f"Total images to eval: {count}")

    if args.dump_stage_csv:
        import csv
        p = Path(args.dump_stage_csv).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "orig_path", "orig_w", "orig_h", "sx", "sy", "pad_x", "pad_y", "staged_path"])
            for i, st in enumerate(stage_infos):
                w.writerow([i, str(st.orig_path), st.orig_w, st.orig_h, st.sx, st.sy, st.pad_x, st.pad_y, str(st.staged_path)])
        logger.info(f"Dumped stage csv: {p}")

    categories = [{"id": i + 1, "name": str(n)} for i, n in enumerate(names)]
    coco_gt: Dict[str, Any] = {"images": [], "annotations": [], "categories": categories}
    ann_id = 1

    for i, st in enumerate(stage_infos):
        coco_gt["images"].append({"id": i, "file_name": str(st.orig_path), "width": WIDTH, "height": HEIGHT})
        label_path = labels_dir / f"{st.orig_path.stem}.txt"
        rows = parse_yolo_label_file(label_path)
        for (cls, x, y, w, h) in rows:
            if len(names) == 1:
                cls = 0
            mapped = yolo_to_staged_xywh(cls, x, y, w, h, st)
            if mapped is None:
                continue
            cls_m, left, top, bw, bh = mapped
            coco_gt["annotations"].append({
                "id": ann_id,
                "image_id": i,
                "category_id": int(cls_m) + 1,
                "bbox": [left, top, bw, bh],
                "area": float(bw * bh),
                "iscrowd": 0
            })
            ann_id += 1

    gt_json = Path("/tmp/rocam_gt_coco.json")
    gt_json.write_text(json.dumps(coco_gt), encoding="utf-8")

    orig_cfg = Path(args.config).expanduser().resolve()
    engine_path = Path(args.engine).expanduser().resolve() if args.engine else None
    tmp_cfg = write_temp_pgie_config(orig_cfg, engine_path)
    logger.info(f"Using temp config: {tmp_cfg} (engine forced to {engine_path})")

    Gst.init(None)
    pipeline_desc = build_minimal_pipeline_desc(pattern, count, tmp_cfg, args.fps)
    pipeline = Gst.parse_launch(pipeline_desc)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    infer = pipeline.get_by_name("infer")
    if infer is None:
        raise RuntimeError("Pipeline element 'infer' not found (nvinfer).")

    infer_src = infer.get_static_pad("src")
    infer_src.add_probe(Gst.PadProbeType.BUFFER, infer_probe, 0)

    logger.info(f"Starting inference over {count} images...")
    detections_raw = []
    frame_counter = 0
    seq_counter = 0

    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)

    # MINIMAL CHANGE: 如果崩了 / 没跑满，直接判定无效退出，避免“半截 mAP”
    if had_error or frame_counter != count:
        logger.error(f"Run invalid: frames_processed={frame_counter}, expected={count}")
        if had_error_msg:
            logger.error(f"Last error:\n{had_error_msg}")
        return 2

    logger.info(f"Collected detections: {len(detections_raw)} (after conf>={args.conf:.4f})")

    dets_pp = postprocess_by_image(
        dets=detections_raw,
        top1=bool(args.top1),
        topk=int(args.topk) if args.topk is not None else 0,
        nms_iou=float(args.nms_iou),
    )
    logger.info(f"Detections after postprocess: {len(dets_pp)} (top1={args.top1}, topk={args.topk}, nms_iou={args.nms_iou})")

    if args.debug_vis and int(args.debug_vis) > 0:
        out_dir = Path("/tmp/rocam_bench_debug")
        draw_debug_overlays(stage_infos, dets_pp, coco_gt, out_dir, int(args.debug_vis))
        logger.info(f"Saved debug overlay images to {out_dir} (N={args.debug_vis})")

    if args.dump_json:
        Path(args.dump_json).expanduser().resolve().write_text(json.dumps(dets_pp), encoding="utf-8")
        logger.info(f"Dumped detections json: {args.dump_json}")

    coco = COCO(str(gt_json))
    if len(dets_pp) == 0:
        logger.error("No detections produced. mAP=0.")
        print("\n=== ACCURACY SUMMARY (COCOeval) ===")
        print(f"split: {args.split}")
        print(f"images: {count}")
        print(f"stage-mode: {args.stage_mode}")
        print(f"bbox-source: {args.bbox_source}")
        print(f"image-id-mode: {args.image_id_mode}")
        print(f"top1: {args.top1}   topk: {args.topk}   nms_iou: {args.nms_iou}")
        print(f"detections (conf>={args.conf:.4f}): 0")
        print("mAP@0.50:0.95 = 0.000000")
        print("mAP@0.50      = 0.000000")
        return 0

    coco_dt = coco.loadRes(dets_pp)
    ev = COCOeval(coco, coco_dt, iouType="bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    map_5095 = float(ev.stats[0])
    map_50 = float(ev.stats[1])

    print("\n=== ACCURACY SUMMARY (COCOeval) ===")
    print(f"split: {args.split}")
    print(f"images: {count}")
    print(f"stage-mode: {args.stage_mode}")
    print(f"bbox-source: {args.bbox_source}")
    print(f"image-id-mode: {args.image_id_mode}")
    print(f"top1: {args.top1}   topk: {args.topk}   nms_iou: {args.nms_iou}")
    print(f"detections (conf>={args.conf:.4f}): {len(dets_pp)}")
    print(f"mAP@0.50:0.95 = {map_5095:.6f}")
    print(f"mAP@0.50      = {map_50:.6f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
