import os
import sys
import torch
import torch.nn as nn
from copy import deepcopy
from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Detect, v10Detect
import ultralytics.utils
import ultralytics.models.yolo
import onnx
import onnxslim

# Compatibility hacks for YOLO models
sys.modules["ultralytics.yolo"] = ultralytics.models.yolo
sys.modules["ultralytics.yolo.utils"] = ultralytics.utils

class DeepStreamOutput(nn.Module):
    """
    Post-processing layer to format YOLO output for DeepStream's nvinfer.
    Concatenates [boxes, scores, labels] into a single output tensor.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.transpose(1, 2)
        boxes = x[:, :, :4]
        scores, labels = torch.max(x[:, :, 4:], dim=-1, keepdim=True)
        return torch.cat([boxes, scores, labels.to(boxes.dtype)], dim=-1)

def export_onnx(input_pt: str, output_onnx: str):
    """
    Exports a YOLO .pt model to a DeepStream-compatible ONNX model.
    Hardcodes the optimized parameters: 540x960 resolution, batch size 1, opset 17.
    """
    # Hardcoded parameters aligned with original convert_model.sh requirements
    img_size = (540, 960)
    batch_size = 1
    opset_version = 17

    print(f"Loading model: {input_pt}")
    model = YOLO(input_pt)
    model = deepcopy(model.model).to("cpu")
    
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()
    model = model.fuse()

    # Configure modules for export
    for k, m in model.named_modules():
        if isinstance(m, (Detect, v10Detect)):
            m.dynamic = False
            m.export = True
            m.format = "onnx"
        elif isinstance(m, C2f):
            m.forward = m.forward_split

    # Append DeepStream formatting layer
    model = nn.Sequential(model, DeepStreamOutput())

    print(f"Exporting to ONNX: {output_onnx} (Size: {img_size}, Batch: {batch_size})")
    dummy_input = torch.zeros(batch_size, 3, *img_size)
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx,
        verbose=False,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"]
    )

    print("Simplifying ONNX model...")
    model_onnx = onnx.load(output_onnx)
    model_onnx = onnxslim.slim(model_onnx)
    onnx.save(model_onnx, output_onnx)
    
    # Invalidate potential stale engines (logic from convert_model.sh)
    # The original script deleted a specific engine path; we try to find it near the ONNX file
    engine_hint = os.path.join(os.path.dirname(output_onnx), "..", "model_b1_gpu0_fp16.engine")
    if os.path.exists(engine_hint):
        os.remove(engine_hint)
        print(f"Removed stale engine: {engine_hint}")

    print(f"Successfully created: {output_onnx}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert YOLO .pt to DeepStream-compatible ONNX")
    parser.add_argument("input", help="Input .pt file path")
    parser.add_argument("output", help="Output .onnx file path")
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
        
    export_onnx(args.input, args.output)
