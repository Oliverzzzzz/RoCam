# Build & Benchmark (YOLO → ONNX → TensorRT Engine)

## 1) Put model in `models/`
```bash
cp /path/to/your_model.pt models/model.pt
```

## 2) Export ONNX
```bash
bash convert_model.sh
# output: models/model.pt.onnx
```

## 3) Build TensorRT engine (via DeepStream nvinfer)
```bash
python3 build_engine.py --rebuild --config pgie_config.txt
# output: model_b1_gpu0_fp16.engine
```

## 4) Run benchmark
```bash
bash run_accuracy_bench.sh
```

### Override defaults (optional)
```bash
DATA_YAML=/path/to/data.yaml SPLIT=val \
ENGINE=model_b1_gpu0_fp16.engine PGIE_CONFIG=pgie_config.txt \
bash run_accuracy_bench.sh
```

## Notes
- `pgie_config.txt` must point to:
  - `onnx-file=models/model.pt.onnx`
  - `model-engine-file=model_b1_gpu0_fp16.engine`
  - `custom-lib-path=libnvdsinfer_custom_impl_Yolo.so`
