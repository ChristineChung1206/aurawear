Place optional model files here.

Face parsing (CelebAMask-HQ 19-class) expected path:
- face_parsing.onnx

Model details:
- Architecture: BiSeNet (ResNet-18 backbone)
- Training data: CelebAMask-HQ (30,000 images, 19 classes)
- Best validation mIoU: **0.7609**
- Export format: ONNX (opset 14)

The pipeline will automatically use it if present (and if `onnxruntime` is installed).
Alternatively set:
- AURAWEAR_FACE_PARSING_ONNX=/absolute/path/to/face_parsing.onnx
