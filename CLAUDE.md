# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository merges YOLOv9 (object detection) with DepthAnythingV2 (depth estimation) to create a unified computer vision pipeline that performs whole-body detection (34 keypoints), depth estimation, and optional people segmentation.

## Key Commands

### Model Preparation
```bash
# Prepare depth model for specific resolution (H and W must be multiples of 14)
H=490
W=644
ONNXSIM_FIXED_POINT_ITERS=10000 onnxsim depth_anything_v2_small.onnx depth_anything_v2_small_${H}x${W}.onnx \
--overwrite-input-shape "pixel_values:1,3,${H},${W}"

# Merge YOLOv9 and DepthAnythingV2 models
python merge_preprocess_onnx.py

# For depth with segmentation support
python merge_preprocess_onnx_depth_seg.py
```

### Running Inference
```bash
# Basic inference with depth estimation
python demo_yolov9_onnx_wholebody34_with_edges_with_depth.py \
-i ./images \
-ep cuda \
-dvw \
-dwk \
-kst 0.25 \
-dnm \
-dgm \
-dlr \
-dhm \
-kdm dot \
-edm

# Inference with segmentation
python demo_yolov9_onnx_wholebody34_with_edges_with_depth_seg.py \
-m yolov9_e_wholebody34_with_depth_seg_post_0100_1x3x480x640.onnx \
-i ./images \
-ep tensorrt
```

### Performance Benchmarking
```bash
sit4onnx \
-if yolov9_e_wholebody34_with_depth_post_0100_1x3x480x640.onnx \
-oep tensorrt \
-fs 1 3 480 640
```

## Architecture Overview

The codebase follows a pipeline architecture that merges two state-of-the-art models:

1. **YOLOv9**: Provides object detection with whole-body keypoint detection (34 keypoints)
2. **DepthAnythingV2**: Adds depth estimation capabilities (relative or metric depth)

### Key Components

- **Model Merging Scripts**: `merge_preprocess_onnx.py` and `merge_preprocess_onnx_depth_seg.py` handle the complex task of combining YOLOv9 and DepthAnythingV2 models into a single ONNX graph
- **Demo Scripts**: Two main inference scripts that support various execution providers (CUDA, TensorRT) and multiple visualization options
- **Pre/Post-processing Models**: Resolution-specific ONNX models for input preprocessing and output postprocessing

### Execution Providers

The system supports multiple execution providers:
- CUDA (GPU acceleration)
- TensorRT (optimized inference)
- CPU (fallback)

### Output Capabilities

- Bounding boxes with object detection
- 34 whole-body keypoints (including face, hands, and body landmarks)
- Depth maps (relative or metric - indoor/outdoor)
- Edge detection
- Optional people segmentation masks

## Dependencies

The project relies on specialized ONNX manipulation tools:
- `sne4onnx`: ONNX node extraction
- `sor4onnx`: ONNX node renaming
- `snc4onnx`: ONNX node connection
- `soa4onnx`: ONNX attribute operations
- `sio4onnx`: ONNX I/O operations
- `onnxsim`: ONNX simplification
- `sit4onnx`: ONNX inference testing

Standard dependencies include OpenCV, NumPy, ONNX Runtime, and PyTorch.

## VS Code Debug Configuration

The repository includes debug configurations in `.vscode/launch.json` for easy development. The configuration supports various model types (standard, metric indoor/outdoor, with segmentation) and execution providers.