# yolo-depthanythingv2-merge

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PINTO0309/yolo-depthanythingv2-merge)

**The model committed to this repository is an incomplete model that has not yet been fully trained, and is therefore a demo model whose detection performance is only about 45% of the original accuracy.**

- TensorRT 10 test

  https://github.com/user-attachments/assets/5d3f904e-b41d-4c55-819b-2e5930e0cb6b

- BBox/Center Point Depth, Full Angle Depth, People Segmentation

  ![image](https://github.com/user-attachments/assets/ccdbb755-d9f7-4388-8d4f-e09a745625ed)

## Overview

A computer vision pipeline that integrates YOLOv9 (object detection with 34 whole-body keypoints) and DepthAnythingV2 (depth estimation) to perform real-time object detection, pose estimation, and depth estimation simultaneously. Optionally supports person segmentation.

## Key Features

- **Object Detection**: High-precision object detection using YOLOv9
- **Pose Estimation**: Detection of 34 whole-body keypoints (face, hands, body)
- **Depth Estimation**: Relative or metric depth estimation (indoor/outdoor) using DepthAnythingV2
- **Segmentation**: Optional person segmentation support
- **Edge Detection**: Edge detection from depth maps
- **Head Distance Measurement**: Distance calculation to head using camera FOV

## Environment Setup

### Requirements

- Python 3.10 or higher
- CUDA-capable GPU (recommended)
- ONNXRuntime or TensorRT (for fast inference)

### Installing Dependencies

```bash
# Basic dependencies
pip install opencv-python
pip install numpy
pip install torch
pip install pyyaml
pip install onnx
pip install onnxruntime-gpu  # GPU version (CUDA environment)
# or
pip install onnxruntime      # CPU version

# ONNX simplification tool
pip install onnxsim

# PINTO0309's ONNX manipulation tools (required for model merging)
pip install sne4onnx
pip install sor4onnx
pip install snc4onnx
pip install soa4onnx
pip install sio4onnx
pip install sit4onnx  # For benchmarking (optional)

# For TensorRT usage (optional)
# TensorRT must be installed separately from NVIDIA official sources
```

### Model Preparation

The following ONNX models are required for this repository. You can download all the necessary ONNX files from:
**https://github.com/PINTO0309/yolo-depthanythingv2-merge/releases/tag/onnx**

Required models:
1. **YOLOv9 Model**: `yolov9_e_wholebody34_post_0100_1x3x480x640.onnx`
2. **DepthAnythingV2 Models**:
   - Relative depth: `depth_anything_v2_small.onnx`
   - Metric depth (indoor): `depth_anything_v2_metric_hypersim_vits_indoor_maxdepth20_1x3x518x518.onnx`
   - Metric depth (outdoor): `depth_anything_v2_metric_vkitti_vits_outdoor_maxdepth80_1x3x518x518.onnx`
3. **Segmentation Model** (optional): `peopleseg_1x3x480x640.onnx`

## 1. Model Merging

Create an integrated model combining YOLOv9 and DepthAnythingV2.

### Basic Merging (Relative Depth)

```bash
# Adjust DepthAnythingV2 model resolution (H and W must be multiples of 14)
H=490 # Multiples of 14
W=644 # Multiples of 14
ONNXSIM_FIXED_POINT_ITERS=10000 onnxsim depth_anything_v2_small.onnx depth_anything_v2_small_${H}x${W}.onnx \
--overwrite-input-shape "pixel_values:1,3,${H},${W}"

# Merge models
python merge_preprocess_onnx.py
```

### Merging with Segmentation

```bash
# Merge with segmentation support
python merge_preprocess_onnx_depth_seg.py
```

Merged model structure:
![image](https://github.com/user-attachments/assets/0bc94ed3-17ad-4b5b-837e-07d20bdb96b2)

## 2. Running Inference

### Basic Usage

Run inference on an image folder:

```bash
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
```

### Video Inference

```bash
# Webcam (device index 0)
python demo_yolov9_onnx_wholebody34_with_edges_with_depth.py \
-v 0 \
-ep cuda

# Video file
python demo_yolov9_onnx_wholebody34_with_edges_with_depth.py \
-v ./video.mp4 \
-ep tensorrt
```

### Inference with Segmentation

```bash
python demo_yolov9_onnx_wholebody34_with_edges_with_depth_seg.py \
-m yolov9_e_wholebody34_with_depth_seg_post_0100_1x3x480x640.onnx \
-i ./images \
-ep cuda \
-edm \
-ehd
```

### Inference Results Examples

![000000000241](https://github.com/user-attachments/assets/b75dcab1-5441-4e05-af10-a05f4ca3a1e3)

![000000012069](https://github.com/user-attachments/assets/5f958051-2893-48f0-8463-3cdd0743298e)

## 3. Command Line Options

### Input Options
- `-m, --model`: Path to ONNX model file to use
- `-v, --video`: Video file path or camera index (0, 1, 2...)
- `-i, --images_dir`: Image folder path (supports jpg, png)

### Execution Provider
- `-ep, --execution_provider`: Provider to use for inference
  - `cpu`: CPU execution
  - `cuda`: CUDA GPU execution (recommended)
  - `tensorrt`: TensorRT execution (fastest)

### Performance Options
- `-it, --inference_type`: Inference precision (`fp16` or `int8`)
- `-dvw, --disable_video_writer`: Disable video writer (improves processing speed)
- `-dwk, --disable_waitKey`: Disable key input wait (for batch processing)

### Detection Thresholds
- `-ost, --object_score_threshold`: Object detection score threshold (default: 0.35)
- `-ast, --attribute_score_threshold`: Attribute score threshold (default: 0.70)
- `-kst, --keypoint_threshold`: Keypoint score threshold (default: 0.25)

### Display Options
- `-kdm, --keypoint_drawing_mode`: Keypoint drawing mode (`dot`, `box`, `both`)
- `-dnm, --disable_generation_identification_mode`: Disable generation identification mode
- `-dgm, --disable_gender_identification_mode`: Disable gender identification mode
- `-dlr, --disable_left_and_right_hand_identification_mode`: Disable left/right hand identification mode
- `-dhm, --disable_headpose_identification_mode`: Disable head pose identification mode
- `-drc, --disable_render_classids`: Disable rendering of specific class IDs (e.g., `-drc 17 18 19`)

### Special Features
- `-efm, --enable_face_mosaic`: Enable face mosaic (toggle with F key)
- `-ebd, --enable_bone_drawing`: Enable bone drawing (toggle with B key)
- `-edm, --enable_depth_map_overlay`: Enable depth map overlay (toggle with D key)
- `-ehd, --enable_head_distance_measurement`: Enable head distance measurement (toggle with M key)
- `-oyt, --output_yolo_format_text`: Output YOLO format text files and images

### Other Options
- `-bblw, --bounding_box_line_width`: Bounding box line width (default: 2)
- `-chf, --camera_horizontal_fov`: Camera horizontal FOV (default: 90 degrees)

### Keyboard Shortcuts (During Execution)
- `N`: Toggle generation identification mode
- `G`: Toggle gender identification mode
- `H`: Toggle left/right hand identification mode
- `P`: Toggle head pose identification mode
- `F`: Toggle face mosaic
- `B`: Toggle bone drawing
- `D`: Toggle depth map overlay
- `M`: Toggle head distance measurement
- `Q` or `ESC`: Exit

## 4. Benchmark

### Performance Measurement

```bash
sit4onnx \
-if yolov9_e_wholebody34_with_depth_post_0100_1x3x480x640.onnx \
-oep tensorrt \
-fs 1 3 480 640
```

### Example Results

```
INFO: file: yolov9_e_wholebody34_with_depth_post_0100_1x3x480x640.onnx
INFO: providers: ['TensorrtExecutionProvider', 'CPUExecutionProvider']
INFO: input_name.1: input_bgr shape: [1, 3, 480, 640] dtype: float32
INFO: test_loop_count: 10
INFO: total elapsed time:  120.57948112487793 ms
INFO: avg elapsed time per pred:  12.057948112487793 ms
INFO: output_name.1: batchno_classid_score_x1y1x2y2_depth shape: [0, 8] dtype: float32
INFO: output_name.2: depth shape: [1, 1, 480, 640] dtype: float32
```

This example shows that using TensorRT to process 480x640 images, inference can be performed at approximately 12ms per frame (about 83 FPS).

## 5. Model Customization

### Supporting Different Resolutions

```bash
# Prepare DepthAnythingV2 model with custom resolution
H=518  # Must be multiple of 14
W=518  # Must be multiple of 14
ONNXSIM_FIXED_POINT_ITERS=10000 onnxsim depth_anything_v2_small.onnx depth_anything_v2_small_${H}x${W}.onnx \
--overwrite-input-shape "pixel_values:1,3,${H},${W}"
```

### Using Metric Depth

For indoor environments:
```bash
python demo_yolov9_onnx_wholebody34_with_edges_with_depth.py \
-m yolov9_e_wholebody34_with_depth_metric_indoor_post_0100_1x3x480x640.onnx \
-i ./images \
-ep cuda \
-ehd  # Enable head distance measurement
```

For outdoor environments:
```bash
python demo_yolov9_onnx_wholebody34_with_edges_with_depth.py \
-m yolov9_e_wholebody34_with_depth_metric_outdoor_post_0100_1x3x480x640.onnx \
-i ./images \
-ep cuda \
-ehd
```

## 6. Troubleshooting

### When CUDA/TensorRT is Unavailable

```bash
# Run on CPU
python demo_yolov9_onnx_wholebody34_with_edges_with_depth.py \
-i ./images \
-ep cpu
```

### When Out of Memory

- Reduce input image resolution
- Keep batch size at 1 (default)

### When Models are Not Found

Verify that the required ONNX model files exist:
```bash
ls *.onnx
```

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.
