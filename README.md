# yolo-depthanythingv2-merge

## 1. Merging YOLOv9 and DepthAnythingV2
```bash
H=490 # Multiples of 14
W=644 # Multiples of 14
ONNXSIM_FIXED_POINT_ITERS=10000 onnxsim depth_anything_v2_small.onnx depth_anything_v2_small_${H}x${W}.onnx \
--overwrite-input-shape "pixel_values:1,3,${H},${W}"

python merge_preprocess_onnx.py
```

## 2. Inference test

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
![000000000241](https://github.com/user-attachments/assets/b75dcab1-5441-4e05-af10-a05f4ca3a1e3)

![000000012069](https://github.com/user-attachments/assets/5f958051-2893-48f0-8463-3cdd0743298e)

## 3. Benchmark

```bash
sit4onnx \
-if yolov9_e_wholebody34_with_depth_post_0100_1x3x480x640.onnx \
-oep tensorrt \
-fs 1 3 480 640
```
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
