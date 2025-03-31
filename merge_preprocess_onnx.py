from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import torch
import yaml
import onnx
import shutil
from onnxsim import simplify
from sor4onnx import rename
from snc4onnx import combine
from soa4onnx import outputs_add
from sne4onnx import extraction

class Pre_model(torch.nn.Module):
    def __init__(
        self,
        h: int,
        w: int,
    ):
        super(Pre_model, self).__init__()
        self.h = h
        self.w = w

    def forward(self, x: torch.Tensor):
        x = torch.nn.functional.interpolate(input=x, size=(self.h, self.w))
        return x

class Post_model(torch.nn.Module):
    def __init__(
        self,
        input_h: int,
        input_w: int,
    ):
        super(Post_model, self).__init__()
        self.input_h = input_h
        self.input_w = input_w

    def forward(self, x: torch.Tensor, input_image_bgr: torch.Tensor):
        n, c, h, w = input_image_bgr.shape
        x = torch.nn.functional.interpolate(input=x, size=(h, w))
        return x

class DepthBBoxProcessor(torch.nn.Module):
    def __init__(self):
        super(DepthBBoxProcessor, self).__init__()

    def forward(self, bboxes: torch.Tensor, depth_map: torch.Tensor):
        """
        Args:
            bboxes (torch.Tensor): Tensor of shape [instances, 7] containing [batchid, classid, score, x1, y1, x2, y2].
            depth_map (torch.Tensor): Tensor of shape [batch, 1, Height, Width] representing pixel-wise depth.

        Returns:
            torch.Tensor: Tensor of shape [instances, 8] containing [batchid, classid, score, x1, y1, x2, y2, depth].
        """
        batch_ids = bboxes[:, 0].long()  # Extract batch indices
        depth_map = depth_map.squeeze(1)  # Shape: [batch, Height, Width]
        height, width = depth_map.shape[1:]

        # Convert normalized coordinates to absolute pixel values
        x1 = (bboxes[:, 3] * width).long()
        y1 = (bboxes[:, 4] * height).long()
        x2 = (bboxes[:, 5] * width).long()
        y2 = (bboxes[:, 6] * height).long()

        cx = ((x1 + x2) // 2).clamp(0, width - 1)
        cy = ((y1 + y2) // 2).clamp(0, height - 1)

        depth_values = depth_map[batch_ids, cy, cx]
        updated_bboxes = torch.cat((bboxes, depth_values.unsqueeze(1)), dim=1)

        return updated_bboxes


def main():
    dpa_H = 490 # 480->490, Multiples of 14
    dpa_W = 644 # 640->644, Multiples of 14

    onnx_file = f"depth_anything_v2_small_{dpa_H}x{dpa_W}.onnx"
    output_onnx_file = onnx_file

    shutil.copy('depth_anything_v2_small.onnx', onnx_file)

    rename(
        old_new=["/", "depthanything/"],
        input_onnx_file_path=onnx_file,
        output_onnx_file_path=output_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["1079", "depthanything/1079"],
        input_onnx_file_path=output_onnx_file,
        output_onnx_file_path=output_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["pretrained.", "depthanything/pretrained."],
        input_onnx_file_path=output_onnx_file,
        output_onnx_file_path=output_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["onnx::", "depthanything/onnx::"],
        input_onnx_file_path=output_onnx_file,
        output_onnx_file_path=output_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["10", "depthanything/10"],
        input_onnx_file_path=output_onnx_file,
        output_onnx_file_path=output_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["depth_head.", "depthanything/depth_head."],
        input_onnx_file_path=output_onnx_file,
        output_onnx_file_path=output_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["onnx::", "depthanything/onnx::"],
        input_onnx_file_path=output_onnx_file,
        output_onnx_file_path=output_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )

    extraction(
        input_op_names=['pixel_values'],
        output_op_names=['depthanything/Relu_output_0'],
        input_onnx_file_path=output_onnx_file,
        output_onnx_file_path=output_onnx_file,
    )
    model_onnx = onnx.load(output_onnx_file)
    model_simp, check = simplify(
        model=model_onnx,
        overwrite_input_shapes={"pixel_values": [1,3,dpa_H,dpa_W]},
    )
    onnx.save(model_simp, output_onnx_file)

    rename(
        old_new=["depthanything/Relu_output_0", "depth"],
        input_onnx_file_path=output_onnx_file,
        output_onnx_file_path=output_onnx_file,
        mode="outputs",
        search_mode="prefix_match",
    )

    ############### pre-process
    yolo_H=480
    yolo_W=640

    pre_onnx_file = f"preprocess_{yolo_H}x{yolo_W}_{dpa_H}x{dpa_W}.onnx"
    pre_model = Pre_model(h=dpa_H, w=dpa_W)
    x = torch.randn(1, 3, yolo_H, yolo_W).cpu()
    torch.onnx.export(
        pre_model,
        args=(x),
        f=pre_onnx_file,
        opset_version=13,
        input_names=['input_pre'],
        output_names=['output_pre'],
    )
    model_onnx1 = onnx.load(pre_onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, pre_onnx_file)
    onnx_graph = rename(
        old_new=["/", "depthanything/pre/"],
        input_onnx_file_path=pre_onnx_file,
        output_onnx_file_path=pre_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    model_onnx2 = onnx.load(pre_onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, pre_onnx_file)

    ############### post-process
    post_onnx_file = f"postprocess_{yolo_H}x{yolo_W}_{dpa_H}x{dpa_W}.onnx"
    post_model = Post_model(input_h=yolo_H, input_w=yolo_W)
    x = torch.randn(1, 1, dpa_H, dpa_W).cpu()
    y = torch.randn(1, 3, yolo_H, yolo_W).cpu()
    torch.onnx.export(
        post_model,
        args=(x, y),
        f=post_onnx_file,
        opset_version=13,
        input_names=['input_post', 'input_image_bgr'],
        output_names=['depth'],
        dynamic_axes={
            'input_image_bgr' : {2: 'H', 3: 'W'},
            'depth' : {0: '1', 1: '1', 2: 'H', 3: 'W'},
        }
    )
    model_onnx1 = onnx.load(post_onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, post_onnx_file)

    rename(
        old_new=["/", "depthanything/post/"],
        input_onnx_file_path=post_onnx_file,
        output_onnx_file_path=post_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["onnx::", "depthanything/onnx::"],
        input_onnx_file_path=post_onnx_file,
        output_onnx_file_path=post_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )

    model_onnx2 = onnx.load(post_onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, post_onnx_file)

    bboxes = torch.tensor([
        [0, 1, 0.9, 0.25, 0.25, 0.75, 0.75],
        [0, 2, 0.8, 0.15, 0.15, 0.5, 0.5]
    ], dtype=torch.float32)

    depth_map = torch.rand(1, 1, dpa_H, dpa_W)  # Example depth map for 1 batches

    ############### BBox+Depth merge post-process
    processor = DepthBBoxProcessor()
    processor.cpu()
    processor.eval()
    bbox_depth_merge_onnx_file = f"bboxes_depth_merge_process.onnx"
    torch.onnx.export(
        processor,
        args=(bboxes, depth_map),
        f=bbox_depth_merge_onnx_file,
        opset_version=13,
        input_names=['input_bboxes', 'inuput_depth_map'],
        output_names=['batchno_classid_score_x1y1x2y2_depth'],
        dynamic_axes={
            'input_bboxes' : {0: 'N'},
            'inuput_depth_map' : {2: 'H', 3: 'W'},
            'batchno_classid_score_x1y1x2y2_depth' : {0: 'N'},
        }
    )
    model_onnx = onnx.load(bbox_depth_merge_onnx_file)
    model_simp, check = simplify(model_onnx)
    onnx.save(model_simp, bbox_depth_merge_onnx_file)

    rename(
        old_new=["/", "bbox_depth_merge/"],
        input_onnx_file_path=bbox_depth_merge_onnx_file,
        output_onnx_file_path=bbox_depth_merge_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )


    ############### YOLO + DepthAnything
    combine(
        srcop_destop = [
            ['output_prep', 'input_pre']
        ],
        input_onnx_file_paths = [
            f'yolov9_e_wholebody34_post_0100_1x3x{yolo_H}x{yolo_W}.onnx',
            pre_onnx_file,
        ],
        output_onnx_file_path = f'yolov9_e_wholebody34_with_depth_post_0100_1x3x{yolo_H}x{yolo_W}.onnx',
    )
    combine(
        srcop_destop = [
            ['output_pre', 'pixel_values']
        ],
        input_onnx_file_paths = [
            f'yolov9_e_wholebody34_with_depth_post_0100_1x3x{yolo_H}x{yolo_W}.onnx',
            f'depth_anything_v2_small_{dpa_H}x{dpa_W}.onnx',
        ],
        output_onnx_file_path = f'yolov9_e_wholebody34_with_depth_post_0100_1x3x{yolo_H}x{yolo_W}.onnx',
    )
    rename(
        old_new=["depth", "yolo_depth"],
        input_onnx_file_path=f'yolov9_e_wholebody34_with_depth_post_0100_1x3x{yolo_H}x{yolo_W}.onnx',
        output_onnx_file_path=f'yolov9_e_wholebody34_with_depth_post_0100_1x3x{yolo_H}x{yolo_W}.onnx',
        mode="outputs",
        search_mode="prefix_match",
    )
    combine(
        srcop_destop = [
            ['yolo_depth', 'input_post', 'input_bgr', 'input_image_bgr'],
        ],
        input_onnx_file_paths = [
            f'yolov9_e_wholebody34_with_depth_post_0100_1x3x{yolo_H}x{yolo_W}.onnx',
            post_onnx_file,
        ],
        output_onnx_file_path = f'yolov9_e_wholebody34_with_depth_post_0100_1x3x{yolo_H}x{yolo_W}.onnx',
    )
    combine(
        srcop_destop = [
            ['batchno_classid_score_x1y1x2y2', 'input_bboxes', 'depth', 'inuput_depth_map'],
        ],
        input_onnx_file_paths = [
            f'yolov9_e_wholebody34_with_depth_post_0100_1x3x{yolo_H}x{yolo_W}.onnx',
            bbox_depth_merge_onnx_file,
        ],
        output_onnx_file_path = f'yolov9_e_wholebody34_with_depth_post_0100_1x3x{yolo_H}x{yolo_W}.onnx',
    )
    outputs_add(
        input_onnx_file_path=f'yolov9_e_wholebody34_with_depth_post_0100_1x3x{yolo_H}x{yolo_W}.onnx',
        output_op_names=["depth"],
        output_onnx_file_path=f'yolov9_e_wholebody34_with_depth_post_0100_1x3x{yolo_H}x{yolo_W}.onnx',
    )

if __name__ == "__main__":
    main()
