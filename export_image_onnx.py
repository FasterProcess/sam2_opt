import sys

sys.path.insert(0, "sam2")

import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np
from torch import nn
import os
import onnx

device = torch.device("cuda")
onnx_path = "models"

onnx_version = 18
enable_dynamo = False
os.makedirs(onnx_path, exist_ok=True)

# large
sam2_checkpoint = "./sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# # tiny
# sam2_checkpoint = "./sam2/checkpoints/sam2.1_hiera_tiny.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)


@torch.no_grad()
def export_forward_image(
    onnx_name="forward_image.onnx", simplify_onnx=True, override=False
):
    global predictor, onnx_path
    os.makedirs(onnx_path, exist_ok=True)

    ori_forward = predictor.model.forward
    predictor.model.forward = predictor.model.inference_image_torch
    predictor.model.eval()

    save_path = os.path.join(onnx_path, onnx_name)
    if os.path.exists(save_path) and not override:
        print(f"skip gen {save_path}")
        return

    input_img = torch.randn(1, 3, 1024, 1024).to(device)
    if not enable_dynamo:
        torch.onnx.export(
            predictor.model,
            input_img,
            save_path,
            export_params=True,
            opset_version=onnx_version,
            do_constant_folding=True,
            input_names=["image"],
            output_names=[
                "vision_features",
                "vision_pos_enc0",
                "vision_pos_enc1",
                "vision_pos_enc2",
                "backbone_fpn0",
                "backbone_fpn1",
                "backbone_fpn2",
            ],
            dynamic_axes={
                "image": {0: "N"},
                "vision_features": {0: "N"},
                "vision_pos_enc0": {0: "N"},
                "vision_pos_enc1": {0: "N"},
                "vision_pos_enc2": {0: "N"},
                "backbone_fpn0": {0: "N"},
                "backbone_fpn1": {0: "N"},
                "backbone_fpn2": {0: "N"},
            },
        )
    else:
        torch.onnx.export(
            predictor.model,
            input_img,
            save_path,
            export_params=True,
            opset_version=onnx_version,
            do_constant_folding=True,
            input_names=["image"],
            output_names=[
                "vision_features",
                "vision_pos_enc0",
                "vision_pos_enc1",
                "vision_pos_enc2",
                "backbone_fpn0",
                "backbone_fpn1",
                "backbone_fpn2",
            ],
            dynamo=True,
        )

    predictor.model.forward = ori_forward

    if simplify_onnx:
        from onnxsim import simplify

        onnx_model = onnx.load(save_path)  # load onnx model
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, save_path.replace(".onnx", "_opt.onnx"))


@torch.no_grad()
def export_set_image_e2e(
    onnx_name="set_image_e2e.onnx", simplify_onnx=True, override=False
):
    """ """
    global predictor, onnx_path
    os.makedirs(onnx_path, exist_ok=True)

    ori_forward = predictor.forward
    predictor.forward = predictor.set_image_e2e
    predictor.eval()

    save_path = os.path.join(onnx_path, onnx_name)
    if os.path.exists(save_path) and not override:
        print(f"skip gen {save_path}")
        return

    input_img = torch.randn(1, 3, 1024, 1024).to(device)
    if not enable_dynamo:
        torch.onnx.export(
            predictor,
            input_img,
            save_path,
            export_params=True,
            opset_version=onnx_version,
            do_constant_folding=True,
            input_names=["image"],
            output_names=[
                "feature0",
                "feature1",
                "feature2",
            ],
            dynamic_axes={
                "image": {0: "N"},
                "feature0": {0: "N"},
                "feature1": {0: "N"},
                "feature2": {0: "N"},
            },
        )
    else:
        torch.onnx.export(
            predictor,
            input_img,
            save_path,
            export_params=True,
            opset_version=onnx_version,
            do_constant_folding=True,
            input_names=["image"],
            output_names=[
                "feature0",
                "feature1",
                "feature2",
            ],
            dynamo=True,
        )

    predictor.forward = ori_forward

    if simplify_onnx:
        from onnxsim import simplify

        onnx_model = onnx.load(save_path)  # load onnx model
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, save_path.replace(".onnx", "_opt.onnx"))


@torch.no_grad()
def export_mask_encoder(
    onnx_name="image_mask_encoder.onnx", simplify_onnx=True, override=False
):
    global predictor, onnx_path
    os.makedirs(onnx_path, exist_ok=True)

    ori_forward = predictor.model.sam_mask_decoder.forward
    predictor.model.sam_mask_decoder.forward = (
        predictor.model.sam_mask_decoder.inference_predict_masks
    )
    predictor.model.sam_mask_decoder.eval().to(device=device)

    save_path = os.path.join(onnx_path, onnx_name)
    if os.path.exists(save_path) and not override:
        print(f"skip gen {save_path}")
        return

    src = torch.randn(1, 256, 64, 64).to(device)
    tokens = torch.randn(1, 9, 256).to(device)
    pos_src = torch.randn(1, 256, 64, 64).to(device)
    high_res_feature0 = torch.randn(1, 32, 256, 256).to(device)
    high_res_feature1 = torch.randn(1, 64, 128, 128).to(device)

    if not enable_dynamo:
        torch.onnx.export(
            predictor.model.sam_mask_decoder,
            (
                src,
                tokens,
                pos_src,
                high_res_feature0,
                high_res_feature1,
            ),
            save_path,
            export_params=True,
            opset_version=onnx_version,
            do_constant_folding=True,
            input_names=[
                "src",
                "tokens",
                "pos_src",
                "high_res_feature0",
                "high_res_feature1",
            ],
            output_names=[
                "masks",
                "iou_pred",
                "mask_tokens_out",
                "object_score_logits",
            ],
            dynamic_axes={
                "src": {0: "N"},
                "tokens": {0: "N", 1: "L"},
                "pos_src": {0: "N"},
                "high_res_feature0": {0: "N"},
                "high_res_feature1": {0: "N"},
                "masks": {0: "N"},
                "iou_pred": {0: "N"},
                "mask_tokens_out": {0: "N"},
                "object_score_logits": {0: "N"},
            },
        )
    else:
        torch.onnx.export(
            predictor.model.sam_mask_decoder,
            (
                src,
                tokens,
                pos_src,
                high_res_feature0,
                high_res_feature1,
            ),
            save_path,
            export_params=True,
            opset_version=onnx_version,
            do_constant_folding=True,
            input_names=[
                "src",
                "tokens",
                "pos_src",
                "high_res_feature0",
                "high_res_feature1",
            ],
            output_names=[
                "masks",
                "iou_pred",
                "mask_tokens_out",
                "object_score_logits",
            ],
            dynamo=True,
        )

    predictor.model.sam_mask_decoder.forward = ori_forward

    if simplify_onnx:
        from onnxsim import simplify

        onnx_model = onnx.load(save_path)  # load onnx model
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, save_path.replace(".onnx", "_opt.onnx"))


export_forward_image(override=False)
export_set_image_e2e(override=True)
export_mask_encoder(override=False)
