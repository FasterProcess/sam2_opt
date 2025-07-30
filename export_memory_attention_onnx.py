import sys

sys.path.insert(0, "sam2")
import os

os.environ.setdefault("EXPORT_ONNX_SEQ_LEN", "4096")
import torch
from sam2.build_sam import build_sam2
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.modeling.memory_attention import MemoryAttention
from PIL import Image
import numpy as np
from torch import nn
import os
import onnx
from onnxconverter_common import float16

device = torch.device("cuda")
onnx_path = "models"
os.makedirs(onnx_path, exist_ok=True)

# large
sam2_checkpoint = "./sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# # tiny
# sam2_checkpoint = "./sam2/checkpoints/sam2.1_hiera_tiny.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

predictor = build_sam2_video_predictor(
    model_cfg, sam2_checkpoint, device=device
)  # type:SAM2VideoPredictor


def export_memory_attention(
    onnx_name="memory_attention.onnx", simplify_onnx=True, override=False, num=64
):
    global predictor, onnx_path, device
    os.makedirs(onnx_path, exist_ok=True)

    model = predictor.memory_attention  # type: MemoryAttention
    ori_forward = model.forward
    model.eval()

    model.forward = model.inference_memory_attention_torch

    save_path = os.path.join(onnx_path, onnx_name)
    if os.path.exists(save_path) and not override:
        print(f"skip gen {save_path}")
        return

    N = 1
    L = 7  # 1~7
    P = num  # 0~64

    curr = torch.randn(4096, N, 256, device=device)
    memory = torch.randn(L, 4096, N, 64, device=device)
    curr_pos = torch.randn(4096, N, 256, device=device)
    memory_pos = torch.randn(L, 4096, N, 64, device=device)

    memory_exclude = torch.randn(P, N, 64, device=device)
    memory_pos_exclude = torch.randn(P, N, 64, device=device)

    # num_obj_ptr_tokens = torch.tensor([num], dtype=torch.int32, device=device)

    torch.onnx.export(
        model,
        (curr, memory, curr_pos, memory_pos, memory_exclude, memory_pos_exclude),
        save_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=[
            "curr",
            "memory",
            "curr_pos",
            "memory_pos",
            "memory_exclude",
            "memory_pos_exclude",
        ],
        output_names=["memory_output"],
        dynamic_axes={
            "curr": {1: "N"},
            "memory": {0: "L", 2: "N"},
            "curr_pos": {1: "N"},
            "memory_pos": {0: "L", 2: "N"},
            "memory_exclude": {0: "P", 1: "N"},
            "memory_pos_exclude": {0: "P", 1: "N"},
            "memory_output": {1: "N"},
        },
    )

    model.forward = ori_forward

    onnx.checker.check_model(save_path)

    if simplify_onnx:
        from onnxsim import simplify

        onnx_model = onnx.load(save_path)  # load onnx model
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, save_path.replace(".onnx", "_opt.onnx"))


if __name__ == "__main__":
    export_memory_attention(
        onnx_name="memory_attention_exclude.onnx",
        override=True,
        simplify_onnx=True,
        num=64,
    )

    export_memory_attention(
        onnx_name="memory_attention_none.onnx",
        override=True,
        simplify_onnx=True,
        num=0,
    )
