import sys
import os
import torch
import onnx
import numpy as np


sys.path.insert(0, os.path.abspath("sam2"))

from sam2.build_sam import build_sam2_video_predictor
import torch
import onnx
import argparse
from onnxsim import simplify
from Module import MemAttention
from sam2.build_sam import build_sam2
from Module import MemAttentionONNX
import torch.nn as nn


onnx_output_path = "models/"
model_config_file = "configs/sam2.1/sam2.1_hiera_l.yaml"
model_checkpoints_file = "sam2/checkpoints/sam2.1_hiera_large.pt"


class MemoryAttentionOnnxWrapper(nn.Module):
    def __init__(self, memory_attention_module: nn.Module):
        super().__init__()
        # 只包含我们要导出的核心模块
        self.memory_attention = memory_attention_module

    def forward(
        self,
        curr: torch.Tensor,             # 形状: [HW, B, C]
        curr_pos: torch.Tensor,         # 形状: [HW, B, C]
        memory: torch.Tensor,           # 形状: [MemLen, B, C_mem]
        memory_pos: torch.Tensor,       # 形状: [MemLen, B, C_mem]
        # num_obj_ptr_tokens 不再是输入
    ):
        # 在这个 Wrapper 中，假设 num_obj_ptr_tokens 是一个固定的值
        # memory_0.shape[0] 是 16，num_obj_ptr_tokens 是 64。
        
        num_obj_ptr_tokens_int = 64
        
        return self.memory_attention(
            curr=curr,
            curr_pos=curr_pos,
            memory=memory,
            memory_pos=memory_pos,
            num_obj_ptr_tokens=num_obj_ptr_tokens_int,
        )


def export_memory_attention_final(model_wrapper, onnx_path):
    # --- 准备与运行时完全一致的示例输入 ---
    B = 1 # Batch size
    HW = 4096
    C = 256
    C_mem = 64
    MemLen = 7 * 4096 + 64 # 示例内存长度

    curr_in = torch.randn(HW, B, C)
    curr_pos_in = torch.randn(HW, B, C)
    memory_in = torch.randn(MemLen, B, C_mem)
    memory_pos_in = torch.randn(MemLen, B, C_mem)
    
    # 定义输入名称
    input_names = [
        "curr",
        "curr_pos",
        "memory",
        "memory_pos",
    ]
    
    dynamic_axes = {
        "curr": {0: "hw", 1: "batch"},
        "curr_pos": {0: "hw", 1: "batch"},
        "memory": {0: "memory_length", 1: "batch"},
        "memory_pos": {0: "memory_length", 1: "batch"},
    }
    torch.onnx.export(
        model_wrapper,
        (curr_in, curr_pos_in, memory_in, memory_pos_in), 
        onnx_path + "memory_attention.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=["pix_feat_with_mem"],
        dynamic_axes=dynamic_axes
    )
    
    # 简化和检查
    original_model = onnx.load(onnx_path + "memory_attention.onnx")
    simplified_model, check = simplify(original_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(simplified_model, onnx_path + "memory_attention_opt.onnx")
    onnx_model = onnx.load(onnx_path + "memory_attention_opt.onnx")
    onnx.checker.check_model(onnx_model)
    print("Final memory_attention.onnx model exported successfully with correct interface!")

if __name__ == "__main__":
    sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2/checkpoints/sam2.1_hiera_large.pt", device="cpu")
    
    wrapper_for_export = MemoryAttentionOnnxWrapper(sam2_model.memory_attention).cpu().eval()

    export_memory_attention_final(wrapper_for_export, "models/")