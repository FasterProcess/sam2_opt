# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
from torch import nn, Tensor

from sam2.modeling.sam.transformer import RoPEAttention

from sam2.modeling.sam2_utils import get_activation_fn, get_clones
from ytools.bench import test_torch_cuda_time
from ytools.executor import ModelExectuor
from ytools.onnxruntime import OnnxRuntimeExecutor


class MemoryAttentionLayer(nn.Module):

    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        self_attention: nn.Module,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)

        # Where to add pos enc
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt, query_pos):
        # Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(
        self,
        tgt,
        memory,
        query_pos,
        pos,
        num_k_exclude_rope: Tensor = torch.tensor([0], dtype=torch.int32),
    ):
        kwds = {}
        if isinstance(self.cross_attn_image, RoPEAttention):
            num_k_exclude_rope = torch.where(
                num_k_exclude_rope > 0, num_k_exclude_rope, 0
            )
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        # if num_k_exclude_rope > 0:
        #     assert isinstance(self.cross_attn_image, RoPEAttention)
        #     kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        # Cross-Attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        num_k_exclude_rope: Tensor = torch.tensor([0], dtype=torch.int32),
    ) -> torch.Tensor:

        # Self-Attn, Cross-Attn
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class MemoryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: nn.Module,
        num_layers: int,
        batch_first: bool = True,  # Do layers expect batch first input?
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first
        # --- new ---
        self.backend_contexts: List[ModelExectuor] = []
        self.inference_memory_attention = self.inference_memory_attention_torch
        self.set_runtime_backend(backend="torch")

    def set_runtime_backend(self, backend="torch", args: dict = None):
        """
        Dynamically sets the runtime backend for MemoryAttention (torch or onnxruntime).
        """
        self.backend_contexts = []
        if backend.lower() == "torch":
            self.inference_memory_attention = self.inference_memory_attention_torch
        elif backend.lower() == "onnxruntime":
            self.inference_memory_attention = (
                self.inference_memory_attention_onnxruntime
            )
            assert (
                args and "model_paths" in args
            ), "The 'model_paths' argument is required to specify the ONNX model path"

            model_path = args["model_paths"][0]
            providers = args.get("providers", None)
            executor = OnnxRuntimeExecutor(model_path, providers=providers)

            curr_warmup = torch.randn(4096, 1, 256)
            curr_pos_warmup = torch.randn(4096, 1, 256)
            num_obj_ptr_tokens_warmup = torch.tensor([64], dtype=torch.int32)

            # 4100-28736
            MIN_MEM_LEN = 4100
            MAX_MEM_LEN = 28736

            for mem_len in [MIN_MEM_LEN, MAX_MEM_LEN]:
                warmup_inputs = [
                    curr_warmup,
                    torch.randn(mem_len, 1, 64),
                    curr_pos_warmup,
                    torch.randn(mem_len, 1, 64),
                    num_obj_ptr_tokens_warmup,
                ]
                executor.warmup(warmup_inputs)

            self.backend_contexts.append(executor)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def forward(
        self,
        curr: torch.Tensor,
        memory: torch.Tensor,
        curr_pos: Optional[Tensor] = None,
        memory_pos: Optional[Tensor] = None,
        num_obj_ptr_tokens: int = 0,
    ):
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = (
                curr[0],
                curr_pos[0],
            )

        assert (
            curr.shape[1] == memory.shape[1]
        ), "Batch size must be the same for curr and memory"

        if not isinstance(num_obj_ptr_tokens, Tensor):
            num_obj_ptr_tokens = torch.tensor([num_obj_ptr_tokens], dtype=torch.int32)

        return self.inference_memory_attention(
            curr, memory, curr_pos, memory_pos, num_obj_ptr_tokens
        )

    @test_torch_cuda_time()
    # --- 4. 创建 Torch 实现 (原始 forward 逻辑) ---
    def inference_memory_attention_torch(
        self,
        curr: torch.Tensor,
        memory: torch.Tensor,
        curr_pos: Optional[Tensor] = None,
        memory_pos: Optional[Tensor] = None,
        num_obj_ptr_tokens: Tensor = torch.tensor([0], dtype=torch.float32),
    ):

        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        if self.batch_first:
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)

        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens.to(dtype=torch.int32)}

            output = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                **kwds,
            )
        normed_output = self.norm(output)

        if self.batch_first:
            normed_output = normed_output.transpose(0, 1)
            # curr_pos = curr_pos.transpose(0, 1) # This line was in your original code but seems unused

        return normed_output

    @test_torch_cuda_time()
    def inference_memory_attention_onnxruntime(
        self,
        curr: torch.Tensor,
        memory: torch.Tensor,
        curr_pos: Optional[Tensor] = None,
        memory_pos: Optional[Tensor] = None,
        num_obj_ptr_tokens: Tensor = torch.tensor([0], dtype=torch.int32),
    ):
        print(f"")
        outputs = self.backend_contexts[0].Inference(
            [curr, memory, curr_pos, memory_pos, num_obj_ptr_tokens],
            output_type="torch",
        )
        return outputs[0].to(curr.device)
