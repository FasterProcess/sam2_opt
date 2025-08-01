# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import math
import warnings
from functools import partial
from typing import Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from sam2.modeling.position_encoding_fix import (
    compute_axial_rope_cos_sin,
    apply_rotary_emb,
)
from sam2.modeling.sam2_utils import MLP
from sam2.utils.misc import get_sdpa_settings
import os

warnings.simplefilter(action="ignore", category=FutureWarning)
# Check whether Flash Attention is available (and use it by default)
OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = get_sdpa_settings()
# A fallback setting to allow all available kernels if Flash Attention fails
ALLOW_ALL_KERNELS = False

# Use matrix version of rotrary enc
USE_MAT_ROTARY_ENC = True


def sdp_kernel_context(dropout_p):
    """
    Get the context for the attention scaled dot-product kernel. We use Flash Attention
    by default, but fall back to all available kernels if Flash Attention fails.
    """
    if ALLOW_ALL_KERNELS:
        return contextlib.nullcontext()

    return torch.backends.cuda.sdp_kernel(
        enable_flash=USE_FLASH_ATTN,
        # if Flash attention kernel is off, then math kernel needs to be enabled
        enable_math=(OLD_GPU and dropout_p > 0.0) or MATH_KERNEL_ON,
        enable_mem_efficient=OLD_GPU,
    )


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLP(
            embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation
        )
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.dropout_p = dropout

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        # try:
        #    with sdp_kernel_context(dropout_p):
        #        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        # except Exception as e:
        if True:
            # Fall back to all kernels if the Flash attention kernel fails
            # warnings.warn(
            #    f"Flash Attention kernel failed due to: {e}\nFalling back to all available "
            #    f"kernels for scaled_dot_product_attention (which may have a slower speed).",
            #    category=UserWarning,
            #    stacklevel=2,
            # )
            global ALLOW_ALL_KERNELS
            ALLOW_ALL_KERNELS = True
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class RoPEAttention(Attention):
    def __init__(
        self,
        *args,
        rope_theta=10000.0,
        rope_k_repeat=False,
        feat_sizes=(64, 64),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.rope_theta = rope_theta
        self.feat_sizes = feat_sizes
        self.rope_k_repeat = rope_k_repeat
        self._cached_shape = None
        self._cached_cos_sin = None
        self.max_seq = 4096

        # self.freq_cos, self.freq_sin = self.get_cos_sin(
        #     self.max_seq, device="cpu", dtype=torch.float32
        # )

        self.cos, self.sin = None, None

        seq_len = int(os.environ.get("EXPORT_ONNX_SEQ_LEN", 0))
        if seq_len > 0:
            self.init_cos_sin(seq_len, device="cuda", dtype=torch.float32)

    def init_cos_sin(self, seq_len, device, dtype):
        self.cos, self.sin = self.get_cos_sin(seq_len, device, dtype)

    def get_cos_sin(self, seq_len, device, dtype):
        if self._cached_shape == (seq_len, device, dtype):
            return self._cached_cos_sin
        w = int(math.sqrt(seq_len))
        h = w
        cos, sin = compute_axial_rope_cos_sin(
            dim=self.internal_dim // self.num_heads,
            end_x=w,
            end_y=h,
            theta=self.rope_theta,
        )
        cos, sin = cos.to(device=device, dtype=dtype), sin.to(
            device=device, dtype=dtype
        )
        self._cached_shape = (seq_len, device, dtype)
        self._cached_cos_sin = (cos, sin)
        return cos, sin

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        num_k_exclude_rope: Tensor = None,
    ) -> Tensor:
        if num_k_exclude_rope is None:
            num_k_exclude_rope = torch.tensor([0], dtype=torch.int32, device=k.device)

        assert isinstance(
            num_k_exclude_rope, Tensor
        ), "num_k_exclude_rope need to be tensor"

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        seq_len = q.size(-2)
        cos, sin = self.get_cos_sin(seq_len, q.device, q.dtype)

        # q: [B, n_head, seq_len, head_dim]
        q = apply_rotary_emb(q, cos, sin)
        # if k.size(-2) != q.size(-2):
        #     assert self.rope_k_repeat
        #     # repeat cos/sin
        #     repeat = k.shape[-2] // q.shape[-2]
        #     cos_k = cos.repeat(repeat, 1)
        #     sin_k = sin.repeat(repeat, 1)
        # else:
        #     cos_k, sin_k = cos, sin

        cos_k = cos.repeat(k.size(-2) // q.size(-2), 1)
        sin_k = sin.repeat(k.size(-2) // q.size(-2), 1)

        # num_k_rope = torch.where(
        #     num_k_exclude_rope > 0, k.size(-2) - num_k_exclude_rope, k.size(-2)
        # ).to(torch.int32)

        # k_rope = apply_rotary_emb(
        #     k[:, :, :num_k_rope, ...], cos_k[:num_k_rope], sin_k[:num_k_rope]
        # )
        # k = torch.cat([k_rope, k[:, :, num_k_rope:, ...]], dim=-2)

        num_k_rope = (k.size(-2) - num_k_exclude_rope).to(dtype=torch.int64)
        # k = torch.cat(
        #         [
        #             apply_rotary_emb(
        #                 k[:, :, :num_k_rope, :],
        #                 cos_k[:num_k_rope, :],
        #                 sin_k[:num_k_rope, :],
        #             ),
        #             k[:, :, num_k_rope:, :],
        #         ],
        #         dim=-2,
        #     )

        if num_k_exclude_rope.item() > 0:
            k = torch.cat(
                [
                    apply_rotary_emb(
                        k[:, :, :num_k_rope, :],
                        cos_k[:num_k_rope, :],
                        sin_k[:num_k_rope, :],
                    ),
                    k[:, :, num_k_rope:, :],
                ],
                dim=-2,
            )
        else:
            k = apply_rotary_emb(k, cos_k, sin_k)

        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out
