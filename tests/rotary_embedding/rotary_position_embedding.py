import importlib
import logging
import math
import os
import unittest
from typing import Optional

import torch
from deepseek_yarn_rotary_embedding import (
    DeepseekV2YarnRotaryEmbedding,
    deepseek_apply_rotary_pos_emb,
    yarn_get_mscale,
)
from torch import nn
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaRotaryEmbedding
from yarn_rotary_embedding import FlashYaRNRotaryEmbedding

from rtp_llm.ops.compute_ops import rtp_llm_ops


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class GPTNeoXRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[GPTNeoXConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logging.warning(
                "`GPTNeoXRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get(
                    "rope_type", config.rope_scaling.get("type")
                )
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config, device, **self.rope_kwargs
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer(
                "inv_freq", inv_freq, persistent=False
            )  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if (
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class QWenRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        if importlib.util.find_spec("einops") is None:
            raise RuntimeError("einops is required for Rotary Embedding")

        self._rotary_pos_emb_cache = None
        self._seq_len_cached = 0
        self._ntk_alpha_cached = 1.0

    def update_rotary_pos_emb_cache(self, max_seq_len, offset=0, ntk_alpha=1.0):
        seqlen = max_seq_len + offset
        if seqlen > self._seq_len_cached or ntk_alpha != self._ntk_alpha_cached:
            base = self.base * ntk_alpha ** (self.dim / (self.dim - 2))
            self.inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, self.dim, 2, device=self.inv_freq.device).float()
                    / self.dim
                )
            )
            self._seq_len_cached = max(2 * seqlen, 16)
            self._ntk_alpha_cached = ntk_alpha
            seq = torch.arange(self._seq_len_cached, device=self.inv_freq.device)
            freqs = torch.outer(seq.type_as(self.inv_freq), self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            from einops import rearrange

            self._rotary_pos_emb_cache = rearrange(emb, "n d -> 1 n 1 d")

    def forward(self, max_seq_len, offset=0, ntk_alpha=1.0):
        self.update_rotary_pos_emb_cache(max_seq_len, offset, ntk_alpha)
        return self._rotary_pos_emb_cache[:, offset : offset + max_seq_len]


def _rotate_half(x):
    from einops import rearrange

    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def qwen_apply_rotary_pos_emb(t, freqs):

    rot_dim = freqs.shape[-1]
    t_, t_pass_ = t[..., :rot_dim], t[..., rot_dim:]
    t_ = t_.float()
    t_pass_ = t_pass_.float()
    t_ = (t_ * freqs.cos()) + (_rotate_half(t_) * freqs.sin())
    return torch.cat((t_, t_pass_), dim=-1).type_as(t)


class TestRope(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        torch.classes.load_library(
            os.environ["TEST_SRCDIR"] + "/rtp_llm/tests/libtest_ops.so"
        )

    def test_LinearScaling(self):
        base = 10000
        dim = 128
        headsize = 32
        batch_size = 8
        max_position_embeddings = 2048
        cuda = torch.device("cuda")
        for scale in range(1, 10, 5):
            for seq_len in range(1024, 4096, 1024):
                self.RopeOP = torch.classes.unittest.RotaryPositionEmbeddingOp(
                    dim, max_position_embeddings, base, scale, 1, 1
                )
                self.Rope = GPTNeoXRotaryEmbedding(
                    dim, max_position_embeddings, base, cuda, scale, rope_type="linear"
                )
                x = torch.rand(batch_size, seq_len, headsize, dim).cuda()
                position_ids = torch.arange(0, seq_len, dtype=torch.long).cuda()
                position_ids = position_ids.unsqueeze(0).view(-1, seq_len)
                cos, sin = self.Rope(x.permute(0, 2, 1, 3), position_ids)
                result, _ = apply_rotary_pos_emb(
                    x.permute(0, 2, 1, 3), x.permute(0, 2, 1, 3), cos, sin, position_ids
                )
                test = self.RopeOP.forward(x, 0)
                torch.testing.assert_close(
                    test, result.permute(0, 2, 1, 3), atol=1e-3, rtol=1e-5
                )

    def test_DynamicNTKScaling(self):
        torch.classes.load_library(
            os.environ["TEST_SRCDIR"] + "/rtp_llm/tests/libtest_ops.so"
        )
        base = 10000
        dim = 64
        headsize = 1
        batch_size = 8
        max_position_embeddings = 200
        cuda = torch.device("cuda")
        for scale in [1, 4]:
            for seq_len in [100, 200, 300]:
                self.RopeOP = torch.classes.unittest.RotaryPositionEmbeddingOp(
                    dim, max_position_embeddings, base, scale, 3, 1
                )
                self.Rope = GPTNeoXRotaryEmbedding(
                    dim, max_position_embeddings, base, cuda, scale, rope_type="dynamic"
                )
                x = torch.rand(batch_size, seq_len, headsize, dim).cuda() + 1
                position_ids = torch.arange(0, seq_len, dtype=torch.long).cuda()
                position_ids = position_ids.unsqueeze(0).view(-1, seq_len)
                cos, sin = self.Rope(x.permute(0, 2, 1, 3), position_ids)
                result, _ = apply_rotary_pos_emb(
                    x.permute(0, 2, 1, 3), x.permute(0, 2, 1, 3), cos, sin, position_ids
                )
                test = self.RopeOP.forward(x.clone(), 0)
                torch.testing.assert_close(
                    test, result.permute(0, 2, 1, 3), atol=1e-3, rtol=1e-5
                )

    def test_QWen(self):
        torch.classes.load_library(
            os.environ["TEST_SRCDIR"] + "/rtp_llm/tests/libtest_ops.so"
        )

        base = 10000
        dim = 128
        headsize = 32
        batch_size = 8
        max_position_embeddings = 2048
        for seq_len in range(1024, 4096, 1024):
            self.RopeOP = torch.classes.unittest.RotaryPositionEmbeddingOp(
                dim, max_position_embeddings, base, 1.0, 4, 1
            )
            self.Rope = QWenRotaryEmbedding(dim, base)
            x = torch.rand(batch_size, seq_len, headsize, dim).cuda()

            context_value = math.log(seq_len / max_position_embeddings, 2) + 1
            ntk_alpha = 2 ** math.ceil(context_value) - 1
            ntk_alpha = max(ntk_alpha, 1)
            freq = self.Rope(seq_len, ntk_alpha=ntk_alpha)
            result = qwen_apply_rotary_pos_emb(x, freq.cuda())
            test = self.RopeOP.forward(x, 0)
            torch.testing.assert_close(test, result, atol=1e-2, rtol=1e-2)

    def test_Yarn(self):
        def get_mscale(scale: float):
            if scale < 1.0:
                return 1.0
            return 0.1 * math.log(scale) + 1.0

        torch.classes.load_library(
            os.environ["TEST_SRCDIR"] + "/rtp_llm/tests/libtest_ops.so"
        )

        base = 10000
        dim = 128
        headsize = 32
        batch_size = 8
        max_position_embeddings = 2048
        for scale in range(1, 10, 5):
            for seq_len in range(1024, 2048, 4096):
                self.RopeOP = torch.classes.unittest.RotaryPositionEmbeddingOp(
                    dim, max_position_embeddings, base, scale, 5, get_mscale(scale)
                )
                self.Rope = FlashYaRNRotaryEmbedding(
                    dim,
                    base,
                    scaling_factor=scale,
                    original_max_position_embeddings=max_position_embeddings,
                )
                x = torch.rand(batch_size, seq_len, headsize, dim).cuda()
                position_ids = torch.arange(0, seq_len, dtype=torch.long).cuda()
                position_ids = position_ids.unsqueeze(0).view(-1, seq_len)
                cos, sin = self.Rope(x.permute(0, 2, 1, 3))
                result, _ = apply_rotary_pos_emb(
                    x.permute(0, 2, 1, 3),
                    x.permute(0, 2, 1, 3),
                    cos.unsqueeze_(0),
                    sin.unsqueeze_(0),
                    position_ids,
                )
                test = self.RopeOP.forward(x, 0)
                torch.testing.assert_close(
                    test, result.permute(0, 2, 1, 3), atol=1e-3, rtol=1e-5
                )

    @unittest.skip("need update transformers")
    def test_Llama3(self):
        dim = 128
        headsize = 1
        batch_size = 1
        max_pos = 200
        config = LlamaConfig()
        config.rope_scaling = {
            "rope_type": "llama3",
            "factor": 8,
            "low_freq_factor": 1,
            "high_freq_factor": 4,
            "original_max_position_embeddings": max_pos,
        }
        self.ropeop = torch.classes.unittest.RotaryPositionEmbeddingOp(
            dim, max_pos, 10000, 8, 6
        )
        self.rope = LlamaRotaryEmbedding(
            dim=dim, rope_type="llama3", device=torch.device("cuda"), config=config
        )
        # long seq has numeric issue. python transformers use double prec in rotary invfreq
        for seq_len in [2]:
            x = torch.rand(batch_size, seq_len, headsize, dim).cuda()
            position_ids = torch.arange(0, seq_len, dtype=torch.long).cuda()
            position_ids = position_ids.unsqueeze(0).view(-1, seq_len)
            cos, sin = self.rope(x.permute(0, 2, 1, 3), position_ids=position_ids)
            sin = sin.reshape(seq_len, dim)
            cos = cos.reshape(seq_len, dim)
            x_permute = x.permute(0, 2, 1, 3)
            result, _ = apply_rotary_pos_emb(
                x_permute, x_permute, cos, sin, position_ids
            )
            result = result.permute(0, 2, 1, 3)
            test = self.ropeop.forward(x.clone(), 0)
            torch.testing.assert_close(test, result, atol=1e-3, rtol=1e-5)

    def test_deepseek_Yarn(self):
        torch.classes.load_library(
            os.environ["TEST_SRCDIR"] + "/rtp_llm/tests/libtest_ops.so"
        )
        base = 10000
        nope_dim = 128
        dim = 64
        headsize = 1
        batch_size = 8
        max_position_embeddings = 2048
        mscale = 0.707
        mscale_all_dim = 0.707
        for scale in [1, 40]:
            for seq_len in [32, 128, 256]:
                real_mscale = float(
                    yarn_get_mscale(scale, mscale)
                    / yarn_get_mscale(scale, mscale_all_dim)
                )
                self.RopeOP = torch.classes.unittest.RotaryPositionEmbeddingOp(
                    dim, max_position_embeddings, base, scale, 5, real_mscale
                )
                self.Rope = DeepseekV2YarnRotaryEmbedding(
                    dim=dim,
                    max_position_embeddings=max_position_embeddings,
                    original_max_position_embeddings=max_position_embeddings,
                    scaling_factor=scale,
                    mscale=mscale,
                    mscale_all_dim=mscale_all_dim,
                )
                # x = torch.arange(0, batch_size * seq_len * headsize * dim, dtype=torch.float32).reshape(batch_size, seq_len, headsize, dim)
                x = torch.randn(batch_size, seq_len, headsize, dim)
                input = torch.rand(batch_size, seq_len, headsize, nope_dim + dim)
                input[:, :, :, nope_dim:] = (
                    x.reshape(batch_size, seq_len, headsize, dim // 2, 2)
                    .transpose(3, 4)
                    .reshape(batch_size, seq_len, headsize, dim)
                )
                input = input.contiguous().cuda()
                x = x.cuda()
                position_ids = torch.arange(0, seq_len, dtype=torch.long).cuda()
                position_ids = position_ids.unsqueeze(0).view(-1, seq_len)
                cos, sin = self.Rope(x.permute(0, 2, 1, 3), seq_len)
                result, _ = deepseek_apply_rotary_pos_emb(
                    x.permute(0, 2, 1, 3), x.permute(0, 2, 1, 3), cos, sin, position_ids
                )
                test = self.RopeOP.forward(input, nope_dim)
                test = test[:, :, :, nope_dim:].contiguous().cuda()
                torch.testing.assert_close(
                    test, result.permute(0, 2, 1, 3), atol=1e-3, rtol=1e-5
                )


if __name__ == "__main__":
    unittest.main()
