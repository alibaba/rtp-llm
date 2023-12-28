import os
import unittest
import torch
import math
import importlib
import einops

from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXRotaryEmbedding
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLinearScalingRotaryEmbedding
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXDynamicNTKScalingRotaryEmbedding
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb


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

    def test_RotaryEmbedding(self):
        torch.classes.load_library(os.environ['TEST_SRCDIR'] + "/maga_transformer/tests/libtest_ops.so")

        base = 10000
        dim = 128
        headsize = 32
        batch_size = 16
        max_position_embeddings = 2048
        cuda = torch.device('cuda')
        for seq_len in range(1024, 4096, 1024):
            self.RopeOP = torch.classes.unittest.RotaryPositionEmbeddingOp(dim, max_position_embeddings, base, 1.0, 0, 0)
            self.Rope = GPTNeoXRotaryEmbedding(dim, max_position_embeddings, base, cuda)
            x = torch.rand(batch_size, seq_len, headsize, dim).cuda()
            position_ids = torch.arange(0, seq_len, dtype=torch.long).cuda()
            position_ids = position_ids.unsqueeze(0).view(-1, seq_len)
            cos, sin = self.Rope(x.permute(0, 2, 1, 3), seq_len)
            result, _ = apply_rotary_pos_emb(x.permute(0, 2, 1, 3), x.permute(0, 2, 1, 3), cos, sin, position_ids)
            test = self.RopeOP.forward(x)
            torch.testing.assert_close(test, result.permute(0, 2, 1, 3), atol=1e-3, rtol=1e-5)

    def test_LinearScaling(self):
        torch.classes.load_library(os.environ['TEST_SRCDIR'] + "/maga_transformer/tests/libtest_ops.so")

        base = 10000
        dim = 128
        headsize = 32
        batch_size = 16
        max_position_embeddings = 2048
        cuda = torch.device('cuda')
        for scalar in range(1, 10, 5):
            for seq_len in range(1024, 4096, 1024):
                self.RopeOP = torch.classes.unittest.RotaryPositionEmbeddingOp(dim, max_position_embeddings, base, scalar, 0, 1)
                self.Rope = GPTNeoXLinearScalingRotaryEmbedding(dim, max_position_embeddings, base, cuda, scalar)
                x = torch.rand(batch_size, seq_len, headsize, dim).cuda()
                position_ids = torch.arange(0, seq_len, dtype=torch.long).cuda()
                position_ids = position_ids.unsqueeze(0).view(-1, seq_len)
                cos, sin = self.Rope(x.permute(0, 2, 1, 3), seq_len)
                result, _ = apply_rotary_pos_emb(x.permute(0, 2, 1, 3), x.permute(0, 2, 1, 3), cos, sin, position_ids)
                test = self.RopeOP.forward(x)
                torch.testing.assert_close(test, result.permute(0, 2, 1, 3), atol=1e-3, rtol=1e-5)

    def test_DynamicNTKScaling(self):
        torch.classes.load_library(os.environ['TEST_SRCDIR'] + "/maga_transformer/tests/libtest_ops.so")
        base = 10000
        dim = 128
        headsize = 32
        batch_size = 16
        max_position_embeddings = 2048
        cuda = torch.device('cuda')
        for scalar in range(1, 10, 5):
            for seq_len in range(1024, 4096, 1024):
                self.RopeOP = torch.classes.unittest.RotaryPositionEmbeddingOp(dim, max_position_embeddings, base, scalar, 0, 2)
                self.Rope = GPTNeoXDynamicNTKScalingRotaryEmbedding(dim, max_position_embeddings, base, cuda, scalar)
                x = torch.rand(batch_size, seq_len, headsize, dim).cuda()
                position_ids = torch.arange(0, seq_len, dtype=torch.long).cuda()
                position_ids = position_ids.unsqueeze(0).view(-1, seq_len)
                cos, sin = self.Rope(x.permute(0, 2, 1, 3), seq_len)
                result, _ = apply_rotary_pos_emb(x.permute(0, 2, 1, 3), x.permute(0, 2, 1, 3), cos, sin, position_ids)
                test = self.RopeOP.forward(x)
                torch.testing.assert_close(test, result.permute(0, 2, 1, 3), atol=1e-3, rtol=1e-5)

    def test_QWen(self):
        torch.classes.load_library(os.environ['TEST_SRCDIR'] + "/maga_transformer/tests/libtest_ops.so")

        base = 10000
        dim = 128
        headsize = 32
        batch_size = 16
        max_position_embeddings = 2048
        max_logn_seq_len = 2048
        for seq_len in range(1024, 10000, 1024):
            self.RopeOP = torch.classes.unittest.RotaryPositionEmbeddingOp(dim, max_position_embeddings, base, 1.0, max_logn_seq_len, 3)
            self.Rope = QWenRotaryEmbedding(dim,base)
            x = torch.rand(batch_size, seq_len, headsize, dim).cuda()

            context_value = math.log(seq_len / max_logn_seq_len, 2) + 1
            ntk_alpha = 2 ** math.ceil(context_value) - 1
            ntk_alpha = max(ntk_alpha, 1)
            freq= self.Rope(seq_len, ntk_alpha = ntk_alpha)
            result = qwen_apply_rotary_pos_emb(x, freq.cuda())
            test = self.RopeOP.forward(x)
            torch.testing.assert_close(test, result, atol=1e-2, rtol=1e-2)


class GLMRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, precision=torch.half, learnable=False):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        # inv_freq = inv_freq.half().cuda()
        inv_freq = inv_freq.cuda()
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.precision = precision

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()

            # [sx, 1 (b * np), hn]
            cos_cached = emb.cos()[:, None, :]
            sin_cached = emb.sin()[:, None, :]
            if self.precision == torch.bfloat16:
                cos_cached = cos_cached.bfloat16()
                sin_cached = sin_cached.bfloat16()
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


def GLMrotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions


def GLM_apply_rotary_pos_emb_index(q, k, cos, sin, position_id):
    # position_id: [sq, b], q, k: [sq, b, np, hn], cos: [sq, 1, hn] -> [sq, b, 1, hn]
    cos, sin = torch.nn.functional.embedding(position_id, cos.squeeze(1)).unsqueeze(2), \
        torch.nn.functional.embedding(position_id, sin.squeeze(1)).unsqueeze(2)
    q, k = (q * cos) + (GLMrotate_half(q) * sin), (k * cos) + (GLMrotate_half(k) * sin)
    return q, k


class TestGLM(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        torch.classes.load_library(os.environ['TEST_SRCDIR'] + "/maga_transformer/tests/libtest_ops.so")

    def test_glm(self):
        base = 10000
        dim = 128
        headsize = 32
        batch_size = 16
        max_position_embeddings = 2048
        cuda = torch.device('cuda')
        for seq_len in range(30, 4096, 1024):
            self.RopeOP = torch.classes.unittest.RotaryPositionEmbeddingOp(dim, max_position_embeddings, base, 1.0, 0, 4)
            self.Rope = GLMRotaryEmbedding(dim/2, base)
            x = torch.rand(batch_size, seq_len, headsize, dim).cuda()
            x1, x2 = x.chunk(2, dim=(x.ndim - 1))
            position_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1).transpose(0,1).cuda()
            cos, sin = self.Rope(x.permute(1, 0, 2, 3), seq_len = seq_len)
            result1, _ = GLM_apply_rotary_pos_emb_index(x1.permute(1, 0, 2, 3), x1.permute(1, 0, 2, 3), cos, sin, position_ids)
            result2, _ = GLM_apply_rotary_pos_emb_index(x2.permute(1, 0, 2, 3), x2.permute(1, 0, 2, 3), cos, sin, position_ids)
            result = torch.concat([result1, result2], dim=(result1.ndim - 1))
            test = self.RopeOP.forward(x)
            torch.testing.assert_close(test, result.permute(1, 0, 2, 3), atol=1e-3, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()