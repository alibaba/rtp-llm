"""DiffusionGemma layer CPU tests against a pure-torch HF reference.

The reference is a direct transcription of transformers
models/diffusion_gemma/modeling_diffusion_gemma.py: fp32 RMSNorm, QK-norm +
dual-parameter RoPE (sliding default full-dim; global "proportional" partial),
gated GeGLU-tanh shared MLP, raw-residual router (softmax -> topk -> renorm ->
per_expert_scale), routed-expert dual branch, per-mode layer scalars.

HF layout weights ([out, in]) drive the reference; model kernels are their
transposes ([in, out]) exactly as the loader produces them. Packed expert
tensors keep the HF 3D layout ([E, 2I, H] / [E, H, I]) which the v1 executor
consumes directly.
"""

from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import ctypes
import glob
import os

import torch


def _preload_libcuda_stub() -> None:
    """CPU-only hosts lack an NVIDIA driver; toolkit stubs satisfy the
    librtp_compute_ops link-time dependencies so the import succeeds. The
    torch/CPU code paths below never call driver APIs. Extra stub locations
    come from the CUDA_STUB_LIB environment variable (colon-separated)."""
    extra_dirs = [
        d for d in os.environ.get("CUDA_STUB_LIB", "").split(":") if d
    ]
    for soname in ("libcuda.so.1", "libnvidia-ml.so.1"):
        candidates = [soname]
        for base in extra_dirs:
            candidates.append(os.path.join(base, soname))
        candidates += sorted(glob.glob(f"/usr/local/cuda*/lib64/stubs/{soname}"))
        for path in candidates:
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                break
            except OSError:
                continue


_preload_libcuda_stub()

from rtp_llm.config.model_config import ModelConfig  # noqa: E402
from rtp_llm.models.diffusion_gemma import DiffusionGemmaConfig  # noqa: E402
from rtp_llm.models_py.model_desc import diffusion_gemma as dgm  # noqa: E402
from rtp_llm.ops import ParallelismConfig  # noqa: E402
from rtp_llm.utils.model_weight import W  # noqa: E402


H = 32          # hidden size
SLIDING_DIM = 16
SLIDING_KV = 2
GLOBAL_DIM = 32
GLOBAL_KV = 1
HEADS = 4
INTER = 24      # shared expert intermediate
MOE_INTER = 12
EXPERTS = 6
TOPK = 3
LAYERS = 3
EPS = 1e-6
WINDOW = 8      # small window so mask semantics are exercised
CANVAS = 6
PAGE = 4


def _make_config() -> DiffusionGemmaConfig:
    config = DiffusionGemmaConfig()
    config.num_layers = LAYERS
    config.hidden_size = H
    config.vocab_size = 64
    config.max_seq_len = 128
    config.layernorm_eps = EPS
    config.attn_config.head_num = HEADS
    config.attn_config.kv_head_num = SLIDING_KV
    config.attn_config.size_per_head = SLIDING_DIM
    config.attn_config.is_causal = True
    config.attn_config.kernel_tokens_per_block = PAGE
    config.inter_size = INTER
    config.moe_inter_size = MOE_INTER
    config.expert_num = EXPERTS
    config.moe_k = TOPK
    config.dgm_head_dim = SLIDING_DIM
    config.dgm_kv_head_num = SLIDING_KV
    config.dgm_global_head_dim = GLOBAL_DIM
    config.dgm_global_kv_head_num = GLOBAL_KV
    config.dgm_global_partial_rope = 0.25
    config.dgm_swa_rope_theta = 10000.0
    config.dgm_global_rope_theta = 1000000.0
    config.dgm_sliding_window = WINDOW
    config.dgm_canvas_length = CANVAS
    config.dgm_final_logit_softcapping = 30.0
    config.dgm_layer_types = dgm.diffusion_gemma_layer_types(LAYERS)
    return config


def _make_hf_weights(layer_type: str, seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    kv = GLOBAL_KV if layer_type == "full_attention" else SLIDING_KV
    head_dim = GLOBAL_DIM if layer_type == "full_attention" else SLIDING_DIM
    return {
        "qkv": torch.randn(HEADS * head_dim + 2 * kv * head_dim, H, generator=generator) * 0.2,
        "o": torch.randn(H, HEADS * head_dim, generator=generator) * 0.2,
        "q_ln": torch.ones(head_dim) + 0.02 * torch.randn(head_dim, generator=generator),
        "k_ln": torch.ones(head_dim) + 0.02 * torch.randn(head_dim, generator=generator),
        "input_ln": torch.ones(H) + 0.05 * torch.randn(H, generator=generator),
        "post_attn_ln": torch.ones(H) + 0.05 * torch.randn(H, generator=generator),
        "pre_ffn_ln": torch.ones(H) + 0.05 * torch.randn(H, generator=generator),
        "post_ffn_ln": torch.ones(H) + 0.05 * torch.randn(H, generator=generator),
        "pre_ffn_ln_2": torch.ones(H) + 0.05 * torch.randn(H, generator=generator),
        "post_ffn_ln_1": torch.ones(H) + 0.05 * torch.randn(H, generator=generator),
        "post_ffn_ln_2": torch.ones(H) + 0.05 * torch.randn(H, generator=generator),
        "gate": torch.randn(INTER, H, generator=generator) * 0.2,
        "up": torch.randn(INTER, H, generator=generator) * 0.2,
        "down": torch.randn(H, INTER, generator=generator) * 0.2,
        "moe_gate_up": torch.randn(EXPERTS, 2 * MOE_INTER, H, generator=generator) * 0.2,
        "moe_down": torch.randn(EXPERTS, H, MOE_INTER, generator=generator) * 0.2,
        "router_proj": torch.randn(EXPERTS, H, generator=generator) * 0.5,
        "router_scale": torch.ones(H) + 0.1 * torch.randn(H, generator=generator),
        "per_expert_scale": torch.ones(EXPERTS) + 0.1 * torch.randn(EXPERTS, generator=generator),
        "layer_scalar_enc": 0.9 + 0.2 * torch.rand(1, generator=generator),
        "layer_scalar_dec": 1.1 + 0.2 * torch.rand(1, generator=generator),
    }


def _model_weights(hf: dict) -> dict:
    """HF [out,in] weights -> model kernels ([in, out]); MoE 3D kept as-is."""
    return {
        W.attn_qkv_w: hf["qkv"].T.contiguous(),
        W.attn_o_w: hf["o"].T.contiguous(),
        W.q_ln_gamma: hf["q_ln"],
        W.k_ln_gamma: hf["k_ln"],
        W.pre_ln_gamma: hf["input_ln"],
        W.post_ln_gamma: hf["post_attn_ln"],
        W.dgm_pre_ffn_ln_gamma: hf["pre_ffn_ln"],
        W.dgm_post_ffn_ln_gamma: hf["post_ffn_ln"],
        W.dgm_pre_ffn_ln2_gamma: hf["pre_ffn_ln_2"],
        W.dgm_post_ffn_ln1_gamma: hf["post_ffn_ln_1"],
        W.dgm_post_ffn_ln2_gamma: hf["post_ffn_ln_2"],
        W.ffn_w1: hf["gate"].T.contiguous(),
        W.ffn_w3: hf["up"].T.contiguous(),
        W.ffn_w2: hf["down"].T.contiguous(),
        W.moe_w1: hf["moe_gate_up"].contiguous(),
        W.moe_w2: hf["moe_down"].contiguous(),
        W.moe_gate: hf["router_proj"].T.contiguous(),
        W.dgm_router_scale: hf["router_scale"],
        W.dgm_router_per_expert_scale: hf["per_expert_scale"],
        W.dgm_layer_scalar_enc: hf["layer_scalar_enc"],
        W.dgm_layer_scalar_dec: hf["layer_scalar_dec"],
    }


# ---------------------------------------------------------------------------
# Reference implementation (HF transcription)
# ---------------------------------------------------------------------------


def _ref_rmsnorm(x, weight=None, eps=EPS):
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    out = x * torch.rsqrt(variance + eps)
    if weight is not None:
        out = out * weight.to(torch.float32)
    return out.to(x.dtype)


def _ref_rope(head_dim, theta, partial, positions):
    rope_angles = int(partial * head_dim // 2)
    inv = 1.0 / (theta ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float64) / head_dim))
    nope = head_dim // 2 - rope_angles
    if nope > 0:
        inv = torch.cat([inv, torch.zeros(nope, dtype=torch.float64)])
    freqs = torch.outer(positions.to(torch.float32), inv.to(torch.float32))
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _ref_apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((x1 * cos[..., :half] - x2 * sin[..., :half],
                      x2 * cos[..., :half] + x1 * sin[..., :half]), dim=-1)


def _ref_attention(q, k, v, prefix_len, causal, window):
    """q/k/v [T, H, D]; GQA kv heads repeat to the query head count like the
    HF eager path."""
    if k.shape[1] != q.shape[1]:
        groups = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    score = torch.einsum("qhd,khd->hqk", q.to(torch.float32), k.to(torch.float32))
    total = q.shape[0]
    q_pos = torch.arange(total)
    k_pos = torch.arange(total)
    allowed = torch.ones(total, total, dtype=torch.bool)
    for qi in range(total):
        for ki in range(total):
            if causal and k_pos[ki] > q_pos[qi]:
                allowed[qi, ki] = False
            if window > 0 and abs(int(k_pos[ki] - q_pos[qi])) > window - 1:
                allowed[qi, ki] = False
    score = score.masked_fill(~allowed.unsqueeze(0), float("-inf"))
    probs = torch.softmax(score, dim=-1)
    return torch.einsum("hqk,khd->qhd", probs.to(q.dtype), v)


def _ref_layer(hidden, hf, layer_type, positions, causal, decoder_mode):
    kv = GLOBAL_KV if layer_type == "full_attention" else SLIDING_KV
    head_dim = GLOBAL_DIM if layer_type == "full_attention" else SLIDING_DIM
    if layer_type == "full_attention":
        cos, sin = _ref_rope(GLOBAL_DIM, 1e6, 0.25, positions)
    else:
        cos, sin = _ref_rope(SLIDING_DIM, 1e4, 1.0, positions)

    residual = hidden
    h = _ref_rmsnorm(residual, hf["input_ln"])
    qkv = torch.nn.functional.linear(h, hf["qkv"])
    q_size = HEADS * head_dim
    kv_size = kv * head_dim
    q = qkv[..., :q_size].view(-1, HEADS, head_dim)
    k = qkv[..., q_size:q_size + kv_size].view(-1, kv, head_dim)
    v = qkv[..., q_size + kv_size:].view(-1, kv, head_dim)
    q = _ref_rmsnorm(q, hf["q_ln"].view(1, 1, -1))
    k = _ref_rmsnorm(k, hf["k_ln"].view(1, 1, -1))
    v = _ref_rmsnorm(v)
    q = _ref_apply_rope(q, cos, sin)
    k = _ref_apply_rope(k, cos, sin)
    window = WINDOW if layer_type == "sliding_attention" else 0
    attn = _ref_attention(q, k, v, 0, causal, window)
    attn = attn.reshape(-1, HEADS * head_dim)
    h = torch.nn.functional.linear(attn, hf["o"])
    h = _ref_rmsnorm(h, hf["post_attn_ln"])
    h = residual + h

    residual = h
    h = _ref_rmsnorm(h, hf["pre_ffn_ln"])
    gate = torch.nn.functional.linear(h, hf["gate"])
    up = torch.nn.functional.linear(h, hf["up"])
    mlp = torch.nn.functional.linear(
        torch.nn.functional.gelu(gate, approximate="tanh") * up, hf["down"]
    )
    h1 = _ref_rmsnorm(mlp, hf["post_ffn_ln_1"])

    flat = residual.reshape(-1, H)
    routed_in = _ref_rmsnorm(flat, hf["pre_ffn_ln_2"])
    router_h = _ref_rmsnorm(flat) * hf["router_scale"] * (H ** -0.5)
    scores = torch.nn.functional.linear(router_h, hf["router_proj"])
    probs = torch.softmax(scores.to(torch.float32), dim=-1)
    top_w, top_i = torch.topk(probs, TOPK, dim=-1)
    top_w = top_w / top_w.sum(dim=-1, keepdim=True)
    top_w = top_w * hf["per_expert_scale"][top_i]
    h2 = torch.zeros_like(routed_in)
    for token in range(routed_in.shape[0]):
        for slot in range(TOPK):
            expert = int(top_i[token, slot])
            gate_up = torch.nn.functional.linear(
                routed_in[token:token + 1], hf["moe_gate_up"][expert]
            )
            gate_e, up_e = gate_up.chunk(2, dim=-1)
            expert_out = torch.nn.functional.linear(
                torch.nn.functional.gelu(gate_e, approximate="tanh") * up_e,
                hf["moe_down"][expert],
            )
            h2[token] += top_w[token, slot].to(routed_in.dtype) * expert_out.squeeze(0)
    h2 = _ref_rmsnorm(h2, hf["post_ffn_ln_2"])

    h = h1 + h2
    h = _ref_rmsnorm(h, hf["post_ffn_ln"])
    h = residual + h
    scalar = hf["layer_scalar_dec"] if decoder_mode else hf["layer_scalar_enc"]
    return h * scalar


class _TorchRMSNorm(torch.nn.Module):
    def __init__(self, weight, eps=EPS):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, x):
        return _ref_rmsnorm(x, self.weight, self.eps)


def _attn_inputs(total, positions, batch=1, prefix_lens=None, block_table=None):
    if prefix_lens is None:
        prefix_lens = torch.zeros(batch, dtype=torch.int32)
    cu = torch.zeros(batch + 1, dtype=torch.int32)
    for b in range(batch):
        cu[b + 1] = cu[b] + total // batch
    return SimpleNamespace(
        prefix_lengths=prefix_lens,
        input_lengths=torch.full((batch,), total // batch, dtype=torch.int32),
        cu_seqlens=cu,
        cu_seqlens_device=cu,
        kv_cache_kernel_block_id=block_table,
        kv_cache_kernel_block_id_device=block_table,
        combo_position_ids=positions.to(torch.int32),
        sequence_lengths=None,
        is_target_verify=False,
    )


class _MockLayerKVCache:
    def __init__(self, k_rows, v_rows):
        pages = (k_rows.shape[0] + PAGE - 1) // PAGE
        cache = torch.zeros(pages, 2, k_rows.shape[1], PAGE, k_rows.shape[2])
        for i in range(k_rows.shape[0]):
            cache[i // PAGE, 0, :, i % PAGE, :] = k_rows[i]
            cache[i // PAGE, 1, :, i % PAGE, :] = v_rows[i]
        self.kv_cache_base = cache
        self.tag = "swa_kv"


class DiffusionGemmaLayerTest(TestCase):
    def _run_layer(self, layer_type, causal, decoder_mode, seed=11):
        config = _make_config()
        hf = _make_hf_weights(layer_type, seed)
        total = CANVAS if not causal else 5
        torch.manual_seed(seed)
        hidden = torch.randn(total, H)
        positions = torch.arange(total)
        if layer_type == "full_attention":
            rope = dgm.DiffusionGemmaRope(1e6, GLOBAL_DIM, 0.25, 128, hidden.device, torch.float32)
        else:
            rope = dgm.DiffusionGemmaRope(1e4, SLIDING_DIM, 1.0, 128, hidden.device, torch.float32)
        attn_inputs = _attn_inputs(total, positions)
        attn_op = dgm.DiffusionGemmaAttnOp(
            layer_type, HEADS,
            GLOBAL_KV if layer_type == "full_attention" else SLIDING_KV,
            GLOBAL_DIM if layer_type == "full_attention" else SLIDING_DIM,
            WINDOW, PAGE, torch.float32, None,
        )
        with patch.object(dgm, "RMSNorm", _TorchRMSNorm):
            layer = dgm.DiffusionGemmaLayer(
                config, ParallelismConfig(), _model_weights(hf), 0, layer_type, None
            )
            out = layer(
                hidden, rope, positions, attn_op, None, attn_inputs,
                causal=causal, write_kv=False, decoder_mode=decoder_mode,
            )
        ref = _ref_layer(hidden, hf, layer_type, positions, causal, decoder_mode)
        torch.testing.assert_close(out, ref, rtol=2e-4, atol=2e-4)
        return out

    def test_sliding_layer_encoder_causal(self):
        self._run_layer("sliding_attention", causal=True, decoder_mode=False)

    def test_sliding_layer_canvas_bidirectional(self):
        self._run_layer("sliding_attention", causal=False, decoder_mode=True)

    def test_global_layer_encoder_causal(self):
        self._run_layer("full_attention", causal=True, decoder_mode=False)

    def test_global_layer_canvas(self):
        self._run_layer("full_attention", causal=False, decoder_mode=True)

    def test_layer_scalar_modes_differ(self):
        hf = _make_hf_weights("sliding_attention", 3)
        self.assertNotEqual(
            hf["layer_scalar_enc"].item(), hf["layer_scalar_dec"].item()
        )

    def test_prefix_window_attention_matches_reference(self):
        """Canvas with a 10-token prefix: only the window tail is visible."""
        layer_type = "sliding_attention"
        torch.manual_seed(5)
        prefix_len = 10
        prefix_k = torch.randn(prefix_len, SLIDING_KV, SLIDING_DIM)
        prefix_v = torch.randn(prefix_len, SLIDING_KV, SLIDING_DIM)
        positions = torch.arange(prefix_len, prefix_len + CANVAS)

        # my attn op reads the prefix from the paged mock cache.
        cache = _MockLayerKVCache(prefix_k, prefix_v)
        pages = (prefix_len + PAGE - 1) // PAGE
        block_table = torch.arange(pages, dtype=torch.int32).view(1, -1)
        attn_inputs = _attn_inputs(
            CANVAS, positions, 1,
            prefix_lens=torch.tensor([prefix_len], dtype=torch.int32),
            block_table=block_table,
        )
        attn_op = dgm.DiffusionGemmaAttnOp(
            layer_type, HEADS, SLIDING_KV, SLIDING_DIM, WINDOW, PAGE, torch.float32, None
        )
        torch.manual_seed(5)
        q = torch.randn(CANVAS, HEADS, SLIDING_DIM)
        k = torch.randn(CANVAS, SLIDING_KV, SLIDING_DIM)
        v = torch.randn(CANVAS, SLIDING_KV, SLIDING_DIM)
        out = attn_op.forward(
            q, k, v, cache, attn_inputs, positions, causal=False, write_kv=False
        )
        # reference: score against [prefix | current] with the window mask.
        keys = torch.cat([prefix_k, k], dim=0)
        values = torch.cat([prefix_v, v], dim=0)
        groups = HEADS // SLIDING_KV
        keys = keys.repeat_interleave(groups, dim=1)
        values = values.repeat_interleave(groups, dim=1)
        score = torch.einsum("qhd,khd->hqk", q, keys)
        allowed = torch.ones(CANVAS, keys.shape[0], dtype=torch.bool)
        for qi in range(CANVAS):
            qpos = prefix_len + qi
            for ki in range(keys.shape[0]):
                if abs(ki - qpos) > WINDOW - 1:
                    allowed[qi, ki] = False
        score = score.masked_fill(~allowed.unsqueeze(0), float("-inf"))
        probs = torch.softmax(score, dim=-1)
        ref = torch.einsum("hqk,khd->qhd", probs, values)
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


class DiffusionGemmaPagedWriteParityTest(TestCase):
    """GPU parity for the torch-attention-path KV write/gather rewrite.

    Global layers run attention on the torch path but now write KV through
    the paged write op (flashinfer.page append) and read the committed
    prefix through the vectorized gather. Both must reproduce the reference
    loop bit-exactly: same rows, same slots, same batch-concatenated order.
    """

    def _gpu(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA device required")
        return torch.device("cuda:0")

    # production global-layer geometry: flashinfer's append kernel requires
    # head_dim in {64, 128, 256, 512}; 512 exercises the real global layers.
    GQ, GKV, GD, GP = 8, 2, 512, 16

    def _mk_op(self, layer_type, device, dtype, paged):
        op = dgm.DiffusionGemmaAttnOp(
            layer_type,
            self.GQ,
            self.GKV,
            self.GD,
            WINDOW,
            self.GP,
            dtype,
            None,
        )
        # Force the production global-layer condition: torch attention with
        # the paged write still enabled (head_dim>256 only disables the
        # flashinfer attention wrappers, not the write op).
        op.use_flashinfer = False
        if not paged:
            op.use_paged_write = False
        return op

    def _attn_inputs(self, device, prefix_lens, span, table, table_dev):
        batch = prefix_lens.numel()
        cu = torch.zeros(batch + 1, dtype=torch.int32, device=device)
        cu[1:] = torch.cumsum(span, dim=0)
        return SimpleNamespace(
            prefix_lengths=prefix_lens,
            input_lengths=span,
            cu_seqlens=cu.cpu(),
            cu_seqlens_device=cu,
            kv_cache_kernel_block_id=table,
            kv_cache_kernel_block_id_device=table_dev,
            combo_position_ids=None,
            sequence_lengths=None,
            is_target_verify=False,
        )

    def test_paged_write_matches_torch_loop(self):
        device = self._gpu()
        torch.manual_seed(23)
        P, Hkv, D = self.GP, self.GKV, self.GD
        # two requests, unequal prefixes and spans, several pages per request
        prefix = torch.tensor([9, 13], dtype=torch.int32, device=device)
        span = torch.tensor([7, 5], dtype=torch.int32, device=device)
        total = int(span.sum())
        pages_needed = [int((prefix[b] + span[b] + P - 1) // P) for b in range(2)]
        num_pages = sum(pages_needed)
        # distinct pages per request; pad columns are never read
        table = torch.zeros(2, max(pages_needed), dtype=torch.int32)
        cur = 0
        for b in range(2):
            table[b, : pages_needed[b]] = torch.arange(cur, cur + pages_needed[b], dtype=torch.int32)
            cur += pages_needed[b]
        table_dev = table.to(device)
        k = torch.randn(total, Hkv, D, device=device, dtype=torch.bfloat16)
        v = torch.randn(total, Hkv, D, device=device, dtype=torch.bfloat16)
        q = torch.randn(total, self.GQ, D, device=device, dtype=torch.bfloat16)
        positions = torch.cat(
            [
                torch.arange(int(prefix[b]), int(prefix[b] + span[b]), device=device)
                for b in range(2)
            ]
        )
        caches = {}
        for name, paged in (("paged", True), ("loop", False)):
            op = self._mk_op("full_attention", device, torch.bfloat16, paged)
            attn_inputs = self._attn_inputs(device, prefix, span, table, table_dev)
            op.plan(attn_inputs, causal=True)
            cache = SimpleNamespace(
                kv_cache_base=torch.zeros(
                    num_pages, 2, Hkv, P, D,
                    dtype=torch.bfloat16, device=device,
                ),
                tag="global_kv",
            )
            out = op.forward(
                q, k, v, cache, attn_inputs, positions, causal=True, write_kv=True
            )
            caches[name] = (cache, out)
        base_p = caches["paged"][0].kv_cache_base.reshape(-1)
        base_l = caches["loop"][0].kv_cache_base.reshape(-1)
        self.assertTrue(torch.equal(base_p, base_l))
        self.assertTrue(torch.equal(caches["paged"][1], caches["loop"][1]))
        # the written rows must be exactly the k/v rows at the right slots
        cache = caches["paged"][0].kv_cache_base.view(-1, 2, Hkv, P, D)
        pos = 0
        for b in range(2):
            for r in range(int(span[b])):
                gpos = int(prefix[b]) + r
                page = int(table[b, gpos // P])
                slot = gpos % P
                self.assertTrue(torch.equal(cache[page, 0, :, slot, :], k[pos + r]))
                self.assertTrue(torch.equal(cache[page, 1, :, slot, :], v[pos + r]))
            pos += int(span[b])

    def test_gather_prefix_matches_loop_reference(self):
        device = self._gpu()
        torch.manual_seed(29)
        P, Hkv, D = self.GP, self.GKV, self.GD
        prefix = torch.tensor([11, 0, 13], dtype=torch.int32, device=device)
        batch = 3
        pages_per = [int((int(p) + P - 1) // P) for p in prefix]
        NB = sum(pages_per) + 4
        table = torch.zeros(batch, max(pages_per) + 2, dtype=torch.int32)
        cur = 0
        for b in range(batch):
            for pb in range(pages_per[b]):
                table[b, pb] = cur
                cur += 1
        paged_cache = torch.randn(NB, 2, Hkv, P, D, device=device, dtype=torch.float32)
        cache = SimpleNamespace(kv_cache_base=paged_cache, tag="global_kv")
        attn_inputs = SimpleNamespace(
            kv_cache_kernel_block_id=None,
            kv_cache_kernel_block_id_device=table.to(device),
        )
        op = self._mk_op("full_attention", device, torch.float32, True)
        got_k, got_v = op._gather_prefix(cache, attn_inputs, prefix, batch)

        def ref_loop():
            bt = table.to(device)
            gathered_k = paged_cache[bt.to(torch.long), 0]
            gathered_v = paged_cache[bt.to(torch.long), 1]
            k_rows, v_rows = [], []
            for b in range(batch):
                plen = int(prefix[b])
                if plen <= 0:
                    k_rows.append(gathered_k.new_zeros(0, Hkv, D))
                    v_rows.append(gathered_v.new_zeros(0, Hkv, D))
                    continue
                k_parts, v_parts = [], []
                for pb in range((plen + P - 1) // P):
                    take = min(plen - pb * P, P)
                    k_parts.append(gathered_k[b, pb, :, :take, :])
                    v_parts.append(gathered_v[b, pb, :, :take, :])
                k_rows.append(torch.cat(k_parts, dim=1).transpose(0, 1))
                v_rows.append(torch.cat(v_parts, dim=1).transpose(0, 1))
            return torch.cat(k_rows, dim=0), torch.cat(v_rows, dim=0)

        ref_k, ref_v = ref_loop()
        self.assertEqual(got_k.shape, ref_k.shape)
        self.assertTrue(torch.equal(got_k, ref_k))
        self.assertTrue(torch.equal(got_v, ref_v))


class DiffusionGemmaRopeTest(TestCase):
    def test_sliding_rope_matches_reference(self):
        rope = dgm.DiffusionGemmaRope(1e4, SLIDING_DIM, 1.0, 64, torch.device("cpu"), torch.float32)
        positions = torch.arange(17)
        q = torch.randn(17, 2, SLIDING_DIM)
        cos, sin = _ref_rope(SLIDING_DIM, 1e4, 1.0, positions)
        ref = _ref_apply_rope(q, cos, sin)
        out, _ = rope.apply(q, q.clone(), positions)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_global_proportional_rope_matches_reference(self):
        rope = dgm.DiffusionGemmaRope(1e6, GLOBAL_DIM, 0.25, 64, torch.device("cpu"), torch.float32)
        positions = torch.arange(13)
        q = torch.randn(13, 2, GLOBAL_DIM)
        cos, sin = _ref_rope(GLOBAL_DIM, 1e6, 0.25, positions)
        ref = _ref_apply_rope(q, cos, sin)
        out, _ = rope.apply(q, q.clone(), positions)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
        # proportional rope: rope_angles = int(0.25*32//2) = 4 -> dims
        # [0:4) and [16:20) rotate; the zero-frequency nope dims in both
        # halves stay identity.
        half = GLOBAL_DIM // 2
        rope_angles = int(0.25 * GLOBAL_DIM // 2)
        torch.testing.assert_close(
            out[..., rope_angles:half], q[..., rope_angles:half], rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            out[..., half + rope_angles:], q[..., half + rope_angles:], rtol=1e-6, atol=1e-6
        )


class EntropyBoundDenoiserTest(TestCase):
    def test_accept_lowest_entropy_under_bound(self):
        from rtp_llm.models_py.model_desc.diffusion_gemma_proposer import (
            EntropyBoundDenoiser,
        )

        generator = torch.Generator().manual_seed(3)
        denoiser = EntropyBoundDenoiser(
            CANVAS, 64, 8, 0.4, 0.8, entropy_bound=0.5,
            confidence_threshold=1e-9, stability_threshold=1, generator=generator,
        )
        canvas = denoiser.initialize_canvas(1, torch.device("cpu"))
        logits = torch.randn(1, CANVAS, 64, generator=generator)
        renoised, mask = denoiser.step(canvas, logits)
        dist = torch.distributions.Categorical(logits=logits)
        entropy = dist.entropy()
        sorted_entropy, sorted_idx = torch.sort(entropy, dim=-1)
        cum = torch.cumsum(sorted_entropy, dim=-1)
        expected = cum - sorted_entropy <= 0.5
        expected_mask = torch.scatter(
            torch.zeros_like(expected), -1, sorted_idx, expected
        )
        torch.testing.assert_close(mask.float(), expected_mask.float())
        self.assertTrue(((renoised >= 0) & (renoised < 64)).all())

    def test_temperature_schedule_matches_reference(self):
        from rtp_llm.models_py.model_desc.diffusion_gemma_proposer import (
            EntropyBoundDenoiser,
        )

        denoiser = EntropyBoundDenoiser(4, 8, 48, 0.4, 0.8, 0.1, 0.005, 1)
        # reference iterates steps REMAINING (reversed range): t = t_min +
        # (t_max - t_min) * (remaining / max_steps); our loop counts elapsed
        # steps, so temperature(0) = t_max and decays to ~t_min at the end.
        self.assertAlmostEqual(denoiser.temperature(0), 0.8, places=6)
        self.assertAlmostEqual(denoiser.temperature(24), 0.4 + 0.4 * (24 / 48), places=6)
        self.assertAlmostEqual(denoiser.temperature(47), 0.4 + 0.4 * (1 / 48), places=6)


if __name__ == "__main__":
    main()
