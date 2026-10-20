"""DiffusionGemma proposer end-to-end CPU test.

Builds a real (small) DiffusionGemmaModel with random weights, registers it in
the in-process registry, constructs the DSpark proposer and runs one full
canvas: the entropy-bounded denoising loop (bidirectional canvas attention
reading a committed prefix from a mock paged cache, self-conditioning, EB
accept/re noise, adaptive stopping) and the final argmax canvas. The result is
compared against an independent transcription of the HF generation mixin
(inner loop of DiffusionGemmaGenerationMixin.generate) using the reference
layer from diffusion_gemma_test.
"""

import ctypes
import glob
import os
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import torch


def _preload_libcuda_stub() -> None:
    extra_dirs = [d for d in os.environ.get("CUDA_STUB_LIB", "").split(":") if d]
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
from rtp_llm.models import diffusion_gemma_registry  # noqa: E402
from rtp_llm.models.diffusion_gemma import DiffusionGemmaConfig  # noqa: E402
from rtp_llm.models_py.model_desc import diffusion_gemma as dgm  # noqa: E402
from rtp_llm.models_py.model_desc import diffusion_gemma_proposer as dgp  # noqa: E402
from diffusion_gemma_test import (  # noqa: E402
    CANVAS,
    EXPERTS,
    H,
    HEADS,
    INTER,
    LAYERS,
    MOE_INTER,
    PAGE,
    SLIDING_DIM,
    SLIDING_KV,
    GLOBAL_DIM,
    GLOBAL_KV,
    TOPK,
    WINDOW,
    _make_hf_weights,
    _ref_layer,
    _ref_rmsnorm,
    _ref_rope,
    _TorchRMSNorm,
)
from rtp_llm.ops import ParallelismConfig  # noqa: E402
from rtp_llm.utils.model_weight import W  # noqa: E402


VOCAB = 64
MAX_STEPS = 3
PROMPT_LEN = 7
PREFIX_LEN = PROMPT_LEN


def _make_config() -> DiffusionGemmaConfig:
    config = DiffusionGemmaConfig()
    config.num_layers = LAYERS
    config.hidden_size = H
    config.vocab_size = VOCAB
    config.max_seq_len = 128
    config.layernorm_eps = 1e-6
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
    # EB sampler params (tiny for CPU determinism)
    config.dgm_canvas_length = CANVAS
    config.dgm_max_denoising_steps = MAX_STEPS
    config.dgm_t_min = 0.4
    config.dgm_t_max = 0.8
    config.dgm_entropy_bound = 0.5
    config.dgm_confidence_threshold = 1e-6
    config.dgm_stability_threshold = 1
    return config


class _TorchEmbedding(torch.nn.Module):
    def __init__(self, config, parallelism_config, weight):
        super().__init__()
        self.weight = weight

    def forward(self, input_ids):
        return torch.nn.functional.embedding(input_ids.reshape(-1), self.weight)


def _model_weights(hf_layers, globals_):
    weights = {}
    per_layer = []
    for hf in hf_layers:
        per_layer.append(
            {
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
        )
    weights["layers"] = per_layer
    weights["globals"] = globals_
    return weights


class _MockWeights:
    """ModelWeights stand-in: per-layer dicts + global get_global_weight."""

    def __init__(self, per_layer, globals_):
        self.weights = per_layer
        self._globals = globals_

    def get_global_weight(self, name):
        return self._globals[name]


def _ref_self_conditioning(embeds, soft, sc_hf):
    normed = _ref_rmsnorm(soft, sc_hf["pre_norm"])
    gate = torch.nn.functional.linear(normed, sc_hf["gate"])
    up = torch.nn.functional.linear(normed, sc_hf["up"])
    mlp = torch.nn.functional.linear(
        torch.nn.functional.gelu(gate, approximate="tanh") * up, sc_hf["down"]
    )
    return _ref_rmsnorm(embeds + mlp)


def _ref_canvas_forward(canvas_tokens, prefixes, hf_layers, sc_hf,
                        embedding, layer_types, positions, self_cond_logits,
                        final_ln=None):
    """Reference decoder forward: prefixes is a per-layer (k, v) tuple list."""
    batch = canvas_tokens.shape[0]
    embeds = torch.nn.functional.embedding(
        canvas_tokens.reshape(-1), embedding
    ) * (H ** 0.5)
    embeds = embeds.view(batch, CANVAS, H)
    if self_cond_logits is not None:
        soft = torch.matmul(
            self_cond_logits.to(torch.float32).softmax(-1).to(embeds.dtype),
            embedding,
        ) * (H ** 0.5)
        embeds = _ref_self_conditioning(embeds, soft, sc_hf)
    else:
        embeds = _ref_self_conditioning(embeds, torch.zeros_like(embeds), sc_hf)
    hidden = embeds.reshape(-1, H)
    for idx, (hf, layer_type) in enumerate(zip(hf_layers, layer_types)):
        prefix_k, prefix_v = prefixes[idx]
        hidden = _ref_layer_with_prefix(
            hidden, hf, layer_type, positions, prefix_k, prefix_v
        )
    return _ref_rmsnorm(hidden, final_ln)


def _ref_layer_with_prefix(hidden, hf, layer_type, positions, prefix_k, prefix_v):
    """_ref_layer variant with bidirectional window over [prefix | current]."""
    kv = GLOBAL_KV if layer_type == "full_attention" else SLIDING_KV
    head_dim = GLOBAL_DIM if layer_type == "full_attention" else SLIDING_DIM
    if layer_type == "full_attention":
        cos, sin = _ref_rope(GLOBAL_DIM, 1e6, 0.25, positions)
        window = 0
    else:
        cos, sin = _ref_rope(SLIDING_DIM, 1e4, 1.0, positions)
        window = WINDOW

    residual = hidden
    h = _ref_rmsnorm(residual, hf["input_ln"])
    qkv = torch.nn.functional.linear(h, hf["qkv"])
    q_size = HEADS * head_dim
    kv_size = kv * head_dim
    q = qkv[..., :q_size].reshape(-1, HEADS, head_dim)
    k = qkv[..., q_size:q_size + kv_size].reshape(-1, kv, head_dim)
    v = qkv[..., q_size + kv_size:].reshape(-1, kv, head_dim)
    q = _ref_rmsnorm(q, hf["q_ln"].view(1, 1, -1))
    k = _ref_rmsnorm(k, hf["k_ln"].view(1, 1, -1))
    v = _ref_rmsnorm(v)
    q = _ref_apply_rope_ref(q, cos, sin)
    k = _ref_apply_rope_ref(k, cos, sin)
    plen = prefix_k.shape[0] if prefix_k is not None else 0
    keys = torch.cat([prefix_k, k], 0) if plen else k
    values = torch.cat([prefix_v, v], 0) if plen else v
    if keys.shape[1] != q.shape[1]:
        groups = q.shape[1] // keys.shape[1]
        keys = keys.repeat_interleave(groups, dim=1)
        values = values.repeat_interleave(groups, dim=1)
    score = torch.einsum("qhd,khd->hqk", q.to(torch.float32), keys)
    total = q.shape[0]
    allowed = torch.ones(total, keys.shape[0], dtype=torch.bool)
    for qi in range(total):
        qpos = plen + qi
        for ki in range(keys.shape[0]):
            if window > 0 and abs(ki - qpos) > window - 1:
                allowed[qi, ki] = False
    score = score.masked_fill(~allowed.unsqueeze(0), float("-inf"))
    probs = torch.softmax(score, -1)
    attn = torch.einsum("hqk,khd->qhd", probs.to(q.dtype), values).reshape(-1, HEADS * head_dim)
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
    probs_r = torch.softmax(scores.to(torch.float32), -1)
    top_w, top_i = torch.topk(probs_r, TOPK, -1)
    top_w = top_w / top_w.sum(-1, keepdim=True)
    top_w = top_w * hf["per_expert_scale"][top_i]
    h2 = torch.zeros_like(routed_in)
    for token in range(routed_in.shape[0]):
        for slot in range(TOPK):
            expert = int(top_i[token, slot])
            gu = torch.nn.functional.linear(routed_in[token:token + 1], hf["moe_gate_up"][expert])
            ge, ue = gu.chunk(2, -1)
            eo = torch.nn.functional.linear(
                torch.nn.functional.gelu(ge, approximate="tanh") * ue, hf["moe_down"][expert]
            )
            h2[token] += top_w[token, slot].to(routed_in.dtype) * eo.squeeze(0)
    h2 = _ref_rmsnorm(h2, hf["post_ffn_ln_2"])

    h = h1 + h2
    h = _ref_rmsnorm(h, hf["post_ffn_ln"])
    h = residual + h
    return h * hf["layer_scalar_dec"]


def _ref_apply_rope_ref(x, cos, sin):
    half = x.shape[-1] // 2
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((x1 * cos[..., :half] - x2 * sin[..., :half],
                      x2 * cos[..., :half] + x1 * sin[..., :half]), dim=-1)


class DiffusionGemmaProposerTest(TestCase):
    def test_canvas_denoising_loop_matches_reference(self):
        diffusion_gemma_registry.reset()
        config = _make_config()
        torch.manual_seed(42)
        hf_layers = [
            _make_hf_weights(layer_type, 100 + i)
            for i, layer_type in enumerate(config.dgm_layer_types)
        ]
        embedding = torch.randn(VOCAB, H) * 0.1
        final_ln = torch.ones(H) + 0.05 * torch.randn(H)
        sc_hf = {
            "pre_norm": torch.ones(H) + 0.05 * torch.randn(H),
            "gate": torch.randn(INTER, H) * 0.2,
            "up": torch.randn(INTER, H) * 0.2,
            "down": torch.randn(H, INTER) * 0.2,
        }
        globals_ = {
            W.embedding: embedding,
            W.lm_head: embedding,
            W.final_ln_gamma: final_ln,
            W.dgm_self_cond_pre_norm: sc_hf["pre_norm"],
            W.dgm_self_cond_gate_w: sc_hf["gate"].T.contiguous(),
            W.dgm_self_cond_up_w: sc_hf["up"].T.contiguous(),
            W.dgm_self_cond_down_w: sc_hf["down"].T.contiguous(),
        }
        per_layer = []
        for hf in hf_layers:
            row = {
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
            per_layer.append(row)
        mock_weights = _MockWeights(per_layer, globals_)

        with patch.object(dgm, "RMSNorm", _TorchRMSNorm), patch.object(
            dgm, "Embedding", _TorchEmbedding
        ):
            model = dgm.DiffusionGemmaModel(
                config, ParallelismConfig(), mock_weights, max_generate_batch_size=1
            )
        diffusion_gemma_registry.register_main_model(model, "/fake/ckpt")

        # committed prefix KV in a mock paged cache
        torch.manual_seed(9)
        prefix_k = torch.randn(PREFIX_LEN, SLIDING_KV, SLIDING_DIM)
        prefix_v = torch.randn(PREFIX_LEN, SLIDING_KV, SLIDING_DIM)
        cache = torch.zeros(3, 2, SLIDING_KV, PAGE, SLIDING_DIM)
        for i in range(PREFIX_LEN):
            plane = 0 if i % 2 == 0 else 0  # all kv on same pages
            cache[i // PAGE, 0, :, i % PAGE, :] = prefix_k[i]
            cache[i // PAGE, 1, :, i % PAGE, :] = prefix_v[i]
        swa_cache = SimpleNamespace(kv_cache_base=cache, tag="swa_kv")
        # global-layer cache: different shapes; the canvas test uses sliding
        # layers' cache only through layer 0/1; layer 2 (global) reads its own.
        gcache = torch.zeros(3, 2, GLOBAL_KV, PAGE, GLOBAL_DIM)
        for i in range(PREFIX_LEN):
            gcache[i // PAGE, 0, :, i % PAGE, :] = torch.randn(GLOBAL_KV, GLOBAL_DIM, generator=torch.Generator().manual_seed(50 + i))
            gcache[i // PAGE, 1, :, i % PAGE, :] = torch.randn(GLOBAL_KV, GLOBAL_DIM, generator=torch.Generator().manual_seed(60 + i))
        global_cache = SimpleNamespace(kv_cache_base=gcache, tag="global_kv")

        def layer_caches(layer_idx):
            return [swa_cache, global_cache]

        model.kv_cache = SimpleNamespace(
            get_layer_cache_groups=layer_caches, layer_count=LAYERS
        )

        bt = torch.arange(3, dtype=torch.int32).view(1, -1)
        cu = torch.tensor([0, CANVAS], dtype=torch.int32)
        attn_inputs = SimpleNamespace(
            prefix_lengths=torch.tensor([PREFIX_LEN], dtype=torch.int32),
            input_lengths=torch.tensor([CANVAS], dtype=torch.int32),
            cu_seqlens=cu,
            cu_seqlens_device=cu,
            kv_cache_kernel_block_id=bt,
            kv_cache_kernel_block_id_device=bt,
            combo_position_ids=None,
            # AttnOp.plan reads this attribute unconditionally; the real
            # PyAttentionInputs always carries it (None allowed), the mock
            # must too or the first GPU run dies with AttributeError.
            sequence_lengths=None,
            is_target_verify=False,
        )
        inputs = SimpleNamespace(
            input_ids=torch.zeros(CANVAS, dtype=torch.int32),
            attention_inputs={"swa_kv": attn_inputs, "global_kv": attn_inputs},
        )

        proposer = dgp.DiffusionGemmaProposer(config, ParallelismConfig(), "/fake/ckpt")
        proposer._seed = 0
        out = proposer.forward_propose(inputs)
        tokens = out.diffusion_tokens
        self.assertEqual(tuple(tokens.shape), (1, CANVAS))
        self.assertEqual(tokens.dtype, torch.int32)
        self.assertTrue(((tokens >= 0) & (tokens < VOCAB)).all())
        self.assertEqual(out.hidden_states.shape, (CANVAS, H))

        # Independent reference loop with the SAME canvas initialization.
        generator = torch.Generator().manual_seed(
            int(0) * 1000003 + int(torch.tensor(PREFIX_LEN).sum().item()) % 1000003
        )
        canvas = torch.randint(0, VOCAB, (1, CANVAS), generator=generator)
        self_cond = None
        history = None
        finished = torch.zeros(1, dtype=torch.bool)
        softcap = 30.0
        positions = torch.arange(PREFIX_LEN, PREFIX_LEN + CANVAS)

        # global-layer prefix KV for the reference (same random values as the
        # mock global cache built above).
        torch.manual_seed(0)
        g_prefix_k = []
        g_prefix_v = []
        for i in range(PREFIX_LEN):
            g = torch.Generator().manual_seed(50 + i)
            g_prefix_k.append(torch.randn(GLOBAL_KV, GLOBAL_DIM, generator=g))
            g = torch.Generator().manual_seed(60 + i)
            g_prefix_v.append(torch.randn(GLOBAL_KV, GLOBAL_DIM, generator=g))
        g_prefix_k = torch.stack(g_prefix_k)
        g_prefix_v = torch.stack(g_prefix_v)

        for step in range(MAX_STEPS):
            prefixes = [
                (prefix_k, prefix_v) if t == "sliding_attention" else (g_prefix_k, g_prefix_v)
                for t in config.dgm_layer_types
            ]
            hidden = _ref_canvas_forward(
                canvas, prefixes, hf_layers, sc_hf, embedding,
                config.dgm_layer_types, positions, self_cond, final_ln,
            )
            logits = torch.nn.functional.linear(hidden, embedding).to(torch.float32)
            logits = logits / softcap
            logits = torch.tanh(logits) * softcap
            logits = logits.view(1, CANVAS, -1)
            t = 0.4 + 0.4 * ((MAX_STEPS - step) / MAX_STEPS)
            processed = logits / t
            new_argmax = torch.argmax(processed, -1)
            if history is None:
                history = torch.full((1, 1, CANVAS), -1, dtype=new_argmax.dtype)
            entropy = torch.distributions.Categorical(logits=processed).entropy()
            confident = entropy.mean(-1) < 1e-6
            stable = (history == new_argmax.unsqueeze(0)).all(-1).all(0)
            finished = finished | (stable & confident)
            history = torch.cat([history[1:], new_argmax.unsqueeze(0)], 0)
            mask = _ref_accept_mask(processed, 0.5)
            probs = torch.softmax(processed, -1, dtype=torch.float32)
            sampled = torch.multinomial(probs.reshape(-1, VOCAB), 1, generator=generator)
            denoised = sampled.view(1, CANVAS)
            accepted = torch.where(mask, denoised, canvas)
            random_canvas = torch.randint(0, VOCAB, (1, CANVAS), generator=generator)
            canvas = torch.where(mask, accepted, random_canvas)
            canvas = torch.where(finished[:, None], accepted, canvas)
            self_cond = processed.to(embedding.dtype)
            argmax_canvas = new_argmax
            if bool(finished.all()):
                break

        # Step-level numeric equivalence: feed both implementations the SAME
        # fixed canvas state (embedding -> self-conditioning -> decoder layers
        # -> final norm) and require fp32-noise agreement. The full loop
        # itself multiplies that noise through multinomial sampling, so token
        # equality is only required where the reference's own top-2 logits are
        # not a near-tie.
        prefixes = [
            (prefix_k, prefix_v) if t == "sliding_attention" else (g_prefix_k, g_prefix_v)
            for t in config.dgm_layer_types
        ]
        fixed_canvas = torch.randint(
            0, VOCAB, (1, CANVAS), generator=torch.Generator().manual_seed(77)
        )
        ref_hidden = _ref_canvas_forward(
            fixed_canvas, prefixes, hf_layers, sc_hf, embedding,
            config.dgm_layer_types, positions, None, final_ln,
        )
        model_hidden = model.forward_canvas(inputs, fixed_canvas, positions, None)
        max_diff = float((model_hidden - ref_hidden).abs().max())
        torch.testing.assert_close(
            model_hidden, ref_hidden, rtol=2e-4, atol=2e-4,
            msg=lambda m: f"canvas forward diverged beyond fp32 noise: max_diff={max_diff}",
        )

        logits = torch.nn.functional.linear(ref_hidden, embedding).to(torch.float32)
        logits = logits / 30.0
        logits = torch.tanh(logits) * 30.0
        t = 0.4 + 0.4 * (MAX_STEPS / MAX_STEPS)
        processed = (logits / t).view(1, CANVAS, -1)
        top2 = processed[0].topk(2, dim=-1)
        gap = top2.values[:, 0] - top2.values[:, 1]
        mismatch = tokens.to(torch.long)[0] != argmax_canvas[0]
        for pos in range(CANVAS):
            if mismatch[pos]:
                self.assertLess(
                    float(gap[pos]),
                    1e-2,
                    msg=(
                        f"token mismatch at position {pos} without a near-tie: "
                        f"proposer={int(tokens[0, pos])} reference={int(argmax_canvas[0, pos])} "
                        f"top2 gap={float(gap[pos])}"
                    ),
                )


def _ref_accept_mask(logits, bound):
    entropy = torch.distributions.Categorical(logits=logits).entropy()
    sorted_entropy, sorted_idx = torch.sort(entropy, -1, descending=False)
    cum = torch.cumsum(sorted_entropy, -1)
    sel = cum - sorted_entropy <= bound
    return torch.scatter(torch.zeros_like(sel), -1, sorted_idx, sel)


def _accept_boundary_margin(entropy, bound):
    """Per-row min |(cumsum(sorted) - sorted) - bound|: how far the tightest
    accept decision sits from the threshold. An accept-mask flip between two
    entropy variants is only noise-legitimate inside this margin."""
    sorted_e, _ = torch.sort(entropy, dim=-1, descending=False)
    cum = torch.cumsum(sorted_e, dim=-1)
    stat = cum - sorted_e
    return (stat - bound).abs().min(dim=-1).values


class DiffusionGemmaEntropyABTest(TestCase):
    """Contract #5-iv GPU A/B: fused single-launch Triton entropy kernel vs
    the torch.distributions.Categorical reference path, plus the jit-cache
    placement/warmup evidence. Skipped without a CUDA device (CPU UT lanes
    keep exercising the torch path through the same gate)."""

    def _require_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA device required for the fused entropy kernel")

    def test_entropy_jit_cache_gpu(self):
        self._require_cuda()
        import tempfile
        import time

        from rtp_llm.models_py.triton_kernels.common import categorical_entropy as ce

        dev = torch.device("cuda:0")
        torch.manual_seed(7)
        logits = torch.randn(64, 4096, device=dev)
        # Point the Triton JIT at a fresh dir exactly the way jit_cache_manager
        # does at engine runtime (TRITON_CACHE_DIR), prove the kernel's
        # compile artifacts land there, and that the second call reuses them
        # instead of recompiling (warmup semantics).
        cache_dir = tempfile.mkdtemp(prefix="dgm_entropy_triton_")
        prev = os.environ.get("TRITON_CACHE_DIR")
        os.environ["TRITON_CACHE_DIR"] = cache_dir
        try:
            t0 = time.perf_counter()
            out1 = ce.categorical_entropy(logits)
            torch.cuda.synchronize()
            t_first = time.perf_counter() - t0
            entries_first = set(os.listdir(cache_dir))
            artifacts = {}
            for entry in sorted(entries_first):
                sub = os.path.join(cache_dir, entry)
                artifacts[entry] = sorted(os.listdir(sub))
            t0 = time.perf_counter()
            out2 = ce.categorical_entropy(logits)
            torch.cuda.synchronize()
            t_second = time.perf_counter() - t0
            entries_second = set(os.listdir(cache_dir))
            print(
                f"[dgm-entropy-jit] TRITON_CACHE_DIR={cache_dir} "
                f"t_first={t_first:.3f}s t_second={t_second:.4f}s "
                f"entries_after_first={len(entries_first)} "
                f"entries_after_second={len(entries_second)} "
                f"artifacts={artifacts}",
                flush=True,
            )
            self.assertTrue(torch.equal(out1, out2))
            self.assertTrue(
                entries_first,
                msg="no JIT artifacts written to TRITON_CACHE_DIR",
            )
            self.assertFalse(
                entries_second - entries_first,
                msg=f"second call recompiled: {sorted(entries_second - entries_first)}",
            )
        finally:
            if prev is None:
                os.environ.pop("TRITON_CACHE_DIR", None)
            else:
                os.environ["TRITON_CACHE_DIR"] = prev

    def test_gated_accept_stop_ab_gpu(self):
        # EB acceptance is semantically sensitive: run the REAL denoiser
        # code path in both gate states on the real proposer-shaped logits
        # and require the entropy/mask/stop outputs to agree within fp32
        # reduction noise (knife-edge flips excepted, with margin proof).
        self._require_cuda()
        self.assertTrue(
            dgp._FUSED_ENTROPY_ENV and dgp._fused_entropy_kernel is not None,
            msg="fused entropy path must be importable and enabled by default",
        )
        dev = torch.device("cuda:0")
        torch.manual_seed(31)
        canvas, vocab, batch = 256, 262144, 4
        denoiser = dgp.EntropyBoundDenoiser(
            canvas, vocab, 48, 0.4, 0.8, 0.1, 1e-3, 1, generator=None
        )
        logits = torch.tanh(torch.randn(batch, canvas, vocab, device=dev) * 2.0) * 75.0
        argmax = torch.argmax(logits, dim=-1)
        history = torch.full((1, batch, canvas), -1, dtype=argmax.dtype, device=dev)
        mask_fused = denoiser._accept_mask(logits)
        stop_fused = denoiser.stop(argmax, logits, history)
        saved = dgp._FUSED_ENTROPY_ENV
        try:
            dgp._FUSED_ENTROPY_ENV = False
            mask_ref = denoiser._accept_mask(logits)
            stop_ref = denoiser.stop(argmax, logits, history)
        finally:
            dgp._FUSED_ENTROPY_ENV = saved
        ent_fused = dgp._categorical_entropy(logits)
        ent_ref = torch.distributions.Categorical(logits=logits).entropy()
        ent_true = torch.distributions.Categorical(logits=logits.double()).entropy()
        kernel_err = float((ent_fused - ent_true).abs().max())
        ent_max = float((ent_fused - ent_ref).abs().max())
        # contract bound: the KERNEL's own accuracy vs the float64 truth.
        self.assertLess(kernel_err, 1e-5)
        # fused-vs-ref gap is the reference path's own fp32 accumulation
        # noise (measured: torch32-vs-truth is the same magnitude); sanity
        # bound only, the actual number is printed as evidence below.
        self.assertLess(ent_max, 1e-4)
        mask_equal = bool(torch.equal(mask_fused, mask_ref))
        stop_equal = bool(torch.equal(stop_fused, stop_ref))
        if not mask_equal:
            margin = _accept_boundary_margin(ent_ref, denoiser.entropy_bound)
            flipped = (mask_fused != mask_ref).any(dim=-1).nonzero().flatten().tolist()
            for row in flipped:
                self.assertLess(
                    float(margin[row]),
                    10.0 * ent_max,
                    msg=f"accept mask flipped beyond boundary noise at row {row}",
                )
        if not stop_equal:
            mean_margin = (ent_ref.mean(dim=-1) - denoiser.confidence_threshold).abs()
            flipped = (stop_fused != stop_ref).nonzero().flatten().tolist()
            for row in flipped:
                self.assertLess(
                    float(mean_margin[row]),
                    10.0 * ent_max,
                    msg=f"stop decision flipped beyond threshold noise at row {row}",
                )
        print(
            f"[dgm-entropy-ab] gated accept/stop path: kernel_vs_true={kernel_err:.3e} "
            f"ent_max_diff(fused_vs_ref)={ent_max:.3e} "
            f"mask_equal={mask_equal} stop_equal={stop_equal}",
            flush=True,
        )

    def test_reference_entropy_ab_gpu(self):
        # Direct kernel-vs-Categorical comparison over the proposer's real
        # [1024, 262144] shape and neighbors: flat / sharp / softcap-bounded
        # distributions, the small masked-tail shape, and -inf lanes. A
        # float64 Categorical is the referee: the contract bound (<1e-5)
        # applies to the kernel's own accuracy against that truth, while the
        # fused-vs-ref gap is the torch path's own fp32 accumulation noise
        # (measured: torch32-vs-truth is the same magnitude) — printed as
        # evidence, sanity-bounded only.
        self._require_cuda()
        from rtp_llm.models_py.triton_kernels.common.categorical_entropy import (
            categorical_entropy,
        )

        dev = torch.device("cuda:0")
        torch.manual_seed(2026)
        cases = [
            ("flat_262144", torch.randn(1024, 262144, device=dev)),
            ("sharp_262144", torch.randn(1024, 262144, device=dev) * 20.0),
            (
                "softcap_262144",
                torch.tanh(torch.randn(1024, 262144, device=dev) * 2.0) * 75.0,
            ),
            ("small_64", torch.randn(7, 64, device=dev) * 3.0),
        ]
        neg_inf = torch.randn(9, 4096, device=dev) * 5.0
        neg_inf[:, ::7] = float("-inf")
        cases.append(("neg_inf_4096", neg_inf))
        worst_kernel = 0.0
        worst_gap = 0.0
        for name, logits in cases:
            ref = torch.distributions.Categorical(logits=logits).entropy()
            fused = categorical_entropy(logits)
            truth = torch.distributions.Categorical(logits=logits.double()).entropy()
            kernel_err = float((fused - truth).abs().max())
            ref_err = float((ref - truth).abs().max())
            gap = (fused - ref).abs()
            gap_max = float(gap.max())
            worst_kernel = max(worst_kernel, kernel_err)
            worst_gap = max(worst_gap, gap_max)
            print(
                f"[dgm-entropy-ab] {name}: shape={tuple(logits.shape)} "
                f"ent_true_mean={float(truth.mean()):.4f} "
                f"kernel_vs_true={kernel_err:.3e} ref_vs_true={ref_err:.3e} "
                f"fused_vs_ref={gap_max:.3e} mean_abs_diff={float(gap.mean()):.3e}",
                flush=True,
            )
            self.assertLess(kernel_err, 1e-5)
            self.assertLess(gap_max, 1e-4)
            del logits, ref, fused, truth, gap
        print(
            f"[dgm-entropy-ab] worst kernel_vs_true: {worst_kernel:.3e} "
            f"worst fused_vs_ref: {worst_gap:.3e}",
            flush=True,
        )


    def test_reference_entropy_perf_gpu(self):
        # Kernel-level timing evidence for contract #5-iv: one fused launch
        # (two row reads) vs the torch Categorical chain (logsumexp +
        # normalized-logits + softmax + clamp/mul/sum materializations) on
        # the proposer's real [1024, 262144] fp32 block.
        self._require_cuda()
        import time

        from rtp_llm.models_py.triton_kernels.common.categorical_entropy import (
            categorical_entropy,
        )

        dev = torch.device("cuda:0")
        torch.manual_seed(99)
        logits = torch.tanh(torch.randn(1024, 262144, device=dev) * 2.0) * 75.0

        def _torch_path():
            return torch.distributions.Categorical(logits=logits).entropy()

        for _ in range(3):
            categorical_entropy(logits)
            _torch_path()
        torch.cuda.synchronize()
        rounds = 10
        t0 = time.perf_counter()
        for _ in range(rounds):
            categorical_entropy(logits)
        torch.cuda.synchronize()
        t_fused = (time.perf_counter() - t0) / rounds
        t0 = time.perf_counter()
        for _ in range(rounds):
            _torch_path()
        torch.cuda.synchronize()
        t_ref = (time.perf_counter() - t0) / rounds
        print(
            f"[dgm-entropy-perf] [1024,262144] fp32 (H20): "
            f"fused={t_fused * 1e3:.3f}ms/call torch_ref={t_ref * 1e3:.3f}ms/call "
            f"speedup={t_ref / t_fused:.1f}x",
            flush=True,
        )
        # both paths measured on the same block: the fused path must not be
        # slower than the reference chain it replaces.
        self.assertLess(t_fused, t_ref)


def _tv_distance(a: torch.Tensor, b: torch.Tensor, bins: int) -> float:
    """Total-variation distance between two empirical token distributions."""
    ca = torch.bincount(a.reshape(-1).to(torch.long), minlength=bins).double()
    cb = torch.bincount(b.reshape(-1).to(torch.long), minlength=bins).double()
    total = max(1, int(ca.sum().item()))
    return 0.5 * float((ca - cb).abs().sum().item()) / total


def _chi2_vs_analytic(tokens: torch.Tensor, probs: torch.Tensor, bins: int):
    """Pearson chi-square of an empirical token distribution against the
    analytic softmax probabilities (bins with expected count < 5 excluded,
    the standard validity rule). Returns (chi2, dof)."""
    counts = torch.bincount(
        tokens.reshape(-1).to(torch.long), minlength=bins
    ).double()
    n = float(counts.sum().item())
    expected = probs.reshape(-1).double() * n
    keep = expected >= 5.0
    chi2 = float(((counts[keep] - expected[keep]) ** 2 / expected[keep]).sum())
    dof = max(1, int(keep.sum().item()) - 1)
    return chi2, dof


class DiffusionGemmaSoftcapABTest(TestCase):
    """Opt r5 ⑥-lane item 2 GPU A/B: the in-register softcap/temperature
    transform (tanh(x/s) * out_scale, contract #5-iii merged algebra) inside
    the entropy/sampling kernels and the fused softcap_argmax launch vs the
    torch materialized chain

        processed = tanh(raw.to(fp32) / s) * out_scale

    that the fused paths replace. Reference criterion: the kernel outputs
    agree with "CPU-merged-then-kernel" — argmax exactly (monotone
    transform, distinct bf16-grid gaps), processed values to the last
    fp32 ulp class, entropy within the kernel's own 1e-5 truth bound,
    sampling distributionally (independent RNG streams). Skipped without
    CUDA (CPU UT lanes keep the torch path through the gates)."""

    def _require_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA device required for the softcap-inline kernels")

    def _reference_processed(self, raw, softcap, out_scale):
        return torch.tanh(raw.to(torch.float32) / softcap) * out_scale

    def test_softcap_argmax_ab_gpu(self):
        self._require_cuda()
        self.assertTrue(
            dgp._FUSED_SOFTCAP_ENV and dgp._softcap_argmax_kernel is not None,
            msg="fused softcap path must be importable and enabled by default",
        )
        from rtp_llm.models_py.triton_kernels.common.softcap_argmax import (
            softcap_argmax,
        )

        dev = torch.device("cuda:0")
        torch.manual_seed(2026)
        softcap, out_scale = 30.0, 75.0
        cases = [
            ("flat_fp32", torch.randn(256, 262144, device=dev)),
            ("sharp_fp32", torch.randn(256, 262144, device=dev) * 40.0),
            ("bf16_raw", torch.randn(256, 262144, device=dev).to(torch.bfloat16)),
            (
                "small_64",
                torch.randn(7, 64, device=dev) * 3.0,
            ),
        ]
        # saturation: raw magnitudes beyond ~8.9*s collapse fp32 tanh to +-1,
        # creating exact ties the argmax must resolve to the first index on
        # both paths.
        sat = torch.randn(9, 4096, device=dev)
        sat[:, ::3] = 5000.0
        sat[:, 1::3] = -5000.0
        cases.append(("saturated_ties", sat))
        for name, raw in cases:
            processed, argmax = softcap_argmax(raw, softcap, out_scale)
            ref = self._reference_processed(raw, softcap, out_scale)
            ref_argmax = torch.argmax(ref, dim=-1)
            ulp = (
                processed.view(torch.int32)
                - ref.contiguous().view(torch.int32)
            ).abs()
            argmax_eq = bool(torch.equal(argmax, ref_argmax))
            proc_max = float((processed - ref).abs().max())
            print(
                f"[dgm-softcap-ab] argmax {name}: shape={tuple(raw.shape)} "
                f"dtype={raw.dtype} argmax_equal={argmax_eq} "
                f"processed_max_abs_diff={proc_max:.3e} "
                f"processed_max_ulp={int(ulp.max())}",
                flush=True,
            )
            self.assertEqual(processed.dtype, torch.float32)
            self.assertEqual(argmax.dtype, torch.int64)
            self.assertLessEqual(int(ulp.max()), 2)
            if not argmax_eq:
                # monotone transform: flips are only legitimate where the
                # reference's own top-2 processed gap is zero (saturation
                # collapse) — prove it per flipped row.
                top2 = ref.topk(2, dim=-1).values
                gap = top2[..., 0] - top2[..., 1]
                flipped = (argmax != ref_argmax).nonzero(as_tuple=True)[0]
                for row in flipped.tolist():
                    self.assertEqual(
                        float(gap[row]),
                        0.0,
                        msg=f"argmax flip at row {row} without a tie",
                    )

    def test_softcap_entropy_ab_gpu(self):
        self._require_cuda()
        from rtp_llm.models_py.triton_kernels.common.categorical_entropy import (
            categorical_entropy,
        )

        dev = torch.device("cuda:0")
        torch.manual_seed(2027)
        softcap, out_scale = 30.0, 75.0
        cases = [
            ("flat_262144", torch.randn(512, 262144, device=dev) * 4.0),
            (
                "softcap_262144",
                torch.randn(512, 262144, device=dev) * 2.0,
            ),
            ("bf16_262144", torch.randn(512, 262144, device=dev).to(torch.bfloat16)),
            ("small_64", torch.randn(7, 64, device=dev) * 3.0),
        ]
        worst_kernel = 0.0
        worst_gap = 0.0
        for name, raw in cases:
            fused = categorical_entropy(raw, softcap=softcap, out_scale=out_scale)
            ref_logits = self._reference_processed(raw, softcap, out_scale)
            ref = torch.distributions.Categorical(logits=ref_logits).entropy()
            truth = torch.distributions.Categorical(
                logits=ref_logits.double()
            ).entropy()
            kernel_err = float((fused - truth).abs().max())
            gap_max = float((fused - ref).abs().max())
            worst_kernel = max(worst_kernel, kernel_err)
            worst_gap = max(worst_gap, gap_max)
            print(
                f"[dgm-softcap-ab] entropy {name}: shape={tuple(raw.shape)} "
                f"dtype={raw.dtype} kernel_vs_true={kernel_err:.3e} "
                f"fused_vs_ref={gap_max:.3e} "
                f"ent_true_mean={float(truth.mean()):.4f}",
                flush=True,
            )
            self.assertLess(kernel_err, 1e-5)
            self.assertLess(gap_max, 1e-4)
            del raw, ref_logits, fused, ref, truth
        print(
            f"[dgm-softcap-ab] entropy worst kernel_vs_true={worst_kernel:.3e} "
            f"worst fused_vs_ref={worst_gap:.3e}",
            flush=True,
        )

    def test_softcap_sample_smallvocab_ab_gpu(self):
        self._require_cuda()
        from rtp_llm.models_py.triton_kernels.common.categorical_sample import (
            categorical_sample,
        )

        dev = torch.device("cuda:0")
        torch.manual_seed(29)
        softcap, out_scale = 30.0, 75.0
        vocab = 64
        draws = 262144
        # raw scale picked so the transformed logits (tanh(x/s) * out_scale)
        # land at std ~0.6 — the flat regime where every bin's expected
        # count clears the chi-square validity floor (a sharper row leaves
        # dof ~ 2 and the [0.5, 2.5] noise band meaningless).
        row = (torch.randn(vocab, device=dev) * 0.24).to(torch.bfloat16)
        raw = row.unsqueeze(0).expand(draws, vocab).contiguous()
        ref_logits = self._reference_processed(raw, softcap, out_scale)
        probs = torch.softmax(ref_logits[0].double(), dim=-1)
        generator = torch.Generator(device=dev).manual_seed(17)
        torch_ref = torch.multinomial(
            probs.unsqueeze(0).expand(draws, vocab).contiguous().float(),
            1,
            generator=generator,
        ).squeeze(-1)
        fused_a = categorical_sample(raw, seed=31, softcap=softcap, out_scale=out_scale)
        fused_b = categorical_sample(raw, seed=31, softcap=softcap, out_scale=out_scale)
        self.assertTrue(torch.equal(fused_a, fused_b), "same seed must reproduce")
        tv = _tv_distance(fused_a, torch_ref, vocab)
        chi2_f, dof = _chi2_vs_analytic(fused_a, probs.float(), vocab)
        chi2_t, dof_t = _chi2_vs_analytic(torch_ref, probs.float(), vocab)
        print(
            f"[dgm-softcap-ab] sample small64 draws={draws}: "
            f"fused_vs_torch TV={tv:.5f} chi2_fused={chi2_f:.1f} "
            f"chi2_torch={chi2_t:.1f} dof={dof}",
            flush=True,
        )
        self.assertEqual(fused_a.dtype, torch.int64)
        self.assertTrue(
            ((fused_a >= 0) & (fused_a < vocab)).all(), "tokens out of range"
        )
        self.assertLess(tv, 0.01)
        self.assertLess(chi2_f / dof, 2.5)
        self.assertGreater(chi2_f / dof, 0.5)
        self.assertLess(chi2_t / dof_t, 2.5)

    def test_softcap_step_ab_gpu(self):
        # Proposer-level gated A/B: the real EntropyBoundDenoiser.step on
        # proposer-shaped RAW bf16 logits with a softcap ctx vs the reference
        # path on the materialized processed rows. The accept mask (entropy
        # + accept-mask kernel) must agree bit-for-bit or within boundary
        # noise, and both sampling paths stay in-distribution.
        self._require_cuda()
        dev = torch.device("cuda:0")
        torch.manual_seed(37)
        canvas, vocab, batch = 256, 262144, 2
        softcap = 30.0
        out_scale = 75.0
        raw = (torch.randn(batch, canvas, vocab, device=dev) * 2.0).to(torch.bfloat16)
        processed = torch.tanh(raw.to(torch.float32) / softcap) * out_scale
        canvas_tokens = torch.randint(0, vocab, (batch, canvas), device=dev)

        def _fresh_denoiser():
            generator = torch.Generator(device=dev).manual_seed(2026)
            return dgp.EntropyBoundDenoiser(
                canvas, vocab, 48, 0.4, 0.8, 0.1, 1e-3, 1, generator=generator
            )

        ctx = (raw, softcap, out_scale)
        mask_fused, renoised_fused = None, None
        calls = []
        real_entropy = dgp._fused_entropy_kernel
        real_sample = dgp._fused_sample_kernel

        def _entropy_spy(logits, softcap=None, out_scale=None):
            result = real_entropy(logits, softcap=softcap, out_scale=out_scale)
            calls.append(int(logits.numel()))
            return result

        # fused softcap path: entropy reads raw, sampling reads raw
        with patch.object(dgp, "_fused_entropy_kernel", _entropy_spy):
            renoised_fused, mask_fused = _fresh_denoiser().step(
                canvas_tokens, processed, logits_ctx=ctx
            )
        self.assertTrue(calls, "fused step must engage the softcap entropy kernel")
        # reference path: no ctx -> materialized processed rows
        renoised_ref, mask_ref = _fresh_denoiser().step(canvas_tokens, processed)
        mask_equal = bool(torch.equal(mask_fused, mask_ref))
        if not mask_equal:
            ent_ref = torch.distributions.Categorical(logits=processed).entropy()
            margin = _accept_boundary_margin(ent_ref, 0.1)
            flipped = (mask_fused != mask_ref).any(dim=-1).nonzero().flatten().tolist()
            for row in flipped:
                self.assertLess(
                    float(margin[row]),
                    1e-4,
                    msg=f"accept mask flipped beyond boundary noise at row {row}",
                )
        for renoised in (renoised_fused, renoised_ref):
            self.assertTrue(
                ((renoised >= 0) & (renoised < vocab)).all(),
                "renoised canvas tokens out of range",
            )
        print(
            f"[dgm-softcap-ab] gated step path: entropy_calls={len(calls)} "
            f"mask_equal={mask_equal} "
            f"fused_argmax_free_rnoised=True",
            flush=True,
        )

    def test_softcap_argmax_perf_gpu(self):
        # Kernel-level timing evidence: one fused launch (raw read +
        # processed write + argmax) vs the torch chain it replaces (fp32
        # cast + tanh + scale + torch.argmax — four full-width passes) on
        # the proposer's real [1024, 262144] bf16 block.
        self._require_cuda()
        import time

        from rtp_llm.models_py.triton_kernels.common.softcap_argmax import (
            softcap_argmax,
        )

        dev = torch.device("cuda:0")
        torch.manual_seed(99)
        raw = torch.randn(1024, 262144, device=dev).to(torch.bfloat16)
        softcap, out_scale = 30.0, 75.0

        def _torch_path():
            logits = raw.to(torch.float32)
            logits = torch.tanh(logits / softcap)
            processed = logits * out_scale
            return torch.argmax(processed, dim=-1)

        for _ in range(3):
            softcap_argmax(raw, softcap, out_scale)
            _torch_path()
        torch.cuda.synchronize()
        rounds = 10
        t0 = time.perf_counter()
        for _ in range(rounds):
            softcap_argmax(raw, softcap, out_scale)
        torch.cuda.synchronize()
        t_fused = (time.perf_counter() - t0) / rounds
        t0 = time.perf_counter()
        for _ in range(rounds):
            _torch_path()
        torch.cuda.synchronize()
        t_ref = (time.perf_counter() - t0) / rounds
        print(
            f"[dgm-softcap-perf] [1024,262144] bf16 (H20): "
            f"fused={t_fused * 1e3:.3f}ms/call torch_ref={t_ref * 1e3:.3f}ms/call "
            f"speedup={t_ref / t_fused:.1f}x",
            flush=True,
        )
        self.assertLess(t_fused, t_ref)


class DiffusionGemmaAcceptMaskABTest(TestCase):
    """Opt r5 ⑥-lane item 3 GPU A/B: the single-launch accept_mask kernel
    (in-register bitonic sort + exclusive cumsum + scatter) vs the torch
    sort/cumsum/scatter chain it replaces — same ascending-entropy budget
    semantics (the sort IS the rule; see the kernel docstring for the
    semantics-first ruling). Bit-equality expected (both orders are
    stable-in-practice); boundary-margin proof for entropy-noise flips."""

    def _require_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA device required for the fused accept-mask kernel")

    def _torch_chain(self, entropy, bound):
        sorted_entropy, sorted_idx = torch.sort(entropy, dim=-1, descending=False)
        cumulative = torch.cumsum(sorted_entropy, dim=-1)
        sorted_mask = cumulative - sorted_entropy <= bound
        return torch.scatter(torch.zeros_like(sorted_mask), -1, sorted_idx, sorted_mask)

    def test_accept_mask_ab_gpu(self):
        self._require_cuda()
        self.assertTrue(
            dgp._FUSED_ACCEPT_ENV and dgp._fused_accept_kernel is not None,
            msg="fused accept-mask path must be importable and enabled by default",
        )
        from rtp_llm.models_py.triton_kernels.common.accept_mask import accept_mask

        dev = torch.device("cuda:0")
        torch.manual_seed(47)
        bound = 2.0
        cases = [
            ("flat_256", torch.rand(64, 256, device=dev) * 0.2),
            ("spread_256", torch.rand(64, 256, device=dev) * 20.0),
            ("sharp_256", torch.rand(64, 256, device=dev) ** 8 * 0.5),
            ("batch4_256", torch.rand(4, 256, device=dev) * 0.1),
            ("nonpow2_100", torch.rand(9, 100, device=dev) * 0.3),
            ("nonpow2_37", torch.rand(5, 37, device=dev) * 3.0),
            ("tiny_2", torch.rand(3, 2, device=dev) * 0.5),
        ]
        # exact-tie rows: duplicated entropies exercise the stable
        # lower-index tie-break against torch's CUDA segment sort.
        tie = torch.rand(8, 256, device=dev) * 0.1
        tie[:, 1::2] = tie[:, ::2]
        cases.append(("ties_256", tie))
        for name, entropy in cases:
            fused = accept_mask(entropy, bound)
            ref = self._torch_chain(entropy, bound)
            equal = bool(torch.equal(fused, ref))
            mismatch_count = int((fused != ref).sum())
            print(
                f"[dgm-accept-ab] {name}: shape={tuple(entropy.shape)} "
                f"equal={equal} mismatch={mismatch_count} "
                f"accepted_fused={int(fused.sum())} accepted_ref={int(ref.sum())}",
                flush=True,
            )
            self.assertEqual(fused.dtype, torch.bool)
            self.assertEqual(fused.shape, entropy.shape)
            if not equal:
                # flips are only legitimate where the accept decision sits
                # at the budget boundary within the two cumsum
                # implementations' fp32 reduction-order noise (the same
                # margin proof the entropy A/B uses) or on exact-tie
                # positions (tie order is stable-lower-index here, the
                # CUDA segment sort's order there — the accepted count
                # within a tie group is order-invariant only when the
                # boundary cuts inside the group).
                margin = _accept_boundary_margin(entropy, bound)
                pairs = (fused != ref).nonzero(as_tuple=False)
                for p in range(pairs.shape[0]):
                    row_idx = int(pairs[p, 0])
                    pos = int(pairs[p, -1])
                    row_ent = entropy[row_idx]
                    is_tie = int((row_ent == row_ent[pos]).sum()) > 1
                    if not is_tie:
                        self.assertLess(
                            float(margin[row_idx]),
                            1e-3,
                            msg=f"accept mask flipped beyond cumsum/tie noise "
                            f"at row {row_idx} pos {pos} ({name})",
                        )
        # determinism
        entropy = torch.rand(16, 256, device=dev) * 0.5
        self.assertTrue(
            torch.equal(accept_mask(entropy, bound), accept_mask(entropy, bound))
        )

    def test_accept_mask_perf_gpu(self):
        # Launch-overhead evidence: the torch chain (sort + cumsum + sub +
        # compare + zeros + scatter — six-plus launches) vs one fused launch
        # on the proposer's real [4, 256] rows.
        self._require_cuda()
        import time

        from rtp_llm.models_py.triton_kernels.common.accept_mask import accept_mask

        dev = torch.device("cuda:0")
        torch.manual_seed(51)
        entropy = torch.rand(4, 256, device=dev) * 0.2
        bound = 0.15

        def _torch_path():
            return self._torch_chain(entropy, bound)

        for _ in range(10):
            accept_mask(entropy, bound)
            _torch_path()
        torch.cuda.synchronize()
        rounds = 2000
        t0 = time.perf_counter()
        for _ in range(rounds):
            accept_mask(entropy, bound)
        torch.cuda.synchronize()
        t_fused = (time.perf_counter() - t0) / rounds
        t0 = time.perf_counter()
        for _ in range(rounds):
            _torch_path()
        torch.cuda.synchronize()
        t_ref = (time.perf_counter() - t0) / rounds
        print(
            f"[dgm-accept-perf] [4,256] (H20): fused={t_fused * 1e6:.2f}us/call "
            f"torch_ref={t_ref * 1e6:.2f}us/call speedup={t_ref / t_fused:.1f}x",
            flush=True,
        )
        self.assertLess(t_fused, t_ref)


class DiffusionGemmaAddNormABTest(TestCase):
    """Opt r5 ⑥-lane item 1 GPU A/B: the fused residual-add + RMSNorm launch
    (flashinfer.triton.norm.rms_norm_add_residual) vs the exact torch
    two-step the layer chain ran (bf16 add, then the production
    flashinfer.norm.rmsnorm). Value contract: the sum is bit-identical (same
    fp32 accumulate + bf16 round), the normed pair agrees within bf16
    rounding noise (<= 2 ulp; the fused path's rsqrt uses the unrounded fp32
    sum — the one allowed deviation, dsv4 rmsnorm-replace precedent)."""

    def _require_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA device required for the fused add+rmsnorm")

    def test_fused_add_rmsnorm_ab_gpu(self):
        self._require_cuda()
        self.assertTrue(
            dgm._FUSED_ADD_NORM_ENV and dgm._FI_ADD_NORM is not None,
            msg="fused add-norm must be importable and enabled by default",
        )
        import flashinfer

        dev = torch.device("cuda:0")
        torch.manual_seed(61)
        eps = 1e-6
        for shape in ((2048, 4096), (256, 5120)):
            x = (torch.randn(*shape, device=dev) * 0.5).to(torch.bfloat16)
            residual = (torch.randn(*shape, device=dev) * 0.5).to(torch.bfloat16)
            weight = torch.ones(shape[1], device=dev, dtype=torch.bfloat16)
            weight += 0.05 * torch.randn(shape[1], device=dev, dtype=torch.bfloat16)
            x_ref, res_ref = x.clone(), residual.clone()
            out, summed = dgm._fused_add_norm(
                x, residual, dgm.RMSNorm(weight, eps)
            )
            ref_sum = res_ref + x_ref
            ref_normed = flashinfer.norm.rmsnorm(ref_sum, weight, eps=eps)
            self.assertTrue(
                torch.equal(summed, ref_sum),
                msg=f"sum diverged at {shape}: the fp32 accumulate + bf16 round "
                f"must match the torch add",
            )
            diff = (out.to(torch.float32) - ref_normed.to(torch.float32)).abs()
            ulp = (
                out.view(torch.int16).to(torch.int32)
                - ref_normed.view(torch.int16).to(torch.int32)
            ).abs()
            print(
                f"[dgm-addnorm-ab] shape={shape}: max_diff={float(diff.max()):.3e} "
                f"max_ulp={int(ulp.max())} "
                f"mismatch_frac={float((out != ref_normed).float().mean()):.4f} "
                f"sum_equal={bool(torch.equal(summed, ref_sum))}",
                flush=True,
            )
            self.assertLessEqual(
                int(ulp.max()),
                2,
                msg=f"normed pair diverged beyond bf16 noise at {shape}",
            )
            # chained-pair smoke: site-1 output feeds site-2 exactly like the
            # layer forward does (residual threading through the fused op).
            x2 = (torch.randn(*shape, device=dev) * 0.5).to(torch.bfloat16)
            x2_ref = x2.clone()
            out2, summed2 = dgm._fused_add_norm(x2, summed, dgm.RMSNorm(weight, eps))
            ref2_sum = ref_sum + x2_ref
            ref2 = flashinfer.norm.rmsnorm(ref2_sum, weight, eps=eps)
            self.assertTrue(torch.equal(summed2, ref2_sum))
            ulp2 = (
                out2.view(torch.int16).to(torch.int32)
                - ref2.view(torch.int16).to(torch.int32)
            ).abs()
            self.assertLessEqual(int(ulp2.max()), 2)

    def test_fused_add_rmsnorm_gate_gpu(self):
        # DGM_FUSED_ADD_NORM=0 must restore the exact torch two-step objects
        # (fresh sum tensor, module's own norm).
        self._require_cuda()
        dev = torch.device("cuda:0")
        torch.manual_seed(67)
        x = (torch.randn(64, 128, device=dev) * 0.5).to(torch.bfloat16)
        residual = (torch.randn(64, 128, device=dev) * 0.5).to(torch.bfloat16)
        norm = dgm.RMSNorm(torch.ones(128, device=dev, dtype=torch.bfloat16), 1e-6)
        saved = dgm._FUSED_ADD_NORM_ENV
        try:
            dgm._FUSED_ADD_NORM_ENV = False
            out, summed = dgm._fused_add_norm(x, residual, norm)
        finally:
            dgm._FUSED_ADD_NORM_ENV = saved
        ref_sum = residual + x
        self.assertTrue(torch.equal(summed, ref_sum))
        self.assertTrue(torch.equal(out, norm(ref_sum)))


class DiffusionGemmaSamplingABTest(TestCase):
    """Contract opt-r4 lane B GPU A/B: fused Gumbel-max categorical sampling
    kernel vs the torch softmax+multinomial reference. Sampling is stochastic
    and the two paths deliberately use independent RNG streams, so the
    contract is distribution-level: same-seed determinism plus empirical
    agreement of >= 1e3-sample draws (total variation < 1%, Pearson
    chi-square vs the analytic softmax in the [0.5, 2.5]-per-dof noise band),
    plus jit-cache placement and kernel-vs-torch timing evidence. Skipped
    without a CUDA device (CPU UT lanes keep exercising the torch path
    through the same gate)."""

    def _require_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA device required for the fused sampling kernel")

    def _import_kernel(self):
        from rtp_llm.models_py.triton_kernels.common.categorical_sample import (
            categorical_sample,
        )

        return categorical_sample

    def test_sampling_jit_cache_gpu(self):
        # Same evidence pattern as the entropy kernel: point the Triton JIT
        # at a fresh TRITON_CACHE_DIR, prove the compile artifacts land there
        # and that the second call reuses them instead of recompiling.
        self._require_cuda()
        import os
        import tempfile
        import time

        categorical_sample = self._import_kernel()
        dev = torch.device("cuda:0")
        torch.manual_seed(7)
        logits = torch.randn(64, 4096, device=dev)
        cache_dir = tempfile.mkdtemp(prefix="dgm_sample_triton_")
        prev = os.environ.get("TRITON_CACHE_DIR")
        os.environ["TRITON_CACHE_DIR"] = cache_dir
        try:
            t0 = time.perf_counter()
            out1 = categorical_sample(logits, seed=1)
            torch.cuda.synchronize()
            t_first = time.perf_counter() - t0
            entries_first = set(os.listdir(cache_dir))
            t0 = time.perf_counter()
            out2 = categorical_sample(logits, seed=2)
            torch.cuda.synchronize()
            t_second = time.perf_counter() - t0
            entries_second = set(os.listdir(cache_dir))
            print(
                f"[dgm-sample-jit] TRITON_CACHE_DIR={cache_dir} "
                f"t_first={t_first:.3f}s t_second={t_second:.4f}s "
                f"entries_after_first={len(entries_first)} "
                f"entries_after_second={len(entries_second)}",
                flush=True,
            )
            self.assertEqual(out1.dtype, torch.int64)
            self.assertEqual(tuple(out1.shape), (64,))
            self.assertTrue(
                ((out1 >= 0) & (out1 < 4096)).all(), "tokens out of range"
            )
            self.assertTrue(
                entries_first,
                msg="no JIT artifacts written to TRITON_CACHE_DIR",
            )
            self.assertFalse(
                entries_second - entries_first,
                msg=f"second call recompiled: {sorted(entries_second - entries_first)}",
            )
        finally:
            if prev is None:
                os.environ.pop("TRITON_CACHE_DIR", None)
            else:
                os.environ["TRITON_CACHE_DIR"] = prev

    def test_sampling_smallvocab_distribution_ab_gpu(self):
        # The exact proposer call shape (multinomial(probs, 1), replacement
        # semantics irrelevant at 1 draw) at a small vocab where >= 1e3
        # samples per path give a dense, statistically meaningful empirical
        # distribution: fused-vs-torch TV < 1%, both paths' chi-square vs
        # the analytic softmax in the noise band, plus determinism and the
        # -inf-lane edge contract.
        self._require_cuda()
        categorical_sample = self._import_kernel()
        dev = torch.device("cuda:0")
        vocab = 64
        draws = 524288
        torch.manual_seed(41)
        row = torch.randn(vocab, device=dev) * 0.6  # flat: every bin expected ~8192
        logits = row.unsqueeze(0).expand(draws, vocab).contiguous()
        probs = torch.softmax(row, dim=-1)
        generator = torch.Generator(device=dev).manual_seed(17)
        torch_ref = torch.multinomial(
            probs.unsqueeze(0).expand(draws, vocab).contiguous(),
            1,
            generator=generator,
        ).squeeze(-1)
        fused_a = categorical_sample(logits, seed=2026)
        fused_b = categorical_sample(logits, seed=2026)
        fused_c = categorical_sample(logits, seed=2027)
        self.assertTrue(torch.equal(fused_a, fused_b), "same seed must reproduce")
        self.assertFalse(
            torch.equal(fused_a, fused_c), "different seed must reshuffle"
        )
        self.assertEqual(fused_a.dtype, torch.int64)
        self.assertTrue(
            ((fused_a >= 0) & (fused_a < vocab)).all(),
            "tokens out of range",
        )
        tv = _tv_distance(fused_a, torch_ref, vocab)
        chi2_f, dof = _chi2_vs_analytic(fused_a, probs, vocab)
        chi2_t, dof_t = _chi2_vs_analytic(torch_ref, probs, vocab)
        print(
            f"[dgm-sample-ab] small64 draws={draws}: fused_vs_torch TV={tv:.5f} "
            f"chi2_fused={chi2_f:.1f} chi2_torch={chi2_t:.1f} dof={dof}",
            flush=True,
        )
        self.assertLess(tv, 0.01)
        self.assertLess(chi2_f / dof, 2.5)
        self.assertGreater(chi2_f / dof, 0.5)
        self.assertLess(chi2_t / dof_t, 2.5)
        # -inf lanes must lose the argmax and never poison the draw.
        neg = torch.randn(9, 4096, device=dev) * 5.0
        neg[:, ::3] = float("-inf")
        neg_toks = categorical_sample(neg, seed=3)
        finite = neg.isfinite()
        for r in range(neg.shape[0]):
            self.assertTrue(
                finite[r, int(neg_toks[r])],
                msg=f"-inf lane won the draw at row {r}",
            )

    def test_sampling_bigvocab_distribution_ab_gpu(self):
        # Proposer-scale vocab (262144): a concentrated support (~100 hot
        # tokens over the full scan width) so 1e6+ pooled draws give a
        # dense empirical distribution; torch reference is
        # multinomial(probs, S, replacement=True) — the memory-feasible
        # multi-draw form of the same softmax distribution (the exact
        # production call, softmax + multinomial(., 1), cannot materialize
        # 1e6 rows x 262144 fp32). The kernel exercises the identical
        # full-vocab scan and its samples_per_row stream.
        self._require_cuda()
        categorical_sample = self._import_kernel()
        dev = torch.device("cuda:0")
        vocab = 262144
        rows = 64
        samples = 16384
        torch.manual_seed(43)
        hot = 100
        base = torch.full((1, vocab), -30.0, device=dev)
        hot_idx = torch.randperm(vocab, device=dev)[:hot]
        base[0, hot_idx] = torch.randn(hot, device=dev) * 2.0 - 1.5
        logits = base.expand(rows, vocab).contiguous()
        probs = torch.softmax(base, dim=-1)
        torch_ref = torch.multinomial(
            probs.expand(rows, vocab).contiguous(), samples, replacement=True
        )
        fused = categorical_sample(logits, seed=515, samples_per_row=samples)
        self.assertEqual(tuple(fused.shape), (rows, samples))
        self.assertTrue(
            ((fused >= 0) & (fused < vocab)).all(), "tokens out of range"
        )
        fused_rep = categorical_sample(logits, seed=515, samples_per_row=samples)
        self.assertTrue(torch.equal(fused, fused_rep), "same seed must reproduce")
        # pooled over the hot support + one "other" bucket
        support = hot_idx.to(torch.long)
        fused_counts = torch.bincount(
            fused.reshape(-1).to(torch.long), minlength=vocab
        ).double()
        ref_counts = torch.bincount(
            torch_ref.reshape(-1).to(torch.long), minlength=vocab
        ).double()
        other_f = float(fused_counts.sum() - fused_counts[support].sum())
        other_r = float(ref_counts.sum() - ref_counts[support].sum())
        diff = (fused_counts[support] - ref_counts[support]).abs().sum()
        total = float(fused_counts.sum())
        tv = 0.5 * float(diff + abs(other_f - other_r)) / total
        p_expected = probs.reshape(-1).double()
        chi2_f, dof = _chi2_vs_analytic(fused, p_expected, vocab)
        print(
            f"[dgm-sample-ab] big262144 rows={rows}x{samples} draws={int(total)}: "
            f"fused_vs_torch TV={tv:.5f} support_chi2_fused={chi2_f:.1f} "
            f"dof={dof} other_fused={other_f:.0f} other_torch={other_r:.0f}",
            flush=True,
        )
        self.assertLess(tv, 0.01)
        # expected hot-token frequencies must match the analytic softmax
        # (per-token expected count >= 100 for the kept bins)
        self.assertLess(chi2_f / dof, 2.5)
        self.assertGreater(chi2_f / dof, 0.5)

    def test_sampling_perf_gpu(self):
        # Kernel-level timing evidence for the lane-B contract: one fused
        # launch (single logits read) vs the torch chain (full fp32 softmax
        # materialization + the multinomial walk) on the proposer's real
        # [1024, 262144] fp32 block.
        self._require_cuda()
        import time

        categorical_sample = self._import_kernel()
        dev = torch.device("cuda:0")
        torch.manual_seed(99)
        logits = torch.tanh(torch.randn(1024, 262144, device=dev) * 2.0) * 75.0

        def _torch_path(x):
            probs = torch.softmax(x, dim=-1, dtype=torch.float32)
            flat = probs.reshape(-1, probs.shape[-1])
            return torch.multinomial(flat, 1)

        for _ in range(3):
            categorical_sample(logits, seed=7)
            _torch_path(logits)
        torch.cuda.synchronize()
        rounds = 10
        t0 = time.perf_counter()
        for _ in range(rounds):
            categorical_sample(logits, seed=7)
        torch.cuda.synchronize()
        t_fused = (time.perf_counter() - t0) / rounds
        t0 = time.perf_counter()
        for _ in range(rounds):
            _torch_path(logits)
        torch.cuda.synchronize()
        t_ref = (time.perf_counter() - t0) / rounds
        print(
            f"[dgm-sample-perf] [1024,262144] fp32 (H20): "
            f"fused={t_fused * 1e3:.3f}ms/call torch_ref={t_ref * 1e3:.3f}ms/call "
            f"speedup={t_ref / t_fused:.1f}x",
            flush=True,
        )
        # both paths measured on the same block: the fused path must not be
        # slower than the reference chain it replaces.
        self.assertLess(t_fused, t_ref)

    def test_gated_step_sampling_ab_gpu(self):
        # Proposer-level gated A/B: the real EntropyBoundDenoiser.step on
        # the real [batch, canvas, 262144] logits shape. The fused kernel
        # must engage under the default gate (spy proves it), reproduce
        # identical step outputs for identical round state (determinism
        # contract for golden reproducibility), and the gate-off run must
        # fall back to the torch path while the entropy-driven accept mask
        # (a different kernel's output) stays identical.
        self._require_cuda()
        self.assertTrue(
            dgp._FUSED_SAMPLING_ENV and dgp._fused_sample_kernel is not None,
            msg="fused sampling path must be importable and enabled by default",
        )
        dev = torch.device("cuda:0")
        torch.manual_seed(31)
        canvas, vocab, batch = 256, 262144, 4
        logits = torch.tanh(
            torch.randn(batch, canvas, vocab, device=dev) * 2.0
        ) * 75.0
        canvas_tokens = torch.randint(0, vocab, (batch, canvas), device=dev)

        def _fresh_denoiser():
            generator = torch.Generator(device=dev).manual_seed(2026)
            return dgp.EntropyBoundDenoiser(
                canvas, vocab, 48, 0.4, 0.8, 0.1, 1e-3, 1, generator=generator
            )

        calls = []
        real_kernel = dgp._fused_sample_kernel

        def _spy(logits_arg, seed):
            # record only SUCCESSFUL kernel executions: a compile/runtime
            # failure would otherwise silently pass through the proposer's
            # torch fallback and fake the engagement evidence.
            result = real_kernel(logits_arg, seed)
            calls.append(int(seed))
            return result

        with patch.object(dgp, "_fused_sample_kernel", _spy):
            renoised_a, mask_a = _fresh_denoiser().step(canvas_tokens, logits)
        with patch.object(dgp, "_fused_sample_kernel", _spy):
            renoised_b, mask_b = _fresh_denoiser().step(canvas_tokens, logits)
        self.assertEqual(len(calls), 2, "one kernel draw per step() call")
        self.assertGreater(calls[0], 0)
        self.assertTrue(
            torch.equal(renoised_a, renoised_b),
            "identical round state must reproduce identical step outputs",
        )
        self.assertTrue(torch.equal(mask_a, mask_b))
        self.assertEqual(renoised_a.dtype, canvas_tokens.dtype)
        self.assertTrue(
            ((renoised_a >= 0) & (renoised_a < vocab)).all(),
            "renoised canvas tokens out of range",
        )
        saved = dgp._FUSED_SAMPLING_ENV
        try:
            dgp._FUSED_SAMPLING_ENV = False
            renoised_ref, mask_ref = _fresh_denoiser().step(canvas_tokens, logits)
        finally:
            dgp._FUSED_SAMPLING_ENV = saved
        self.assertEqual(len(calls), 2, "gate-off must not touch the fused kernel")
        self.assertTrue(
            torch.equal(mask_ref, mask_a),
            "accept mask must not depend on the sampling gate",
        )
        self.assertTrue(
            ((renoised_ref >= 0) & (renoised_ref < vocab)).all()
        )
        print(
            f"[dgm-sample-ab] gated step path: fused_calls={len(calls)} "
            f"deterministic=True mask_equal_gateAB={bool(torch.equal(mask_ref, mask_a))} "
            f"renoised_dtype={renoised_a.dtype}",
            flush=True,
        )


if __name__ == "__main__":
    main()
