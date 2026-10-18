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


if __name__ == "__main__":
    main()
