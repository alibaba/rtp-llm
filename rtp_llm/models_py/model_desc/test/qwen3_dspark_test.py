"""Model-level UT: Qwen3DSparkModel vs the dense reference golden.

The golden is dumped by dspark_reference.py (same package) over the real
converted draft ckpt with synthetic aux hidden states:

    python3 dspark_reference.py --ckpt <converted_ckpt> --out <golden_dir>

Stages compared (names match the oracle):
    A. combine_hidden_states  -> fused_features
    B. inject_context_kv      -> ctx_k / ctx_v (read back from the paged cache)
    C. block forward          -> head_hidden, base_logits
    D. markov_correct         -> draft_tokens (exact), corrected_logits

The model computes in bf16 over real weights while the golden is fp32 over
the same (bf16) weights, so stage comparisons use scale-aware tolerances;
the greedy draft tokens must match exactly.

Env overrides (test skips if the paths are absent):
    DSPARK_TEST_CKPT    converted ckpt dir
    DSPARK_TEST_GOLDEN  golden dir with dspark_golden.safetensors + manifest.json
"""

import json
import math
import os
import unittest
from typing import Dict, List

import torch

DEFAULT_CKPT = "/data0/caihaowen.chw/dspark-work/models/dspark_draft_rtp"
DEFAULT_GOLDEN = "/data0/caihaowen.chw/dspark-work/golden/ctx96_seed42"
PAGE_SIZE = 64


def _ckpt_dir() -> str:
    return os.environ.get("DSPARK_TEST_CKPT", DEFAULT_CKPT)


def _golden_dir() -> str:
    return os.environ.get("DSPARK_TEST_GOLDEN", DEFAULT_GOLDEN)


def _load_raw_ckpt(ckpt_dir: str) -> Dict[str, torch.Tensor]:
    from safetensors import safe_open

    raw: Dict[str, torch.Tensor] = {}
    with safe_open(
        os.path.join(ckpt_dir, "model.safetensors"), framework="pt", device="cpu"
    ) as f:
        for name in f.keys():
            raw[name] = f.get_tensor(name)
    return raw


def _build_model_weights(raw: Dict[str, torch.Tensor], num_layers: int, device: str):
    """Hand-build ModelWeights in the loader's runtime layout (tp=1, bf16).

    Orientations mirror the transforms Qwen3DSparkWeight declares:
    merge_qkv_hf for qkv, transpose for o/ffn/fc, identity for norms and
    vocab-sized tensors.  The declared name->ckpt mapping itself is pinned by
    DSparkWeightInfoMappingTest below.
    """
    from rtp_llm.model_loader.model_weight_info import ModelWeights
    from rtp_llm.utils.model_weight import W, merge_qkv_hf

    dt = torch.bfloat16

    def t(name: str) -> torch.Tensor:
        return raw[name].to(device=device, dtype=dt)

    mw = ModelWeights(num_layers, device, dt)
    mw.set_global_weight(W.embedding, t("model.embed_tokens.weight"))
    mw.set_global_weight(W.lm_head, t("lm_head.weight"))
    mw.set_global_weight(W.final_ln_gamma, t("model.norm.weight"))
    mw.set_global_weight(W.dspark_fc_w, t("fc.weight").T.contiguous())
    mw.set_global_weight(W.dspark_hidden_norm_gamma, t("model.hidden_norm.weight"))
    if "markov_head.markov_w1.weight" in raw:
        mw.set_global_weight(W.dspark_markov_w1, t("markov_head.markov_w1.weight"))
        mw.set_global_weight(W.dspark_markov_w2, t("markov_head.markov_w2.weight"))

    for i in range(num_layers):
        p = f"model.layers.{i}."
        lw = mw.weights[i]
        lw[W.pre_ln_gamma] = t(p + "input_layernorm.weight")
        lw[W.post_ln_gamma] = t(p + "post_attention_layernorm.weight")
        lw[W.attn_qkv_w] = merge_qkv_hf(
            [
                t(p + "self_attn.q_proj.weight"),
                t(p + "self_attn.k_proj.weight"),
                t(p + "self_attn.v_proj.weight"),
            ]
        )
        lw[W.attn_o_w] = t(p + "self_attn.o_proj.weight").T.contiguous()
        lw[W.q_ln_gamma] = t(p + "self_attn.q_norm.weight")
        lw[W.k_ln_gamma] = t(p + "self_attn.k_norm.weight")
        lw[W.ffn_w1] = t(p + "mlp.gate_proj.weight").T.contiguous()
        lw[W.ffn_w3] = t(p + "mlp.up_proj.weight").T.contiguous()
        lw[W.ffn_w2] = t(p + "mlp.down_proj.weight").T.contiguous()
    return mw


class _TestKVCache:
    """Duck-typed stand-in for the engine KVCache (get_layer_cache only)."""

    def __init__(self, layer_caches: List[object]):
        self._layer_caches = layer_caches

    def get_layer_cache(self, layer_idx: int):
        return self._layer_caches[layer_idx]


def _make_layer_caches(num_layers: int, num_pages: int, nkv: int, hd: int):
    from rtp_llm.ops.compute_ops import LayerKVCache

    caches = []
    for _ in range(num_layers):
        cache = LayerKVCache()
        cache.kv_cache_base = torch.zeros(
            num_pages, 2, nkv, PAGE_SIZE, hd, dtype=torch.bfloat16, device="cuda"
        )
        caches.append(cache)
    return caches


def _make_block_attn_inputs(prefix_len: int, block_width: int):
    """Chunked-prefill metadata for the draft block forward (batch=1)."""
    from rtp_llm.ops.compute_ops import PyAttentionInputs, get_typemeta

    seq_len = prefix_len + block_width
    num_blocks = math.ceil(seq_len / PAGE_SIZE)
    block_ids = torch.arange(num_blocks, dtype=torch.int32).view(1, -1)

    ai = PyAttentionInputs()
    ai.is_prefill = True
    ai.input_lengths = torch.tensor(
        [block_width], dtype=torch.int32, device="cpu"
    ).pin_memory()
    ai.prefix_lengths = torch.tensor(
        [prefix_len], dtype=torch.int32, device="cpu"
    ).pin_memory()
    ai.sequence_lengths = torch.tensor(
        [seq_len], dtype=torch.int32, device="cpu"
    ).pin_memory()
    ai.kv_cache_block_id_host = block_ids
    ai.kv_cache_block_id_device = block_ids.cuda()
    ai.kv_cache_kernel_block_id_host = block_ids
    ai.kv_cache_kernel_block_id_device = block_ids.cuda()
    ai.cu_seqlens = torch.tensor(
        [0, block_width], dtype=torch.int32, device="cuda"
    )
    ai.dtype = get_typemeta(torch.zeros(1, dtype=torch.bfloat16))
    return ai


class DSparkModelGoldenTest(unittest.TestCase):
    """Runs the full draft round once, then compares stage by stage."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        ckpt_dir, golden_dir = _ckpt_dir(), _golden_dir()
        golden_st = os.path.join(golden_dir, "dspark_golden.safetensors")
        if not os.path.isdir(ckpt_dir) or not os.path.isfile(golden_st):
            raise unittest.SkipTest(
                f"dspark ckpt/golden not available: {ckpt_dir}, {golden_st}"
            )

        from safetensors import safe_open

        from rtp_llm.models.qwen_3_dspark import Qwen3DSpark
        from rtp_llm.models_py.model_desc.qwen3_dspark import Qwen3DSparkModel
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
            PyFlashinferPagedPrefillImpl,
        )
        from rtp_llm.ops import ParallelismConfig
        from rtp_llm.ops.compute_ops import PyModelInputs

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        cls.golden = {}
        with safe_open(golden_st, framework="pt", device="cpu") as f:
            for name in f.keys():
                cls.golden[name] = f.get_tensor(name)
        with open(os.path.join(golden_dir, "manifest.json")) as f:
            cls.manifest = json.load(f)

        config = Qwen3DSpark._create_config(ckpt_dir)
        config.attn_config.tokens_per_block = PAGE_SIZE
        config.attn_config.kernel_tokens_per_block = PAGE_SIZE
        cls.config = config
        ds = config.dspark_config
        assert ds is not None

        raw = _load_raw_ckpt(ckpt_dir)
        weights = _build_model_weights(raw, config.num_layers, "cuda")

        parallelism_config = ParallelismConfig()
        model = Qwen3DSparkModel(
            config, parallelism_config, weights, max_generate_batch_size=4
        )
        # The block-forward path needs the rope + cache-write stages enabled.
        assert model.attn_configs.is_causal is False
        assert model.attn_configs.kernel_tokens_per_block == PAGE_SIZE, (
            f"kernel_tokens_per_block={model.attn_configs.kernel_tokens_per_block}"
        )
        model.attn_configs.dtype = torch.bfloat16
        model.attn_configs.need_rope_kv_cache = True
        cls.model = model

        ctx_len = cls.manifest["ctx_len"]  # committed_len == ctx_len in the dump
        width = ds.block_width
        num_pages = math.ceil((ctx_len + width) / PAGE_SIZE)
        layer_caches = _make_layer_caches(
            config.num_layers,
            num_pages,
            model.attn_configs.kv_head_num,
            model.attn_configs.size_per_head,
        )
        cls.layer_caches = layer_caches
        model.kv_cache = _TestKVCache(layer_caches)

        attn_inputs = _make_block_attn_inputs(ctx_len, width)
        fmha_impl = PyFlashinferPagedPrefillImpl(model.attn_configs, attn_inputs)

        anchor_id = cls.manifest["anchor_id"]
        mask_id = cls.manifest["mask_token_id"]
        inputs = PyModelInputs()
        inputs.input_ids = torch.tensor(
            [anchor_id] + [mask_id] * ds.speculative_tokens,
            dtype=torch.int32,
            device="cuda",
        )
        inputs.input_hiddens = (
            cls.golden["aux_concat"].to(device="cuda", dtype=torch.bfloat16)
        )
        inputs.attention_inputs = attn_inputs

        with torch.no_grad():
            cls.proposal = model.propose(inputs, fmha_impl=fmha_impl)
        cls.ctx_len = ctx_len
        cls.inputs = inputs
        cls.fmha_impl = fmha_impl

    # ---- helpers ------------------------------------------------------

    def _assert_close(self, got: torch.Tensor, want: torch.Tensor, name: str,
                      rtol: float, atol_scale: float):
        """allclose with atol scaled to the golden tensor's magnitude."""
        got = got.detach().float().cpu()
        want = want.detach().float().cpu()
        self.assertEqual(tuple(got.shape), tuple(want.shape), name)
        atol = atol_scale * want.abs().mean().item()
        diff = (got - want).abs()
        ok = torch.allclose(got, want, rtol=rtol, atol=atol)
        self.assertTrue(
            ok,
            f"{name}: max_diff={diff.max().item():.5f} "
            f"mean_diff={diff.mean().item():.5f} "
            f"(atol={atol:.5f}, rtol={rtol}, scale={want.abs().mean().item():.4f})",
        )

    # ---- Stage A: feature combine --------------------------------------

    def test_stage_a_fused_features(self):
        aux = self.golden["aux_concat"].to(device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            fused = self.model.combine_hidden_states(aux)
        self._assert_close(
            fused, self.golden["fused_features"], "fused_features",
            rtol=3e-2, atol_scale=3e-2,
        )

    # ---- Stage B: feature KV in the paged cache -------------------------

    def _read_cache(self, layer_idx: int, kv: int) -> torch.Tensor:
        """[ctx_len, nkv, hd] from the paged cache (HND, sequential pages)."""
        base = self.layer_caches[layer_idx].kv_cache_base  # [P,2,nkv,page,hd]
        flat = base[:, kv].permute(0, 2, 1, 3).reshape(
            -1, base.shape[2], base.shape[4]
        )
        return flat[: self.ctx_len]

    def test_stage_b_ctx_kv_written(self):
        for layer_idx in range(self.config.num_layers):
            self._assert_close(
                self._read_cache(layer_idx, 0),
                self.golden["ctx_k"][layer_idx],
                f"ctx_k[{layer_idx}]",
                rtol=3e-2, atol_scale=3e-2,
            )
            self._assert_close(
                self._read_cache(layer_idx, 1),
                self.golden["ctx_v"][layer_idx],
                f"ctx_v[{layer_idx}]",
                rtol=3e-2, atol_scale=3e-2,
            )

    # ---- Stage C: block forward ----------------------------------------

    def test_stage_c_head_hidden(self):
        self._assert_close(
            self.proposal.head_hidden, self.golden["head_hidden"], "head_hidden",
            rtol=5e-2, atol_scale=5e-2,
        )

    def test_stage_c_base_logits(self):
        self._assert_close(
            self.proposal.base_logits.squeeze(0),
            self.golden["base_logits"],
            "base_logits",
            rtol=5e-2, atol_scale=5e-2,
        )

    # ---- Stage D: markov correction -------------------------------------

    def test_stage_d_draft_tokens_exact(self):
        got = self.proposal.draft_tokens.squeeze(0).cpu().tolist()
        want = self.golden["draft_tokens"].tolist()
        self.assertEqual(got, want, "greedy draft tokens must match the oracle")

    def test_stage_d_corrected_logits(self):
        self._assert_close(
            self.proposal.corrected_logits.squeeze(0),
            self.golden["corrected_logits"],
            "corrected_logits",
            rtol=5e-2, atol_scale=5e-2,
        )

    # ---- Engine-facing forward() output contract -------------------------

    def test_forward_fills_draft_proposal_outputs(self):
        """forward() must expose draft_tokens/draft_probs on PyModelOutputs
        (G3: sampling lives in the model).  Re-running is safe: the feature
        injection overwrites the same cache positions with the same values."""
        with torch.no_grad():
            outputs = self.model.forward(self.inputs, fmha_impl=self.fmha_impl)
        k = self.config.dspark_config.speculative_tokens
        vocab = self.config.vocab_size
        self.assertEqual(tuple(outputs.draft_tokens.shape), (1, k))
        self.assertEqual(
            outputs.draft_tokens.squeeze(0).cpu().tolist(),
            self.golden["draft_tokens"].tolist(),
        )
        self.assertEqual(tuple(outputs.draft_probs.shape), (1, k, vocab))
        sums = outputs.draft_probs.sum(dim=-1)
        torch.testing.assert_close(
            sums, torch.ones_like(sums), atol=1e-4, rtol=0,
            msg="draft_probs rows must be a probability distribution",
        )


class DSparkWeightInfoMappingTest(unittest.TestCase):
    """Pins the declared ckpt-name mapping against the real ckpt inventory.

    Pure metadata: instantiates Qwen3DSparkWeight, walks every declared
    CkptWeightInfo name (with {i} expanded) and asserts it exists in the
    checkpoint.  Loader transforms are exercised end-to-end elsewhere.
    """

    def test_declared_names_exist_in_ckpt(self):
        ckpt_dir = _ckpt_dir()
        if not os.path.isdir(ckpt_dir):
            raise unittest.SkipTest(f"dspark ckpt not available: {ckpt_dir}")

        from safetensors import safe_open

        from rtp_llm.models.qwen_3_dspark import Qwen3DSpark
        from rtp_llm.ops import HWKernelConfig, KVCacheConfig, ParallelismConfig

        with safe_open(
            os.path.join(ckpt_dir, "model.safetensors"), framework="pt", device="cpu"
        ) as f:
            inventory = set(f.keys())

        config = Qwen3DSpark._create_config(ckpt_dir)
        weight_info = Qwen3DSpark.get_weight_cls()(
            model_config=config,
            parallelism_config=ParallelismConfig(),
            hw_kernel_config=HWKernelConfig(),
            kv_cache_config=KVCacheConfig(),
        )
        weight_info._process_meta([{}], list(inventory))
        self.assertTrue(weight_info.has_markov_head)

        model_weight_info = weight_info._get_weight_info()

        def collect_names(module) -> List[str]:
            names = []
            sub = getattr(module, "sub_weights", None)
            if sub:
                sub_iter = sub.values() if isinstance(sub, dict) else sub
                for child in sub_iter:
                    names.extend(collect_names(child))
                return names
            for info in getattr(module, "weights", []) or []:
                names.append(info.name)
            return names

        declared: List[str] = []
        for module in model_weight_info.weights:
            declared.extend(collect_names(module))
        for layer_id, layer_modules in enumerate(model_weight_info.layer_weights):
            for module in layer_modules:
                declared.extend(
                    name.format(i=layer_id, i_1=layer_id + 1)
                    for name in collect_names(module)
                )

        missing = [name for name in declared if name not in inventory]
        self.assertFalse(missing, f"declared ckpt names missing: {missing}")

        # Confidence head must stay unmapped (phase-1 No Goal).
        mapped = set(declared)
        confidence = [n for n in inventory if n.startswith("confidence_head.")]
        self.assertTrue(confidence, "expected confidence_head.* in the test ckpt")
        self.assertFalse(mapped & set(confidence))


if __name__ == "__main__":
    unittest.main()
