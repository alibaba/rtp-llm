"""Parity against the upstream transformers implementation.

The sibling tests in this directory compare against a hand transcription of the
upstream forward, which by construction cannot catch a transcription error. This
one executes the upstream source itself, so it is a real truth anchor.

Point ``QWEN4_EXP_UPSTREAM_SRC`` at transformers' ``modeling_qwen4_exp.py``
(``src/transformers/models/qwen4_exp/``); the tests skip when it is absent, so CI
stays green without it. The file is exec'd rather than imported because upstream
``main`` carries a config framework released transformers cannot load -- only the
leaf ``nn.Module`` definitions are taken, and those need nothing but torch.

Set ``QWEN4_EXP_CKPT_DIR`` as well to additionally drive the comparison with the
checkpoint's real layer-0 gated-residual tensors.

Everything is compared in float64 so a reported difference is a difference in the
math, not accumulated bf16 noise. Parity here is exact -- see
``test_folding_plus_one_into_the_weight_would_break_exactness`` for why these
norms deliberately skip the loader's ``plus_one``.
"""

import ast
import math
import os
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.models_py.modules.qwen4_exp.gated_residual import (
    Qwen4ExpGatedResidual,
    inject_into_residual,
)
from rtp_llm.models_py.modules.qwen4_exp.indexer import (
    Qwen4ExpQSAIndexer,
    apply_partial_rope,
)
from rtp_llm.models_py.modules.qwen4_exp.norm import Qwen4ExpFusedQKRMSNorm
from rtp_llm.models_py.modules.qwen4_exp.ple import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLELayer,
)

_UPSTREAM_SRC = os.environ.get("QWEN4_EXP_UPSTREAM_SRC", "")
_CKPT_DIR = os.environ.get("QWEN4_EXP_CKPT_DIR", "")

_UPSTREAM_NAMES = (
    "Qwen4ExpTextRMSNorm",
    "Qwen4ExpTextGatedResidual",
    "Qwen4ExpTextPLELayer",
    "Qwen4ExpTextNGramEmbedding",
    "Qwen4ExpTextQSAIndexer",
    "apply_rotary_pos_emb",
    "rotate_half",
    "_splitmix64",
    "_build_layer_multipliers",
    "_is_prime",
    "_find_nth_prime_after",
    "apply_mask_to_padding_states",
)


def _load_upstream(path):
    """Exec the named upstream definitions in an otherwise empty namespace."""
    tree = ast.parse(open(path).read())
    body = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            if node.name in _UPSTREAM_NAMES:
                body.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id.isupper() for t in node.targets
        ):
            # splitmix64 seeds and masks.
            body.append(node)
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "math": math,
        # Referenced only from annotations.
        "Cache": object,
        "Qwen4ExpTextConfig": object,
        "__name__": "qwen4_exp_upstream_ref",
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), namespace)
    missing = [name for name in _UPSTREAM_NAMES if name not in namespace]
    if missing:
        raise AssertionError(f"upstream source lacks {missing}; refresh the file")
    return SimpleNamespace(**{name: namespace[name] for name in _UPSTREAM_NAMES})


def _randomize(module, scale=0.05):
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.randn_like(param) * scale)


def _gamma(weight):
    """Gamma as our modules take it: the raw checkpoint tensor.

    ``grouped_rms_norm`` applies upstream's ``+1`` itself, in fp32 after the cast,
    so nothing is folded at load time.
    """
    return weight.detach()


@unittest.skipUnless(
    os.path.isfile(_UPSTREAM_SRC), "QWEN4_EXP_UPSTREAM_SRC not set to a file"
)
class GatedResidualParityTest(unittest.TestCase):
    HIDDEN, HC, LOWRANK, EPS = 64, 4, 16, 1e-6

    @classmethod
    def setUpClass(cls):
        cls.up = _load_upstream(_UPSTREAM_SRC)

    def _pair(self, use_combine):
        torch.manual_seed(0)
        config = SimpleNamespace(
            hidden_size=self.HIDDEN,
            hc_count=self.HC,
            hc_lowrank=self.LOWRANK,
            rms_norm_eps=self.EPS,
        )
        ref = self.up.Qwen4ExpTextGatedResidual(config, use_combine=use_combine)
        ref = ref.double()
        _randomize(ref)
        ours = Qwen4ExpGatedResidual(
            _gamma(ref.hc_norm.weight),
            ref.input_mix_weight_down.weight.detach(),
            ref.input_mix_weight_up.weight.detach(),
            ref.block_inject_weight.weight.detach() if use_combine else None,
            hc_mult=self.HC,
            norm_eps=self.EPS,
        )
        return ref, ours

    def _stream(self, batch=2, seq=5):
        return torch.randn(batch, seq, self.HC * self.HIDDEN, dtype=torch.float64)

    def test_layer_unit_is_bit_exact(self):
        ref, ours = self._pair(use_combine=True)
        x = self._stream()
        ref_mixed, ref_hyper, ref_inject = ref(x)
        mixed, hyper, inject = ours(x)
        self.assertEqual((ref_mixed - mixed).abs().max().item(), 0.0)
        self.assertEqual((ref_hyper - hyper).abs().max().item(), 0.0)
        self.assertEqual((ref_inject - inject).abs().max().item(), 0.0)

    def test_global_mixer_is_bit_exact(self):
        """``use_combine=False``: upstream returns the collapsed stream only."""
        ref, ours = self._pair(use_combine=False)
        x = self._stream()
        mixed, _, inject = ours(x)
        self.assertEqual((ref(x) - mixed).abs().max().item(), 0.0)
        self.assertIsNone(inject)

    def test_injection_matches_upstream_writeback(self):
        """Upstream writes back into the pre-norm stream, not the normalized one."""
        ref, ours = self._pair(use_combine=True)
        x = self._stream()
        _, ref_hyper, ref_inject = ref(x)
        sublayer_out = torch.randn(2, 5, self.HIDDEN, dtype=torch.float64)
        expected = ref_hyper + (
            sublayer_out.unsqueeze(-2) * ref_inject.unsqueeze(-1)
        ).flatten(-2)
        _, hyper, inject = ours(x)
        got = inject_into_residual(hyper, sublayer_out, inject)
        self.assertEqual((expected - got).abs().max().item(), 0.0)

    def test_folding_plus_one_into_the_weight_would_break_exactness(self):
        """Why these norms skip the loader's ``plus_one``.

        Upstream adds the one in fp32 after the cast. Folding it into the bf16
        checkpoint tensor instead -- what ``plus_one`` does, and what the rest of
        the repo's RMSNorms accept -- perturbs the gain enough to be visible in the
        unit's output. Measured on the real ``layers.0`` gamma: 7520/10240 elements
        differ, up to 3.9e-3 relative. This test fails if anyone reintroduces the
        fold, which would silently cost bit-exactness.
        """
        ref, ours = self._pair(use_combine=True)
        x = self._stream()
        expected = ref(x)[0]
        raw = ref.hc_norm.weight.detach()

        self.assertEqual((expected - ours(x)[0]).abs().max().item(), 0.0)

        prefolded = Qwen4ExpGatedResidual(
            # Emulate a loader that folded in bf16, then undo the module's own
            # ``1 +`` so only the folding error remains.
            (raw.to(torch.bfloat16) + 1).double() - 1.0,
            ref.input_mix_weight_down.weight.detach(),
            ref.input_mix_weight_up.weight.detach(),
            ref.block_inject_weight.weight.detach(),
            hc_mult=self.HC,
            norm_eps=self.EPS,
        )(x)[0]
        self.assertGreater((expected - prefolded).abs().max().item(), 1e-4)


@unittest.skipUnless(
    os.path.isfile(_UPSTREAM_SRC), "QWEN4_EXP_UPSTREAM_SRC not set to a file"
)
class PLEParityTest(unittest.TestCase):
    HIDDEN, HC, EPS = 32, 4, 1e-6
    NGRAM, HEADS_PER_NGRAM, EMBED, KERNEL = 3, 2, 16, 4
    EOS = 2

    @classmethod
    def setUpClass(cls):
        cls.up = _load_upstream(_UPSTREAM_SRC)

    def _build(self):
        torch.manual_seed(0)
        config = SimpleNamespace(
            hidden_size=self.HIDDEN,
            hc_count=self.HC,
            rms_norm_eps=self.EPS,
            ple_embed_dim=self.EMBED,
            ple_conv_kernel_size=self.KERNEL,
            ngram_size=self.NGRAM,
            heads_per_ngram=self.HEADS_PER_NGRAM,
            vocab_size=128,
            ngram_vocab_size_base=97,
            seed=1234,
            eos_token_id=self.EOS,
            make_ngram_vocab_size_divisible_by=8,
        )
        ref = self.up.Qwen4ExpTextPLELayer(config, layer_idx=1, ple_layer_index=0)
        ref = ref.double()
        _randomize(ref, scale=0.1)
        embedding = Qwen4ExpNGramEmbedding(
            # One shard standing in for the checkpoint's 128-way row split;
            # shard-vs-whole-table equivalence is covered by ple_test.
            [ref.ple_embedding.ngram_embedding.weight.detach()],
            ref.ple_embedding.ngram_heads_vocab_sizes,
            ref.ple_embedding.ngram_heads_offsets,
            ref.ple_embedding.layer_multipliers,
            ngram_size=self.NGRAM,
            eos_token_id=self.EOS,
        )
        ours = Qwen4ExpPLELayer(
            embedding,
            ref.key_proj.weight.detach(),
            ref.value_proj.weight.detach(),
            ref.conv1d.weight.detach(),
            _gamma(ref.norm_key.weight),
            _gamma(ref.norm_query.weight),
            _gamma(ref.norm_conv.weight),
            hc_mult=self.HC,
            hidden_size=self.HIDDEN,
            conv_kernel_size=self.KERNEL,
            norm_eps=self.EPS,
        )
        return ref, ours

    def _history(self, input_ids):
        """Upstream seeds the absent n-gram context with EOS, not zeros."""
        context = input_ids.new_full((input_ids.shape[0], self.NGRAM - 1), self.EOS)
        return torch.cat([context, input_ids], dim=1)

    def test_layer_is_bit_exact(self):
        ref, ours = self._build()
        input_ids = torch.randint(0, 128, (2, 7))
        hyper = torch.randn(2, 7, self.HC * self.HIDDEN, dtype=torch.float64)
        expected = ref(hyper, input_ids, None)
        got = ours(hyper, self._history(input_ids))
        self.assertEqual((expected - got).abs().max().item(), 0.0)

    def test_ngram_hashing_is_bit_exact(self):
        ref, ours = self._build()
        input_ids = torch.randint(0, 128, (2, 7))
        expected = ref.ple_embedding(input_ids, None)
        got = ours.ple_embedding(self._history(input_ids), input_ids.shape[1])
        self.assertEqual((expected - got).abs().max().item(), 0.0)

    def test_eos_restart_is_bit_exact(self):
        """EOS restarts the n-gram context; the shift masking is easy to get wrong."""
        ref, ours = self._build()
        input_ids = torch.tensor([[5, 9, self.EOS, 11, 12, 13, 14]])
        hyper = torch.randn(1, 7, self.HC * self.HIDDEN, dtype=torch.float64)
        expected = ref(hyper, input_ids, None)
        got = ours(hyper, self._history(input_ids))
        self.assertEqual((expected - got).abs().max().item(), 0.0)


@unittest.skipUnless(
    os.path.isfile(_UPSTREAM_SRC), "QWEN4_EXP_UPSTREAM_SRC not set to a file"
)
class FusedQKRMSNormParityTest(unittest.TestCase):
    """q/k norm on the fused qkv tensor, against upstream's per-head RMSNorm.

    Upstream norms q and k separately with ``Qwen4ExpTextRMSNorm(head_dim)`` (no
    group_size). Ours has to do it on the packed ``[rows, q|k|v]`` layout, so this
    checks the packing and the slicing as much as the arithmetic.
    """

    HEADS, KV_HEADS, HEAD_DIM, EPS = 6, 2, 16, 1e-6

    @classmethod
    def setUpClass(cls):
        cls.up = _load_upstream(_UPSTREAM_SRC)

    def _reference(self, qkv, q_gamma, k_gamma):
        """Upstream RMSNorm applied per head to the q and k slices."""
        q_norm = self.up.Qwen4ExpTextRMSNorm(self.HEAD_DIM, eps=self.EPS).double()
        k_norm = self.up.Qwen4ExpTextRMSNorm(self.HEAD_DIM, eps=self.EPS).double()
        with torch.no_grad():
            q_norm.weight.copy_(q_gamma)
            k_norm.weight.copy_(k_gamma)
        rows = qkv.shape[0]
        heads = qkv.reshape(rows, self.HEADS + 2 * self.KV_HEADS, self.HEAD_DIM)
        out = heads.clone()
        out[:, : self.HEADS] = q_norm(heads[:, : self.HEADS])
        out[:, self.HEADS : self.HEADS + self.KV_HEADS] = k_norm(
            heads[:, self.HEADS : self.HEADS + self.KV_HEADS]
        )
        return out.reshape(rows, -1)

    def _gammas(self):
        torch.manual_seed(0)
        return (
            torch.randn(self.HEAD_DIM, dtype=torch.float64) * 0.1,
            torch.randn(self.HEAD_DIM, dtype=torch.float64) * 0.1,
        )

    def _qkv(self, rows=5):
        width = (self.HEADS + 2 * self.KV_HEADS) * self.HEAD_DIM
        return torch.randn(rows, width, dtype=torch.float64)

    def test_is_bit_exact(self):
        q_gamma, k_gamma = self._gammas()
        qkv = self._qkv()
        expected = self._reference(qkv, q_gamma, k_gamma)
        got = Qwen4ExpFusedQKRMSNorm(
            q_gamma, k_gamma, self.HEADS, self.KV_HEADS, self.HEAD_DIM, self.EPS
        )(qkv.clone())
        self.assertEqual((expected - got).abs().max().item(), 0.0)

    def test_value_slice_is_untouched(self):
        """Only q and k are normed; v must pass through byte-for-byte."""
        q_gamma, k_gamma = self._gammas()
        qkv = self._qkv()
        got = Qwen4ExpFusedQKRMSNorm(
            q_gamma, k_gamma, self.HEADS, self.KV_HEADS, self.HEAD_DIM, self.EPS
        )(qkv.clone())
        v_start = (self.HEADS + self.KV_HEADS) * self.HEAD_DIM
        self.assertEqual((qkv[:, v_start:] - got[:, v_start:]).abs().max().item(), 0.0)

    def test_folding_plus_one_would_break_exactness(self):
        """Guard the reason this module exists instead of the fused kernel."""
        q_gamma, k_gamma = self._gammas()
        qkv = self._qkv()
        expected = self._reference(qkv, q_gamma, k_gamma)
        prefolded = Qwen4ExpFusedQKRMSNorm(
            (q_gamma.to(torch.bfloat16) + 1).double() - 1.0,
            (k_gamma.to(torch.bfloat16) + 1).double() - 1.0,
            self.HEADS,
            self.KV_HEADS,
            self.HEAD_DIM,
            self.EPS,
        )(qkv.clone())
        self.assertGreater((expected - prefolded).abs().max().item(), 1e-4)

    def test_replacing_copies_geometry_from_fused(self):
        q_gamma, k_gamma = self._gammas()
        fused = SimpleNamespace(
            q_weight=q_gamma,
            k_weight=k_gamma,
            head_num=self.HEADS,
            kv_head_num=self.KV_HEADS,
            size_per_head=self.HEAD_DIM,
            eps=self.EPS,
        )
        swapped = Qwen4ExpFusedQKRMSNorm.replacing(fused)
        qkv = self._qkv()
        expected = self._reference(qkv, q_gamma, k_gamma)
        self.assertEqual((expected - swapped(qkv.clone())).abs().max().item(), 0.0)


@unittest.skipUnless(
    os.path.isfile(_UPSTREAM_SRC) and os.path.isdir(_CKPT_DIR),
    "QWEN4_EXP_UPSTREAM_SRC / QWEN4_EXP_CKPT_DIR not both set",
)
class RealWeightGatedResidualParityTest(unittest.TestCase):
    """Same parity, driven by the released checkpoint's own tensors."""

    HIDDEN, HC, LOWRANK, EPS = 2560, 4, 320, 1e-6
    PREFIX = "model.language_model.layers.0.attn_hyper_connection."
    NAMES = (
        "hc_norm.weight",
        "input_mix_weight_down.weight",
        "input_mix_weight_up.weight",
        "block_inject_weight.weight",
    )

    @classmethod
    def setUpClass(cls):
        import json

        from safetensors.torch import load_file

        cls.up = _load_upstream(_UPSTREAM_SRC)
        index = os.path.join(_CKPT_DIR, "model.safetensors.index.json")
        weight_map = json.load(open(index))["weight_map"]
        keys = [cls.PREFIX + name for name in cls.NAMES]
        shards = {weight_map[key] for key in keys}
        absent = [s for s in shards if not os.path.exists(os.path.join(_CKPT_DIR, s))]
        if absent:
            raise unittest.SkipTest(f"shards not downloaded yet: {sorted(absent)}")
        cls.tensors = {}
        for shard in shards:
            loaded = load_file(os.path.join(_CKPT_DIR, shard))
            cls.tensors.update({k: loaded[k] for k in keys if k in loaded})

    def _get(self, name):
        return self.tensors[self.PREFIX + name].double()

    def test_real_layer0_unit_is_bit_exact(self):
        config = SimpleNamespace(
            hidden_size=self.HIDDEN,
            hc_count=self.HC,
            hc_lowrank=self.LOWRANK,
            rms_norm_eps=self.EPS,
        )
        ref = self.up.Qwen4ExpTextGatedResidual(config).double()
        with torch.no_grad():
            ref.hc_norm.weight.copy_(self._get("hc_norm.weight"))
            ref.input_mix_weight_down.weight.copy_(
                self._get("input_mix_weight_down.weight")
            )
            ref.input_mix_weight_up.weight.copy_(
                self._get("input_mix_weight_up.weight")
            )
            ref.block_inject_weight.weight.copy_(
                self._get("block_inject_weight.weight")
            )
        ours = Qwen4ExpGatedResidual(
            _gamma(self._get("hc_norm.weight")),
            self._get("input_mix_weight_down.weight"),
            self._get("input_mix_weight_up.weight"),
            self._get("block_inject_weight.weight"),
            hc_mult=self.HC,
            norm_eps=self.EPS,
        )
        torch.manual_seed(0)
        x = torch.randn(1, 4, self.HC * self.HIDDEN, dtype=torch.float64)
        ref_mixed, _, ref_inject = ref(x)
        mixed, _, inject = ours(x)
        self.assertEqual((ref_mixed - mixed).abs().max().item(), 0.0)
        self.assertEqual((ref_inject - inject).abs().max().item(), 0.0)

    def test_real_qk_norm_is_bit_exact(self):
        """The 26 q/k norm tensors are the ones the Qwen3.5 base would fold."""
        import json

        from safetensors.torch import load_file

        index = os.path.join(_CKPT_DIR, "model.safetensors.index.json")
        weight_map = json.load(open(index))["weight_map"]
        pairs = {}
        for key, shard in weight_map.items():
            if not key.endswith(("self_attn.q_norm.weight", "self_attn.k_norm.weight")):
                continue
            if not os.path.exists(os.path.join(_CKPT_DIR, shard)):
                continue
            layer = key.rsplit(".self_attn.", 1)[0]
            pairs.setdefault(layer, {})[key.rsplit(".", 2)[-2]] = (key, shard)
        complete = {l: v for l, v in pairs.items() if {"q_norm", "k_norm"} <= v.keys()}
        if not complete:
            raise unittest.SkipTest("no layer has both q_norm and k_norm downloaded")

        layer, entry = sorted(complete.items())[0]
        gammas = {}
        for role, (key, shard) in entry.items():
            gammas[role] = load_file(os.path.join(_CKPT_DIR, shard))[key].double()
        head_dim = gammas["q_norm"].shape[0]
        heads, kv_heads = 3, 2
        torch.manual_seed(0)
        qkv = torch.randn(4, (heads + 2 * kv_heads) * head_dim, dtype=torch.float64)

        def upstream(x, gamma):
            norm = self.up.Qwen4ExpTextRMSNorm(head_dim, eps=self.EPS).double()
            with torch.no_grad():
                norm.weight.copy_(gamma)
            return norm(x)

        packed = qkv.reshape(4, heads + 2 * kv_heads, head_dim)
        expected = packed.clone()
        expected[:, :heads] = upstream(packed[:, :heads], gammas["q_norm"])
        expected[:, heads : heads + kv_heads] = upstream(
            packed[:, heads : heads + kv_heads], gammas["k_norm"]
        )
        got = Qwen4ExpFusedQKRMSNorm(
            gammas["q_norm"], gammas["k_norm"], heads, kv_heads, head_dim, self.EPS
        )(qkv.clone())
        self.assertEqual(
            (expected.reshape(4, -1) - got).abs().max().item(), 0.0, msg=layer
        )

    def test_checkpoint_shapes_match_config(self):
        self.assertEqual(
            tuple(self._get("input_mix_weight_down.weight").shape),
            (self.LOWRANK, self.HC * self.HIDDEN),
        )
        self.assertEqual(
            tuple(self._get("block_inject_weight.weight").shape),
            (self.HC, self.HC * self.HIDDEN),
        )


@unittest.skipUnless(
    os.path.isfile(_UPSTREAM_SRC), "QWEN4_EXP_UPSTREAM_SRC not set to a file"
)
class QSAIndexerParityTest(unittest.TestCase):
    """Block selection against upstream ``Qwen4ExpTextQSAIndexer``.

    Upstream only returns the additive mask, so mask equality is the contract
    checked here. That pins the selected *set* but not its order; ordering comes
    from mirroring upstream's ``index_select`` on the topk output and is not
    independently verified.

    Geometry is scaled down (head_dim 8, rotary 4, budget 8) but keeps the ratio
    that matters: rotary_dim is half the head dim, so the partial-RoPE split gets
    exercised the same way 64-of-128 is in production.
    """

    HIDDEN, N_HEADS, KV_HEADS, HEAD_DIM = 48, 4, 1, 8
    BUDGET, RATIO, ROTARY, EPS = 8, 4, 4, 1e-6

    @classmethod
    def setUpClass(cls):
        cls.up = _load_upstream(_UPSTREAM_SRC)

    def _pair(self):
        torch.manual_seed(0)
        config = SimpleNamespace(
            hidden_size=self.HIDDEN,
            indexer_n_heads=self.N_HEADS,
            indexer_kv_heads=self.KV_HEADS,
            indexer_head_dim=self.HEAD_DIM,
            indexer_budget=self.BUDGET,
            indexer_compress_ratio=self.RATIO,
            rms_norm_eps=self.EPS,
        )
        ref = self.up.Qwen4ExpTextQSAIndexer(config, layer_idx=3).double()
        _randomize(ref, scale=0.1)
        ours = Qwen4ExpQSAIndexer(
            ref.index_qk_proj.weight.detach(),
            ref.q_layernorm.weight.detach(),
            ref.k_layernorm.weight.detach(),
            n_heads=self.N_HEADS,
            kv_heads=self.KV_HEADS,
            head_dim=self.HEAD_DIM,
            token_budget=self.BUDGET,
            compress_ratio=self.RATIO,
            norm_eps=self.EPS,
        )
        return ref, ours

    def _inputs(self, seq_len, batch=2):
        hidden = torch.randn(batch, seq_len, self.HIDDEN, dtype=torch.float64)
        cos = torch.randn(batch, seq_len, self.ROTARY, dtype=torch.float64)
        sin = torch.randn(batch, seq_len, self.ROTARY, dtype=torch.float64)
        causal = (
            torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(batch, 1, seq_len, seq_len)
        )
        return hidden, cos, sin, causal

    def _assert_mask_matches(self, seq_len):
        ref, ours = self._pair()
        hidden, cos, sin, causal = self._inputs(seq_len)
        expected = ref(hidden, (cos, sin), causal, None)
        got = ours.selection_mask(ours(hidden, cos, sin, causal[:, 0]), seq_len)
        self.assertEqual(tuple(expected.shape), tuple(got.shape))
        self.assertTrue(torch.equal(expected, got), msg=f"seq_len={seq_len}")

    def test_mask_matches_with_tail(self):
        """seq_len % ratio != 0 -- exercises the unscored trailing block."""
        self._assert_mask_matches(14)

    def test_mask_matches_without_tail(self):
        """seq_len % ratio == 0 -- tail is empty on the last row."""
        self._assert_mask_matches(12)

    def test_mask_matches_when_context_below_budget(self):
        """Fewer complete blocks than block_topk -- topk must clamp."""
        self._assert_mask_matches(6)

    def test_mask_matches_when_no_complete_block(self):
        """Shorter than one block: every row is pure tail."""
        self._assert_mask_matches(3)

    def test_vectorized_path_matches_upstream(self):
        """``forward_causal`` selects the same set as upstream, loop-free."""
        for seq_len in (3, 6, 11, 12, 14, 20, 33):
            with self.subTest(seq_len=seq_len):
                ref, ours = self._pair()
                hidden, cos, sin, causal = self._inputs(seq_len)
                expected = ref(hidden, (cos, sin), causal, None)
                got = ours.selection_mask(
                    ours.forward_causal(hidden, cos, sin), seq_len
                )
                self.assertTrue(torch.equal(expected, got))

    def test_vectorized_path_matches_reference_loop(self):
        """The two of ours agree, so either can serve as the other's baseline."""
        for seq_len in (7, 12, 19):
            with self.subTest(seq_len=seq_len):
                _, ours = self._pair()
                hidden, cos, sin, causal = self._inputs(seq_len)
                loop = ours.selection_mask(
                    ours(hidden, cos, sin, causal[:, 0]), seq_len
                )
                vec = ours.selection_mask(
                    ours.forward_causal(hidden, cos, sin), seq_len
                )
                self.assertTrue(torch.equal(loop, vec))

    def test_vectorized_decode_step_matches_full_prefill(self):
        """One-token decode with cached keys == the last row of a full prefill."""
        _, ours = self._pair()
        total = 17
        hidden = torch.randn(1, total, self.HIDDEN, dtype=torch.float64)
        cos = torch.randn(1, total, self.ROTARY, dtype=torch.float64)
        sin = torch.randn(1, total, self.ROTARY, dtype=torch.float64)
        prefill = ours.selection_mask(ours.forward_causal(hidden, cos, sin), total)
        _, raw_keys = ours.project(hidden)
        step = ours.forward_causal(
            hidden[:, -1:], cos, sin, past_raw_keys=raw_keys[:, :-1]
        )
        self.assertTrue(
            torch.equal(ours.selection_mask(step, total)[:, :, 0], prefill[:, :, -1])
        )

    def test_vectorized_layout_puts_tail_at_a_fixed_offset(self):
        """Static layout is the reason the vectorized path is capturable."""
        _, ours = self._pair()
        seq_len = 14
        hidden, cos, sin, _ = self._inputs(seq_len, batch=1)
        selected = ours.forward_causal(hidden, cos, sin)
        self.assertEqual(selected.shape[-1], ours.max_selected)
        tail_base = ours.block_topk * self.RATIO
        for query in range(seq_len):
            visible = query + 1
            tail = selected[0, query, tail_base:].tolist()
            expected = [
                (visible // self.RATIO) * self.RATIO + slot
                for slot in range(self.RATIO - 1)
            ]
            expected = [t if t < visible else -1 for t in expected]
            self.assertEqual(tail, expected, msg=f"query={query}")

    def test_tail_tokens_are_always_selected(self):
        """The trailing partial block bypasses scoring entirely."""
        _, ours = self._pair()
        seq_len = 14
        hidden, cos, sin, causal = self._inputs(seq_len, batch=1)
        selected = ours(hidden, cos, sin, causal[:, 0])
        for query in range(seq_len):
            visible = query + 1
            tail_start = (visible // self.RATIO) * self.RATIO
            row = selected[0, query].tolist()
            for token in range(tail_start, visible):
                self.assertIn(token, row, msg=f"query={query} tail={token}")

    def test_partial_rope_leaves_the_tail_untouched(self):
        """Only the leading ``cos.shape[-1]`` features rotate."""
        torch.manual_seed(0)
        x = torch.randn(2, 3, self.HEAD_DIM, dtype=torch.float64)
        cos = torch.randn(2, 3, self.ROTARY, dtype=torch.float64)
        sin = torch.randn(2, 3, self.ROTARY, dtype=torch.float64)
        out = apply_partial_rope(x, cos, sin)
        self.assertEqual(
            (out[..., self.ROTARY :] - x[..., self.ROTARY :]).abs().max().item(), 0.0
        )
        self.assertGreater(
            (out[..., : self.ROTARY] - x[..., : self.ROTARY]).abs().max().item(), 0.0
        )

    def test_production_geometry_derives_documented_widths(self):
        """Real config: budget 2048 / ratio 4 -> topk 512, output width 2051."""
        indexer = Qwen4ExpQSAIndexer(
            torch.zeros((4 + 1) * 128, 2560),
            torch.zeros(128),
            torch.zeros(128),
            n_heads=4,
            kv_heads=1,
            head_dim=128,
            token_budget=2048,
            compress_ratio=4,
            norm_eps=1e-6,
        )
        self.assertEqual(indexer.block_topk, 512)
        self.assertEqual(indexer.max_selected, 2051)


if __name__ == "__main__":
    unittest.main()
