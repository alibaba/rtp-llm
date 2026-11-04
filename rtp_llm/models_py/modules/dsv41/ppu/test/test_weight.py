"""Tests for the PPU V4.1 weight module (weight.py, L2 deliverable).

Covers:
  * synthetic checkpoint materialization (files, key families, seed
    determinism, lm_head optionality);
  * dense FP8 32x32 UE8M0 -> resident BF16 dequant cross-checked against the
    ``test_mxfp8_dense.py`` oracle formula (the same convention
    ``test_linear.py`` uses);
  * routed-expert FP4 shard stacking per EP rank (ep_size=2 split sanity
    with a tiny expert count);
  * Engram host shard slicing + ``EngramShardLayout`` math;
  * mHC config presence and the ``PPUWeightBundle.bind`` wiring;
  * import discipline: the standalone core loads without the RTP framework
    server stack / accelerator wheels.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv41.ppu.engram import EngramShardLayout
from rtp_llm.models_py.modules.dsv41.ppu.linear import PPUResidentBF16Linear
from rtp_llm.models_py.modules.dsv41.ppu.mhc import MHCConfig
from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
from rtp_llm.models_py.modules.dsv41.ppu.weight import (
    DeepSeekV41PPUWeight,
    PPUWeightBundle,
    V41Geometry,
    load_ppu_weights,
    read_safetensors_file,
    synthetic_config,
    write_synthetic_checkpoint,
)

# Reference FP8 32x32 UE8M0 dequant oracle, the test_mxfp8_dense.py:104-106
# convention (also used by test_linear.py):
# w_bf16 = (w.float() * scales.float()
#           .repeat_interleave(32, 0).repeat_interleave(32, 1)).to(bf16)


def _oracle_bf16(payload, scale_bytes):
    scale_values = torch.exp2(scale_bytes.to(torch.float32) - 127.0)
    return (
        payload.float() * scale_values.repeat_interleave(32, 0).repeat_interleave(32, 1)
    ).to(torch.bfloat16)


class _RecordingOps:
    """Bind-seam recorder: the three PPUOps injection points."""

    def __init__(self):
        self.mhc_config = None
        self.experts = []
        self.engram = []

    def bind_mhc_config(self, config):
        self.mhc_config = config

    def bind_expert_weights(self, layer_id, w13_pair, w2_pair):
        self.experts.append((layer_id, w13_pair, w2_pair))

    def bind_engram_shard(self, layer_id, shard, layout, group=None):
        self.engram.append((layer_id, shard, layout, group))


class SyntheticCheckpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = synthetic_config()
        cls.root = Path(tempfile.mkdtemp(prefix="dsv41_w1_"))
        cls.checkpoint = write_synthetic_checkpoint(cls.config, cls.root, seed=123)
        cls.tensors = read_safetensors_file(cls.checkpoint / "weights.safetensors")
        cls.other_root = Path(tempfile.mkdtemp(prefix="dsv41_w2_"))
        write_synthetic_checkpoint(cls.config, cls.other_root, seed=123)
        write_synthetic_checkpoint(cls.config, cls.other_root / "seed9", seed=9)
        write_synthetic_checkpoint(
            cls.config, cls.other_root / "nohead", seed=123, with_lm_head=False
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)
        shutil.rmtree(cls.other_root, ignore_errors=True)

    def test_checkpoint_files_exist_and_config_round_trips(self):
        self.assertTrue((self.checkpoint / "weights.safetensors").is_file())
        on_disk = json.loads((self.checkpoint / "config.json").read_text())
        self.assertEqual(on_disk, self.config)

    def test_expected_key_families_present(self):
        keys = set(self.tensors)
        for name in (
            "embed.weight",
            "norm.weight",
            "head.weight",
            "layers.0.attn.wq_a.weight",
            "layers.0.attn.wq_a.scale",
            "layers.0.attn.wkv.weight",
            "layers.0.ffn.gate.weight",
            "layers.0.ffn.experts.0.w1.weight",
            "layers.0.ffn.experts.0.w1.scale",
            "layers.0.ffn.experts.7.w2.weight",
            "layers.0.ffn.shared_experts.w2.weight",
            "layers.0.hc_attn_fn",
            "layers.1.engram.embed.weight",
            "layers.1.engram.embed.scale",
            "layers.1.engram.wkv.weight",
            "layers.1.engram.q_weight",
            "layers.2.attn.compressor.wkv.weight",
            "layers.2.attn.compressor.wgate.weight",
            "layers.2.attn.indexer.wk.weight",
            "layers.2.attn.indexer.wq_b.weight",
        ):
            self.assertIn(name, keys)
        # layer 0 is not a source: no compressor weights for it.
        self.assertNotIn("layers.0.attn.compressor.wkv.weight", keys)

    def test_seed_determinism(self):
        same = read_safetensors_file(self.other_root / "weights.safetensors")
        other = read_safetensors_file(self.other_root / "seed9" / "weights.safetensors")
        for name in ("embed.weight", "layers.0.attn.wq_a.weight"):
            self.assertTrue(torch.equal(self.tensors[name], same[name]), name)
            self.assertFalse(torch.equal(self.tensors[name], other[name]), name)

    def test_lm_head_is_optional(self):
        no_head = read_safetensors_file(
            self.other_root / "nohead" / "weights.safetensors"
        )
        self.assertNotIn("head.weight", no_head)
        bundle = load_ppu_weights(self.config, self.other_root / "nohead")
        self.assertIsNone(bundle.lm_head)
        with_head = load_ppu_weights(self.config, self.checkpoint)
        self.assertIsNotNone(with_head.lm_head)


class GeometryRolesTest(unittest.TestCase):
    """The synthetic config parses into every real-model layer role."""

    def setUp(self):
        self.geometry = V41Geometry.from_config(synthetic_config())

    def test_layer_roles(self):
        geometry = self.geometry
        self.assertEqual(geometry.num_layers, 8)
        self.assertEqual(
            [geometry.ratio(layer) for layer in range(8)], [0, 0, 2, 2, 2, 2, 1, 1]
        )
        self.assertEqual(geometry.kv_source_layer_ids, (2, 6))
        self.assertEqual(geometry.index_source_layer_ids, (2, 6))
        self.assertEqual(geometry.engram_layer_ids, (1,))
        self.assertEqual(geometry.kv_source_for(5), 2)
        self.assertEqual(geometry.kv_source_for(7), 6)
        self.assertIsNone(geometry.kv_source_for(1))

    def test_num_local_experts_split(self):
        self.assertEqual(self.geometry.num_local_experts(1), 8)
        self.assertEqual(self.geometry.num_local_experts(2), 4)
        with self.assertRaises(ValueError):
            self.geometry.num_local_experts(3)

    def test_mhc_config_presence(self):
        config = self.geometry.mhc_config()
        self.assertIsInstance(config, MHCConfig)
        self.assertEqual(config.hc_mult, 4)
        self.assertEqual(config.sinkhorn_iters, 4)
        self.assertEqual(config.hc_eps, 1e-6)
        self.assertEqual(config.rms_eps, 1e-6)

    def test_trailing_compress_ratios_are_truncated(self):
        # The real checkpoint's compress_ratios carries entries past
        # num_hidden_layers (multimodal composite config: 43 entries for 40
        # text layers). V41Geometry must match CacheGeometry and the fixture
        # manifest: truncate to num_layers.
        config = dict(synthetic_config())
        config["compress_ratios"] = list(config["compress_ratios"]) + [0, 0, 0]
        geometry = V41Geometry.from_config(config)
        self.assertEqual(geometry.num_layers, 8)
        self.assertEqual(
            len(geometry.compress_ratios),
            geometry.num_layers,
        )
        self.assertEqual(
            [geometry.ratio(layer) for layer in range(8)],
            [0, 0, 2, 2, 2, 2, 1, 1],
        )


class DenseDequantTest(unittest.TestCase):
    """Dense FP8 32x32 UE8M0 weights land as resident BF16 exactly."""

    @classmethod
    def setUpClass(cls):
        cls.config = synthetic_config()
        cls.root = Path(tempfile.mkdtemp(prefix="dsv41_w3_"))
        cls.checkpoint = write_synthetic_checkpoint(cls.config, cls.root, seed=5)
        cls.bundle = load_ppu_weights(cls.config, cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_dense_weight_matches_mxfp8_dense_oracle(self):
        tensors = read_safetensors_file(self.checkpoint / "weights.safetensors")
        for prefix in (
            "layers.0.attn.wq_a",
            "layers.0.attn.wq_b",
            "layers.0.attn.wkv",
            "layers.0.attn.wo_a",
            "layers.0.attn.wo_b",
            "layers.2.attn.compressor.wkv",
            "layers.2.attn.compressor.wgate",
            "layers.2.attn.indexer.wk",
            "layers.2.attn.indexer.wq_b",
            "layers.1.engram.wkv",
            "layers.0.ffn.shared_experts.w1",
        ):
            with self.subTest(prefix=prefix):
                payload = tensors[f"{prefix}.weight"]
                scale = tensors[f"{prefix}.scale"]
                self.assertEqual(payload.dtype, torch.float8_e4m3fn)
                expected = _oracle_bf16(payload, scale)
                linear = self._resolve(prefix)
                self.assertIsInstance(linear, PPUResidentBF16Linear)
                self.assertEqual(linear.weight.dtype, torch.bfloat16)
                torch.testing.assert_close(linear.weight, expected, rtol=0, atol=0)

    def _resolve(self, prefix):
        parts = prefix.split(".")
        layer = self.bundle.layers[int(parts[1])]
        rest = ".".join(parts[2:])
        if rest.startswith("attn.compressor."):
            return getattr(layer, "compressor_" + rest.split(".")[-1])
        if rest.startswith("attn.indexer."):
            return getattr(
                layer, "index_" + {"wk": "wk", "wq_b": "wq"}[rest.split(".")[-1]]
            )
        if rest.startswith("engram.wkv"):
            return layer.engram_wkv
        if rest.startswith("ffn.shared_experts.w1"):
            return PPUResidentBF16Linear(
                layer.shared_w13.weight[: layer.shared_w13.weight.shape[0] // 2]
            )
        return getattr(layer, rest.replace("attn.", "").replace(".", "_"))

    def test_all_dense_projections_are_resident_bf16(self):
        layer = self.bundle.layers[0]
        for linear in (layer.wq_a, layer.wq_b, layer.wkv, layer.wo_a, layer.wo_b):
            self.assertIsInstance(linear, PPUResidentBF16Linear)
            self.assertEqual(linear.weight.dtype, torch.bfloat16)
        source = self.bundle.layers[2]
        for linear in (source.compressor_wkv, source.compressor_wgate, source.index_wk):
            self.assertIsInstance(linear, PPUResidentBF16Linear)
        self.assertIsInstance(source.index_wq, PPUResidentBF16Linear)

    def test_forward_uses_dequantized_weight(self):
        tensors = read_safetensors_file(self.checkpoint / "weights.safetensors")
        prefix = "layers.0.attn.wkv"
        expected = _oracle_bf16(tensors[f"{prefix}.weight"], tensors[f"{prefix}.scale"])
        x = torch.randn(4, self.bundle.geometry.hidden_size).to(torch.bfloat16)
        out = self.bundle.layers[0].wkv(x)
        torch.testing.assert_close(
            out, torch.nn.functional.linear(x, expected), rtol=0, atol=0
        )


class RoutedExpertShardTest(unittest.TestCase):
    """ep_size=2: each rank stacks only its own local experts (FP4 native)."""

    @classmethod
    def setUpClass(cls):
        cls.config = synthetic_config(n_routed_experts=4, n_activated_experts=2)
        cls.root = Path(tempfile.mkdtemp(prefix="dsv41_w4_"))
        cls.checkpoint = write_synthetic_checkpoint(cls.config, cls.root, seed=7)
        cls.tensors = read_safetensors_file(cls.checkpoint / "weights.safetensors")
        cls.ranks = [
            load_ppu_weights(cls.config, cls.checkpoint, ep_rank=rank, ep_size=2)
            for rank in (0, 1)
        ]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def _expert(self, layer, expert, sub):
        return self.tensors[f"layers.{layer}.ffn.experts.{expert}.{sub}.weight"]

    def test_shard_shapes(self):
        inter, hidden = 32, 64
        for rank, bundle in enumerate(self.ranks):
            layer = bundle.layers[0]
            with self.subTest(rank=rank):
                self.assertEqual(
                    tuple(layer.routed_w13_weight.shape), (2, 2 * inter, hidden // 2)
                )
                self.assertEqual(
                    tuple(layer.routed_w13_scale.shape), (2, 2 * inter, hidden // 32)
                )
                self.assertEqual(
                    tuple(layer.routed_w2_weight.shape), (2, hidden, inter // 2)
                )
                self.assertEqual(
                    tuple(layer.routed_w2_scale.shape), (2, hidden, inter // 32)
                )
                self.assertEqual(layer.routed_w13_weight.dtype, torch.int8)

    def test_rank0_stacks_experts_0_and_1(self):
        layer = self.ranks[0].layers[0]
        for local, expert in enumerate((0, 1)):
            with self.subTest(expert=expert):
                expected_w13 = torch.cat(
                    (
                        self._expert(0, expert, "w1"),
                        self._expert(0, expert, "w3"),
                    ),
                    dim=0,
                )
                self.assertTrue(
                    torch.equal(layer.routed_w13_weight[local], expected_w13)
                )
                self.assertTrue(
                    torch.equal(
                        layer.routed_w2_weight[local], self._expert(0, expert, "w2")
                    )
                )

    def test_rank1_stacks_experts_2_and_3_and_is_disjoint(self):
        layer = self.ranks[1].layers[0]
        for local, expert in enumerate((2, 3)):
            expected_w13 = torch.cat(
                (self._expert(0, expert, "w1"), self._expert(0, expert, "w3")), dim=0
            )
            self.assertTrue(torch.equal(layer.routed_w13_weight[local], expected_w13))
        # The two ranks never share expert payload bytes.
        self.assertFalse(
            torch.equal(
                self.ranks[0].layers[0].routed_w13_weight,
                self.ranks[1].layers[0].routed_w13_weight,
            )
        )

    def test_every_moe_layer_is_sharded(self):
        for bundle in self.ranks:
            for layer_id, layer in bundle.layers.items():
                self.assertIsNotNone(layer.routed_w13_weight, layer_id)

    def test_undivisible_expert_count_fails_loud(self):
        config = synthetic_config(n_routed_experts=6, n_activated_experts=2)
        root = Path(tempfile.mkdtemp(prefix="dsv41_w5_"))
        try:
            write_synthetic_checkpoint(config, root, seed=1)
            with self.assertRaises(ValueError):
                load_ppu_weights(config, root, ep_rank=0, ep_size=4)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_bind_into_two_rank_ops(self):
        geometry = self.ranks[0].geometry
        for rank, bundle in enumerate(self.ranks):
            with self.subTest(rank=rank):
                ops = PPUOps(
                    ep_rank=rank,
                    ep_size=2,
                    num_routed_experts=geometry.n_routed_experts,
                    num_local_experts=geometry.num_local_experts(2),
                    hidden_size=geometry.hidden_size,
                    inter_dim=geometry.moe_inter_dim,
                    swiglu_limit=geometry.swiglu_limit,
                    mhc_config=geometry.mhc_config(),
                    sm_scale=geometry.sm_scale(),
                    window=geometry.window,
                )
                bundle.bind(ops)


class EngramShardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = synthetic_config()
        cls.root = Path(tempfile.mkdtemp(prefix="dsv41_w6_"))
        cls.checkpoint = write_synthetic_checkpoint(cls.config, cls.root, seed=11)
        cls.tensors = read_safetensors_file(cls.checkpoint / "weights.safetensors")
        cls.single = load_ppu_weights(cls.config, cls.checkpoint)
        cls.ranks = [
            load_ppu_weights(cls.config, cls.checkpoint, ep_rank=rank, ep_size=2)
            for rank in (0, 1)
        ]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_single_rank_shard_covers_the_whole_table(self):
        shard, layout = self.single.engram_shards[1]
        payload = self.tensors["layers.1.engram.embed.weight"]
        scale = self.tensors["layers.1.engram.embed.scale"]
        self.assertEqual(layout.enp_size, 1)
        self.assertEqual(layout.num_embeddings, 2048)
        self.assertEqual(shard.dim, self.single.geometry.engram_head_dim)
        self.assertTrue(
            torch.equal(shard.payload.view(torch.uint8), payload.view(torch.uint8))
        )
        self.assertTrue(torch.equal(shard.scale, scale))

    def test_layout_math_for_two_ranks(self):
        layout = EngramShardLayout.from_table(2048, enp_size=2)
        self.assertEqual(layout.alignment, 256)
        self.assertEqual(layout.padded_rows, 2048)
        self.assertEqual(layout.num_per_part, 1024)
        self.assertEqual(layout.row_start(0), 0)
        self.assertEqual(layout.row_start(1), 1024)
        self.assertEqual(layout.valid_rows(0), 1024)
        self.assertEqual(layout.valid_rows(1), 1024)
        self.assertEqual(layout.owner(1023), 0)
        self.assertEqual(layout.owner(1024), 1)

    def test_odd_table_size_keeps_valid_rows_bookkeeping(self):
        layout = EngramShardLayout.from_table(2050, enp_size=2)
        self.assertEqual(layout.padded_rows, 2304)
        self.assertEqual(layout.num_per_part, 1152)
        self.assertEqual(layout.valid_rows(0), 1152)
        self.assertEqual(layout.valid_rows(1), 2050 - 1152)

    def test_two_rank_shards_slice_the_table(self):
        payload = self.tensors["layers.1.engram.embed.weight"].view(torch.uint8)
        scale = self.tensors["layers.1.engram.embed.scale"]
        for rank, bundle in enumerate(self.ranks):
            shard, layout = bundle.engram_shards[1]
            start = 1024 * rank
            with self.subTest(rank=rank):
                self.assertEqual(layout.enp_size, 2)
                self.assertTrue(
                    torch.equal(
                        shard.payload.view(torch.uint8), payload[start : start + 1024]
                    )
                )
                self.assertTrue(torch.equal(shard.scale, scale[start : start + 1024]))

    def test_hash_params_are_built(self):
        bundle = self.single
        self.assertIsNotNone(bundle.engram_hash_params)
        self.assertEqual(
            bundle.engram_hash_params.n_hash_cols,
            (bundle.geometry.engram_max_ngram_size - 1)
            * bundle.geometry.engram_n_heads,
        )

    def test_missing_engram_table_fails_loud(self):
        tensors = dict(self.tensors)
        del tensors["layers.1.engram.embed.weight"]
        with self.assertRaises(KeyError):
            load_ppu_weights(self.config, tensors)


class BindWiringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = synthetic_config()
        cls.root = Path(tempfile.mkdtemp(prefix="dsv41_w7_"))
        cls.checkpoint = write_synthetic_checkpoint(cls.config, cls.root, seed=3)
        cls.bundle = load_ppu_weights(cls.config, cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_bind_injects_every_seam(self):
        ops = _RecordingOps()
        self.bundle.bind(ops)
        self.assertEqual(ops.mhc_config, self.bundle.geometry.mhc_config())
        self.assertEqual(
            sorted(layer_id for layer_id, _, _ in ops.experts), list(range(8))
        )
        for layer_id, (w13_w, _), (w2_w, _) in ops.experts:
            self.assertEqual(w13_w.shape[0], 8)  # all experts on ep_size=1
            self.assertEqual(w13_w.shape[0], w2_w.shape[0])
        self.assertEqual([layer_id for layer_id, *_ in ops.engram], [1])
        layer_id, shard, layout, group = ops.engram[0]
        self.assertIs(shard, self.bundle.engram_shards[1][0])
        self.assertIs(layout, self.bundle.engram_shards[1][1])
        self.assertIsNone(group)

    def test_bind_into_real_ops(self):
        geometry = self.bundle.geometry
        ops = PPUOps(
            ep_rank=0,
            ep_size=1,
            num_routed_experts=geometry.n_routed_experts,
            num_local_experts=geometry.num_local_experts(1),
            hidden_size=geometry.hidden_size,
            inter_dim=geometry.moe_inter_dim,
            swiglu_limit=geometry.swiglu_limit,
            mhc_config=geometry.mhc_config(),
            sm_scale=geometry.sm_scale(),
            window=geometry.window,
        )
        self.bundle.bind(ops)

    def test_freqs_tables_are_precomputed(self):
        bundle = self.bundle
        self.assertEqual(
            bundle.freqs_cis_main.shape[-1], bundle.geometry.rope_head_dim // 2
        )
        self.assertEqual(
            bundle.freqs_cis_compressed.shape[-1],
            bundle.geometry.rope_head_dim // 2,
        )
        self.assertIs(bundle.freqs_for(0), bundle.freqs_cis_main)
        self.assertIs(bundle.freqs_for(2), bundle.freqs_cis_compressed)


class ImportDisciplineTest(unittest.TestCase):
    """The standalone core must not drag in the framework/accelerator stack."""

    def test_weight_and_model_import_light(self):
        script = (
            "import sys;"
            "import rtp_llm.models_py.modules.dsv41.ppu.weight as weight;"
            "import rtp_llm.models_py.modules.dsv41.ppu.model as model;"
            "bad = [m for m in ('deep_gemm', 'deep_ep', 'flash_mla', 'triton',"
            " 'tilelang', 'rtp_llm.ops') if m in sys.modules];"
            "assert not bad, bad;"
            "assert callable(weight.load_ppu_weights);"
            "assert callable(weight.write_synthetic_checkpoint);"
            "assert model.PPUModelCore;"
            "print('ok')"
        )
        subprocess.run([sys.executable, "-c", script], check=True)

    @unittest.skipIf(
        DeepSeekV41PPUWeight.framework_base_available(),
        "full RTP framework chain importable here",
    )
    def test_framework_descriptor_fails_loud_without_framework(self):
        weight = DeepSeekV41PPUWeight()
        with self.assertRaises(RuntimeError) as ctx:
            weight._get_weight_info()
        self.assertIn("load_ppu_weights", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
