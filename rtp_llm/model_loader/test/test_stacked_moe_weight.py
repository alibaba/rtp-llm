"""Unit tests for stacked MoE weight loading infrastructure.

Covers:
  - StackSplitTensorSource: cache hit/miss, split correctness, passthrough
  - MoeAtomicWeight._build_split_config: per-expert key generation
  - ModelLoader._build_stacked_key_config: stacked key mapping construction
"""

import unittest
from typing import Dict, List
from unittest.mock import MagicMock

import torch

from rtp_llm.config.quant_config import (
    Fp8BlockWiseQuantConfig,
    Fp8MxBlockWiseQuantConfig,
)
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight
from rtp_llm.model_loader.ffn_weight import (
    FfnAtomicWeight,
    FfnConfig,
    FfnWeight,
    MoeAtomicWeight,
    MoeConfig,
    MoeWeight,
    iter_stacked_moe_weights,
)
from rtp_llm.model_loader.offline_modelopt_fp4_quant_weight import (
    OfflineMegaMoeFp4MoeWeight,
    OfflineMegaMoeFp8SharedExpertWeight,
    wrap_for_offline_fp4,
    wrap_shared_expert_for_offline_fp4,
)
from rtp_llm.model_loader.mxfp8_quant_weight import Mxfp8Weight
from rtp_llm.model_loader.online_modelopt_fp4_quant_weight import (
    OnlineMegaMoeFp4FromFp8Weight,
    wrap_moe_for_mega_moe,
)
from rtp_llm.model_loader.per_block_fp8_quant_weight import (
    PerBlockFp8Weight,
    V4PerBlockFp8Weight,
)
from rtp_llm.model_loader.tensor_source import StackSplitTensorSource, TensorSource
from rtp_llm.utils.model_weight import CkptWeightInfo, W, concat_0, identity


def _swap_stacked_halves(ts: List[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(ts)
    first, second = stacked.chunk(2, dim=1)
    return torch.cat((second, first), dim=1)


class FakeTensorSource(TensorSource):
    """In-memory TensorSource for testing."""

    def __init__(self, tensors: Dict[str, torch.Tensor]):
        self._tensors = tensors

    def load_tensor(self, name: str, data_type=torch.float16) -> List[torch.Tensor]:
        if name not in self._tensors:
            raise KeyError(f"Tensor {name!r} not found")
        return [self._tensors[name].to(data_type)]

    def has_tensor(self, name: str) -> bool:
        return name in self._tensors

    def get_database(self):
        return None


class TestV4SharedExpertW13Weight(unittest.TestCase):
    def test_v4_shared_w13_derives_weight_and_scale_inputs(self):
        src = AttnAtomicWeight(
            W.v4_shared_w13_w,
            [
                CkptWeightInfo("layers.{i}.ffn.shared_experts.w1.weight", identity),
                CkptWeightInfo("layers.{i}.ffn.shared_experts.w3.weight", identity),
            ],
            concat_0,
        )

        wrapped = src.create(src, Fp8BlockWiseQuantConfig(is_quanted=True))

        self.assertIsInstance(wrapped, V4PerBlockFp8Weight)
        self.assertEqual(wrapped.kernel.name, W.v4_shared_w13_w)
        self.assertEqual(wrapped.scale.name, W.v4_shared_w13_s)
        self.assertEqual(
            [w.name for w in wrapped.kernel.weights],
            [
                "layers.{i}.ffn.shared_experts.w1.weight",
                "layers.{i}.ffn.shared_experts.w3.weight",
            ],
        )
        self.assertEqual(
            [w.name for w in wrapped.scale.weights],
            [
                "layers.{i}.ffn.shared_experts.w1.scale",
                "layers.{i}.ffn.shared_experts.w3.scale",
            ],
        )
        self.assertIs(wrapped.kernel.process_fun, concat_0)
        self.assertIs(wrapped.scale.process_fun, concat_0)


class TestDirectModelOptMxfp4Weight(unittest.TestCase):
    def test_fused_weight_and_raw_ue8m0_scale_are_loaded_byte_exact(self):
        src = MoeAtomicWeight(
            W.moe_w1,
            [CkptWeightInfo("model.layers.{i}.mlp.experts.w13_weight")],
            _swap_stacked_halves,
            config=MoeConfig(expert_num=2),
            stacked_ckpt_keys=True,
        )
        offline = OfflineMegaMoeFp4MoeWeight(
            src, scale_dtype=torch.float8_e8m0fnu
        )
        self.assertEqual(
            offline.scale.weights[0].name,
            "model.layers.{i}.mlp.experts.w13_weight_scale",
        )
        self.assertTrue(offline.scale.stacked_ckpt_keys)
        self.assertEqual(offline.scale.data_type, torch.uint8)

        packed = torch.arange(2 * 4 * 2, dtype=torch.uint8).reshape(2, 4, 2)
        scale_bits = torch.tensor(
            [
                [[123], [124], [125], [126]],
                [[127], [128], [129], [130]],
            ],
            dtype=torch.uint8,
        )
        source = FakeTensorSource(
            {
                "model.layers.0.mlp.experts.w13_weight": packed,
                "model.layers.0.mlp.experts.w13_weight_scale": scale_bits,
            }
        )
        load_config = MagicMock()
        load_config.get_selected_experts.return_value = [0, 1]
        load_config.compute_dtype = torch.bfloat16

        raw = offline._load_raw_tensor(source, 0, "cpu", load_config)
        processed = offline._postprocess(raw, "cpu", load_config)
        expected_packed = torch.cat((packed[:, 2:], packed[:, :2]), dim=1)
        expected_scale_bits = torch.cat(
            (scale_bits[:, 2:], scale_bits[:, :2]), dim=1
        )
        torch.testing.assert_close(
            processed[W.moe_w1].view(torch.uint8), expected_packed
        )
        self.assertEqual(processed[W.moe_s1].dtype, torch.float8_e8m0fnu)
        torch.testing.assert_close(
            processed[W.moe_s1].view(torch.uint8), expected_scale_bits
        )


class TestOnlineMxfp8ToMegaMoeFp4Weight(unittest.TestCase):
    def test_routed_expert_selects_1x32_fp4_converter(self):
        src = MoeAtomicWeight(
            W.moe_w1,
            [CkptWeightInfo("model.layers.{i}.mlp.experts.gate_up_proj")],
            identity,
            config=MoeConfig(expert_num=256),
            stacked_ckpt_keys=True,
        )
        mxfp8 = Mxfp8Weight(
            src,
            Fp8MxBlockWiseQuantConfig(
                is_quanted=True,
                checkpoint_scale_suffix=".weight_scale",
                packed_scale_suffix="_scale",
            ),
            name=src.name,
        )

        wrapped = wrap_moe_for_mega_moe(mxfp8)

        self.assertIsInstance(wrapped, OnlineMegaMoeFp4FromFp8Weight)
        self.assertEqual(wrapped._source_block_size, 32)
        self.assertTrue(wrapped._scale_is_ue8m0_exponent)
        self.assertEqual(
            [item.name for item in wrapped.fp8_scale.weights],
            ["model.layers.{i}.mlp.experts.gate_up_proj_scale"],
        )


class TestOfflineFp4SharedExpertWeight(unittest.TestCase):
    def _make_shared_ffn(self):
        config = FfnConfig(align_size=0, is_gated_activation=True)
        return FfnWeight(
            sub_weights=[
                FfnAtomicWeight(
                    W.ffn_w1,
                    [
                        CkptWeightInfo(
                            "model.layers.{i}.mlp.shared_experts.gate_proj.weight",
                            identity,
                        )
                    ],
                    identity,
                    config=config,
                ),
                FfnAtomicWeight(
                    W.ffn_w2,
                    [
                        CkptWeightInfo(
                            "model.layers.{i}.mlp.shared_experts.down_proj.weight",
                            identity,
                        )
                    ],
                    identity,
                    config=config,
                ),
                FfnAtomicWeight(
                    W.ffn_w3,
                    [
                        CkptWeightInfo(
                            "model.layers.{i}.mlp.shared_experts.up_proj.weight",
                            identity,
                        )
                    ],
                    identity,
                    config=config,
                ),
            ],
            config=config,
        )

    def test_unwraps_fp8_shared_w13_to_offline_fp8_scale_names(self):
        ffn = self._make_shared_ffn()
        fp8_wrapped = ffn.w13.create(ffn.w13, Fp8BlockWiseQuantConfig(is_quanted=True))

        self.assertIsInstance(fp8_wrapped, PerBlockFp8Weight)
        offline = wrap_shared_expert_for_offline_fp4(fp8_wrapped)

        self.assertIsInstance(offline, OfflineMegaMoeFp8SharedExpertWeight)
        self.assertEqual(offline.kernel.data_type, torch.float8_e4m3fn)
        self.assertEqual(
            [w.name for w in offline.scale.weights],
            [
                "model.layers.{i}.mlp.shared_experts.gate_proj.weight_scale_inv",
                "model.layers.{i}.mlp.shared_experts.up_proj.weight_scale_inv",
            ],
        )

    def test_fastsafetensors_loader_keeps_fp8_quant_postprocess_boundary(self):
        """Do not split a QuantWeight into independent kernel/scale load units.

        Loading the scale as an AtomicWeight bypasses PerBlockFp8Weight._postprocess,
        leaving a checkpoint UE8M0 tensor unconverted when generic DeepGEMM expects
        a float32 or packed int32 scale.
        """
        from rtp_llm.model_loader.loader import ModelLoader

        ffn = self._make_shared_ffn()
        fp8_wrapped = ffn.w13.create(ffn.w13, Fp8BlockWiseQuantConfig(is_quanted=True))
        self.assertIsInstance(fp8_wrapped, PerBlockFp8Weight)

        loader = object.__new__(ModelLoader)
        loader._load_config = MagicMock()
        loader._load_config.vit_separation = object()
        loader._model_weights_info = MagicMock()
        loader._model_weights_info.layer_weights = [fp8_wrapped]
        loader._model_weights_info.weights = []
        loader._misc_weights_info = []

        tensor_to_weight, load_units = loader._generate_weight_info()
        expected_keys = fp8_wrapped.get_tensor_names(0, loader._load_config)

        self.assertEqual(len(load_units), 1)
        self.assertIs(load_units[0].weight, fp8_wrapped)
        self.assertEqual(set(tensor_to_weight), expected_keys)
        self.assertTrue(
            all(infos == [load_units[0]] for infos in tensor_to_weight.values())
        )

    def test_unwraps_fp8_shared_w2_to_offline_fp8_scale_name(self):
        ffn = self._make_shared_ffn()
        fp8_wrapped = ffn.w2.create(ffn.w2, Fp8BlockWiseQuantConfig(is_quanted=True))

        self.assertIsInstance(fp8_wrapped, PerBlockFp8Weight)
        offline = wrap_shared_expert_for_offline_fp4(fp8_wrapped)

        self.assertIsInstance(offline, OfflineMegaMoeFp8SharedExpertWeight)
        self.assertEqual(offline.kernel.data_type, torch.float8_e4m3fn)
        self.assertEqual(
            [w.name for w in offline.scale.weights],
            ["model.layers.{i}.mlp.shared_experts.down_proj.weight_scale_inv"],
        )

    def test_strategy_wrapper_skips_shared_expert_unless_requested(self):
        ffn = self._make_shared_ffn()
        fp8_wrapped = ffn.w13.create(ffn.w13, Fp8BlockWiseQuantConfig(is_quanted=True))

        self.assertIs(wrap_for_offline_fp4(fp8_wrapped), fp8_wrapped)

        offline = wrap_for_offline_fp4(fp8_wrapped, include_shared_expert=True)
        self.assertIsInstance(offline, OfflineMegaMoeFp8SharedExpertWeight)

    def test_strategy_wrapper_always_wraps_routed_moe(self):
        moe = MoeAtomicWeight(
            W.moe_w1,
            [
                CkptWeightInfo(
                    "model.layers.{i}.mlp.experts.0.gate_proj.weight",
                    identity,
                )
            ],
            identity,
            config=MoeConfig(expert_num=1),
        )

        offline = wrap_for_offline_fp4(moe)

        self.assertIsInstance(offline, OfflineMegaMoeFp4MoeWeight)
        self.assertEqual(
            [w.name for w in offline.scale.weights],
            ["model.layers.{i}.mlp.experts.0.gate_proj.weight_scale"],
        )

    def test_shared_expert_split_keeps_full_ue8m0_tensor_on_ep_ranks(self):
        ffn = self._make_shared_ffn()
        offline = OfflineMegaMoeFp8SharedExpertWeight(
            ffn.w13, scale_dtype=torch.float8_e8m0fnu
        )
        kernel = torch.empty((16, 8), dtype=torch.float8_e4m3fn)
        scale = torch.zeros((2, 1), dtype=torch.uint8).view(torch.float8_e8m0fnu)
        load_config = MagicMock(tp_size=1, ffn_tp_size=1, ep_size=4, dp_size=1)

        split = offline._split(
            {offline.kernel.name: kernel, offline.scale.name: scale}, load_config
        )

        self.assertIs(split[offline.kernel.name], kernel)
        self.assertIs(split[offline.scale.name], scale)
        self.assertEqual(split[offline.scale.name].dtype, torch.float8_e8m0fnu)

    def test_modelopt_mxfp8_down_proj_split_uses_ffn_tp(self):
        ffn = self._make_shared_ffn()
        mxfp8 = Mxfp8Weight(
            ffn.w2,
            Fp8MxBlockWiseQuantConfig(
                is_quanted=True,
                checkpoint_scale_suffix=".weight_scale",
            ),
            name=ffn.w2.name,
        )
        kernel = torch.empty((8, 64), dtype=torch.float8_e4m3fn)
        scale = torch.empty((8, 2), dtype=torch.float32)
        load_config = MagicMock(
            tp_size=8,
            tp_rank=3,
            ffn_tp_size=1,
            ffn_tp_rank=0,
            ep_size=8,
            ep_rank=3,
            dp_size=1,
            dp_rank=0,
        )

        split = mxfp8._split(
            {mxfp8.kernel.name: kernel, mxfp8.scale.name: scale}, load_config
        )

        self.assertEqual(tuple(split[mxfp8.kernel.name].shape), (8, 64))
        self.assertEqual(tuple(split[mxfp8.scale.name].shape), (8, 2))


class TestStackSplitTensorSource(unittest.TestCase):
    def _make_stacked(self, num_experts: int, expert_dim: int):
        return torch.randn(num_experts, expert_dim, dtype=torch.float32)

    def test_split_correctness(self):
        """Per-expert slices should match manual indexing."""
        num_experts = 4
        stacked = self._make_stacked(num_experts, 8)
        base = FakeTensorSource({"stacked_w": stacked})

        split_config = {}
        for eid in range(num_experts):
            split_config[f"expert.{eid}"] = ("stacked_w", eid, identity)

        src = StackSplitTensorSource(base, split_config)
        for eid in range(num_experts):
            result = src.load_tensor(f"expert.{eid}", torch.float32)
            self.assertEqual(len(result), 1)
            torch.testing.assert_close(result[0], stacked[eid])

    def test_cache_hit(self):
        """Stacked tensor should be loaded from base only once."""
        stacked = self._make_stacked(2, 4)
        base = FakeTensorSource({"stacked_w": stacked})
        original_load = base.load_tensor
        call_count = 0

        def counting_load(name, data_type=torch.float16):
            nonlocal call_count
            call_count += 1
            return original_load(name, data_type)

        base.load_tensor = counting_load

        split_config = {
            "expert.0": ("stacked_w", 0, identity),
            "expert.1": ("stacked_w", 1, identity),
        }
        src = StackSplitTensorSource(base, split_config)
        src.load_tensor("expert.0", torch.float32)
        src.load_tensor("expert.1", torch.float32)
        self.assertEqual(call_count, 1)

    def test_passthrough_for_non_stacked_keys(self):
        """Keys not in split_config should pass through to base."""
        regular_tensor = torch.randn(3, 3)
        base = FakeTensorSource({"regular_w": regular_tensor})
        src = StackSplitTensorSource(base, {})
        result = src.load_tensor("regular_w", torch.float32)
        self.assertEqual(len(result), 1)
        torch.testing.assert_close(result[0], regular_tensor)

    def test_has_tensor_delegates(self):
        stacked = self._make_stacked(2, 4)
        base = FakeTensorSource({"stacked_w": stacked})
        split_config = {"expert.0": ("stacked_w", 0, identity)}
        src = StackSplitTensorSource(base, split_config)

        self.assertTrue(src.has_tensor("expert.0"))
        self.assertFalse(src.has_tensor("nonexistent"))

    def test_single_expert(self):
        """Boundary case: num_experts=1."""
        stacked = self._make_stacked(1, 16)
        base = FakeTensorSource({"stacked_w": stacked})
        split_config = {"expert.0": ("stacked_w", 0, identity)}
        src = StackSplitTensorSource(base, split_config)
        result = src.load_tensor("expert.0", torch.float32)
        torch.testing.assert_close(result[0], stacked[0])

    def test_custom_merge_fun(self):
        """Non-identity merge_fun should be applied to the loaded tensor."""
        raw = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        base = FakeTensorSource({"stacked_w": raw})

        def double_merge(tensors):
            return tensors[0] * 2

        split_config = {"expert.0": ("stacked_w", 0, double_merge)}
        src = StackSplitTensorSource(base, split_config)
        result = src.load_tensor("expert.0", torch.float32)
        expected = raw * 2
        torch.testing.assert_close(result[0], expected[0])


class TestBuildSplitConfig(unittest.TestCase):
    def _make_load_config(self, expert_num: int):
        lc = MagicMock()
        lc.get_selected_experts.return_value = list(range(expert_num))
        return lc

    def test_basic_config(self):
        """Verify per-expert keys and stacked key mapping."""
        expert_num = 3
        config = MoeConfig(expert_num=expert_num)
        ckpt_weights = [CkptWeightInfo("model.layers.{i}.moe.gate_up")]
        moe_w = MoeAtomicWeight(
            name=W.moe_w1,
            weights=ckpt_weights,
            config=config,
            stacked_ckpt_keys=True,
        )

        lc = self._make_load_config(expert_num)
        split_config = moe_w._build_split_config(layer_id=0, load_config=lc)

        self.assertEqual(len(split_config), expert_num)
        for eid in range(expert_num):
            key = f"layers.0.moe.{W.moe_w1}.{eid}.0"
            self.assertIn(key, split_config)
            stacked_key, expert_id, _ = split_config[key]
            self.assertEqual(stacked_key, "model.layers.0.moe.gate_up")
            self.assertEqual(expert_id, eid)

    def test_multiple_ckpt_weights(self):
        """Each ckpt_weight produces its own set of per-expert keys."""
        expert_num = 2
        config = MoeConfig(expert_num=expert_num)
        ckpt_weights = [
            CkptWeightInfo("model.layers.{i}.w_gate"),
            CkptWeightInfo("model.layers.{i}.w_up"),
        ]
        moe_w = MoeAtomicWeight(
            name=W.moe_w1,
            weights=ckpt_weights,
            config=config,
            stacked_ckpt_keys=True,
        )

        lc = self._make_load_config(expert_num)
        split_config = moe_w._build_split_config(layer_id=1, load_config=lc)

        self.assertEqual(len(split_config), expert_num * 2)
        self.assertIn(f"layers.1.moe.{W.moe_w1}.0.0", split_config)
        self.assertIn(f"layers.1.moe.{W.moe_w1}.1.1", split_config)


def _make_moe_weight(config, w1_ckpt, w2_ckpt, stacked):
    """Helper: create a MoeWeight with w1 + w2 sub_weights."""
    moe_w1 = MoeAtomicWeight(
        name=W.moe_w1,
        weights=w1_ckpt,
        config=config,
        stacked_ckpt_keys=stacked,
    )
    moe_w2 = MoeAtomicWeight(
        name=W.moe_w2,
        weights=w2_ckpt,
        config=config,
        stacked_ckpt_keys=stacked,
    )
    return MoeWeight(sub_weights=[moe_w1, moe_w2], config=config)


class TestBuildStackedKeyConfig(unittest.TestCase):
    """Tests for ModelLoader._build_stacked_key_config (static method)."""

    def test_basic_mapping(self):
        from rtp_llm.model_loader.loader import ModelLoader

        config = MoeConfig(expert_num=4)
        moe_weight = _make_moe_weight(
            config,
            [CkptWeightInfo("model.layers.{i}.moe.w1")],
            [CkptWeightInfo("model.layers.{i}.moe.w2")],
            stacked=True,
        )

        wi = MagicMock()
        wi.weight = moe_weight
        wi.layer_id = 0

        result = ModelLoader._build_stacked_key_config([wi])
        self.assertIn("model.layers.0.moe.w1", result)
        self.assertIn("model.layers.0.moe.w2", result)
        template = result["model.layers.0.moe.w1"]
        self.assertIn("{expert_id}", template)
        formatted = template.format(expert_id=2)
        self.assertIn("2", formatted)

    def test_non_identity_merge_fun_skipped(self):
        from rtp_llm.model_loader.loader import ModelLoader

        config = MoeConfig(expert_num=2)

        def custom_merge(ts):
            return torch.stack(ts)

        moe_weight = _make_moe_weight(
            config,
            [CkptWeightInfo("model.layers.{i}.w1", merge_fun=custom_merge)],
            [CkptWeightInfo("model.layers.{i}.w2", merge_fun=custom_merge)],
            stacked=True,
        )

        wi = MagicMock()
        wi.weight = moe_weight
        wi.layer_id = 0

        result = ModelLoader._build_stacked_key_config([wi])
        self.assertEqual(len(result), 0)

    def test_non_stacked_weights_ignored(self):
        from rtp_llm.model_loader.loader import ModelLoader

        config = MoeConfig(expert_num=4)
        moe_weight = _make_moe_weight(
            config,
            [CkptWeightInfo("model.layers.{i}.moe.w1")],
            [CkptWeightInfo("model.layers.{i}.moe.w2")],
            stacked=False,
        )

        wi = MagicMock()
        wi.weight = moe_weight
        wi.layer_id = 0

        result = ModelLoader._build_stacked_key_config([wi])
        self.assertEqual(len(result), 0)


class TestIterStackedMoeWeights(unittest.TestCase):
    def test_yields_stacked(self):
        config = MoeConfig(expert_num=2)
        w = MoeAtomicWeight(
            name=W.moe_w1,
            weights=[CkptWeightInfo("x.{i}")],
            config=config,
            stacked_ckpt_keys=True,
        )
        results = list(iter_stacked_moe_weights(w))
        self.assertEqual(len(results), 1)
        self.assertIs(results[0], w)

    def test_skips_non_stacked(self):
        config = MoeConfig(expert_num=2)
        w = MoeAtomicWeight(
            name=W.moe_w1,
            weights=[CkptWeightInfo("x.{i}")],
            config=config,
            stacked_ckpt_keys=False,
        )
        results = list(iter_stacked_moe_weights(w))
        self.assertEqual(len(results), 0)

    def test_recurse_composite(self):
        config = MoeConfig(expert_num=2)
        w1 = MoeAtomicWeight(
            name=W.moe_w1,
            weights=[CkptWeightInfo("x.{i}")],
            config=config,
            stacked_ckpt_keys=True,
        )
        w2 = MoeAtomicWeight(
            name=W.moe_w2,
            weights=[CkptWeightInfo("y.{i}")],
            config=config,
            stacked_ckpt_keys=True,
        )
        moe = MoeWeight(sub_weights=[w1, w2], config=config)
        results = list(iter_stacked_moe_weights(moe))
        self.assertEqual(len(results), 2)
        self.assertIs(results[0], w1)
        self.assertIs(results[1], w2)


class TestGpuPreallocate(unittest.TestCase):
    """Tests for MoeAtomicWeight._load_raw_tensor_gpu_preallocate."""

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_stack_moe_w1_gpu_preallocate(self):
        """GPU preallocate for stack_moe_w1 should match CPU fallback."""
        from rtp_llm.utils.model_weight import stack_moe_w1

        num_experts = 4
        intermediate = 8
        hidden = 16
        # Build fake per-expert tensors: gate[expert_id] and up[expert_id]
        tensors = {}
        for eid in range(num_experts):
            tensors[f"layers.0.gate.{eid}"] = torch.randn(intermediate, hidden)
            tensors[f"layers.0.up.{eid}"] = torch.randn(intermediate, hidden)

        config = MoeConfig(expert_num=num_experts)
        ckpt_weights = [
            CkptWeightInfo("layers.{i}.gate.{expert_id}"),
            CkptWeightInfo("layers.{i}.up.{expert_id}"),
        ]
        moe_w = MoeAtomicWeight(
            name=W.moe_w1,
            weights=ckpt_weights,
            config=config,
            process_fun=stack_moe_w1,
        )

        source = FakeTensorSource(tensors)
        lc = MagicMock()
        lc.get_selected_experts.return_value = list(range(num_experts))
        lc.compute_dtype = torch.float16

        # CPU fallback path
        cpu_result = moe_w._load_raw_tensor(source, 0, "cpu", lc)
        cpu_tensor = cpu_result[W.moe_w1]

        # GPU preallocate path
        gpu_result = moe_w._load_raw_tensor(source, 0, "cuda:0", lc)
        gpu_tensor = gpu_result[W.moe_w1]

        self.assertEqual(cpu_tensor.shape, gpu_tensor.shape)
        torch.testing.assert_close(cpu_tensor, gpu_tensor.cpu())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_stack_gpu_preallocate(self):
        """GPU preallocate for stack_ should match CPU fallback."""
        from rtp_llm.utils.model_weight import stack_

        num_experts = 4
        dim0, dim1 = 8, 16
        tensors = {}
        for eid in range(num_experts):
            tensors[f"layers.0.w2.{eid}"] = torch.randn(dim0, dim1)

        config = MoeConfig(expert_num=num_experts)
        ckpt_weights = [CkptWeightInfo("layers.{i}.w2.{expert_id}")]
        moe_w = MoeAtomicWeight(
            name=W.moe_w2,
            weights=ckpt_weights,
            config=config,
            process_fun=stack_,
        )

        source = FakeTensorSource(tensors)
        lc = MagicMock()
        lc.get_selected_experts.return_value = list(range(num_experts))
        lc.compute_dtype = torch.float16

        cpu_result = moe_w._load_raw_tensor(source, 0, "cpu", lc)
        gpu_result = moe_w._load_raw_tensor(source, 0, "cuda:0", lc)

        torch.testing.assert_close(cpu_result[W.moe_w2], gpu_result[W.moe_w2].cpu())

    def test_fallback_on_cpu_device(self):
        """When device is CPU, should NOT use GPU preallocate path."""
        from rtp_llm.utils.model_weight import stack_

        num_experts = 4
        dim0, dim1 = 8, 16
        tensors = {}
        for eid in range(num_experts):
            tensors[f"layers.0.w2.{eid}"] = torch.randn(dim0, dim1)

        config = MoeConfig(expert_num=num_experts)
        ckpt_weights = [CkptWeightInfo("layers.{i}.w2.{expert_id}")]
        moe_w = MoeAtomicWeight(
            name=W.moe_w2,
            weights=ckpt_weights,
            config=config,
            process_fun=stack_,
        )

        source = FakeTensorSource(tensors)
        lc = MagicMock()
        lc.get_selected_experts.return_value = list(range(num_experts))
        lc.compute_dtype = torch.float16

        result = moe_w._load_raw_tensor(source, 0, "cpu", lc)
        # Should still produce correct result via fallback
        self.assertEqual(result[W.moe_w2].shape[0], num_experts)
        self.assertEqual(result[W.moe_w2].device.type, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_stack_moe_w1_1d_scale_fallback(self):
        """1D expert tensors (e.g. per-tensor quant scales) should fall back
        to the normal path instead of crashing in GPU preallocate."""
        from rtp_llm.utils.model_weight import stack_moe_w1

        num_experts = 4
        tensors = {}
        for eid in range(num_experts):
            tensors[f"layers.0.gate_s.{eid}"] = torch.tensor([1.0])
            tensors[f"layers.0.up_s.{eid}"] = torch.tensor([2.0])

        config = MoeConfig(expert_num=num_experts)
        ckpt_weights = [
            CkptWeightInfo("layers.{i}.gate_s.{expert_id}"),
            CkptWeightInfo("layers.{i}.up_s.{expert_id}"),
        ]
        moe_w = MoeAtomicWeight(
            name=W.moe_w1,
            weights=ckpt_weights,
            config=config,
            process_fun=stack_moe_w1,
        )

        source = FakeTensorSource(tensors)
        lc = MagicMock()
        lc.get_selected_experts.return_value = list(range(num_experts))
        lc.compute_dtype = torch.float16

        # Should not crash — falls back to serial path for 1D tensors
        result = moe_w._load_raw_tensor(source, 0, "cuda:0", lc)
        self.assertIn(W.moe_w1, result)


if __name__ == "__main__":
    unittest.main()
