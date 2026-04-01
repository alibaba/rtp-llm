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

from rtp_llm.model_loader.ffn_weight import (
    MoeAtomicWeight,
    MoeConfig,
    MoeWeight,
    iter_stacked_moe_weights,
)
from rtp_llm.model_loader.tensor_source import StackSplitTensorSource, TensorSource
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity


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


if __name__ == "__main__":
    unittest.main()
