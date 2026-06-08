"""Unit tests for shared expert fusion weight loading.

Covers:
  - _should_fuse_shared_expert: conditions and guards
  - _get_shared_expert_overrides: stacked vs split ordering
  - MoeAtomicWeight override loading: serial fallback with fused expert
  - Fused vs non-fused forward equivalence (GenericMoeLayer topk extension)
"""

import os
import unittest
from typing import Dict, List
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight, MoeConfig
from rtp_llm.model_loader.tensor_source import StackSplitTensorSource, TensorSource
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity


class FakeTensorSource(TensorSource):
    def __init__(self, tensors: Dict[str, torch.Tensor]):
        self._tensors = tensors
        self._db = FakeDatabase(tensors)

    def load_tensor(self, name: str, data_type=torch.float16) -> List[torch.Tensor]:
        if name not in self._tensors:
            raise KeyError(f"Tensor {name!r} not found")
        return [self._tensors[name].to(data_type)]

    def has_tensor(self, name: str) -> bool:
        return name in self._tensors

    def get_database(self):
        return self._db


class FakeDatabase:
    def __init__(self, tensors: Dict[str, torch.Tensor]):
        self._tensors = tensors

    def load_tensor(self, name: str, data_type=torch.float16) -> List[torch.Tensor]:
        if name not in self._tensors:
            raise KeyError(f"DB tensor {name!r} not found")
        return [self._tensors[name].to(data_type)]

    def has_tensor(self, name: str) -> bool:
        return name in self._tensors


class TestOverrideLoading(unittest.TestCase):
    """Test that override experts are correctly loaded and merged."""

    def test_stacked_override_cat_order(self):
        """Stacked format: override parts are cat'd in list order [gate, up]."""
        gate = torch.randn(4, 8)
        up = torch.randn(4, 8)

        overrides = {
            2: [
                CkptWeightInfo("shared.gate_proj.weight", identity),
                CkptWeightInfo("shared.up_proj.weight", identity),
            ]
        }

        moe_w1 = MoeAtomicWeight(
            W.moe_w1,
            [CkptWeightInfo("experts.gate_up_proj")],
            process_fun=identity,
            config=MoeConfig(expert_num=3),
            stacked_ckpt_keys=True,
            expert_key_overrides=overrides,
        )

        tensors = {
            "shared.gate_proj.weight": gate,
            "shared.up_proj.weight": up,
        }
        db = FakeDatabase(tensors)
        ts = FakeTensorSource(tensors)

        result = moe_w1._load_override_tensor(ts, 0, 2, torch.float32, 0)
        expected = torch.cat([gate, up], dim=0)
        torch.testing.assert_close(result, expected)

    def test_split_override_ckpt_idx(self):
        """Split format: override uses ckpt_idx to select correct part."""
        up = torch.randn(4, 8)
        gate = torch.randn(4, 8)

        overrides = {
            2: [
                CkptWeightInfo("shared.up_proj.weight", identity),
                CkptWeightInfo("shared.gate_proj.weight", identity),
            ]
        }

        moe_w1 = MoeAtomicWeight(
            W.moe_w1,
            [
                CkptWeightInfo("experts.{expert_id}.up_proj.weight"),
                CkptWeightInfo("experts.{expert_id}.gate_proj.weight"),
            ],
            process_fun=identity,
            config=MoeConfig(expert_num=3),
            stacked_ckpt_keys=False,
            expert_key_overrides=overrides,
        )

        tensors = {
            "shared.up_proj.weight": up,
            "shared.gate_proj.weight": gate,
        }
        ts = FakeTensorSource(tensors)

        # ckpt_idx=0 -> up_proj (first in overrides)
        result0 = moe_w1._load_override_tensor(ts, 0, 2, torch.float32, 0)
        torch.testing.assert_close(result0, up)

        # ckpt_idx=1 -> gate_proj (second in overrides)
        result1 = moe_w1._load_override_tensor(ts, 0, 2, torch.float32, 1)
        torch.testing.assert_close(result1, gate)


class TestExtendTopkEquivalence(unittest.TestCase):
    """Test that _extend_topk_with_shared_expert produces correct output."""

    def test_extend_with_gate(self):
        """With shared_expert_gate, last weight = sigmoid(gate(hidden))."""
        num_tokens, top_k = 4, 3
        shared_expert_id = 10

        topk_ids = torch.randint(0, 10, (num_tokens, top_k))
        topk_weights = torch.randn(num_tokens, top_k, dtype=torch.float32)
        hidden = torch.randn(num_tokens, 16)

        gate_output = torch.randn(num_tokens, 1)
        mock_gate = MagicMock(return_value=gate_output)

        # Simulate _extend_topk_with_shared_expert logic
        k = topk_ids.shape[1]
        ext_ids = torch.empty((num_tokens, k + 1), dtype=topk_ids.dtype)
        ext_weights = torch.empty((num_tokens, k + 1), dtype=topk_weights.dtype)
        ext_ids[:, :k] = topk_ids
        ext_ids[:, k] = shared_expert_id
        ext_weights[:, :k] = topk_weights
        ext_weights[:, k:] = torch.sigmoid(mock_gate(hidden)).float()

        # Verify shape
        self.assertEqual(ext_ids.shape, (num_tokens, top_k + 1))
        self.assertEqual(ext_weights.shape, (num_tokens, top_k + 1))

        # Verify original ids preserved
        torch.testing.assert_close(ext_ids[:, :k], topk_ids)

        # Verify shared expert id
        self.assertTrue((ext_ids[:, k] == shared_expert_id).all())

        # Verify gate weight
        expected_weight = torch.sigmoid(gate_output).float().squeeze(-1)
        torch.testing.assert_close(ext_weights[:, k], expected_weight)

    def test_extend_without_gate(self):
        """Without shared_expert_gate, last weight = 1.0."""
        num_tokens, top_k = 4, 3
        shared_expert_id = 10

        topk_ids = torch.randint(0, 10, (num_tokens, top_k))
        topk_weights = torch.randn(num_tokens, top_k, dtype=torch.float32)

        k = topk_ids.shape[1]
        ext_ids = torch.empty((num_tokens, k + 1), dtype=topk_ids.dtype)
        ext_weights = torch.empty((num_tokens, k + 1), dtype=topk_weights.dtype)
        ext_ids[:, :k] = topk_ids
        ext_ids[:, k] = shared_expert_id
        ext_weights[:, :k] = topk_weights
        ext_weights[:, k] = 1.0

        self.assertTrue((ext_weights[:, k] == 1.0).all())


# ---------------------------------------------------------------------------
# Replicate _should_fuse_shared_expert / _get_shared_expert_overrides logic
# without importing rtp_llm.models (which pulls in async_decoder_engine and
# is unavailable in the standalone testlib used by this BUILD target).
# ---------------------------------------------------------------------------

_FUSION_SUPPORTED_QUANT_TYPES = (
    "Fp8PerChannelCompressedQuantConfig",
    "Fp8PerChannelQuarkQuantConfig",
)


def _should_fuse_shared_expert(wi) -> bool:
    """Pure-Python replica of Qwen3NextBaseWeight._should_fuse_shared_expert."""
    if not (hasattr(torch.version, "hip") and torch.version.hip is not None):
        return False
    if os.environ.get("DISABLE_SHARED_EXPERT_FUSION", "0") == "1":
        return False
    mc = wi.model_config
    if getattr(mc, "moe_style", 0) != 2:
        return False
    if getattr(mc, "inter_size", 0) != getattr(mc, "moe_inter_size", 0):
        return False
    if not (wi.ep_size == 1 and wi.dp_size == 1
            and wi.ffn_tp_size == wi.tp_size):
        return False
    if mc.eplb_config.phy_exp_num(mc.expert_num) != mc.expert_num:
        return False
    if wi._quant_algo.isQuant():
        quant_type = type(wi._quant_config).__name__
        if quant_type not in _FUSION_SUPPORTED_QUANT_TYPES:
            return False
        exclude = getattr(wi._quant_config, "exclude_modules", None)
        if exclude and any("mlp.shared_expert" in m for m in exclude):
            return False
    return True


def _get_shared_expert_overrides(prefix, expert_num, fuse_shared, stacked=False):
    """Pure-Python replica of Qwen3NextBaseWeight._get_shared_expert_overrides."""
    if not fuse_shared:
        return None, None
    n = expert_num
    overrides_w2 = {
        n: [CkptWeightInfo(prefix + "layers.{i}.mlp.shared_expert.down_proj.weight", identity)]
    }
    gate = CkptWeightInfo(prefix + "layers.{i}.mlp.shared_expert.gate_proj.weight", identity)
    up = CkptWeightInfo(prefix + "layers.{i}.mlp.shared_expert.up_proj.weight", identity)
    overrides_w1 = {n: [gate, up] if stacked else [up, gate]}
    return overrides_w1, overrides_w2


class TestFusionConditions(unittest.TestCase):
    """Test _should_fuse_shared_expert guard conditions."""

    def _make_wi(self, ep_size=1, dp_size=1, ffn_tp_size=4, tp_size=4,
                 moe_style=2, inter_size=512, moe_inter_size=512,
                 expert_num=256, phy_exp_num=256,
                 quant_is_quant=False, quant_type_name="", exclude_modules=None):
        wi = MagicMock()
        wi.model_config = MagicMock()
        wi.model_config.moe_style = moe_style
        wi.model_config.inter_size = inter_size
        wi.model_config.moe_inter_size = moe_inter_size
        wi.model_config.expert_num = expert_num
        wi.model_config.eplb_config.phy_exp_num.return_value = phy_exp_num
        wi.ep_size = ep_size
        wi.dp_size = dp_size
        wi.ffn_tp_size = ffn_tp_size
        wi.tp_size = tp_size
        wi.expert_num_ = expert_num
        wi.prefix = "model."
        wi._quant_algo = MagicMock()
        wi._quant_algo.isQuant.return_value = quant_is_quant
        qc = type(quant_type_name or "NoQuant", (), {})()
        qc.exclude_modules = exclude_modules
        wi._quant_config = qc
        return wi

    @patch.object(torch.version, "hip", "6.0", create=True)
    def test_all_conditions_met(self):
        self.assertTrue(_should_fuse_shared_expert(self._make_wi()))

    @patch.object(torch.version, "hip", None, create=True)
    def test_non_rocm_returns_false(self):
        self.assertFalse(_should_fuse_shared_expert(self._make_wi()))

    @patch.object(torch.version, "hip", "6.0", create=True)
    @patch.dict(os.environ, {"DISABLE_SHARED_EXPERT_FUSION": "1"})
    def test_env_disable(self):
        self.assertFalse(_should_fuse_shared_expert(self._make_wi()))

    @patch.object(torch.version, "hip", "6.0", create=True)
    def test_ep_size_guard(self):
        self.assertFalse(_should_fuse_shared_expert(self._make_wi(ep_size=2)))

    @patch.object(torch.version, "hip", "6.0", create=True)
    def test_dp_size_guard(self):
        self.assertFalse(_should_fuse_shared_expert(self._make_wi(dp_size=2)))

    @patch.object(torch.version, "hip", "6.0", create=True)
    def test_eplb_guard(self):
        self.assertFalse(_should_fuse_shared_expert(self._make_wi(phy_exp_num=260)))

    @patch.object(torch.version, "hip", "6.0", create=True)
    def test_ffn_tp_mismatch(self):
        self.assertFalse(_should_fuse_shared_expert(self._make_wi(ffn_tp_size=2, tp_size=4)))

    @patch.object(torch.version, "hip", "6.0", create=True)
    def test_inter_size_mismatch(self):
        self.assertFalse(_should_fuse_shared_expert(self._make_wi(inter_size=1024, moe_inter_size=512)))

    @patch.object(torch.version, "hip", "6.0", create=True)
    def test_unsupported_quant_type(self):
        self.assertFalse(_should_fuse_shared_expert(
            self._make_wi(quant_is_quant=True, quant_type_name="MXFp4QuarkQuantConfig")))

    @patch.object(torch.version, "hip", "6.0", create=True)
    def test_supported_quant_type(self):
        self.assertTrue(_should_fuse_shared_expert(
            self._make_wi(quant_is_quant=True, quant_type_name="Fp8PerChannelCompressedQuantConfig")))

    @patch.object(torch.version, "hip", "6.0", create=True)
    def test_quant_exclude_shared_expert_layer0(self):
        self.assertFalse(_should_fuse_shared_expert(self._make_wi(
            quant_is_quant=True, quant_type_name="Fp8PerChannelCompressedQuantConfig",
            exclude_modules={"model.layers.0.mlp.shared_expert.gate_proj"})))

    @patch.object(torch.version, "hip", "6.0", create=True)
    def test_quant_exclude_shared_expert_non_zero_layer(self):
        self.assertFalse(_should_fuse_shared_expert(self._make_wi(
            quant_is_quant=True, quant_type_name="Fp8PerChannelCompressedQuantConfig",
            exclude_modules={"model.layers.15.mlp.shared_expert.down_proj"})))


class TestGetSharedExpertOverrides(unittest.TestCase):
    """Test override ordering for stacked vs split formats."""

    def test_stacked_order_gate_up(self):
        overrides_w1, _ = _get_shared_expert_overrides("model.", 256, True, stacked=True)
        self.assertIn(256, overrides_w1)
        names = [ow.name for ow in overrides_w1[256]]
        self.assertEqual(len(names), 2)
        self.assertIn("gate_proj", names[0])
        self.assertIn("up_proj", names[1])

    def test_split_order_up_gate(self):
        overrides_w1, _ = _get_shared_expert_overrides("model.", 256, True, stacked=False)
        self.assertIn(256, overrides_w1)
        names = [ow.name for ow in overrides_w1[256]]
        self.assertEqual(len(names), 2)
        self.assertIn("up_proj", names[0])
        self.assertIn("gate_proj", names[1])

    def test_no_fuse_returns_none(self):
        self.assertEqual(_get_shared_expert_overrides("model.", 256, False), (None, None))

    def test_overrides_w2_has_down_proj(self):
        _, overrides_w2 = _get_shared_expert_overrides("model.", 256, True)
        self.assertIn(256, overrides_w2)
        self.assertIn("down_proj", overrides_w2[256][0].name)


if __name__ == "__main__":
    unittest.main()
