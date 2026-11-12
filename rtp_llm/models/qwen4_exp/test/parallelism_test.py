import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.config.quant_config import QWEN4_FP8_NVFP4_RECIPE, ModelOptFp4Config
from rtp_llm.config.qwen4_exp_config import resolve_qwen4_parallelism
from rtp_llm.config.server_config_setup import set_parallelism_config
from rtp_llm.models_py.modules.factory.fused_moe.utils.nvfp4_parallelism import (
    use_nvfp4_tensor_parallel,
)
from rtp_llm.ops import ParallelismConfig


def recipe():
    return ModelOptFp4Config(
        bits=4,
        group_size=16,
        is_quanted=False,
        source_fp8_block_size=128,
        require_contiguous_experts=True,
        model_recipe=QWEN4_FP8_NVFP4_RECIPE,
    )


def topology(tp, ep, rank=0):
    pc = ParallelismConfig()
    pc.tp_size = pc.world_size = pc.local_world_size = tp
    pc.ep_size = ep
    pc.world_rank = rank
    set_parallelism_config(pc)
    return pc


class Qwen4ParallelismTest(unittest.TestCase):
    def test_nvfp4_tensor_parallel_uses_native_topology(self):
        pc = topology(2, 1)
        config = SimpleNamespace(
            parallelism_config=pc,
            model_config=SimpleNamespace(quant_config=recipe()),
            tp_size=2,
            ep_size=1,
        )
        self.assertTrue(use_nvfp4_tensor_parallel(config))

    def test_effective_layout_is_model_owned_and_repeatable(self):
        for tp, ep in ((1, 1), (2, 1), (2, 2), (8, 8)):
            for rank in range(tp):
                with self.subTest(tp=tp, ep=ep, rank=rank):
                    original = topology(tp, ep, rank)
                    config = SimpleNamespace(quant_config=recipe())
                    target = resolve_qwen4_parallelism(config, original)
                    draft = resolve_qwen4_parallelism(config, original)
                    expected = tp if ep == 1 else 1
                    self.assertEqual(target.get_ffn_tp_size(), expected)
                    self.assertEqual(target.get_ffn_tp_rank(), rank if ep == 1 else 0)
                    self.assertEqual(draft.get_ffn_tp_size(), expected)
                    self.assertEqual(target.get_attn_tp_size(), tp)
                    self.assertEqual(target.tp_rank, rank)
                    self.assertEqual(target.local_rank, original.local_rank)
                    self.assertEqual(target.world_rank, original.world_rank)
                    self.assertEqual(target.ep_rank, original.ep_rank)
                    self.assertEqual(target.dp_rank, original.dp_rank)
                    self.assertEqual(original.get_ffn_tp_size(), tp)
                    self.assertIsNot(target, original)
                    self.assertIsNot(target, draft)
                    repeated = resolve_qwen4_parallelism(config, target)
                    self.assertEqual(repeated.get_ffn_tp_size(), expected)

    def test_nonrecipe_keeps_common_layout(self):
        pc = topology(2, 2, 1)
        resolved = resolve_qwen4_parallelism(SimpleNamespace(quant_config=None), pc)
        self.assertEqual(resolved.get_ffn_tp_size(), 2)
        self.assertEqual(resolved.get_ffn_tp_rank(), 1)

    def test_unsupported_layout_is_rejected_before_loading(self):
        config = SimpleNamespace(quant_config=recipe())
        for field, value in (("dp_size", 2), ("pp_size", 2), ("world_size", 4)):
            pc = topology(2, 1)
            setattr(pc, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                resolve_qwen4_parallelism(config, pc)
        with self.assertRaises(ValueError):
            resolve_qwen4_parallelism(config, topology(8, 1))

    def test_layout_is_resolved_before_snapshot_and_loader(self):
        from rtp_llm.models.qwen3_next.qwen3_next import Qwen35Moe
        from rtp_llm.models.qwen4_exp.qwen4_exp import Qwen4Exp

        model = Qwen4Exp.__new__(Qwen4Exp)
        model.model_config = SimpleNamespace(quant_config=recipe())
        original = model.parallelism_config = topology(2, 2)
        calls = []

        def record(*args, **kwargs):
            self.assertEqual(model.parallelism_config.get_ffn_tp_size(), 1)
            self.assertEqual(original.get_ffn_tp_size(), 2)
            calls.append(True)

        with (
            patch("rtp_llm.models.qwen4_exp.qwen4_exp.validate_qwen4_runtime", record),
            patch("rtp_llm.models.qwen4_exp.qwen4_exp.freeze_runtime_config", record),
            patch.object(Qwen4Exp, "_validate_checkpoint", record),
            patch.object(Qwen35Moe, "load", record),
        ):
            model.load()
        self.assertEqual(len(calls), 4)


if __name__ == "__main__":
    unittest.main()
