"""CPU tests for K3 quantization semantics and heterogeneous TP layouts."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.model_loader.attn_weight import MlaAttnAtomicWeight, MlaConfig
from rtp_llm.model_loader.linear_attn_weight import LinearAttnAtomicWeight
from rtp_llm.model_loader.per_block_fp8_quant_weight import per_block_cast_to_fp8
from rtp_llm.models.kimi_k3.fp8_weight import KimiK3LoadFp8Weight
from rtp_llm.utils.model_weight import W, identity


class KimiK3Fp8WeightTest(unittest.TestCase):
    def test_tail_scale_and_single_rounding(self):
        for rows in (96, 576, 2112):
            x = torch.zeros(rows, 128, dtype=torch.bfloat16)
            x[-1] = 3.0
            q, s = per_block_cast_to_fp8(x, 128, use_ue8m0=True)
            self.assertEqual(q.shape, x.shape)
            self.assertEqual(s.shape, ((rows + 127) // 128, 1))
            expected_s = torch.tensor(2.0**-7)
            torch.testing.assert_close(s[-1, 0], expected_s, rtol=0, atol=0)
            torch.testing.assert_close(q[-1].float(), x[-1].float() / expected_s)
            self.assertTrue(torch.isfinite(s).all())
            torch.testing.assert_close(
                torch.log2(s), torch.log2(s).round(), rtol=0, atol=0
            )

    def test_kda_shards_keep_replicated_fa_beta_and_tail_scale(self):
        cfg = SimpleNamespace(
            linear_num_key_heads=96,
            linear_num_value_heads=96,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
        )
        source = LinearAttnAtomicWeight(
            W.linear_attn_qkvg_fa_beta_w, [], identity, config=cfg
        )
        wrapper = KimiK3LoadFp8Weight(source, Fp8BlockWiseQuantConfig())
        width = 4 * 12288 + 128 + 96
        weight = torch.arange(width, dtype=torch.float32)[:, None].expand(width, 128)
        scales = torch.arange((width + 127) // 128, dtype=torch.float32)[:, None]
        for tp in (1, 2, 4, 8, 16):
            for rank in range(tp):
                out = wrapper._split(
                    {wrapper.kernel.name: weight, wrapper.scale.name: scales},
                    SimpleNamespace(tp_size=tp, tp_rank=rank),
                )
                w, s = out[wrapper.kernel.name], out[wrapper.scale.name]
                self.assertEqual(w.shape, (4 * 12288 // tp + 224, 128))
                torch.testing.assert_close(w[-224:], weight[-224:], rtol=0, atol=0)
                torch.testing.assert_close(s[-2:], scales[-2:], rtol=0, atol=0)
                for part in range(4):
                    local = 12288 // tp
                    begin = part * 12288 + rank * local
                    torch.testing.assert_close(
                        w[part * local : (part + 1) * local],
                        weight[begin : begin + local],
                        rtol=0,
                        atol=0,
                    )

    def test_bounded_scratch_quantization_is_bitwise_equal(self):
        torch.manual_seed(19)
        for rows in (96, 2112, 3296, 6368):
            raw = torch.randn(rows, 256, dtype=torch.bfloat16)
            raw[: min(rows, 128)] = 0
            for ue8m0 in (False, True):
                expected = per_block_cast_to_fp8(raw, 128, use_ue8m0=ue8m0)
                actual = KimiK3LoadFp8Weight._quantize_matrix(raw, use_ue8m0=ue8m0)
                self.assertTrue(
                    torch.equal(
                        actual[0].view(torch.uint8), expected[0].view(torch.uint8)
                    )
                )
                torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
                self.assertEqual(actual[0].untyped_storage().nbytes(), raw.numel())

    def test_shards_and_replicated_tails_own_only_logical_storage(self):
        # A contiguous row view otherwise retains every other TP rank's rows.
        for name, rows in ((W.mla_q_b_w, 2048), (W.mla_fusedqkrope_w, 2112)):
            source = MlaAttnAtomicWeight(name, [], config=MlaConfig())
            wrapper = KimiK3LoadFp8Weight(source, Fp8BlockWiseQuantConfig())
            raw = torch.ones(rows, 128, dtype=torch.bfloat16)
            q, scales = per_block_cast_to_fp8(raw, 128, use_ue8m0=True)
            for tp in (1, 2, 4, 8, 16):
                for rank in range(tp):
                    out = wrapper._split(
                        {wrapper.kernel.name: q, wrapper.scale.name: scales},
                        SimpleNamespace(tp_size=tp, tp_rank=rank),
                    )
                    for tensor in out.values():
                        self.assertEqual(tensor.storage_offset(), 0)
                        self.assertEqual(
                            tensor.untyped_storage().nbytes(),
                            tensor.numel() * tensor.element_size(),
                        )
                    expected = q if name == W.mla_fusedqkrope_w else q.chunk(tp)[rank]
                    torch.testing.assert_close(
                        out[wrapper.kernel.name].float(),
                        expected.float(),
                        rtol=0,
                        atol=0,
                    )

    def test_kvb_derived_weights_use_final_quantized_values(self):
        cfg = MlaConfig(head_num=32, nope_head_dim=128, v_head_dim=128, kv_lora_rank=128)
        source = MlaAttnAtomicWeight(W.mla_kv_b_w, [], config=cfg)
        wrapper = KimiK3LoadFp8Weight(
            source, Fp8BlockWiseQuantConfig(), derive_mla=True
        )
        wrapper.use_ue8m0 = False
        torch.manual_seed(7)
        raw = torch.randn(8192, 128, dtype=torch.bfloat16)
        q, scale = per_block_cast_to_fp8(raw, 128)
        dense = (q.float() * scale.repeat_interleave(128, 0)).to(torch.bfloat16)
        for tp in (1, 2, 4, 8, 16):
            for rank in range(tp):
                load = SimpleNamespace(tp_size=tp, tp_rank=rank)
                split = wrapper._split(
                    {wrapper.kernel.name: q, wrapper.scale.name: scale}, load
                )
                result = wrapper._postprocess(split, "cpu", load)
                local = dense.chunk(tp)[rank].reshape(32 // tp, 256, 128)
                torch.testing.assert_close(
                    result[W.mla_kc], local[:, :128], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    result[W.mla_vc], local[:, 128:].transpose(1, 2), rtol=0, atol=0
                )

    def test_cached_bf16_rs_workspace_validates_fp8_abi(self):
        import importlib.util
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[2] / "kimi_k3"
        with patch.dict(sys.modules):
            for name, filename in (
                ("rtp_llm.models_py.modules.kimi_k3._collective_gemm", "_collective_gemm.py"),
                ("_k3_fp8_rs_abi_test", "gemm_reduce_scatter.py"),
            ):
                spec = importlib.util.spec_from_file_location(name, root / filename)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
            rs = module
        for tp in (2, 4, 8):
            group = Mock()
            group.size.return_value = tp
            workspace = SimpleNamespace()
            state = rs._GemmReduceScatterState(
                True, group, torch.device("cuda", 0), tp, 32768, 7168,
                SimpleNamespace(_C=SimpleNamespace(bf16_gemm_rs_reduce=Mock())),
                workspace,
            )
            with patch.dict(rs._STATES, {(group, 0): state}, clear=True):
                kwargs = dict(enabled=True, max_m=32768, n=7168)
                self.assertTrue(rs.configure_gemm_reduce_scatter(group, "cuda:0", **kwargs))
                with self.assertRaisesRegex(RuntimeError, "FP8 peer-output RS ABI"):
                    rs.configure_gemm_reduce_scatter(group, "cuda:0", fp8=True, **kwargs)
                workspace._data_offset_bytes = 128
                workspace._mapping_handle = object()
                workspace._launch_lock = object()
                workspace._last_stream = None
                workspace._barrier = Mock()
                self.assertTrue(rs.configure_gemm_reduce_scatter(group, "cuda:0", fp8=True, **kwargs))

    def test_attention_config_enables_mtp_but_not_eagle3_or_global_quantization(self):
        from rtp_llm.config.model_config import ModelConfig
        from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3ModelConfig

        for model_type in ("kimi_k3", "kimi_k3_mtp", "kimi_k3_mla_swa_eagle3"):
            config = KimiK3ModelConfig()
            config.model_type = model_type
            config.data_type = "bf16"
            config.attn_config.use_mla = True
            config.quant_config = None
            with patch.dict(
                os.environ, {"KIMI_K3_ATTENTION_QUANTIZATION": "fp8_per_block", "KIMI_K3_MLA_FP8": "1"}
            ):
                with patch.object(
                    ModelConfig, "init_precision_config", return_value=None
                ):
                    config.init_precision_config(None, None)
            self.assertIsNone(config.quant_config)
            self.assertEqual(config.attn_config.mla_fp8_compute, "eagle3" not in model_type)
            if "eagle3" not in model_type:
                from rtp_llm.ops import KvCacheDataType
                self.assertEqual(config.attn_config.kv_cache_dtype, KvCacheDataType.FP8)
            self.assertEqual(
                config.k3_attention_quant_config is not None, "eagle3" not in model_type
            )

    def test_reference_mode_ignores_workspace_created_by_bf16_draft(self):
        import importlib.util
        import sys
        from pathlib import Path

        # Test the CPU control flow without importing the CUDA factory registry.
        # The real collective functions are loaded; only their I/O is mocked below.
        root = Path(__file__).resolve().parents[2] / "kimi_k3"
        def load(name, filename):
            spec = importlib.util.spec_from_file_location(name, root / filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        with patch.dict(sys.modules):
            load("rtp_llm.models_py.modules.kimi_k3._collective_gemm", "_collective_gemm.py")
            ag = load("_k3_fp8_test_ag", "all_gather_gemm.py")
            rs = load("_k3_fp8_test_rs", "gemm_reduce_scatter.py")
        group = Mock()
        group.size.return_value = 8
        source = Mock(is_cuda=True, device=torch.device("cuda", 0), shape=(4096, 1))
        source.new_empty.side_effect = lambda shape: torch.empty(shape)
        gathered = torch.zeros(32768, 1)
        expected = torch.ones(32768, 1)
        projection = Mock()
        projection.quantize_input.return_value = (gathered, torch.ones(1))
        projection.forward_quantized.return_value = expected
        with (
            patch.dict(os.environ, {"KIMI_K3_FP8_COLLECTIVE_GEMM": "0"}),
            patch.object(ag, "get_process_group", return_value=group),
            patch.dict(
                ag._STATES, {(group, 0): SimpleNamespace(enabled=True)}, clear=True
            ),
            patch.object(ag, "all_gather_into", return_value=gathered) as gather,
            patch.object(ag, "fused_all_gather_fp8_linear") as fused_ag,
            patch.object(rs, "_fp8_remote_gemm_reduce_scatter") as fused_rs,
        ):
            actual = ag.all_gather_gemm(source, [projection], logical_m=32768)[0]
            torch.testing.assert_close(actual, expected)
            gather.assert_called_once()
            fused_ag.assert_not_called()
            self.assertIsNone(
                rs.gemm_reduce_scatter(source, projection, group, pad_rows=False)
            )
            fused_rs.assert_not_called()

    def test_native_moe_cannot_enter_attention_policy(self):
        self.assertNotIn(W.moe_w1, KimiK3LoadFp8Weight.w8a8_weight_list)
        self.assertNotIn(W.ffn_w1, KimiK3LoadFp8Weight.w8a8_weight_list)
        self.assertFalse(KimiK3LoadFp8Weight.support(Fp8BlockWiseQuantConfig(), None))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
