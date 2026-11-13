"""Mixed precision contracts through the production loader and linear."""

import os
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models_py.modules.dsv41.ppu.linear import PPUPTPCLinear, quantize_ptpc_activation
from rtp_llm.models_py.modules.dsv41.ppu.model_weights import PPUPTPCWeight, _StorageTensorSource
from rtp_llm.models_py.modules.dsv41.ppu.moe_precision import moe_precision
from rtp_llm.models_py.modules.dsv41.ppu.quant_conversion import convert_dense_ptpc, convert_engram_wkv
from rtp_llm.models_py.modules.dsv41.ppu.quant_mode import MXFP4_FP8_PTPC, QUANT_MODE_ENV, quant_mode
from rtp_llm.utils.model_weight import CkptWeightInfo, concat_0


class Source(TensorSource):
    def __init__(self, tensors):
        self.tensors, self.calls = tensors, []

    def load_tensor(self, name, data_type=None):
        self.calls.append(name)
        value = self.tensors[name]
        return [value if data_type is None else value.to(data_type)]

    def has_tensor(self, name):
        return name in self.tensors


class PTPCContracts(unittest.TestCase):
    def test_grouped_rebind_clears_provider_selection(self):
        weight = torch.ones(64, 128).to(torch.float8_e4m3fn)
        op = PPUPTPCLinear(weight, torch.ones(64, 1))
        op.prepare_grouped(2, 32)
        op._group_gemm = object()
        op.prepare_grouped(4, 16)
        self.assertIsNone(op._group_gemm)
        self.assertEqual(len(op._groups), 4)

    def test_backend_u16_receipt_preserves_packed_bytes(self):
        from rtp_llm.models_py.modules.dsv41.ppu.quant_conversion import audit_loaded_tensor

        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"RTP_V41_PPU_WEIGHT_AUDIT_DIR": temporary},
        ):
            packed = torch.tensor([0x0123, 0xFEDC], dtype=torch.uint16)
            audit_loaded_tensor("packed-expert", packed, rank=3, layout=True)
            row = json.loads((Path(temporary) / "layouts-rank00003.jsonl").read_text())
            self.assertEqual(row["dtype"], "U16")
            self.assertEqual(row["sha256"], hashlib.sha256(b"\x23\x01\xdc\xfe").hexdigest())
            self.assertEqual((row["shape"], row["stride"], row["storage_offset"]), ([2], [1], 0))

    def test_commit_only_runtime_binds_attention_without_mhc(self):
        from rtp_llm.models_py.modules.dsv41.ppu.model_runtime import SharedNativeRuntime

        class Model:
            pass

        model = Model()
        layer = SimpleNamespace(attn=SimpleNamespace())
        model.v4 = SimpleNamespace(embed=None, commit_only=True, layers=[layer])
        model.main_proj = SimpleNamespace(weight=torch.empty(1))
        model.shared_weight_views = SimpleNamespace(
            geometry=SimpleNamespace(engram_enabled=lambda: False), config={},
        )
        runtime = SharedNativeRuntime(model)
        self.assertIs(layer.attn.runtime, runtime)
        self.assertIs(model.v4._ppu_runtime, runtime)
        self.assertFalse(hasattr(layer, "attn_hc"))

    def test_ptpc_norm_uses_shared_provider_across_runtime_modes(self):
        from rtp_llm.models_py.modules.dsv41.ppu.layers import PPUSharedRMSNorm

        with patch.dict(os.environ, {QUANT_MODE_ENV: MXFP4_FP8_PTPC}):
            op = PPUSharedRMSNorm(torch.ones(1280, dtype=torch.bfloat16))
        x = torch.randn(4, 1280, dtype=torch.bfloat16)
        op.runtime = SimpleNamespace(current=None)
        with patch(
            "rtp_llm.models_py.modules.dsv41.ppu.layers.rms_norm", return_value=x
        ) as native:
            self.assertIs(op(x), x)
            op.runtime.current = SimpleNamespace(is_graph=True, verify=False)
            with patch.dict(os.environ, {QUANT_MODE_ENV: "legacy"}):
                self.assertIs(op(x), x)
        self.assertEqual(native.call_count, 2)
        native.assert_called_with(x, op.weight, op.variance_epsilon)

    def test_layout_check_rejects_two_missing_inventories(self):
        from rtp_llm.tools.v41.validate_ptpc import check_layouts

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "layout comparison failed"):
                check_layouts(root / "source", root / "target", root / "report")

    def test_selection_and_conflicts(self):
        self.assertEqual(quant_mode({}), "legacy")
        env = {QUANT_MODE_ENV: MXFP4_FP8_PTPC}
        self.assertEqual(moe_precision(env), "w4a4")
        for extra in ({"RTP_V41_BACKEND": "cuda"}, {"RTP_V41_PPU_MOE_PRECISION": "w4a8"}):
            with self.assertRaises(ValueError):
                quant_mode({**env, **extra})
        with self.assertRaises(ValueError):
            quant_mode({QUANT_MODE_ENV: "unknown"})

    def test_exact_scale_and_chunk_boundaries(self):
        w = torch.full((65, 96), 1.125).to(torch.float8_e4m3fn)
        s = torch.full((3, 3), 123, dtype=torch.uint8)
        a, scale = convert_dense_ptpc(w, s, chunk_rows=17)
        b, other = convert_dense_ptpc(w, s)
        self.assertTrue(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))
        self.assertTrue(torch.equal(scale.view(torch.int32), other.view(torch.int32)))
        # Independently inspected reference: amax=.0703125 -> 4a922439 LE.
        self.assertEqual(scale[0].view(torch.int32).item(), 0x3924924A)
        self.assertTrue(torch.equal(a.float(), torch.full((65, 96), 448.0)))

    def test_quant_mode_separates_native_cache_identity(self):
        from rtp_llm.models_py.modules.dsv41.ppu.native_protocol import native_cache_key_hash_seed

        with patch.dict(os.environ, {QUANT_MODE_ENV: MXFP4_FP8_PTPC}):
            legacy = native_cache_key_hash_seed({"_ppu_quant_mode": "legacy"})
            ptpc = native_cache_key_hash_seed({"_ppu_quant_mode": MXFP4_FP8_PTPC})
        self.assertNotEqual(legacy, ptpc)
        with patch.dict(os.environ, {QUANT_MODE_ENV: "legacy"}):
            self.assertEqual(ptpc, native_cache_key_hash_seed({"_ppu_quant_mode": MXFP4_FP8_PTPC}))

    def test_cpu_metadata_parser_restores_ppu_backend(self):
        from rtp_llm.tools.v41.validate_ptpc import metadata_config

        class Model:
            @classmethod
            def _create_config(cls, checkpoint):
                return os.environ["RTP_V41_BACKEND"], checkpoint

        with patch.dict(os.environ, {"RTP_V41_BACKEND": "ppu"}):
            self.assertEqual(metadata_config(Model, "/weights"), ("reference", "/weights"))
            self.assertEqual(os.environ["RTP_V41_BACKEND"], "ppu")

    def test_actual_loader_pairs_and_gate_up_order(self):
        raw = {}
        for name, value, exponent in (("gate", 1.0, 127), ("up", -1.5, 126)):
            raw[name + ".weight"] = torch.full((32, 64), value).to(torch.float8_e4m3fn)
            raw[name + ".scale"] = torch.full((1, 2), exponent, dtype=torch.uint8)
        source = Source(raw)
        descriptor = PPUPTPCWeight(AtomicWeight(
            "test.weight", [CkptWeightInfo(n + ".weight", lambda ts: ts[0]) for n in ("gate", "up")], concat_0
        ), "test.scale")
        config = SimpleNamespace(merge_lora=False, tp_size=1)
        result = descriptor.load(source, 0, "cpu", config)
        for i, n in enumerate(("gate", "up")):
            q, s = convert_dense_ptpc(raw[n + ".weight"], raw[n + ".scale"])
            self.assertTrue(torch.equal(result["test.weight"][i * 32:(i + 1) * 32].view(torch.uint8), q.view(torch.uint8)))
            self.assertTrue(torch.equal(result["test.scale"][i * 32:(i + 1) * 32], s))
        self.assertEqual(len(source.calls), 4)
        self.assertEqual(len(descriptor.get_tensor_names(0, config)), 4)
        again = descriptor.load(Source(raw), 0, "cpu", config)
        self.assertTrue(torch.equal(again["test.weight"].view(torch.uint8), result["test.weight"].view(torch.uint8)))

    def test_passthrough_and_expert_bytes(self):
        q = torch.tensor([[0.0, -0.0, 1.0, -448.0]]).to(torch.float8_e4m3fn)
        s = torch.tensor([[0.003]], dtype=torch.float32)
        a, b = convert_dense_ptpc(q, s)
        self.assertEqual(a.data_ptr(), q.data_ptr())
        self.assertEqual(b.data_ptr(), s.data_ptr())
        raw = torch.arange(256).to(torch.uint8).view(torch.int8)
        actual = _StorageTensorSource(Source({"expert": raw})).load_tensor("expert", torch.uint8)[0]
        self.assertTrue(torch.equal(actual, raw.view(torch.uint8)))
        with self.assertRaises(ValueError):
            convert_dense_ptpc(q, torch.ones(1, 2))

    def test_engram_dequant_and_bf16_passthrough(self):
        w = torch.full((33, 64), 1.5).to(torch.float8_e4m3fn)
        s = torch.tensor([[126, 128], [127, 129]], dtype=torch.uint8)
        expected = w.float() * torch.tensor([[0.5] * 32 + [2.0] * 32] * 32 + [[1.0] * 32 + [4.0] * 32])
        actual = convert_engram_wkv(w, s, chunk_rows=17)
        self.assertTrue(torch.equal(actual, expected.to(torch.bfloat16)))
        self.assertIs(convert_engram_wkv(actual, None), actual)

    def test_grouped_linear_and_zero_rows(self):
        w = torch.arange(8 * 64).reshape(8, 64).float().remainder(7).to(torch.float8_e4m3fn)
        s = torch.arange(1, 9).float()[:, None] * 0.001
        op = PPUPTPCLinear(w, s)
        op.prepare_grouped(2, 4)
        x = torch.randn(3, 2, 64, dtype=torch.bfloat16)
        actual = op.forward_grouped(x)
        for i in range(2):
            q, a = quantize_ptpc_activation(x[:, i])
            expected = ((q.float() @ w[i * 4:(i + 1) * 4].float().T) * a * s[i * 4:(i + 1) * 4].T).to(torch.bfloat16)
            self.assertTrue(torch.equal(actual[:, i], expected))
        self.assertEqual(op(torch.empty(0, 64, dtype=torch.bfloat16)).shape, (0, 8))
        q, a = quantize_ptpc_activation(torch.zeros(2, 64, dtype=torch.bfloat16))
        self.assertTrue(torch.isfinite(a).all())
        self.assertTrue((q.float() == 0).all())


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_PTPC_DEVICE") == "1", "explicit PPU execution only")
class PTPCDevice(unittest.TestCase):
    @torch.no_grad()
    def test_grouped_native_and_changed_input_graph_are_exact(self):
        torch.manual_seed(4119)
        groups, width, rank = 8, 4096, 1024
        weight = torch.randn(groups * rank, width, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scales = torch.full((groups * rank, 1), 0.01, device="cuda", dtype=torch.float32)
        op = PPUPTPCLinear(weight, scales)
        op.prepare_grouped(groups, rank)

        def original(value, owner=op):
            outputs = []
            for index, part in enumerate(owner._groups):
                values = value[:, index].contiguous().float()
                maximum = values.abs().amax(-1, keepdim=True).clamp_min(1e-10)
                scale = maximum * 0.0022321429569274187
                quant = (values * (448.0 / maximum)).clamp(-448, 448).to(torch.float8_e4m3fn)
                out = torch.empty((value.shape[0], part.weight.shape[0]), dtype=torch.bfloat16, device=value.device)
                part._gemm((quant, scale), (part.weight, part.scale), out)
                outputs.append(out)
            return torch.stack(outputs, 1)

        def assert_grouped_exact(actual, expected, label):
            actual_bits, expected_bits = actual.view(torch.int16), expected.view(torch.int16)
            if not torch.equal(actual_bits, expected_bits):
                position = tuple((actual_bits != expected_bits).nonzero()[0].tolist())
                self.fail(
                    f"{label} at {position}: actual=0x{int(actual_bits[position].item()) & 0xffff:04x}, "
                    f"expected=0x{int(expected_bits[position].item()) & 0xffff:04x}"
                )

        for rows in (1, 3, 4, 8, 16, 24, 32, 72, 144, 288, 2048, 4096, 8192):
            value = torch.randn(rows, groups, width, device="cuda", dtype=torch.bfloat16)
            assert_grouped_exact(op.forward_grouped(value), original(value), f"wo_a M{rows}")
        for rows in (24, 32, 72):
            value = torch.randn(rows, groups, width, device="cuda", dtype=torch.bfloat16)
            op.forward_grouped(value)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = op.forward_grouped(value)
            for magnitude in (0.0, 0.125, 2.0):
                value.copy_(torch.randn_like(value) * magnitude)
                graph.replay()
                assert_grouped_exact(output, original(value), f"wo_a Graph M{rows} magnitude{magnitude}")
        self.assertEqual(op.forward_grouped(value[:0]).shape, (0, groups, rank))
        fallback = PPUPTPCLinear(
            torch.randn(128, 128, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn),
            torch.full((128, 1), 0.01, device="cuda", dtype=torch.float32),
        )
        fallback.prepare_grouped(2, 64)
        self.assertIsNone(fallback._group_gemm)
        small = torch.randn(4, 2, 128, device="cuda", dtype=torch.bfloat16)
        fallback.forward_grouped(small)
        fallback_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(fallback_graph):
            fallback_output = fallback.forward_grouped(small)
        for magnitude in (0.0, 0.125, 2.0):
            small.copy_(torch.randn_like(small) * magnitude)
            fallback_graph.replay()
            assert_grouped_exact(fallback_output, original(small, fallback), "wo_a fallback Graph")

    def test_native_gemm_and_graph_changed_inputs(self):
        from rtp_llm.models_py.modules.dsv41.ppu.layers import PPUSharedRMSNorm

        norm = PPUSharedRMSNorm(torch.randn(5120, device="cuda", dtype=torch.bfloat16))
        norm_input = torch.randn(4, 5120, device="cuda", dtype=torch.bfloat16)
        norm(norm_input)
        norm_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(norm_graph):
            normalized = norm(norm_input)
        norm_graph.replay()
        self.assertTrue(torch.equal(normalized, norm(norm_input)))
        q, s = convert_dense_ptpc(
            torch.randn(64, 128).to(torch.float8_e4m3fn), torch.full((2, 4), 124, dtype=torch.uint8)
        )
        op = PPUPTPCLinear(q.cuda(), s.cuda())
        x = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
        op(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = op(x)
        for factor in (1.0, 0.125, 2.0):
            x.copy_(torch.randn_like(x) * factor)
            graph.replay()
            self.assertTrue(torch.equal(actual, op(x)))
            aq, a = quantize_ptpc_activation(x.cpu())
            ref = ((aq.float() @ q.float().T) * a * s.T).to(torch.bfloat16)
            error = (actual.cpu().float() - ref.float()).square().sum()
            energy = ref.float().square().sum().clamp_min(1e-20)
            self.assertLess((error / energy).item(), 5e-5)

    @torch.no_grad()
    def test_production_norm_linear_changed_input_graph_is_exact(self):
        from rtp_llm.models_py.modules.dsv41.ppu.layers import PPUSharedRMSNorm

        for width in (1280, 5120):
            gamma = torch.randn(width, device="cuda", dtype=torch.bfloat16)
            norm = PPUSharedRMSNorm(gamma)
            self.assertTrue(norm._ptpc)
            weight = torch.randn(1280, width, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            scales = torch.full((1280, 1), 0.01, device="cuda", dtype=torch.float32)
            op = PPUPTPCLinear(weight, scales)

            def original(value):
                values = value.float()
                inverse = torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
                normalized = (values * inverse * gamma.float()).to(torch.bfloat16).float()
                maximum = normalized.abs().amax(-1, keepdim=True).clamp_min(1e-10)
                scale = maximum * 0.0022321429569274187
                payload = (normalized * (448.0 / maximum)).clamp(-448, 448).to(torch.float8_e4m3fn)
                out = torch.empty((value.shape[0], 1280), device="cuda", dtype=torch.bfloat16)
                op._gemm((payload, scale), (op.weight, op.scale), out)
                return out

            for rows in (24, 32):
                value = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
                op(norm(value))
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = op(norm(value))
                for magnitude in (0.0, 0.125, 2.0):
                    value.copy_(torch.randn_like(value) * magnitude)
                    gamma.copy_(torch.randn_like(gamma))
                    graph.replay()
                    expected = original(value)
                    actual_bits, expected_bits = output.view(torch.int16), expected.view(torch.int16)
                    if not torch.equal(actual_bits, expected_bits):
                        position = tuple((actual_bits != expected_bits).nonzero()[0].tolist())
                        self.fail(
                            f"norm+linear Graph M{rows}/K{width} at {position}: "
                            f"actual=0x{int(actual_bits[position].item()) & 0xffff:04x}, "
                            f"expected=0x{int(expected_bits[position].item()) & 0xffff:04x}"
                        )


if __name__ == "__main__":
    unittest.main()
