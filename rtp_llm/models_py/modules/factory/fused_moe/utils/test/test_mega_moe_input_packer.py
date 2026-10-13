import os
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
    MegaMoeExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se import (
    MegaMoeSEExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4 import (
    gate as gate_module,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.gate import Gate
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.input_packer import (
    FusedMegaMoEInputPacker,
    TorchMegaMoEInputPacker,
    get_mega_moe_input_packer,
)


@contextmanager
def _env(key: str, value: str):
    old = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def _make_buf(tokens, dim, topk, device):
    return SimpleNamespace(
        x=torch.empty((tokens, dim), dtype=torch.float8_e4m3fn, device=device),
        x_sf=torch.empty((tokens, dim // 128), dtype=torch.int32, device=device),
        topk_idx=torch.empty((tokens, topk), dtype=torch.int64, device=device),
        topk_weights=torch.empty((tokens, topk), dtype=torch.float32, device=device),
    )


class TestMegaMoEInputPacker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("test_mega_moe_input_packer requires CUDA")
        if torch.cuda.get_device_capability()[0] != 10:
            raise RuntimeError("test_mega_moe_input_packer requires SM100")

    def test_dispatch(self):
        old = os.environ.pop("MEGA_MOE_INPUT_PACKER", None)
        try:
            self.assertIsInstance(get_mega_moe_input_packer(), FusedMegaMoEInputPacker)
        finally:
            if old is not None:
                os.environ["MEGA_MOE_INPUT_PACKER"] = old
        with _env("MOE_STRICT_FUSED", "0"), _env("MEGA_MOE_INPUT_PACKER", "torch"):
            self.assertIsInstance(get_mega_moe_input_packer(), TorchMegaMoEInputPacker)
        with _env("MEGA_MOE_INPUT_PACKER", "fused"):
            self.assertIsInstance(get_mega_moe_input_packer(), FusedMegaMoEInputPacker)

    def test_fused_rejects_unsupported_without_fallback(self):
        tokens = 2
        dim = 128
        topk = 8
        x = torch.randn(tokens, dim, dtype=torch.bfloat16)
        weights = torch.randn(tokens, topk, dtype=torch.float32)
        indices = torch.randint(0, 256, (tokens, topk), dtype=torch.int64)
        buf = _make_buf(tokens, dim, topk, "cpu")
        with self.assertRaisesRegex(RuntimeError, "requires CUDA bf16"):
            FusedMegaMoEInputPacker().pack(x, weights, indices, buf, tokens)

    def test_strict_rejects_torch_packer(self):
        with _env("MOE_STRICT_FUSED", "1"), _env("MEGA_MOE_INPUT_PACKER", "torch"):
            with self.assertRaisesRegex(RuntimeError, "forbids"):
                get_mega_moe_input_packer()

    def test_non_strict_allows_torch_packer(self):
        with _env("MOE_STRICT_FUSED", "0"), _env("MEGA_MOE_INPUT_PACKER", "torch"):
            self.assertIsInstance(get_mega_moe_input_packer(), TorchMegaMoEInputPacker)

    def test_legacy_pack_impl_disables_gate_pack_fast_path(self):
        executor = MegaMoeExecutor.__new__(MegaMoeExecutor)
        torch.nn.Module.__init__(executor)
        executor._input_packer = FusedMegaMoEInputPacker()
        executor.cfg = SimpleNamespace(n_activated_experts=8)
        with _env("MEGA_MOE_INPUT_PACKER_IMPL", "optimized"):
            self.assertTrue(executor.supports_gate_pack)
        with _env("MEGA_MOE_INPUT_PACKER_IMPL", "legacy"):
            self.assertFalse(executor.supports_gate_pack)

    def test_gate_pack_capability_checks_topk_bounds(self):
        for executor_cls in (MegaMoeExecutor, MegaMoeSEExecutor):
            executor = executor_cls.__new__(executor_cls)
            torch.nn.Module.__init__(executor)
            executor._input_packer = FusedMegaMoEInputPacker()
            for topk, supported in ((0, False), (1, True), (32, True), (33, False)):
                executor.cfg = SimpleNamespace(n_activated_experts=topk)
                with (
                    self.subTest(executor=executor_cls, topk=topk),
                    _env("MEGA_MOE_INPUT_PACKER_IMPL", "optimized"),
                ):
                    self.assertEqual(executor.supports_gate_pack, supported)

    def test_topk_33_uses_eager_gate_and_ordinary_gpu_pack(self):
        from rtp_llm.utils.model_weight import W

        torch.manual_seed(33)
        tokens, dim, experts, topk = 3, 256, 64, 33
        x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
        weights = {
            W.moe_gate: torch.randn(experts, dim, device="cuda", dtype=torch.bfloat16),
            W.moe_gate_bias: torch.randn(experts, device="cuda"),
            W.moe_gate_tid2eid: torch.stack(
                [torch.randperm(experts, device="cuda")[:topk] for _ in range(tokens)]
            ),
        }
        input_ids = torch.arange(tokens, device="cuda")
        for hash_routing in (False, True):
            with self.subTest(hash_routing=hash_routing):
                gate = Gate(
                    0,
                    dim,
                    experts,
                    topk,
                    n_hash_layers=int(hash_routing),
                    vocab_size=tokens,
                    layer_weights=weights,
                )
                gate.fuse_hash_gate = True
                with _env("MOE_GATE_FUSED", "0"):
                    expected_weights, expected_ids = gate(x, input_ids)
                with (
                    _env("MOE_GATE_FUSED", "1"),
                    _env("MOE_GATE_FP32", "0"),
                    patch.object(gate_module, "fused_sqrtsoftplus_gate") as fused,
                    patch.object(gate_module, "fused_sqrtsoftplus_hash_gate") as hashed,
                ):
                    self.assertIsNone(gate.prepare_gate_payload(x, input_ids))
                    route_weights, ids = gate(x, input_ids)
                fused.assert_not_called()
                hashed.assert_not_called()
                torch.testing.assert_close(
                    route_weights, expected_weights, rtol=0, atol=0
                )
                self.assertTrue(torch.equal(ids, expected_ids))
                buf = _make_buf(tokens, dim, topk, "cuda")
                FusedMegaMoEInputPacker().pack(x, route_weights, ids, buf, tokens)
                self.assertTrue(torch.equal(buf.topk_idx, expected_ids))
                self.assertTrue(torch.equal(buf.topk_weights, expected_weights))
                self.assertTrue(torch.isfinite(buf.x.float()).all())

    def test_fused_matches_torch_buffer_bits(self):
        torch.manual_seed(3)
        for tokens in (1, 17, 128):
            with self.subTest(tokens=tokens):
                dim = 256
                topk = 8
                x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
                weights = torch.randn(tokens, topk, device="cuda", dtype=torch.float32)
                indices = torch.randint(
                    0, 256, (tokens, topk), device="cuda", dtype=torch.int64
                )
                ref = _make_buf(tokens, dim, topk, "cuda")
                got = _make_buf(tokens, dim, topk, "cuda")
                with _env("MOE_STRICT_FUSED", "0"):
                    TorchMegaMoEInputPacker().pack(x, weights, indices, ref, tokens)
                FusedMegaMoEInputPacker().pack(x, weights, indices, got, tokens)
                self.assertTrue(
                    torch.equal(
                        ref.x.view(torch.uint8).cpu(), got.x.view(torch.uint8).cpu()
                    )
                )
                self.assertTrue(torch.equal(ref.x_sf.cpu(), got.x_sf.cpu()))
                self.assertTrue(torch.equal(ref.topk_idx.cpu(), got.topk_idx.cpu()))
                self.assertTrue(
                    torch.equal(ref.topk_weights.cpu(), got.topk_weights.cpu())
                )

    def test_zero_tokens_noop(self):
        buf = _make_buf(1, 128, 8, "cuda")
        FusedMegaMoEInputPacker().pack(
            torch.empty((0, 128), device="cuda", dtype=torch.bfloat16),
            torch.empty((0, 8), device="cuda", dtype=torch.float32),
            torch.empty((0, 8), device="cuda", dtype=torch.int64),
            buf,
            0,
        )

    def test_nonfinite_activations_are_zeroed(self):
        tokens, dim, topk = 3, 256, 8
        x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
        x[0, 0] = float("nan")
        x[1, 1] = float("inf")
        x[2, 2] = -float("inf")
        weights = torch.rand(tokens, topk, device="cuda", dtype=torch.float32)
        indices = torch.randint(
            0, 256, (tokens, topk), device="cuda", dtype=torch.int64
        )
        ref = _make_buf(tokens, dim, topk, "cuda")
        got = _make_buf(tokens, dim, topk, "cuda")

        with _env("MOE_STRICT_FUSED", "0"):
            TorchMegaMoEInputPacker().pack(x, weights, indices, ref, tokens)
        FusedMegaMoEInputPacker().pack(x, weights, indices, got, tokens)
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(ref.x.view(torch.uint8), got.x.view(torch.uint8)))
        self.assertTrue(torch.equal(ref.x_sf, got.x_sf))
        self.assertTrue(torch.isfinite(got.x.float()).all().item())

    def test_fused_rejects_noncontiguous_input_without_copying(self):
        tokens, dim, topk = 2, 128, 8
        x = torch.randn(dim, tokens, device="cuda", dtype=torch.bfloat16).transpose(
            0, 1
        )
        self.assertNotEqual(x.stride(-1), 1)
        weights = torch.randn(tokens, topk, device="cuda", dtype=torch.float32)
        indices = torch.randint(
            0, 256, (tokens, topk), device="cuda", dtype=torch.int64
        )
        buf = _make_buf(tokens, dim, topk, "cuda")
        with self.assertRaisesRegex(ValueError, "x must be contiguous"):
            FusedMegaMoEInputPacker().pack(x, weights, indices, buf, tokens)

    def test_fused_rejects_noncontiguous_output_without_copying(self):
        tokens, dim, topk = 2, 128, 8
        x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
        weights = torch.randn(tokens, topk, device="cuda", dtype=torch.float32)
        indices = torch.randint(
            0, 256, (tokens, topk), device="cuda", dtype=torch.int64
        )
        buf = _make_buf(tokens, dim, topk, "cuda")
        buf.x = torch.empty(
            dim, tokens, device="cuda", dtype=torch.float8_e4m3fn
        ).transpose(0, 1)
        self.assertNotEqual(buf.x.stride(-1), 1)
        with self.assertRaisesRegex(ValueError, "out_fp8 must be contiguous"):
            FusedMegaMoEInputPacker().pack(x, weights, indices, buf, tokens)


if __name__ == "__main__":
    unittest.main()
