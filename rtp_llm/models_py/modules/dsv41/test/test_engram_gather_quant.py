"""Shared Engram gather/quant boundaries and Graph ownership on real CUDA."""

import gc
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.model_loader.host_shared_cuda import SharedEngramLookup
from rtp_llm.model_loader.host_shared_weights import (
    HostSharedWeightStore,
    SharedWeightSlice,
)
from rtp_llm.models_py.modules.dsv41._engram_lookup_triton import (
    engram_gather_quantize_kernel,
)
from rtp_llm.models_py.modules.dsv41.engram import Engram
from rtp_llm.models_py.modules.dsv41.linear import V41Block32Linear, quantize_block32


def quantize_reference(values):
    groups = values.cpu().float().reshape(-1, 32)
    maxima = groups.abs().amax(-1).clamp_min(1e-4) * (1.0 / 448.0)
    scales = []
    for maximum in maxima.tolist():
        fraction, exponent = math.frexp(maximum)
        scales.append(math.ldexp(1.0, exponent - (fraction == 0.5)))
    scales = torch.tensor(scales, dtype=torch.float32).reshape(*values.shape[:-1], 8)
    encoded = (groups / scales.reshape(-1, 1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return encoded.reshape(values.shape), scales


def lookup_reference(shared, layer, indices, valid=None):
    ids = indices.cpu().reshape(-1).tolist()
    active = [True] * len(ids) if valid is None else valid.cpu().reshape(-1).tolist()
    prefix = f"layers.{layer}.engram.embed."
    with shared.view(prefix + "weight") as weight, shared.view(
        prefix + "scale"
    ) as scale:
        data = bytearray(
            b"".join(
                weight[(row if live else 0) * 256 : ((row if live else 0) + 1) * 256]
                for row, live in zip(ids, active)
            )
        )
        codes = bytearray(
            b"".join(
                scale[(row if live else 0) * 8 : ((row if live else 0) + 1) * 8]
                for row, live in zip(ids, active)
            )
        )
    if not ids:
        return torch.empty((*indices.shape, 256), dtype=torch.bfloat16)
    raw = (
        torch.frombuffer(data, dtype=torch.uint8)
        .view(torch.float8_e4m3fn)
        .double()
        .reshape(-1, 8, 32)
    )
    factors = (
        torch.frombuffer(codes, dtype=torch.uint8)
        .view(torch.float8_e8m0fnu)
        .double()
        .reshape(-1, 8, 1)
    )
    decoded = (raw * factors).float().bfloat16().reshape(-1, 256)
    decoded[~torch.tensor(active, dtype=torch.bool)] = 0
    return decoded.reshape(*indices.shape, 256)


def fixture_slices(root):
    slices = []
    rows = 512
    for layer in (1, 14):
        codes = (torch.arange(rows * 8).reshape(rows, 8) % 255).to(torch.uint8)
        raw = ((torch.arange(rows * 256).reshape(rows, 8, 32) + layer) % 256).to(
            torch.uint8
        )
        raw[(raw == 127) | (raw == 255)] = 0
        # Large UE8M0 factors still exercise the BF16 finite upper boundary.
        raw = torch.where(codes[..., None] > 246, raw & 0xB7, raw)
        raw[-1, 0, 0], codes[-1, 1], raw[-1, 2, 0], codes[-1, 2] = 127, 255, 126, 254
        for kind, data, dim, dtype in (
            ("weight", raw, 256, "F8_E4M3"),
            ("scale", codes, 8, "F8_E8M0"),
        ):
            name = f"layers.{layer}.engram.embed.{kind}"
            path = root / name
            path.write_bytes(data.numpy().tobytes())
            slices.append(
                SharedWeightSlice(name, path, 0, data.numel(), (rows, dim), dtype)
            )
    return slices


class EngramGatherQuantTest(unittest.TestCase):
    def setUp(self):
        self.assertNotEqual(os.getuid(), 0)
        shared_root = os.environ.get("DSV41_TEST_SHARED_ROOT", "/dev/shm")
        if shared_root:
            os.makedirs(shared_root, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=shared_root)
        root = Path(self.temporary.name)
        self.store = HostSharedWeightStore(root / "shared")
        self.shared = self.store.open_or_publish("a" * 40, fixture_slices(root))
        self.lookup = SharedEngramLookup(self.shared, device=0)
        self.lookup.warmup(quantized=True)

    def tearDown(self):
        self.lookup.close()
        self.temporary.cleanup()

    def check(self, layer, ids, valid=None, output=None):
        expected = lookup_reference(self.shared, layer, ids, valid)
        expected_encoded, expected_scales = quantize_reference(expected)
        output, encoded, scales = self.lookup.lookup_quantized(
            layer, ids, valid_mask=valid, out=output
        )
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(
            encoded.view(torch.uint8).cpu(),
            expected_encoded.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(scales.cpu(), expected_scales, rtol=0, atol=0)
        baseline = self.lookup.lookup(layer, ids, valid_mask=valid)
        old_encoded, old_scales = quantize_block32(baseline.reshape(-1, 256))
        self.assertTrue(
            torch.equal(
                encoded.view(torch.uint8).reshape(-1, 256),
                old_encoded.view(torch.uint8),
            )
        )
        torch.testing.assert_close(scales.reshape(-1, 8), old_scales, rtol=0, atol=0)
        return output, encoded, scales

    def test_finite_codes_and_within_group_rounding(self):
        for layer in (1, 14):
            self.check(layer, torch.arange(511, dtype=torch.int64, device="cuda"))

    def test_empty_padding_and_leading_dimensions(self):
        for shape in ((0, 24), (1, 24), (2, 3, 24)):
            ids = (
                torch.arange(
                    math.prod(shape), device="cuda", dtype=torch.int64
                ).reshape(shape)
                % 511
            )
            valid = ids % 3 != 0
            ids = ids.masked_fill(~valid, -(1 << 63))
            self.check(1, ids, valid)
            ids.fill_((1 << 63) - 1)
            self.check(14, ids, torch.zeros_like(valid))

    def test_alias_and_layout_rejections(self):
        ids = torch.arange(24, device="cuda", dtype=torch.int64)
        valid = torch.ones_like(ids, dtype=torch.bool)
        out = torch.empty(24, 256, device="cuda", dtype=torch.bfloat16)
        for kwargs in (
            dict(indices=ids.int()),
            dict(indices=ids, valid_mask=valid.int()),
            dict(indices=ids, out=out.t()),
            dict(indices=ids.repeat(2)[::2]),
            dict(indices=ids, valid_mask=valid.repeat(2)[::2]),
            dict(indices=out.view(torch.int64).flatten()[:24], out=out),
            dict(indices=ids, valid_mask=out.view(torch.bool).flatten()[:24], out=out),
        ):
            with self.subTest(kwargs=list(kwargs)), self.assertRaises(ValueError):
                self.lookup.lookup_quantized(1, **kwargs)
        with self.assertRaises(KeyError):
            self.lookup.lookup_quantized(2, ids)

    def test_nonfinite_status_and_masking(self):
        ids = torch.tensor([511, 511], device="cuda", dtype=torch.int64)
        valid = torch.tensor([True, False], device="cuda")
        out = torch.empty(2, 256, device="cuda", dtype=torch.bfloat16)
        encoded = torch.empty_like(out, dtype=torch.float8_e4m3fn)
        scales = torch.empty(2, 8, device="cuda")
        finite = torch.empty_like(scales, dtype=torch.bool)
        prefix = "layers.1.engram.embed."
        engram_gather_quantize_kernel[(2,)](
            self.lookup._buffers[prefix + "weight"],
            self.lookup._buffers[prefix + "scale"],
            ids,
            valid,
            out,
            encoded,
            scales,
            finite,
            ROWS=512,
            HAS_VALID=True,
        )
        reference = lookup_reference(self.shared, 1, ids, valid)
        torch.testing.assert_close(out.cpu(), reference, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(
            finite.cpu(), torch.isfinite(reference.reshape(2, 8, 32)).all(-1)
        )
        self.assertFalse(bool(finite[0, :3].any()))
        self.assertTrue(bool(finite[1].all()))
        self.check(1, ids, torch.zeros_like(valid))

    def test_graph_replay_and_owner_lifetime(self):
        records = []
        for layer in (1, 14):
            ids = torch.arange(48, device="cuda", dtype=torch.int64).reshape(2, 24)
            valid = torch.ones_like(ids, dtype=torch.bool)
            out = torch.empty(2, 24, 256, device="cuda", dtype=torch.bfloat16)
            graph = self.lookup.graph()
            torch.cuda.synchronize()
            with graph.capture(torch.cuda.Stream()):
                result = self.lookup.lookup_quantized(
                    layer, ids, valid_mask=valid, out=out
                )
            pointers = tuple(value.data_ptr() for value in result)
            records.append((layer, ids, valid, graph, result, pointers))
        gc.collect()
        for step in (17, 0, 131, 47, 0):
            for layer, ids, valid, graph, result, pointers in records:
                ids.copy_(
                    (torch.arange(48, device="cuda").reshape(2, 24) * 7 + step + layer)
                    % 511
                )
                valid.copy_((ids % 3 != 1) & (step != 0))
                ids.masked_fill_(~valid, -1)
                graph.replay()
                torch.cuda.synchronize()
                expected = lookup_reference(self.shared, layer, ids, valid)
                expected_q, expected_s = quantize_reference(expected)
                torch.testing.assert_close(result[0].cpu(), expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    result[1].view(torch.uint8).cpu(),
                    expected_q.view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(result[2].cpu(), expected_s, rtol=0, atol=0)
                self.assertEqual(pointers, tuple(value.data_ptr() for value in result))
                self.assertFalse(
                    self.store.remove_if_unused(self.shared.manifest["identity"])
                )
        self.lookup.close()
        for _, _, _, graph, _, _ in records:
            with self.assertRaisesRegex(RuntimeError, "closed"):
                graph.replay()
        self.assertTrue(self.store.remove_if_unused(self.shared.manifest["identity"]))

    def test_capture_requires_binding_and_fixed_bf16_output(self):
        ids = torch.arange(24, device="cuda", dtype=torch.int64)
        out = torch.empty(24, 256, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with self.assertRaisesRegex(RuntimeError, "binding"):
            with torch.cuda.graph(graph):
                self.lookup.lookup_quantized(1, ids, out=out)
        graph.reset()
        binding = self.lookup.graph()
        with self.assertRaisesRegex(RuntimeError, "fixed output"):
            with binding.capture(torch.cuda.Stream()):
                self.lookup.lookup_quantized(1, ids)
        binding.close()

    def test_engram_dispatch_preserves_projection_and_lookup_output(self):
        torch.manual_seed(607)
        weight = (torch.randn(640, 6144, device="cuda") * 0.125).to(torch.float8_e4m3fn)
        scales = torch.full((20, 192), 121, device="cuda", dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        projection = V41Block32Linear(weight, scales)
        gate = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
        module = Engram(1, self.lookup, projection, gate, gate)
        for count in (0, 1, 7):
            hidden = torch.randn(count, 4, 128, device="cuda", dtype=torch.bfloat16)
            # Restrict this model test to moderate finite rows; the codec test covers all codes.
            ids = (
                torch.arange(count * 24, device="cuda", dtype=torch.int64).reshape(
                    count, 24
                )
                % 8
            ) + 12
            valid = torch.arange(count, device="cuda") % 3 != 1
            out = torch.empty(count, 24, 256, device="cuda", dtype=torch.bfloat16)
            with patch.object(module, "_fused_lookup_supported", lambda: False):
                expected = module(hidden, ids, valid, lookup_output=out)
                expected_out = out.clone()
            self.assertTrue(module._fused_lookup_supported())
            actual = module(hidden, ids, valid, lookup_output=out)
            torch.testing.assert_close(out, expected_out, rtol=0, atol=0)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
