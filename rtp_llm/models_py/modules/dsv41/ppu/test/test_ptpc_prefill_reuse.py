"""Actual Prefill projection consumers, ending before attention or collectives.

Real norms and resident GEMMs execute through SharedPPUAttention._body and
_select. Transport, RoPE, cache publication and the scoring/attention boundary
are test fixtures. This component does not run a model or validate Decode.
"""

from contextlib import ExitStack, contextmanager
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import attention_layer
from rtp_llm.models_py.modules.dsv41.ppu.layers import PPUSharedRMSNorm
from rtp_llm.models_py.modules.dsv41.ppu.linear import (
    PPUPTPCLinear,
    PPUResidentBF16Linear,
)


ENV = {
    "RTP_V41_PPU_QUANT_MODE": "mxfp4_fp8_ptpc",
    "RTP_V41_PPU_PTPC_FUSION": "1",
    "RTP_V41_PPU_CP8_GROUPED_QO_ROPE": "0",
    "RTP_V41_PPU_CP8_INDEXER_REQUEST_SLICES": "1",
    "RTP_V41_PPU_OPTIMIZED_DIAGNOSTICS": "0",
}


class _ProjectionBoundary(Exception):
    pass


def assert_bits(test, actual, expected):
    test.assertEqual(actual.dtype, expected.dtype)
    test.assertEqual(actual.shape, expected.shape)
    left = actual.contiguous().view(torch.uint8)
    right = expected.contiguous().view(torch.uint8)
    if not torch.equal(left, right):
        position = tuple((left != right).nonzero()[0].tolist())
        test.fail(
            f"byte mismatch at {position}: "
            f"actual={int(left[position])}, expected={int(right[position])}"
        )


def original_linear(op, value):
    if not isinstance(op, PPUPTPCLinear):
        return torch.nn.functional.linear(value, op.weight, op.bias)
    values = value.float()
    maximum = values.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    scale = maximum * 0.0022321429569274187
    payload = (values * (448.0 / maximum)).clamp(-448, 448).to(torch.float8_e4m3fn)
    output = value.new_empty((value.shape[0], op.weight.shape[0]))
    if value.is_cuda:
        op._gemm((payload.contiguous(), scale.contiguous()), (op.weight, op.scale), output)
    else:
        output.copy_((payload.float() @ op.weight.float().T) * scale * op.scale.T)
    return output


def original_norm(norm, value):
    values = value.float()
    inverse = torch.rsqrt(values.square().mean(-1, keepdim=True) + norm.variance_epsilon)
    return (values * inverse * norm.weight.float()).to(torch.bfloat16)


def make_linear(width, output, device):
    weight = torch.randn(output, width, dtype=torch.bfloat16, device=device)
    scale = torch.linspace(0.002, 0.012, output, device=device)[:, None]
    return PPUPTPCLinear(weight.to(torch.float8_e4m3fn), scale)


@contextmanager
def quantizations():
    calls = []
    original = PPUPTPCLinear.quantize_input

    def record(owner, value):
        calls.append((owner, tuple(value.shape)))
        return original(owner, value)

    with patch.object(PPUPTPCLinear, "quantize_input", new=record):
        yield calls


@contextmanager
def prefill_component(rows, *, indexed=True, device="cpu", mixed=False, compact=False):
    """Keep the production projection and selection call graph intact.

    Fixed row maps represent one CP rank; no collective is launched. The
    boundary exception is caught inside run(), including during Graph capture.
    """
    hidden, latent, kv = (5120, 1280, 512) if device == "cuda" else (64, 32, 16)
    q_width, index_width = (128, 128) if device == "cuda" else (32, 16)
    attn = attention_layer.SharedPPUAttention.__new__(attention_layer.SharedPPUAttention)
    torch.nn.Module.__init__(attn)
    for name, width, output in (
        ("wq_a", hidden, latent), ("wq_b", latent, q_width),
        ("wkv", hidden, kv), ("index_wq", latent, index_width),
    ):
        setattr(attn, name, make_linear(width, output, device))
    if mixed:
        attn.wkv = PPUResidentBF16Linear(attn.wkv.weight.float().to(torch.bfloat16))
    with patch.dict(os.environ, ENV):
        attn.q_norm = PPUSharedRMSNorm(torch.randn(latent, device=device, dtype=torch.bfloat16), 1e-6)
        attn.kv_norm = PPUSharedRMSNorm(torch.randn(kv, device=device, dtype=torch.bfloat16), 1e-6)
    attn.index_weights = torch.randn(1, hidden, device=device, dtype=torch.bfloat16)
    attn.geometry = SimpleNamespace(
        hidden_size=hidden, n_heads=1, head_dim=q_width,
        index_n_heads=1, index_head_dim=index_width, index_topk=2, window=128,
        ratio=lambda source: 1,
    )
    attn.layer_id, attn.kv_source_layer_id = 1, 0
    attn.compress_ratio, attn.is_index_source = int(indexed), indexed
    attn.is_kv_source = False
    attn.v41_config = {}
    attn.ops = SimpleNamespace(encode_kv=lambda value, **kwargs: value)
    positions = torch.arange(rows, device=device)
    requests = torch.zeros(rows, device=device, dtype=torch.int64)
    ends = torch.tensor([0, rows], device=device, dtype=torch.int32)
    keep = torch.arange(0, rows, 2, device=device)
    cp = SimpleNamespace(cp_size=8, gather_restore_positions=None, chunk_lengths_per_req=(rows,))
    receipt = SimpleNamespace(outputs=[], projections=[], selections=[], crops=[], staged=[], pairs=[])

    def boundary(*args, **kwargs):
        raise _ProjectionBoundary()

    def stage(layer, pos, req, values, *args, **kwargs):
        receipt.staged.append(values.shape[0])

    cache = SimpleNamespace(
        sharded=True, lengths=(max(rows, 3),), stage_swa=stage,
        prepare_source=lambda source: None, read_source=boundary,
    )
    state = SimpleNamespace(
        cache=cache, prefill=True, idle=False, compacted=False, shared={}, query_prefixes=(0,),
        use_compact_cp8_ops=lambda context: False,
        full_rows=lambda context, **kwargs: (positions, requests, slice(None)),
    )
    row_map = [positions, requests, ends, None]
    state.rows = lambda context: tuple(row_map)

    def crop(value):
        result = value[keep]
        receipt.crops.append(result)
        return result

    state.plan = SimpleNamespace(exchange=SimpleNamespace(compact=crop))

    def finish_crop():
        row_map[:] = [positions[keep], requests[keep], ends, None]
        state.context = SimpleNamespace(
            cp_size=8, gather_restore_positions=None, chunk_lengths_per_req=(keep.numel(),),
        )

    freqs = torch.ones(max(rows, 1), 1, device=device, dtype=torch.complex64)
    attn.runtime = SimpleNamespace(
        current=state, freqs_for=lambda layer: freqs,
        policy=SimpleNamespace(mode="full"), qkv_streams=None,
    )
    attn._cp_ctx = cp
    original_project = attention_layer.SharedPPUAttention._project_indexer
    original_select = attention_layer.SharedPPUAttention._select
    original_query_kv = attention_layer.SharedPPUAttention.project_query_kv

    def rope(value, table):
        receipt.outputs.append(value)
        return value

    def project(owner, x, qr, pos, shared, *, quantized_input=None):
        receipt.projections.append(x.shape[0])
        receipt.pairs.append(quantized_input)
        result = original_project(owner, x, qr, pos, shared, quantized_input=quantized_input)
        receipt.outputs.append(result[1])
        return result

    def select(owner, x, qr, context, cache, shared, **kwargs):
        receipt.selections.append((x.shape[0], kwargs.get("quantized_input"), kwargs.get("indexer_inputs")))
        return original_select(owner, x, qr, context, cache, shared, **kwargs)

    reuse = [True]

    def query_kv(owner, value, **kwargs):
        kwargs["reuse_ptpc"] = reuse[0]
        return original_query_kv(owner, value, **kwargs)

    def run(value, *, share=True):
        reuse[0] = share
        receipt.outputs = []
        receipt.projections = []
        receipt.selections = []
        receipt.crops = []
        receipt.staged = []
        receipt.pairs = []
        row_map[:] = [positions, requests, ends, None]
        try:
            attn._body(value, compact_after_producer=finish_crop if compact else None)
        except _ProjectionBoundary:
            return tuple(receipt.outputs)
        raise AssertionError("Prefill must reach the scoring/attention projection boundary")

    def reference(value):
        qr = original_norm(attn.q_norm, original_linear(attn.wq_a, value))
        q = original_linear(attn.wq_b, qr).reshape(rows, 1, q_width)
        kv_values = original_norm(attn.kv_norm, original_linear(attn.wkv, value))
        if not indexed:
            return q, kv_values
        index_q = original_linear(attn.index_wq, qr).reshape(rows, 1, index_width)
        weights = torch.nn.functional.linear(value, attn.index_weights).float() * index_width ** -0.5
        return q, kv_values, index_q, weights

    with ExitStack() as stack:
        environment = {**ENV, attention_layer.FUSION_ENV: "0", attention_layer.STREAMS_ENV: "0"}
        stack.enter_context(patch.dict(os.environ, environment))
        for name, replacement in (
            ("optimized_diagnostics_enabled", lambda: False),
            ("cp8_yoco_small_ops_enabled", lambda state: False),
            ("cp_all_gather_full", lambda value, context, **kwargs: value),
            ("_rope_tail", rope),
            ("build_cp_prefill_workspace", boundary),
        ):
            stack.enter_context(patch.object(attention_layer, name, new=replacement))
        stack.enter_context(patch.object(attention_layer.SharedPPUAttention, "_project_indexer", new=project))
        stack.enter_context(patch.object(attention_layer.SharedPPUAttention, "_select", new=select))
        stack.enter_context(patch.object(attention_layer.SharedPPUAttention, "project_query_kv", new=query_kv))
        yield attn, receipt, run, reference, hidden


class PTPCPrefillReuseCPU(unittest.TestCase):
    @torch.no_grad()
    def test_body_consumers_exact_three_or_four_quantizers_become_two(self):
        for indexed, old_count in ((False, 3), (True, 4)):
            with self.subTest(indexed=indexed), prefill_component(8, indexed=indexed) as (_, receipt, run, reference, hidden):
                value = torch.randn(8, hidden, dtype=torch.bfloat16)
                for share, count in ((False, old_count), (True, 2)):
                    with quantizations() as calls:
                        actual = run(value, share=share)
                    self.assertEqual(len(calls), count)
                    self.assertEqual(len(actual), 4 if indexed else 2)
                    for left, right in zip(actual, reference(value)):
                        assert_bits(self, left, right)
                    self.assertEqual(receipt.staged, [8])
                    if indexed:
                        self.assertEqual(receipt.projections, [8])
                        self.assertEqual(receipt.pairs[0] is not None, share)

    @torch.no_grad()
    def test_changed_contents_mixed_modules_and_fusion_rollback(self):
        for mixed, fusion, count in ((False, "1", 2), (True, "1", 2), (False, "0", 4)):
            with self.subTest(mixed=mixed, fusion=fusion), prefill_component(8, mixed=mixed) as (_, receipt, run, reference, hidden):
                value = torch.empty(8, hidden, dtype=torch.bfloat16)
                previous_pair = None
                for magnitude in (0.0, 0.125, 2.0):
                    value.copy_(torch.randn_like(value) * magnitude)
                    with patch.dict(os.environ, {"RTP_V41_PPU_PTPC_FUSION": fusion}), quantizations() as calls:
                        actual = run(value)
                    self.assertEqual(len(calls), count)
                    for left, right in zip(actual, reference(value)):
                        assert_bits(self, left, right)
                    pair = receipt.pairs[0]
                    if pair is not None:
                        self.assertIsNot(pair, previous_pair)
                    previous_pair = pair

    @torch.no_grad()
    def test_producer_consumes_full_pair_before_compaction(self):
        with prefill_component(8, compact=True) as (_, receipt, run, reference, hidden):
            value = torch.randn(8, hidden, dtype=torch.bfloat16)
            with quantizations() as calls:
                actual = run(value)
            self.assertEqual(len(calls), 2)
            for left, right in zip(actual, reference(value)):
                assert_bits(self, left, right)
            self.assertEqual(receipt.staged, [8])
            self.assertEqual(receipt.projections, [8])
            self.assertEqual(receipt.pairs[0][0].shape[0], 8)
            rows, pair, inputs = receipt.selections[0]
            self.assertEqual(rows, 4)
            self.assertIsNone(pair)
            for left, right in zip(inputs, (actual[2][::2], actual[3][::2])):
                assert_bits(self, left, right)
            for left, right in zip(receipt.crops, (actual[2][::2], actual[3][::2], actual[0][::2], value[::2])):
                assert_bits(self, left, right)


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_PTPC_DEVICE") == "1", "explicit PPU component test")
class PTPCPrefillReuseDevice(unittest.TestCase):
    @torch.no_grad()
    def test_body_native_exact_and_changed_input_graph(self):
        for rows, indexed, compact in ((24, False, False), (32, True, False), (144, True, True)):
            with self.subTest(rows=rows, indexed=indexed, compact=compact), prefill_component(
                rows, indexed=indexed, device="cuda", compact=compact,
            ) as (_, _, run, reference, hidden):
                value = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
                for _ in range(3):
                    run(value)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with quantizations() as calls:
                    with torch.cuda.graph(graph):
                        actual = run(value)
                self.assertEqual(len(calls), 2)
                for magnitude in (0.0, 0.125, 2.0):
                    value.copy_(torch.randn_like(value) * magnitude)
                    graph.replay()
                    self.assertEqual(len(actual), 4 if indexed else 2)
                    for left, right in zip(actual, reference(value)):
                        assert_bits(self, left, right)


if __name__ == "__main__":
    unittest.main()
