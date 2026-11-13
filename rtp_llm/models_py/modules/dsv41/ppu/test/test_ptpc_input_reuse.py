"""Component-only PTPC producer reuse, exact outputs and replay kernel counts.

Real projection/norm/GEMM implementations run with small resident weights.
The public Commit loop is represented by its three layer callbacks and KV
publication is stubbed; no model, checkpoint, service or RDMA is exercised.
"""

from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu.layers import (
    PPUAttentionProjections,
    PPUSharedRMSNorm,
)
from rtp_llm.models_py.modules.dsv41.ppu.linear import (
    PPUPTPCLinear,
    PPUResidentBF16Linear,
    prepare_shared_ptpc_input,
)


ENV = {
    "RTP_V41_PPU_QUANT_MODE": "mxfp4_fp8_ptpc",
    "RTP_V41_PPU_PTPC_FUSION": "1",
}


def old_quant(value):
    values = value.float()
    maximum = values.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    scale = maximum * 0.0022321429569274187
    payload = (values * (448.0 / maximum)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return payload.contiguous(), scale.contiguous()


def old_norm(value, norm):
    if not value.numel():
        return torch.empty_like(value)
    if value.is_cuda:
        # Standalone norm now has the library's numerical contract. Reuse
        # tests compare the projection/quant scheduling around that boundary.
        from flashinfer.norm import rmsnorm

        return rmsnorm(value, norm.weight, eps=norm.variance_epsilon)
    values = value.float()
    inverse = torch.rsqrt(values.square().mean(-1, keepdim=True) + norm.variance_epsilon)
    return (values * inverse * norm.weight.float()).to(torch.bfloat16)


def old_linear(op, value):
    if not isinstance(op, PPUPTPCLinear):
        return torch.nn.functional.linear(value, op.weight, op.bias)
    shape = (*value.shape[:-1], op.weight.shape[0])
    flat = value.reshape(-1, value.shape[-1])
    output = torch.empty((flat.shape[0], op.weight.shape[0]), dtype=torch.bfloat16, device=value.device)
    if flat.shape[0]:
        payload, scale = old_quant(flat)
        if value.is_cuda:
            op._gemm((payload, scale), (op.weight, op.scale), output)
        else:
            output.copy_((payload.float() @ op.weight.float().T) * scale * op.scale.T)
    return output.reshape(shape)


def assert_bits(case, actual, expected):
    case.assertEqual(actual.dtype, expected.dtype)
    case.assertEqual(actual.shape, expected.shape)
    if not actual.numel():
        return
    left, right = actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    if not torch.equal(left, right):
        position = tuple((left != right).nonzero()[0].tolist())
        case.fail(f"byte mismatch at {position}: actual={int(left[position])}, expected={int(right[position])}")


def make_linear(width, output, device, seed):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(output, width, generator=generator).to(torch.float8_e4m3fn).to(device)
    scale = torch.linspace(0.002, 0.012, output, device=device, dtype=torch.float32)[:, None]
    return PPUPTPCLinear(weight, scale)


def make_norm(width, device, seed):
    generator = torch.Generator().manual_seed(seed)
    gamma = torch.randn(width, generator=generator).to(torch.bfloat16).to(device)
    return PPUSharedRMSNorm(gamma, 1e-20)


def make_projection(device="cpu", *, mixed=False):
    hidden, latent, kv = (5120, 1280, 512) if device == "cuda" else (64, 32, 16)
    op = PPUAttentionProjections.__new__(PPUAttentionProjections)
    torch.nn.Module.__init__(op)
    op.wq_a = make_linear(hidden, latent, device, 4130)
    op.q_norm = make_norm(latent, device, 4131)
    op.wq_b = make_linear(latent, 128 if device == "cuda" else 32, device, 4132)
    op.wkv = make_linear(hidden, kv, device, 4133)
    op.kv_norm = make_norm(kv, device, 4134)
    op.index_wq = make_linear(latent, 128 if device == "cuda" else 16, device, 4135)
    if mixed:
        op.wkv = PPUResidentBF16Linear(op.wkv.weight.float().to(torch.bfloat16))
    return op, hidden


def projection(op, value, *, indexed, reuse):
    qr, q, kv, pair = op.project_query_kv(value, prepare_indexer=indexed, reuse_ptpc=reuse)
    result = [qr, q, kv]
    if indexed:
        result.append(op.project_indexer_query(qr, quantized_input=pair))
    return tuple(result)


def old_projection(op, value, *, indexed):
    qr = old_norm(old_linear(op.wq_a, value), op.q_norm)
    q = old_linear(op.wq_b, qr)
    kv = old_norm(old_linear(op.wkv, value), op.kv_norm)
    return (qr, q, kv, old_linear(op.index_wq, qr)) if indexed else (qr, q, kv)


@contextmanager
def quantizations():
    calls = []
    original = PPUPTPCLinear.quantize_input

    def counted(owner, value):
        calls.append((owner, tuple(value.shape)))
        return original(owner, value)

    with patch.object(PPUPTPCLinear, "quantize_input", new=counted):
        yield calls


@contextmanager
def commit_component(device="cpu"):
    from rtp_llm.models_py.model_desc.deepseek_v41_dspark_model import DeepSeekV41DSparkModel
    from rtp_llm.models_py.modules.dsv41.ppu import dspark_attention
    from rtp_llm.models_py.modules.dsv41.ppu.dspark_model import DeepSeekV41PPUDSparkModel

    hidden, kv = (5120, 512) if device == "cuda" else (64, 16)
    model = DeepSeekV41PPUDSparkModel.__new__(DeepSeekV41PPUDSparkModel)
    if isinstance(model, torch.nn.Module):
        torch.nn.Module.__init__(model)
    model._is_decode_role = True
    model._commit_ptpc_input = None
    model.main_proj = make_linear(128, hidden, device, 4140)
    model.main_norm = make_norm(hidden, device, 4141)
    layers = [SimpleNamespace(attn=SimpleNamespace(
        wkv=make_linear(hidden, kv, device, 4142 + i),
        kv_norm=make_norm(kv, device, 4150 + i),
    )) for i in range(3)]
    model.v4 = SimpleNamespace(layers=layers)
    state = SimpleNamespace(is_graph_commit=True)
    model.shared_runtime = SimpleNamespace(current=state)
    model.shared_weight_views = SimpleNamespace(geometry=SimpleNamespace())
    result = SimpleNamespace(outputs=[], pairs=[], fail_layer=None, foreign_rows=False)
    row_maps = {}

    def write_rows(owner, current, layer, main_x, requests, positions, ends, *, quantized_input=None):
        result.pairs.append(quantized_input)
        # Publication is outside this component: exercise the actual KV
        # projection/norm consumer before replacing only its cache write.
        value = dspark_attention._project_commit_kv(owner.v4.layers[layer].attn, main_x, quantized_input)
        result.outputs.append(value)
        if result.fail_layer == layer:
            raise RuntimeError("injected component publication failure")

    def public_loop(owner, main_x, requests, positions, ends, inputs, commit_ctx=None):
        if not ends.numel() or not main_x.shape[0]:
            return
        rows = main_x.clone() if result.foreign_rows else main_x
        for layer in range(3):
            owner._commit_layer_features(layer, rows, requests, positions, ends, None, 256, 1)

    def run(features, *, reuse=True):
        result.outputs = []
        result.pairs = []
        rows = features.shape[0]
        if rows not in row_maps:
            # Warmup prepares static row maps, as the production graph batch
            # does. Capture must not copy a temporary pageable CPU tensor.
            row_maps[rows] = (
                torch.zeros(rows, dtype=torch.int64, device=features.device),
                torch.arange(rows, device=features.device),
                torch.full((1,), rows, dtype=torch.int32, device=features.device),
            )
        requests, positions, ends = row_maps[rows]
        main_x = model.combine_hidden_states(features)
        call = model.commit_feature_rows if reuse else lambda *a, **k: public_loop(model, *a, **k)
        call(main_x, requests, positions, ends, SimpleNamespace())
        return (main_x, *result.outputs)

    def original(features):
        main_x = old_norm(old_linear(model.main_proj, features), model.main_norm)
        return (main_x, *(old_norm(old_linear(layer.attn.wkv, main_x), layer.attn.kv_norm) for layer in layers))

    with patch.object(DeepSeekV41DSparkModel, "commit_feature_rows", new=public_loop), patch.object(
        dspark_attention, "graph_commit_layer", new=write_rows
    ):
        yield model, result, run, original


class _PTPCInputReuseBase(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, ENV)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def assert_projection(self, op, value, indexed, reuse=True):
        actual = projection(op, value, indexed=indexed, reuse=reuse)
        for left, right in zip(actual, old_projection(op, value, indexed=indexed)):
            assert_bits(self, left, right)


class PTPCInputReuseCPU(_PTPCInputReuseBase):
    def setUp(self):
        super().setUp()

        def test_norm(value, weight, *, eps):
            return old_norm(value, SimpleNamespace(weight=weight, variance_epsilon=eps))

        # CPU exercises orchestration using a test-only provider. Production
        # dispatch has no CPU arithmetic fallback and still requires FlashInfer.
        provider = patch(
            "rtp_llm.models_py.modules.dsv41.ppu.rmsnorm._flashinfer_rmsnorm",
            return_value=test_norm,
        )
        provider.start()
        self.addCleanup(provider.stop)

    @torch.no_grad()
    def test_exact_projection_reduces_three_or_four_quantizers_to_two(self):
        op, hidden = make_projection()
        value = torch.randn(3, hidden, dtype=torch.bfloat16)
        for indexed, old_count in ((False, 3), (True, 4)):
            with self.subTest(indexed=indexed):
                with quantizations() as calls:
                    self.assert_projection(op, value, indexed, reuse=False)
                self.assertEqual(len(calls), old_count)
                with quantizations() as calls:
                    self.assert_projection(op, value, indexed)
                self.assertEqual(len(calls), 2)

    @torch.no_grad()
    def test_shape_zero_empty_and_explicit_fallbacks_keep_results(self):
        op, hidden = make_projection()
        for value in (torch.zeros(3, hidden, dtype=torch.bfloat16), torch.empty(0, hidden, dtype=torch.bfloat16)):
            self.assert_projection(op, value, True)
        value = torch.randn(3, hidden * 2, dtype=torch.bfloat16)[:, ::2]
        self.assert_projection(op, value, True)
        with patch.dict(os.environ, {"RTP_V41_PPU_PTPC_FUSION": "0"}), quantizations() as calls:
            self.assert_projection(op, value, True)
        self.assertEqual(len(calls), 4)
        self.assertIsNone(prepare_shared_ptpc_input(value, (op.wq_a, object())))
        mixed, _ = make_projection(mixed=True)
        self.assert_projection(mixed, value, True)

    @torch.no_grad()
    def test_same_tensor_changed_contents_does_not_reuse_old_payloads(self):
        op, hidden = make_projection()
        value = torch.randn(3, hidden, dtype=torch.bfloat16)
        for magnitude in (0.0, 0.125, 2.0):
            value.copy_(torch.randn_like(value) * magnitude)
            with quantizations() as calls:
                self.assert_projection(op, value, True)
            self.assertEqual(len(calls), 2)

    @torch.no_grad()
    def test_commit_scopes_pair_to_one_call_and_cleans_after_failure(self):
        with commit_component() as (model, receipt, run, original):
            features = torch.randn(8, 128, dtype=torch.bfloat16)
            with quantizations() as calls:
                actual = run(features)
            self.assertEqual(len(calls), 2)
            for left, right in zip(actual, original(features)):
                assert_bits(self, left, right)
            self.assertIsNone(model._commit_ptpc_input)
            self.assertIsNotNone(receipt.pairs[0])
            self.assertTrue(all(pair is receipt.pairs[0] for pair in receipt.pairs))
            previous = receipt.pairs[0]
            features.copy_(torch.randn_like(features) * 0.125)
            actual = run(features)
            self.assertIsNot(receipt.pairs[0], previous)
            for left, right in zip(actual, original(features)):
                assert_bits(self, left, right)
            receipt.fail_layer = 1
            with self.assertRaisesRegex(RuntimeError, "injected component"):
                run(features)
            self.assertIsNone(model._commit_ptpc_input)
            receipt.fail_layer = None
            receipt.foreign_rows = True
            with self.assertRaisesRegex(RuntimeError, "bound feature rows"):
                run(features)
            self.assertIsNone(model._commit_ptpc_input)
            receipt.foreign_rows = False
            run(features)
            self.assertIsNone(model._commit_ptpc_input)

    @torch.no_grad()
    def test_empty_commit_does_not_prepare_unused_inputs(self):
        with commit_component() as (model, receipt, _, _):
            for rows, requests in ((8, 0), (0, 1)):
                main_x = torch.zeros(rows, 64, dtype=torch.bfloat16)
                ids = torch.zeros(rows, dtype=torch.int64)
                positions = torch.arange(rows)
                ends = torch.zeros(requests, dtype=torch.int32)
                with quantizations() as calls:
                    model.commit_feature_rows(main_x, ids, positions, ends, SimpleNamespace())
                self.assertEqual(len(calls), 0)
                self.assertEqual(receipt.outputs, [])
                self.assertIsNone(model._commit_ptpc_input)

    @torch.no_grad()
    def test_commit_prefill_and_fusion_rollback_do_not_share(self):
        with commit_component() as (model, receipt, run, original):
            features = torch.randn(8, 128, dtype=torch.bfloat16)
            for decode, fusion in ((False, "1"), (True, "0")):
                model._is_decode_role = decode
                with patch.dict(os.environ, {"RTP_V41_PPU_PTPC_FUSION": fusion}), quantizations() as calls:
                    actual = run(features)
                self.assertEqual(len(calls), 4)
                self.assertTrue(all(pair is None for pair in receipt.pairs))
                self.assertIsNone(model._commit_ptpc_input)
                for left, right in zip(actual, original(features)):
                    assert_bits(self, left, right)
            model._is_decode_role = True
            legacy = model.v4.layers[1].attn.wkv
            model.v4.layers[1].attn.wkv = PPUResidentBF16Linear(legacy.weight.float().to(torch.bfloat16))
            with quantizations() as calls:
                actual = run(features)
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(pair is None for pair in receipt.pairs))
            for left, right in zip(actual, original(features)):
                assert_bits(self, left, right)

    @torch.no_grad()
    def test_declared_forty_eight_three_and_commit_consumers_save_fifty_three(self):
        # This is an explicit component schedule, not full-model execution.
        op, hidden = make_projection()
        value = torch.randn(2, hidden, dtype=torch.bfloat16)
        counts = []
        for reuse in (False, True):
            with quantizations() as calls:
                for layer in range(40):
                    projection(op, value, indexed=layer in (2, 8, 14, 20, 24, 28, 32, 36), reuse=reuse)
            verify = len(calls)
            with quantizations() as calls:
                for _ in range(3):
                    projection(op, value, indexed=False, reuse=reuse)
            draft = len(calls)
            with commit_component() as (_, _, run, _), quantizations() as calls:
                run(torch.randn(2, 128, dtype=torch.bfloat16), reuse=reuse)
            counts.append((verify, draft, len(calls)))
        self.assertEqual(counts, [(128, 9, 4), (80, 6, 2)])
        self.assertEqual(sum(a - b for a, b in zip(*counts)), 53)


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_PTPC_DEVICE") == "1", "explicit PPU component test")
class PTPCInputReuseDevice(_PTPCInputReuseBase):
    @torch.no_grad()
    def test_projection_native_exact_and_changed_input_graph(self):
        op, hidden = make_projection("cuda")
        for rows, indexed in ((24, False), (32, True)):
            value = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
            self.assert_projection(op, value, indexed)
            projection(op, value, indexed=indexed, reuse=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = projection(op, value, indexed=indexed, reuse=True)
            for magnitude in (0.0, 0.125, 2.0):
                value.copy_(torch.randn_like(value) * magnitude)
                graph.replay()
                for left, right in zip(actual, old_projection(op, value, indexed=indexed)):
                    assert_bits(self, left, right)
        self.assert_projection(op, value[:0], True)

    @torch.no_grad()
    def test_commit_dynamic_features_native_and_changed_input_graph(self):
        for rows in (8, 32):
            with commit_component("cuda") as (model, _, run, original):
                features = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
                for left, right in zip(run(features), original(features)):
                    assert_bits(self, left, right)
                run(features)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = run(features)
                self.assertIsNone(model._commit_ptpc_input)
                for magnitude in (0.0, 0.125, 2.0):
                    features.copy_(torch.randn_like(features) * magnitude)
                    graph.replay()
                    for left, right in zip(actual, original(features)):
                        assert_bits(self, left, right)

    def profile_replay(self, name, rows, call, expected_quant, expected_gemm):
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = call()
        graph.replay()
        torch.cuda.synchronize()
        with tempfile.TemporaryDirectory(prefix="ptpc-reuse-") as temporary:
            root = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", temporary))
            root.mkdir(parents=True, exist_ok=True)
            path = root / f"input-reuse-{name}-M{rows}.json"
            with torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
            ]) as profiler:
                graph.replay()
                torch.cuda.synchronize()
            profiler.export_chrome_trace(str(path))
            events = json.loads(path.read_text())["traceEvents"]
        names = Counter(e["name"] for e in events if e.get("cat") == "kernel" and e.get("ph") == "X")
        self.assertTrue(names, "profiler must contain actual device kernels")
        quant = sum(count for name, count in names.items() if "quantize_rows" in name)
        gemm = sum(count for name, count in names.items() if "fp8_dense_gemm" in name)
        self.assertEqual((quant, gemm), (expected_quant, expected_gemm), names)
        print(json.dumps(dict(component=name, rows=rows, actual_graph_quant_kernels=quant,
                              actual_graph_gemm_kernels=gemm, trace=path.name)), flush=True)
        del output

    @torch.no_grad()
    def test_replay_has_fewer_quant_kernels_and_unchanged_native_gemms(self):
        op, hidden = make_projection("cuda")
        for rows, indexed, old_count in ((24, False, 3), (32, True, 4)):
            value = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
            for reuse, quant in ((False, old_count), (True, 2)):
                call = lambda: projection(op, value, indexed=indexed, reuse=reuse)
                self.profile_replay(f"projection-{int(indexed)}-{int(reuse)}", rows, call, quant, old_count)
        with commit_component("cuda") as (_, _, run, _):
            features = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
            for reuse, quant in ((False, 4), (True, 2)):
                self.profile_replay(f"commit-{int(reuse)}", 8, lambda: run(features, reuse=reuse), quant, 4)


if __name__ == "__main__":
    unittest.main()
