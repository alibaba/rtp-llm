"""Run on an explicitly reserved GPU; covers pinned host reads in graph replay."""

import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.engram import (
    Engram,
    EngramLayout,
    HostEngramEmbedding,
    NgramHashState,
    _PinnedSharedTable,
    gated_engram_residual,
)


class EngramViewTest(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and torch.version.hip is None, "requires NVIDIA CUDA"
    )
    @torch.inference_mode()
    def test_checkpoint_ownership_does_not_change_gate_dispatch(self):
        from rtp_llm.models_py.new_models.deepseek_v41.engram import OwnedEngram

        if torch.cuda.get_device_capability()[0] < 8:
            self.skipTest("requires native BF16")
        cfg = dict(
            engram_layer_ids=[1],
            engram_num_embeddings=[8],
            engram_max_ngram_size=2,
            engram_n_heads=1,
            engram_head_dim=32,
            engram_compressed_vocab_size=8,
            engram_pad_token_id=0,
            engram_vocab_size=7,
            hidden_size=5120,
            hc_mult=4,
            rms_norm_eps=1e-20,
        )
        owner = OwnedEngram(cfg, 1).cuda()
        rng = torch.Generator(device="cuda").manual_seed(427)

        def sample(*shape):
            return torch.randn(
                shape, device="cuda", generator=rng, dtype=torch.bfloat16
            )

        owner.load_weights({"q_weight": sample(4, 5120), "k_weight": sample(4, 5120)})
        hidden, kv = sample(97, 4, 5120), sample(97, 25600)
        ids = torch.arange(len(kv), device="cuda")
        mask = ids % 7 != 0
        # Isolate the gate: both owners receive exactly the same projected KV.
        owner.embed_tokens = lambda indices, device: kv.unsqueeze(1)
        projection = torch.nn.Identity()
        object.__setattr__(owner.wkv, "_executor", projection)
        baseline = Engram(
            owner.layout,
            0,
            owner.embed_tokens,
            projection,
            owner.q_weight.detach(),
            owner.k_weight.detach(),
            owner.eps,
        )
        for strided in (False, True):
            with self.subTest(strided=strided):
                x = (
                    torch.stack((hidden, hidden), dim=-1)[..., 0]
                    if strided
                    else torch.empty_like(hidden)
                )
                x.copy_(hidden)
                expected = baseline(x, ids, mask)
                aliases = expected.data_ptr() == x.data_ptr()
                expected = expected.clone()
                x.copy_(hidden)
                result = owner(x, ids, mask)
                self.assertEqual(result.data_ptr() == x.data_ptr(), aliases)
                self.assertTrue(torch.equal(result, expected))

    @torch.inference_mode()
    def test_cpu_reference_accepts_views(self):
        h = torch.arange(24).reshape(3, 2, 4).float()[..., ::2]
        kv = torch.ones(3, 6)
        w = torch.ones(2, 2)
        self.assertTrue(
            torch.equal(
                gated_engram_residual(h, kv, w, w, 1e-20, normalize_views=True),
                gated_engram_residual(h.contiguous(), kv, w, w, 1e-20),
            )
        )

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.version.hip is None, "requires NVIDIA CUDA"
    )
    @torch.inference_mode()
    def test_native_gate_matches_tight_for_each_strided_operand(self):
        if torch.cuda.get_device_capability()[0] < 8:
            self.skipTest("requires native BF16")
        rng = torch.Generator(device="cuda").manual_seed(427)
        base = [
            torch.randn(shape, generator=rng, device="cuda", dtype=torch.bfloat16)
            for shape in ((19, 4, 5120), (19, 25600), (4, 5120), (4, 5120))
        ]
        base.append(torch.arange(19, device="cuda") % 4 != 0)

        def run(seq, **kwargs):
            return gated_engram_residual(*seq[:4], 1e-20, seq[4], **kwargs)

        expected = run(base)
        for i in range(5):
            with self.subTest(operand=i):
                seq = base.copy()
                seq[i] = torch.stack((seq[i], seq[i]), dim=-1)[..., 0]
                orig = seq[0].clone()
                self.assertTrue(torch.equal(expected, run(seq, normalize_views=True)))
                self.assertTrue(torch.equal(orig, seq[0]))


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA GPU")
class EngramProjectionWarmupTest(unittest.TestCase):
    @torch.inference_mode()
    def test_loaded_wkv_is_discovered_without_registering_another_weight(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("requires SM100 group32 GEMM")
        from safetensors.torch import save_file

        from rtp_llm.models_py.modules.dsv41.dsv41_kernel_jit_warmup import (
            collect_v41_dense_shapes,
        )
        from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear
        from rtp_llm.models_py.new_models.deepseek_v41.engram import (
            OwnedEngramProjection,
        )
        from rtp_llm.models_py.weight_mapper import get_all_weights

        n, k = 640, 512
        w = torch.randn(n, k).to(torch.float8_e4m3fn)
        s = torch.ones(n // 32, k // 32).to(torch.float8_e8m0fnu)
        model = torch.nn.Module()
        model.wkv = OwnedEngramProjection(k, n).cuda()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model.safetensors")
            save_file({"weight": w, "scale": s}, path)
            model.wkv.load_weights(get_all_weights([path]))
        self.assertEqual(collect_v41_dense_shapes(model)[0], {})
        model.wkv.process_weights_after_loading()
        native = model.wkv.v41_runtime_linear
        self.assertIs(type(native), V41MXFP8Linear)
        linears, _ = collect_v41_dense_shapes(model)
        self.assertEqual(linears, {(n, k): ("wkv", native)})
        self.assertEqual(
            set(dict(model.named_parameters())), {"wkv.weight", "wkv.scale"}
        )
        self.assertIs(native.weight, model.wkv.weight)
        x = torch.randn(392, k, device="cuda", dtype=torch.bfloat16)
        self.assertTrue(
            torch.equal(model.wkv(x), V41MXFP8Linear(w.cuda(), s.cuda())(x))
        )


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA GPU")
class EngramCudaTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        config = dict(
            engram_layer_ids=[1, 14],
            engram_num_embeddings=[4000, 4000],
            engram_max_ngram_size=4,
            engram_n_heads=4,
            engram_head_dim=32,
            engram_compressed_vocab_size=32,
            engram_pad_token_id=2,
            engram_vocab_size=13,
        )
        self.layout = EngramLayout(config)
        self.hash = NgramHashState(
            self.layout, token_map=list(range(32)), device="cuda:0"
        )
        self.weight = torch.randn(4000, 32).to(torch.float8_e4m3fn)
        self.scales = torch.randint(125, 130, (4000, 1), dtype=torch.uint8)
        self.pinned = _PinnedSharedTable(
            self.weight, self.scales, "test-" + uuid.uuid4().hex, "cuda:0"
        )
        self.embedding = HostEngramEmbedding(
            self.pinned.weight, self.pinned.scales, pinned=self.pinned
        )

    @patch.dict(os.environ, {"DSV41_ENGRAM_UVA": "0"})
    def test_hash_and_host_lookup_capture_replay(self):
        self.assertTrue(self.pinned.storage.is_pinned())
        windows = torch.tensor(
            [[7, 6, 5, 4], [9, 8, -1, -1]], dtype=torch.int32, device="cuda"
        )
        dead = windows == 6

        def forward():
            return self.embedding(self.hash(windows, dead)[:, 0], "cuda")

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                forward()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = forward()
        windows.copy_(torch.tensor([[1, 0, -1, -1], [17, 16, 15, 14]], device="cuda"))
        dead.copy_(windows == 16)
        graph.replay()
        torch.cuda.synchronize()
        expected_hashes = self.hash(windows.cpu(), dead.cpu())
        torch.testing.assert_close(self.hash(windows, dead).cpu(), expected_hashes)
        expected_rows = HostEngramEmbedding(self.weight, self.scales)(
            expected_hashes[:, 0], "cpu"
        )
        torch.testing.assert_close(output.cpu(), expected_rows, rtol=0, atol=0)

    def test_gated_engram_residual_native_matches_reference(self):
        # Production shape (hc=4, dim=5120): the fused Triton kernel (inference
        # mode) must match the torch reference (grad mode keeps the reference
        # path) bitwise for >= 99.9% of elements. The few differing elements
        # are either 1-ulp bf16 rounding flips (fp32 summation order) or
        # catastrophic-cancellation noise (h + gate*value at bf16 input
        # quantization level); both are bounded: no element may exceed 1 ulp
        # AND 1e-5 absolute difference at once.
        for index, tokens in enumerate((1, 33, 257)):
            torch.manual_seed(57 + index)
            hidden = torch.randn(tokens, 4, 5120, device="cuda").bfloat16()
            kv = torch.randn(tokens, 5 * 5120, device="cuda").bfloat16()
            q_weight = torch.randn(4, 5120, device="cuda").bfloat16()
            k_weight = torch.randn(4, 5120, device="cuda").bfloat16()
            mask = torch.rand(tokens, device="cuda") < 0.8
            for token_mask in (None, mask):
                expected = gated_engram_residual(
                    hidden, kv, q_weight, k_weight, 1e-20, token_mask
                )
                with torch.inference_mode():
                    actual = gated_engram_residual(
                        hidden, kv, q_weight, k_weight, 1e-20, token_mask
                    )
                    inplace_hidden = hidden.clone()
                    inplace = gated_engram_residual(
                        inplace_hidden,
                        kv,
                        q_weight,
                        k_weight,
                        1e-20,
                        token_mask,
                        out=inplace_hidden.view_as(inplace_hidden),
                    )
                self.assertEqual(inplace.data_ptr(), inplace_hidden.data_ptr())
                torch.testing.assert_close(inplace, actual, rtol=0, atol=0)
                ai = actual.view(torch.int16).to(torch.int32)
                bi = expected.view(torch.int16).to(torch.int32)
                ulp = (
                    torch.where(ai >= 0, ai, -32768 - ai)
                    - torch.where(bi >= 0, bi, -32768 - bi)
                ).abs()
                diff = (actual.float() - expected.float()).abs()
                exact = (actual == expected).float().mean().item()
                bad = (ulp > 1) & (diff > 1e-5)
                self.assertGreaterEqual(exact, 0.999, (tokens, exact))
                self.assertEqual(int(bad.sum()), 0, (tokens, int(bad.sum())))

    def test_gated_engram_residual_native_mask_suppression(self):
        torch.manual_seed(63)
        hidden = torch.ones(2, 4, 5120, device="cuda").bfloat16()
        kv = torch.ones(2, 5 * 5120, device="cuda").bfloat16()
        qk = torch.ones(4, 5120, device="cuda").bfloat16()
        mask = torch.tensor([False, True], device="cuda")
        with torch.inference_mode():
            result = gated_engram_residual(hidden, kv, qk, qk, 1e-20, mask)
        self.assertTrue(torch.equal(result[0], hidden[0]))
        self.assertTrue(torch.all(result[1] > hidden[1]).item())

    def test_gated_engram_residual_native_capture_replay(self):
        # The fused kernel is captured by the decode CUDA graph: replay must
        # track input changes exactly like the eager kernel path.
        torch.manual_seed(71)
        hidden = torch.randn(2, 4, 5120, device="cuda").bfloat16()
        kv = torch.randn(2, 5 * 5120, device="cuda").bfloat16()
        q_weight = torch.randn(4, 5120, device="cuda").bfloat16()
        k_weight = torch.randn(4, 5120, device="cuda").bfloat16()

        def call():
            return gated_engram_residual(hidden, kv, q_weight, k_weight, 1e-20)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(3):
                call()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream), torch.inference_mode():
            output = call()
        hidden.copy_(torch.randn_like(hidden))
        kv.copy_(torch.randn_like(kv))
        with torch.inference_mode():
            expected = call()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_gated_engram_residual_inplace_capture_replay(self):
        hidden = torch.randn(7, 4, 5120, device="cuda").bfloat16()
        kv = torch.randn(7, 5 * 5120, device="cuda").bfloat16()
        q = torch.randn(4, 5120, device="cuda").bfloat16()
        k = torch.randn(4, 5120, device="cuda").bfloat16()
        mask = torch.arange(7, device="cuda") % 2 == 0

        def call():
            return gated_engram_residual(hidden, kv, q, k, 1e-20, mask, out=hidden)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(3):
                call()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream), torch.inference_mode():
            output = call()
        self.assertEqual(output.data_ptr(), hidden.data_ptr())
        for _ in range(3):
            hidden.normal_()
            kv.normal_()
            mask.logical_not_()
            with torch.inference_mode():
                expected = gated_engram_residual(hidden, kv, q, k, 1e-20, mask)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_chunked_forward_reuses_hidden_with_option_and_grad_fallback(self):
        chunk_sizes = []
        peak_bytes = {}

        def embedding(ids, device):
            chunk_sizes.append(ids.shape[0])
            return torch.zeros(
                (ids.shape[0], 1, 1), device=device, dtype=torch.bfloat16
            )

        class ConstantProjection(torch.nn.Module):
            def forward(self, rows):
                kv = rows.new_zeros((rows.shape[0], 5 * 5120))
                kv[:, 4 * 5120 :] = 1
                return kv

        q = k = torch.ones((4, 5120), dtype=torch.bfloat16, device="cuda")

        def out_of_place_reference(model, hidden, hashes, mask):
            # Allocation baseline belongs to this memory test, not a serving flag.
            if hidden.shape[0] <= 32768:
                return model._forward_rows(hidden, hashes, mask)
            output = torch.empty_like(hidden)
            for begin in range(0, hidden.shape[0], 32768):
                end = min(begin + 32768, hidden.shape[0])
                model._forward_rows(
                    hidden[begin:end],
                    hashes[begin:end],
                    mask[begin:end],
                    out=output[begin:end],
                )
            return output

        for setting in ("0", "1"):
            model = Engram(self.layout, 0, embedding, ConstantProjection(), q, k, 1e-20)
            for count in (3, 32769):
                with self.subTest(setting=setting, count=count), torch.inference_mode():
                    hidden = torch.zeros(
                        (count, 4, 5120), dtype=torch.bfloat16, device="cuda"
                    )
                    hashes = torch.zeros(
                        (count, self.layout.n_hash_cols),
                        dtype=torch.int64,
                        device="cuda",
                    )
                    mask = torch.arange(count, device="cuda") % 3 != 0
                    chunk_sizes.clear()
                    baseline = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    output = (
                        out_of_place_reference(model, hidden, hashes, mask)
                        if setting == "0"
                        else model(hidden, hashes, mask)
                    )
                    peak_bytes[setting, count] = (
                        torch.cuda.max_memory_allocated() - baseline
                    )
                    self.assertEqual(
                        output.data_ptr() == hidden.data_ptr(), setting == "1"
                    )
                    expected = torch.sigmoid(torch.tensor(0.001)).bfloat16().item()
                    torch.testing.assert_close(
                        output[:, 0, 0],
                        mask.to(torch.bfloat16) * expected,
                        rtol=0,
                        atol=0,
                    )
                    self.assertLessEqual(max(chunk_sizes), 32768)
                    self.assertEqual(sum(chunk_sizes), count)
                    if setting == "0":
                        self.assertEqual(hidden.count_nonzero().item(), 0)
                    del hidden, output
            hidden = torch.zeros(
                (3, 4, 5120), dtype=torch.bfloat16, device="cuda", requires_grad=True
            )
            output = model(hidden, hashes[:3])
            self.assertNotEqual(output.data_ptr(), hidden.data_ptr())
            self.assertEqual(hidden.count_nonzero().item(), 0)
            output.float().sum().backward()
            self.assertTrue(hidden.grad.isfinite().all().item())
            del hidden, output
        self.assertGreaterEqual(
            peak_bytes["0", 32769] - peak_bytes["1", 32769],
            32769 * 4 * 5120 * torch.bfloat16.itemsize,
        )

    def test_complete_engram_capture_changes_input_and_preserves_mask(self):
        from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear

        projection = V41MXFP8Linear(
            (torch.randn(640, 384, device="cuda") * 0.02).to(torch.float8_e4m3fn),
            torch.ones(20, 12, device="cuda").to(torch.float8_e8m0fnu),
        )
        model = Engram(
            self.layout,
            0,
            self.embedding,
            projection,
            torch.randn(4, 128, dtype=torch.bfloat16, device="cuda"),
            torch.randn(4, 128, dtype=torch.bfloat16, device="cuda"),
            1e-20,
        )
        hidden = torch.randn(2, 4, 128, dtype=torch.bfloat16, device="cuda")
        windows = torch.tensor(
            [[7, 6, 5, 4], [-1, -1, -1, -1]], dtype=torch.int32, device="cuda"
        )

        def forward():
            return model(hidden, self.hash(windows)[:, 0], windows[:, 0] >= 0)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                forward()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = forward()
        hidden.copy_(torch.randn_like(hidden))
        windows[0].copy_(torch.tensor([17, 16, 15, 14], device="cuda"))
        expected = forward()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(output[1], hidden[1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
