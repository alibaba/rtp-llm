"""Run on an explicitly reserved GPU; covers pinned host reads in graph replay."""

import os
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.engram import (
    Engram,
    EngramLayout,
    HostEngramEmbedding,
    NgramHashState,
    _PinnedSharedTable,
    gated_engram_residual,
)
from rtp_llm.models_py.modules.dsv4.utils import V41MXFP8Linear


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

    def _prefetch_model(self):
        model = Engram(
            self.layout,
            0,
            self.embedding,
            torch.nn.Linear(384, 640, bias=False, device="cuda", dtype=torch.bfloat16),
            torch.randn(4, 128, dtype=torch.bfloat16, device="cuda"),
            torch.randn(4, 128, dtype=torch.bfloat16, device="cuda"),
            1e-20,
        )
        with patch.dict(os.environ, {"DSV41_ASYNC_ENGRAM_LOOKUP": "1"}):
            from cuda.bindings import driver

            from rtp_llm.models_py.modules.dsv4 import _engram_triton
            from rtp_llm.models_py.modules.dsv4.dsv41_kernel_jit_warmup import (
                warmup_v41_engram_jit,
            )

            startup = SimpleNamespace(
                _engram_hash_state=self.hash,
                v4=SimpleNamespace(layers=[SimpleNamespace(engram=model)]),
            )
            warmup_v41_engram_jit(startup, max_m=32768, device="cuda:0")
            self.assertIsNotNone(model._lookup_stream)
            self.assertTrue(_engram_triton._PREFETCH_LOOKUP_PREPARED)
            attribute = (
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
            )
            for kernel in _engram_triton._PREFETCH_LOOKUP_PREPARED:
                status, value = driver.cuFuncGetAttribute(
                    attribute, driver.CUfunction(kernel.function)
                )
                self.assertEqual(status, driver.CUresult.CUDA_SUCCESS)
                self.assertEqual(value, 100)
        return model

    def test_prefetch_startup_leaves_sync_function_attributes_unchanged(self):
        from cuda.bindings import driver

        from rtp_llm.models_py.modules.dsv4 import _engram_triton

        ids = torch.zeros(
            (16, self.layout.n_hash_cols), dtype=torch.int64, device="cuda"
        )
        output = torch.empty(
            (*ids.shape, self.layout.head_dim), dtype=torch.bfloat16, device="cuda"
        )
        rows = ids.numel()
        grid = min((rows + 15) // 16, self.embedding._num_sms)
        sync_kernel = _engram_triton._lookup_host_kernel[(grid,)](
            *self.embedding._uva,
            ids,
            output,
            rows,
            self.weight.shape[0],
            ids.stride(0),
            ids.stride(1),
            HEADS=ids.shape[1],
            DIM=self.layout.head_dim,
            QUANT_BLOCK=32,
            BLOCK_R=16,
            GRID=grid,
            num_warps=4,
        )
        attribute = (
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
        )
        function = driver.CUfunction(sync_kernel.function)
        before = driver.cuFuncGetAttribute(attribute, function)
        self.assertEqual(before[0], driver.CUresult.CUDA_SUCCESS)
        self._prefetch_model()
        after = driver.cuFuncGetAttribute(attribute, function)
        self.assertEqual(after, before)
        self.assertNotIn(sync_kernel, _engram_triton._PREFETCH_LOOKUP_PREPARED)
        actual = self.embedding(ids, "cuda", prefetch=True)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, output, rtol=0, atol=0)

    def test_prefetch_nondefault_producer_consumer_and_exact_mask(self):
        model = self._prefetch_model()
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        for count in (1, 16, 257):
            windows = torch.full((count, 4), 7, dtype=torch.int32, device="cuda")
            hidden = torch.randn(count, 4, 128, device="cuda", dtype=torch.bfloat16)
            ids = self.hash(windows)[:, 0].contiguous()
            mask = torch.ones(count, dtype=torch.bool, device="cuda")
            # Compile the original specialization before testing side scheduling.
            model(hidden, ids, mask)
            torch.cuda.synchronize()
            for token in (3, 11):
                producer.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(producer):
                    torch.cuda._sleep(1_000_000)
                    windows.fill_(token)
                    windows[::3] = -1
                    windows[1::7, 0] = 9  # Synthetic image/dead-token position.
                    dead = windows == 9
                    ids.copy_(self.hash(windows, dead)[:, 0])
                    mask.copy_((windows[:, 0] >= 0) & (windows[:, 0] != 9))
                    self.assertIsNotNone(model.prefetch_lookup(ids))
                # Consumer deliberately does not wait on producer: lookup's
                # event is the transitive producer -> lookup -> consumer edge.
                with torch.cuda.stream(consumer):
                    actual = model(hidden, ids, mask)
                consumer.synchronize()
                expected = model(hidden, ids, mask)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(actual[~mask], hidden[~mask], rtol=0, atol=0)
                self.assertIsNone(model._lookup_work)

    def test_prefetch_event_reuse_without_synchronizing_between_forwards(self):
        model = self._prefetch_model()
        hidden = torch.randn(16, 4, 128, dtype=torch.bfloat16, device="cuda")
        ids = torch.zeros(16, self.layout.n_hash_cols, dtype=torch.int64, device="cuda")
        mask = torch.arange(16, device="cuda") % 3 != 0
        expected = []
        for token in (7, 11, 3):
            ids.fill_(token)
            expected.append(model(hidden, ids, mask))
        torch.cuda.synchronize()
        actual = []
        for token in (7, 11, 3):
            ids.fill_(token)
            self.assertIsNotNone(model.prefetch_lookup(ids))
            actual.append(model(hidden, ids, mask))
        torch.cuda.synchronize()
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)

    def test_prefetch_records_output_stream_under_allocator_reuse(self):
        model = self._prefetch_model()
        count = 4096
        ids = torch.randint(0, 4000, (count, self.layout.n_hash_cols), device="cuda")
        expected = self.embedding(ids, "cuda")
        torch.cuda.synchronize()
        consumer = torch.cuda.Stream()
        work = model.prefetch_lookup(ids)
        with torch.cuda.stream(consumer):
            rows = work.consume(ids, torch.empty(count, device="cuda"))
            torch.cuda._sleep(5_000_000)
            actual = rows.clone()
        model._lookup_work = None
        del ids, rows, work
        # Reuse pressure on the output allocation's original stream, while the
        # delayed consumer still reads it. Missing record_stream corrupts actual.
        with torch.cuda.stream(model._lookup_stream):
            pressure = [torch.empty_like(expected).fill_(42) for _ in range(8)]
        consumer.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        del pressure

    def test_prefetch_boundaries_and_strided_fallback_match_original(self):
        model = self._prefetch_model()
        for count in (0, 1, 16, 32768, 32769):
            ids = torch.randint(
                0, 4000, (count, self.layout.n_hash_cols), device="cuda"
            )
            expected = self.embedding(ids, "cuda")
            work = model.prefetch_lookup(ids)
            if 0 < count <= 32768:
                self.assertIsNotNone(work)
                actual = work.consume(ids, torch.empty(count, device="cuda"))
                model._lookup_work = None
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            else:
                self.assertIsNone(work)
        ids = torch.randint(0, 4000, (16, self.layout.n_hash_cols * 2), device="cuda")[
            :, ::2
        ]
        self.assertIsNone(model.prefetch_lookup(ids))
        hidden = torch.randn(16, 4, 128, dtype=torch.bfloat16, device="cuda")
        torch.testing.assert_close(
            model(hidden, ids), model(hidden, ids.contiguous()), rtol=0, atol=0
        )

    def test_prefetch_capture_falls_back_and_replay_uses_changed_inputs(self):
        model = self._prefetch_model()
        hidden = torch.randn(16, 4, 128, dtype=torch.bfloat16, device="cuda")
        windows = torch.ones(16, 4, dtype=torch.int32, device="cuda")

        def forward():
            ids = self.hash(windows)[:, 0].contiguous()
            work = model.prefetch_lookup(ids)
            if torch.cuda.is_current_stream_capturing():
                self.assertIsNone(work)
            return model(hidden, ids, windows[:, 0] >= 0)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                forward()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = forward()
        for token in (7, 11, 3):
            windows.fill_(token)
            windows[::3] = -1
            hidden.normal_()
            expected = forward()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            self.assertIsNone(model._lookup_work)


if __name__ == "__main__":
    unittest.main()
