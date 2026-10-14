"""Run on an explicitly reserved GPU; covers pinned host reads in graph replay."""

import os
import threading
import unittest
import uuid
import weakref
from types import SimpleNamespace
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
from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear


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

    def _cpu_prefetch_model(self, capacity=128, *, deadline=False):
        from rtp_llm.models_py.modules.dsv41.engram import prepare_engram_cpu_lookup

        layers = [SimpleNamespace(engram=None) for _ in range(15)]
        for index, layer_id in enumerate((1, 14)):
            layers[layer_id] = SimpleNamespace(
                engram=Engram(
                    self.layout,
                    index,
                    self.embedding,
                    torch.nn.Linear(
                        384, 640, bias=False, device="cuda", dtype=torch.bfloat16
                    ),
                    torch.randn(4, 128, dtype=torch.bfloat16, device="cuda"),
                    torch.randn(4, 128, dtype=torch.bfloat16, device="cuda"),
                    1e-20,
                ),
                engram_hashes=None,
            )
        model = SimpleNamespace(
            _engram_hash_state=self.hash,
            _engram_layers=((1, 0), (14, 1)),
            _engram_cpu_runtime=None,
            v4=SimpleNamespace(layers=layers),
        )
        with patch.dict(os.environ, {"DSV41_CPU_ENGRAM_LOOKUP": "1"}):
            prepare_engram_cpu_lookup(model, max_m=capacity)
        # The existing H2D/allocator tests must exercise CPU consumption even
        # when their tiny synthetic workloads leave no GPU submission lead.
        # The separate deadline test explicitly enables fallback.
        for slot in model._engram_cpu_runtime.slots.values():
            slot.deadline_enabled = deadline
        self.addCleanup(model._engram_cpu_runtime.close)
        return model

    def test_cpu_prefetch_hash_boundaries_and_full_engram_match_gpu(self):
        from rtp_llm.models_py.modules.dsv41.engram import (
            EngramCPULookupPrefetch,
            _hash_token_windows_cpu,
        )

        model = self._cpu_prefetch_model()
        for count in (1, 16, 128):
            windows = torch.randint(-1, 32, (count, 8), dtype=torch.int32)[:, ::2]
            windows[::7, 1] = 2**31 - 1
            windows[::3, 0] = 9
            gpu_windows = windows.cuda()
            hashes = self.hash(gpu_windows, gpu_windows == 9)
            torch.testing.assert_close(
                _hash_token_windows_cpu(self.hash, windows, 9),
                hashes.cpu(),
                rtol=0,
                atol=0,
            )
            hidden = torch.randn(count, 4, 128, dtype=torch.bfloat16, device="cuda")
            mask = (gpu_windows[:, 0] >= 0) & (gpu_windows[:, 0] != 9)
            expected = []
            for layer_id, index in model._engram_layers:
                layer = model.v4.layers[layer_id]
                layer.engram_hashes = hashes[:, index].contiguous()
                expected.append(layer.engram(hidden, layer.engram_hashes, mask))
            plan = EngramCPULookupPrefetch(model, windows, 9)
            self.assertEqual(len(plan.work), 2)
            actual = [
                model.v4.layers[layer_id].engram(
                    hidden, model.v4.layers[layer_id].engram_hashes, mask
                )
                for layer_id, _ in model._engram_layers
            ]
            plan.clear()
            for result, reference in zip(actual, expected):
                torch.testing.assert_close(result, reference, rtol=0, atol=0)
                torch.testing.assert_close(result[~mask], hidden[~mask], rtol=0, atol=0)

    def test_cpu_prefetch_dynamic_outputs_reuse_without_cpu_synchronization(self):
        from rtp_llm.models_py.modules.dsv41.engram import EngramCPULookupPrefetch

        model = self._cpu_prefetch_model(16)
        runtime = model._engram_cpu_runtime
        pointers = [slot.host.data_ptr() for slot in runtime.slots.values()]
        self.assertTrue(
            all(not hasattr(slot, "output") for slot in runtime.slots.values())
        )
        consumers = [torch.cuda.Stream(), torch.cuda.Stream()]
        hidden = torch.randn(16, 4, 128, dtype=torch.bfloat16, device="cuda")
        references, actual = [], []
        for iteration, token in enumerate((7, 11, 3, 7)):
            windows = torch.full((16, 4), token, dtype=torch.int32)
            hashes = self.hash(windows.cuda())
            expected = []
            for layer_id, index in model._engram_layers:
                layer = model.v4.layers[layer_id]
                layer.engram_hashes = hashes[:, index].contiguous()
                expected.append(layer.engram(hidden, layer.engram_hashes))
            references.append(expected)
            consumer = consumers[iteration % 2]
            consumer.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(consumer):
                plan = EngramCPULookupPrefetch(model, windows, 9)
                torch.cuda._sleep(2_000_000)
                actual.append(
                    [
                        model.v4.layers[layer_id].engram(
                            hidden, model.v4.layers[layer_id].engram_hashes
                        )
                        for layer_id, _ in model._engram_layers
                    ]
                )
                # Pressure the owning stream's allocator after WKV released
                # the lookup outputs, while the kernels may still be running.
                pressure = torch.empty((16, 384), device="cuda", dtype=torch.bfloat16)
                pressure.fill_(float(iteration))
                del pressure
            # Deliberately no device synchronization between requests. Host
            # reuse waits for H2D; device reuse follows its allocation stream.
            plan.clear()
        torch.cuda.synchronize()
        for outputs, expected in zip(actual, references):
            for output, reference in zip(outputs, expected):
                torch.testing.assert_close(output, reference, rtol=0, atol=0)
        self.assertEqual(
            pointers,
            [slot.host.data_ptr() for slot in runtime.slots.values()],
        )

    @torch.inference_mode()
    def test_cpu_prefetch_outputs_release_after_each_consumer_and_cross_stream_fallback(
        self,
    ):
        from rtp_llm.models_py.modules.dsv41.engram import EngramCPULookupPrefetch

        model = self._cpu_prefetch_model(128)
        for count in (16, 128, 1, 7):
            windows = torch.full((count, 4), 7, dtype=torch.int32)
            hashes = self.hash(windows.cuda())
            hidden = torch.randn(count, 4, 128, dtype=torch.bfloat16, device="cuda")
            for layer_id, index in model._engram_layers:
                model.v4.layers[layer_id].engram_hashes = hashes[:, index].contiguous()
            plan = EngramCPULookupPrefetch(model, windows, 9)
            references = []
            for module, work in plan.work:
                rows = work.future.result()
                self.assertEqual(rows.shape, (count, 384))
                references.append(weakref.ref(rows))
                del rows
            # Ensure the executor has also dropped the last completed WorkItem
            # argument; the request plan must be the remaining output owner.
            model._engram_cpu_runtime.executor.submit(lambda: None).result()
            for index, (layer_id, _) in enumerate(model._engram_layers):
                layer = model.v4.layers[layer_id]
                result = layer.engram(hidden, layer.engram_hashes)
                self.assertIsNone(plan.work[index][1].future)
                self.assertIsNone(references[index]())
                del result
            plan.clear()

        # A different consumer stream cannot use stream-local allocator reuse.
        # It drains the prefetched copy and falls back to the original lookup.
        expected = model.v4.layers[1].engram(hidden, model.v4.layers[1].engram_hashes)
        plan = EngramCPULookupPrefetch(model, windows, 9)
        consumer = torch.cuda.Stream()
        consumer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(consumer):
            actual = model.v4.layers[1].engram(hidden, model.v4.layers[1].engram_hashes)
        self.assertFalse(plan.work[0][1].consumed)
        plan.clear()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_cpu_prefetch_target_shape_allocates_only_actual_rows(self):
        from rtp_llm.models_py.modules.dsv41.engram import _EngramCPULookupSlot

        # Exercise the real allocation method at production H=24, D=256
        # without loading its 94 GiB host table or running a lookup benchmark.
        slots = []
        for _ in range(2):
            slot = _EngramCPULookupSlot.__new__(_EngramCPULookupSlot)
            slot.device, slot.width, slot.capacity = torch.device("cuda:0"), 6144, 32768
            slot.allocated = torch.cuda.Event()
            slot.allocated.record(torch.cuda.current_stream())
            slots.append(slot)
        before = torch.cuda.memory_allocated()
        outputs = [slot.allocate_output(16384)[0] for slot in slots]
        self.assertEqual(torch.cuda.memory_allocated() - before, 384 * 1024**2)
        self.assertEqual([row.shape for row in outputs], [(16384, 6144)] * 2)
        del outputs
        self.assertEqual(torch.cuda.memory_allocated(), before)

    @torch.inference_mode()
    def test_cpu_prefetch_wkv_partial_failure_preserves_queued_reads(self):
        from rtp_llm.models_py.modules.dsv41.engram import EngramCPULookupPrefetch

        model = self._cpu_prefetch_model(16)
        windows = torch.full((16, 4), 7, dtype=torch.int32)
        hashes = self.hash(windows.cuda())
        hidden = torch.randn(16, 4, 128, dtype=torch.bfloat16, device="cuda")
        for layer_id, index in model._engram_layers:
            model.v4.layers[layer_id].engram_hashes = hashes[:, index].contiguous()
        layer = model.v4.layers[1]
        expected = self.embedding(layer.engram_hashes, hidden.device).flatten(-2)
        reads = []

        def failing_wkv(rows):
            torch.cuda._sleep(2_000_000)
            reads.append(rows.clone())
            raise RuntimeError("partial WKV submission")

        plan = EngramCPULookupPrefetch(model, windows, 9)
        with patch.object(layer.engram.wkv, "forward", side_effect=failing_wkv):
            with self.assertRaisesRegex(RuntimeError, "partial WKV submission"):
                layer.engram(hidden, layer.engram_hashes)
        self.assertIsNone(plan.work[0][1].future)
        plan.clear()
        pressure = torch.empty_like(expected)
        pressure.zero_()
        torch.cuda.synchronize()
        torch.testing.assert_close(reads[0], expected, rtol=0, atol=0)

    def test_cpu_prefetch_abandonment_and_capacity_capture_fallback(self):
        from rtp_llm.models_py.modules.dsv41.engram import EngramCPULookupPrefetch

        model = self._cpu_prefetch_model(16)
        for token in (7, 11):
            windows = torch.full((16, 4), token, dtype=torch.int32)
            hashes = self.hash(windows.cuda())
            for layer_id, index in model._engram_layers:
                model.v4.layers[layer_id].engram_hashes = hashes[:, index].contiguous()
            plan = EngramCPULookupPrefetch(model, windows, 9)
            plan.clear()
            self.assertTrue(
                all(
                    model.v4.layers[i].engram._cpu_lookup_work is None
                    for i, _ in model._engram_layers
                )
            )
        for windows in (
            windows.cuda(),
            torch.zeros(0, 4, dtype=torch.int32),
            torch.zeros(17, 4, dtype=torch.int32),
        ):
            plan = EngramCPULookupPrefetch(model, windows, 9)
            self.assertFalse(plan.work)
            plan.clear()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan = EngramCPULookupPrefetch(
                model, torch.zeros(16, 4, dtype=torch.int32), 9
            )
            self.assertFalse(plan.work)
            plan.clear()
            sentinel = torch.ones(1, device="cuda")
        graph.replay()
        self.assertEqual(sentinel.item(), 1)

    @torch.inference_mode()
    def test_cpu_prefetch_deadline_abandons_late_lookup_and_recovers_next_request(self):
        from rtp_llm.models_py.modules.dsv41 import engram_cpu
        from rtp_llm.models_py.modules.dsv41.engram import EngramCPULookupPrefetch

        model = self._cpu_prefetch_model(16, deadline=True)
        runtime = model._engram_cpu_runtime
        windows = torch.full((16, 4), 7, dtype=torch.int32)
        hashes = self.hash(windows.cuda())
        hidden = torch.randn(16, 4, 128, dtype=torch.bfloat16, device="cuda")
        for layer_id, index in model._engram_layers:
            model.v4.layers[layer_id].engram_hashes = hashes[:, index].contiguous()
        layer = model.v4.layers[1]
        expected = layer.engram(hidden, layer.engram_hashes)
        torch.cuda.current_stream().synchronize()

        entered, release = threading.Event(), threading.Event()
        native = engram_cpu.lookup_host_rows_cpu

        def late_native(*args, **kwargs):
            entered.set()
            if not release.wait(10):
                raise RuntimeError("test failed to release native lookup")
            return native(*args, **kwargs)

        plan = None
        with (
            patch.object(engram_cpu, "lookup_host_rows_cpu", side_effect=late_native),
            patch.object(
                runtime.slots[1], "_copy", wraps=runtime.slots[1]._copy
            ) as copy,
        ):
            try:
                plan = EngramCPULookupPrefetch(model, windows, 9)
                self.assertTrue(entered.wait(5))
                # Finish only the GPU predecessors/allocation events. Native
                # lookup is blocked before H2D, so this cannot wait for it.
                torch.cuda.current_stream().synchronize()
                self.assertTrue(torch.cuda.current_stream().query())
                work = plan.work[0][1]
                actual = layer.engram(hidden, layer.engram_hashes)
                self.assertFalse(release.is_set())
                self.assertTrue(work.abandoned)
                self.assertFalse(work.consumed)
                self.assertTrue(work.future.running())
                copy.assert_not_called()
                self.assertFalse(runtime.reap())
                # A still-running abandoned lookup must not queue another
                # batch's hashes or permit early pinned-buffer reuse.
                for _ in range(3):
                    skipped = EngramCPULookupPrefetch(model, windows, 9)
                    self.assertIsNone(skipped.hashes)
                    self.assertEqual(skipped.work, [])
                    skipped.clear()
                self.assertEqual(len(runtime.retired), 1)
            finally:
                # Release before drain/close, including any assertion failure.
                release.set()
                if plan is not None:
                    plan.clear()
                runtime.executor.submit(lambda: None).result(timeout=10)
                runtime.reap()
            copy.assert_not_called()

        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertIsNone(work.future)
        self.assertEqual(runtime.retired, [])

        # Reuse the GPU hash tensor identities with different token values;
        # the next request must receive new CPU rows, never the retired result.
        windows.fill_(11)
        fresh_hashes = self.hash(windows.cuda())
        for layer_id, index in model._engram_layers:
            model.v4.layers[layer_id].engram_hashes.copy_(fresh_hashes[:, index])
        expected = layer.engram(hidden, layer.engram_hashes)
        fresh = EngramCPULookupPrefetch(model, windows, 9)
        try:
            self.assertEqual(len(fresh.work), 2)
            for _, new_work in fresh.work:
                new_work.future.result(timeout=10)
            actual = layer.engram(hidden, layer.engram_hashes)
            self.assertTrue(fresh.work[0][1].consumed)
            self.assertFalse(fresh.work[0][1].abandoned)
        finally:
            fresh.clear()
            runtime.executor.submit(lambda: None).result(timeout=10)
            runtime.reap()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

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
        with patch.dict(
            os.environ,
            {"DSV41_ASYNC_ENGRAM_LOOKUP": "1", "DSV41_CPU_ENGRAM_LOOKUP": "0"},
        ):
            from cuda.bindings import driver

            from rtp_llm.models_py.modules.dsv41 import _engram_triton
            from rtp_llm.models_py.modules.dsv41.dsv41_kernel_jit_warmup import (
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

        from rtp_llm.models_py.modules.dsv41 import _engram_triton

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
