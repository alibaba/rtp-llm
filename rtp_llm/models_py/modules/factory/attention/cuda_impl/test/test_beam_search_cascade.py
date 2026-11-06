import gc
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel

import torch
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

from rtp_llm.config.quant_config import DEFAULT_FP8_BLOCK_WISE_QUANT_CONFIG
from rtp_llm.models_py.modules.factory.attention.attn_factory import AttnImplFactory
from rtp_llm.models_py.modules.factory.attention.cuda_impl.beam_search_cascade import (
    BeamSearchCascadeAttention,
    PyFlashinferCascadeDecodeAttnOp,
    PyFlashinferCascadeDecodeImpl,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    _compact_graph_workspace,
)


class BeamSearchCascadeTest(unittest.TestCase):
    def test_quantized_models_select_fp32_cascade_partials(self):
        model = SimpleNamespace(
            getAttentionConfigs=lambda tp: SimpleNamespace(use_mla=False),
            quant_config=DEFAULT_FP8_BLOCK_WISE_QUANT_CONFIG,
            max_seq_len=512,
            headwise_config=None,
        )
        parallel = SimpleNamespace(get_attn_tp_size=lambda: 1)
        inputs = SimpleNamespace(is_prefill=False)
        fallback = Mock(return_value=sentinel.reference_attention)
        cascade = Mock(return_value=sentinel.precise_cascade)
        cascade.support.return_value = True
        cascade.support_parallelism_config.return_value = True
        with (
            patch.dict(AttnImplFactory.FMHA_IMPL_REGISTRY, {"mha": fallback}),
            patch(
                "rtp_llm.models_py.modules.factory.attention.cuda_impl.beam_search_cascade.PyFlashinferCascadeDecodeImpl",
                cascade,
            ),
        ):
            result = AttnImplFactory.get_fmha_impl(
                model,
                parallel,
                None,
                inputs,
                is_cuda_graph=True,
                decode_cascade_max_requests=4096,
            )
        self.assertIs(result, sentinel.precise_cascade)
        cascade.support.assert_called_once()
        cascade.assert_called_once()
        self.assertTrue(cascade.call_args.kwargs["use_fp32_partials"])
        fallback.assert_not_called()

    def test_fp32_partials_preserve_cancelling_prefix_and_tail(self):
        batch, heads, kv_heads, dim, page, history = 32, 4, 2, 128, 16, 128
        prefix_pages = history // page
        configs = SimpleNamespace(
            head_num=heads,
            kv_head_num=kv_heads,
            size_per_head=dim,
            kernel_tokens_per_block=page,
            dtype=torch.bfloat16,
        )
        table = torch.empty(batch, prefix_pages + 1, dtype=torch.int32, pin_memory=True)
        table[:, :prefix_pages] = torch.arange(prefix_pages, dtype=torch.int32)
        table[:, prefix_pages] = torch.arange(batch, dtype=torch.int32) + prefix_pages
        lengths = torch.full((batch,), history, dtype=torch.int32, pin_memory=True)
        inputs = SimpleNamespace(
            input_lengths=lengths,
            sequence_lengths=lengths,
            kv_cache_kernel_block_id=table,
            is_cuda_graph=True,
        )
        kv = torch.zeros(
            batch + prefix_pages,
            2,
            kv_heads,
            page,
            dim,
            dtype=configs.dtype,
            device="cuda",
        )
        # The shared prefix has mean 1.00390625, halfway between BF16 values.
        # Rounding it before merging the negative tail loses about 0.002 even
        # though the final result, close to zero, has much finer BF16 spacing.
        kv[:prefix_pages, 1].fill_(1.0)
        kv[:prefix_pages, 1, :, 1::2] = 1.0078125
        kv[prefix_pages:, 1].fill_(-1.0)
        kv[prefix_pages:, 0, :, 0, 0] = math.log(history) * math.sqrt(dim)
        q = torch.zeros(batch, heads, dim, dtype=configs.dtype, device="cuda")
        q[:, :, 0] = 1.0
        op = PyFlashinferCascadeDecodeAttnOp(
            configs,
            inputs,
            max_requests=4096,
            max_seq_len=history + page,
            use_fp32_partials=True,
        )
        op.prepare_for_cuda_graph_replay(inputs)
        self.assertGreater(op.cascade.shared_tokens, 0)
        for _ in range(3):
            op.cascade.run(q, kv)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = op.cascade.run(q, kv)
        graph.replay()
        torch.cuda.synchronize()

        keys = kv[table[0].long().cuda(), 0, 0].flatten(0, 1)[: history + 1].double()
        values = kv[table[0].long().cuda(), 1, 0].flatten(0, 1)[: history + 1].double()
        scores = keys @ q[0, 0].double() / math.sqrt(dim)
        reference = (scores.softmax(0)[:, None] * values).sum(0)
        torch.testing.assert_close(
            actual.double(),
            reference.expand_as(actual),
            atol=2e-5,
            rtol=0.0,
        )
        rounded = BeamSearchCascadeAttention(
            batch,
            heads,
            kv_heads,
            dim,
            page,
            4096,
            prefix_pages + 1,
            q.dtype,
            False,
            torch.zeros(128 * 1024**2, dtype=torch.uint8, device="cuda"),
        )
        rounded.plan(table, lengths)
        rounded_output = rounded.run(q, kv)
        self.assertGreater(
            (rounded_output.double() - reference).abs().max().item(), 1e-3
        )

        # Nonuniform prefix logits put one probability near a BF16 midpoint.
        # Opposite V signs expose the low probability bits instead of letting
        # a common V cancel their error in the normalization denominator.
        kv[:prefix_pages, 0, :, 1::2, 0] = math.log(512 / 511) * math.sqrt(dim)
        kv[:prefix_pages, 1].fill_(16.0)
        kv[:prefix_pages, 1, :, 1::2] = -16.0
        kv[prefix_pages:, 1].fill_(1 / 64)
        graph.replay()
        torch.cuda.synchronize()
        keys = kv[table[0].long().cuda(), 0, 0].flatten(0, 1)[: history + 1].double()
        values = kv[table[0].long().cuda(), 1, 0].flatten(0, 1)[: history + 1].double()
        scores = keys @ q[0, 0].double() / math.sqrt(dim)
        reference = (scores.softmax(0)[:, None] * values).sum(0)
        torch.testing.assert_close(
            actual.double(), reference.expand_as(actual), atol=2e-5, rtol=0.0
        )

    def test_fp32_cascade_replays_layout_and_group_changes(self):
        torch.manual_seed(20261104)
        batch, kv_heads, history = 64, 2, 130
        for group, dim, dtype, page in (
            (2, 64, torch.bfloat16, 4),
            (1, 64, torch.bfloat16, 16),
            (2, 128, torch.bfloat16, 16),
            (4, 128, torch.float16, 16),
            (8, 256, torch.bfloat16, 64),
            (16, 64, torch.bfloat16, 16),
            (32, 256, torch.bfloat16, 64),
        ):
            with self.subTest(group=group, dim=dim, dtype=dtype, page=page):
                heads, prefix_pages = group * kv_heads, history // page
                configs = SimpleNamespace(
                    head_num=heads,
                    kv_head_num=kv_heads,
                    size_per_head=dim,
                    kernel_tokens_per_block=page,
                    dtype=dtype,
                )
                lengths = torch.full(
                    (batch,), history, dtype=torch.int32, pin_memory=True
                )
                table = torch.empty(
                    batch, prefix_pages + 1, dtype=torch.int32, pin_memory=True
                )
                table[:, :prefix_pages] = torch.arange(prefix_pages, dtype=torch.int32)
                table[:, prefix_pages] = prefix_pages + torch.arange(
                    batch, dtype=torch.int32
                )
                kv = torch.randn(
                    prefix_pages + 2 * batch,
                    2,
                    kv_heads,
                    page,
                    dim,
                    dtype=dtype,
                    device="cuda",
                )
                live_tail = (history + 1) % page
                kv[prefix_pages : prefix_pages + batch, 1, :, live_tail:] = float("nan")
                packed = torch.randn(
                    batch, heads + 2 * kv_heads, dim, dtype=dtype, device="cuda"
                )
                q = packed[:, :heads]
                inputs = SimpleNamespace(
                    input_lengths=lengths,
                    sequence_lengths=lengths,
                    kv_cache_kernel_block_id=table,
                    is_cuda_graph=True,
                )
                op = PyFlashinferCascadeDecodeAttnOp(
                    configs,
                    inputs,
                    max_requests=4096,
                    max_seq_len=4096 * page,
                    use_fp32_partials=True,
                )
                op.cascade.table.fill_(-1)
                op.prepare_for_cuda_graph_replay(inputs)
                self.assertGreater(op.cascade.shared_tokens, 0)
                for _ in range(3):
                    op.cascade.run(q, kv)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = op.cascade.run(q, kv)
                pointers = tuple(
                    t.data_ptr()
                    for t in (
                        op.cascade.table,
                        op.cascade.lengths,
                        op.cascade.cuts,
                        op.cascade.prefix_qptr,
                        op.cascade.row_cuts,
                        actual,
                    )
                )
                for private in (False, True):
                    if private:
                        # Break every shared history without changing graph
                        # addresses or launch capacity. The former partial
                        # tail's poisoned padding remains outside valid KV.
                        table[:, 0] = (
                            prefix_pages
                            + batch
                            + torch.arange(batch, dtype=torch.int32)
                        )
                    op.prepare_for_cuda_graph_replay(inputs)
                    self.assertEqual(op.cascade.shared_tokens == 0, private)
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assertTrue(torch.isfinite(actual).all().item())
                    self.assertEqual(
                        pointers,
                        tuple(
                            t.data_ptr()
                            for t in (
                                op.cascade.table,
                                op.cascade.lengths,
                                op.cascade.cuts,
                                op.cascade.prefix_qptr,
                                op.cascade.row_cuts,
                                actual,
                            )
                        ),
                    )
                    for row in (0, batch // 2, batch - 1):
                        data = kv[table[row].long().cuda()].permute(1, 2, 0, 3, 4)
                        data = data.flatten(2, 3)[:, :, : history + 1].double()
                        k, v = (t.repeat_interleave(group, 0) for t in data)
                        scores = torch.einsum(
                            "hd,hld->hl", q[row].double(), k
                        ) / math.sqrt(dim)
                        reference = torch.einsum("hl,hld->hd", scores.softmax(-1), v)
                        torch.testing.assert_close(
                            actual[row].double(), reference, atol=0.004, rtol=0.02
                        )

    def test_fp32_cascade_buckets_share_graph_pool(self):
        torch.manual_seed(20261104)
        buckets = tuple(range(512, 399, -16))
        heads, kv_heads, dim, page, history = 16, 8, 128, 16, 128
        configs = SimpleNamespace(
            head_num=heads,
            kv_head_num=kv_heads,
            size_per_head=dim,
            kernel_tokens_per_block=page,
            dtype=torch.bfloat16,
        )
        q = torch.randn(buckets[0], heads, dim, dtype=configs.dtype, device="cuda")
        kv = torch.randn(
            buckets[0] + 16,
            2,
            kv_heads,
            page,
            dim,
            dtype=configs.dtype,
            device="cuda",
        )
        pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        captures = []
        torch.cuda.synchronize()
        # graph() also clears ordinary allocator caches before capture. Do it
        # before the baseline so released storage cannot hide pool growth.
        gc.collect()
        torch.cuda.empty_cache()
        allocated_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()
        for batch in buckets:
            table = torch.empty(batch, 17, dtype=torch.int32, pin_memory=True)
            table[:, :16] = torch.arange(16, dtype=torch.int32)
            table[:, 16] = 16 + torch.arange(batch, dtype=torch.int32)
            lengths = torch.full((batch,), history, dtype=torch.int32, pin_memory=True)
            inputs = SimpleNamespace(
                input_lengths=lengths,
                sequence_lengths=lengths,
                kv_cache_kernel_block_id=table,
                is_cuda_graph=True,
            )
            op = PyFlashinferCascadeDecodeAttnOp(
                configs,
                inputs,
                max_requests=4096,
                max_seq_len=page * table.shape[1],
                use_fp32_partials=True,
            )
            op.prepare_for_cuda_graph_replay(inputs)
            result = torch.empty_like(q[:batch])
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    op.cascade.run(q[:batch], kv)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            # Match the runner: descending captures, one pool/stream, and a
            # copy into a persistent result before temporary references die.
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                result.copy_(op.cascade.run(q[:batch], kv))
            captures.append((op, inputs, graph, result))

        torch.cuda.synchronize()
        retained = torch.cuda.memory_allocated() - allocated_before
        largest_temporaries = buckets[0] * heads * (4 * (dim + 1) * 4 + dim * 2)
        persistent_results = sum(buckets) * heads * dim * 2
        self.assertLess(retained, 2 * largest_temporaries + persistent_results)
        # Also bound storage retained by the Graph pool after local Tensor
        # references die. Allow the ordinary warmup allocator its own scratch
        # and segment rounding; active allocations alone would miss this.
        self.assertLess(
            torch.cuda.memory_reserved() - reserved_before,
            4 * largest_temporaries + persistent_results,
        )

        # Intermediate states never escape a graph and each replay initializes
        # its own state, so buckets can alternate without retaining old data.
        for index, private, replay_history in (
            (7, False, 0),
            (0, False, 257),
            (4, True, 129),
            (1, False, 127),
            (7, False, 128),
            (0, False, 129),
        ):
            op, inputs, graph, result = captures[index]
            table = inputs.kv_cache_kernel_block_id
            table[:, 0] = table[:, 16] if private else 0
            inputs.sequence_lengths.fill_(replay_history)
            if replay_history == 257:
                inputs.sequence_lengths[0] = 0
                inputs.sequence_lengths[result.shape[0] // 2] = 129
            kv[:, 1].mul_(-0.75)
            op.prepare_for_cuda_graph_replay(inputs)
            self.assertEqual(
                op.cascade.shared_tokens == 0, private or replay_history < 128
            )
            graph.replay()
            torch.cuda.synchronize()
            for row in (0, result.shape[0] // 2, result.shape[0] - 1):
                data = kv[table[row].long().cuda()].permute(1, 2, 0, 3, 4)
                length = int(inputs.sequence_lengths[row]) + 1
                data = data.flatten(2, 3)[:, :, :length].double()
                k, v = (t.repeat_interleave(heads // kv_heads, 0) for t in data)
                scores = torch.einsum("hd,hld->hl", q[row].double(), k) / math.sqrt(dim)
                reference = torch.einsum("hl,hld->hd", scores.softmax(-1), v)
                torch.testing.assert_close(
                    result[row].double(), reference, atol=0.004, rtol=0.02
                )

    def test_large_request_capacity_grows_workspace_before_capture(self):
        torch.manual_seed(20261104)
        batch, heads, kv_heads, dim, page, max_pages = 4096, 16, 8, 128, 16, 256
        configs = SimpleNamespace(
            head_num=heads,
            kv_head_num=kv_heads,
            size_per_head=dim,
            kernel_tokens_per_block=page,
            dtype=torch.bfloat16,
        )
        table = torch.zeros(batch, max_pages, dtype=torch.int32, pin_memory=True)
        lengths = torch.zeros(batch, dtype=torch.int32, pin_memory=True)
        inputs = SimpleNamespace(
            input_lengths=lengths,
            sequence_lengths=lengths,
            kv_cache_kernel_block_id=table,
            is_cuda_graph=True,
        )
        op = PyFlashinferCascadeDecodeAttnOp(
            configs, inputs, max_requests=batch, max_seq_len=page * max_pages
        )
        # The initial all-empty graph plan still reserves its maximum launch
        # shape. A fixed 128 MiB buffer overflows with this request capacity.
        self.assertEqual(op.workspace.numel(), 128 * 1024**2)
        op.prepare_for_cuda_graph_replay(inputs)
        self.assertGreater(op.workspace.numel(), 128 * 1024**2)
        workspace_ptr = op.workspace.data_ptr()

        table[:, :8] = torch.arange(8, dtype=torch.int32)
        table[:, 8] = torch.arange(batch, dtype=torch.int32) + 8
        lengths.fill_(128)
        op.prepare_for_cuda_graph_replay(inputs)
        self.assertEqual(op.workspace.data_ptr(), workspace_ptr)
        self.assertGreater(op.cascade.shared_tokens, 0)
        q = torch.randn(batch, heads, dim, device="cuda", dtype=configs.dtype)
        kv = torch.randn(
            batch + 8, 2, kv_heads, page, dim, device="cuda", dtype=configs.dtype
        )
        for _ in range(3):
            op.cascade.run(q, kv)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = op.cascade.run(q, kv)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        reference = BatchDecodeWithPagedKVCacheWrapper(
            torch.zeros(128 * 1024**2, dtype=torch.uint8, device="cuda"), "HND"
        )
        reference.plan(
            torch.arange(batch + 1, dtype=torch.int32) * 9,
            table[:, :9].flatten().cuda(),
            lengths % page + 1,
            heads,
            kv_heads,
            dim,
            page,
            q_data_type=configs.dtype,
            kv_data_type=configs.dtype,
        )
        torch.testing.assert_close(output, reference.run(q, kv), atol=0.004, rtol=0.02)

    def test_small_graph_buckets_with_reused_workspace(self):
        torch.manual_seed(20261010)
        for batch in (128, 256, 512, 1024):
            with self.subTest(batch=batch):
                # Exercise dirty pool storage and different graph launch sizes.
                work = torch.full(
                    (128 * 1024**2,), 255, dtype=torch.uint8, device="cuda"
                )
                op = BeamSearchCascadeAttention(
                    batch, 16, 8, 128, 16, 8, 32, torch.bfloat16, True, work
                )
                table = torch.zeros(batch, 32, dtype=torch.int32, pin_memory=True)
                table[: batch // 2, :31] = torch.arange(31, dtype=torch.int32)
                table[batch // 2 :, :31] = torch.arange(31, dtype=torch.int32) + 32
                table[:, 31] = torch.arange(batch, dtype=torch.int32) + 64
                lengths = torch.full((batch,), 501, dtype=torch.int32, pin_memory=True)
                q = torch.randn(batch, 16, 128, dtype=torch.bfloat16, device="cuda")
                kv = torch.randn(
                    batch + 64, 2, 8, 16, 128, dtype=q.dtype, device="cuda"
                )
                op.plan(table, lengths)
                for _ in range(3):
                    op.run(q, kv)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    result = op.run(q, kv)
                for _ in range(10):
                    graph.replay()
                torch.cuda.synchronize()
                reference_work = torch.zeros_like(work)
                reference = BatchDecodeWithPagedKVCacheWrapper(reference_work, "HND")
                reference.plan(
                    torch.arange(batch + 1, dtype=torch.int32) * 32,
                    table.flatten().cuda(),
                    lengths % 16 + 1,
                    16,
                    8,
                    128,
                    16,
                    q_data_type=q.dtype,
                    kv_data_type=q.dtype,
                )
                torch.testing.assert_close(
                    result, reference.run(q, kv), atol=0.004, rtol=0.02
                )

    def test_dynamic_groups_reuse_one_graph(self):
        torch.manual_seed(20261010)
        batch, page, dim, max_pages = 2048, 16, 128, 33
        for heads, kv_heads, dtype in (
            (8, 8, torch.bfloat16),
            (16, 8, torch.bfloat16),
            (16, 8, torch.float16),
        ):
            with self.subTest(heads=heads, dtype=dtype):
                work = torch.empty(128 * 1024**2, dtype=torch.uint8, device="cuda")
                op = BeamSearchCascadeAttention(
                    batch,
                    heads,
                    kv_heads,
                    dim,
                    page,
                    8,
                    max_pages,
                    dtype,
                    True,
                    work,
                )
                q = torch.randn(batch, heads, dim, device="cuda", dtype=dtype)
                # Full prefix pages are shared; every beam owns two private pages.
                kv = torch.randn(
                    8 * max_pages + 2 * batch + 1,
                    2,
                    kv_heads,
                    page,
                    dim,
                    device="cuda",
                    dtype=dtype,
                )
                normal = BatchDecodeWithPagedKVCacheWrapper(
                    torch.zeros_like(work), "HND", use_tensor_cores=False
                )
                graph = torch.cuda.CUDAGraph()
                cases = [
                    ([512] * 4, 500, "shared"),
                    ([512, 512, 512, 256, 256], 511, "shared"),
                    ([256] * 8, 512, "shared"),
                    ([512, 256], 257, "shared"),  # padded query rows
                    ([512] * 4, 31, "shared"),  # too-short prefix -> full tail
                    ([512] * 4, 500, "distinct"),
                    ([512] * 4, 500, "partial"),  # diverging later pages
                    ([512, 512, 512, 256, 256], 500, "mixed_lengths"),
                    ([512] * 4, 500, "shared"),  # original partition again
                ]
                for ci, (widths, history, layout) in enumerate(cases):
                    table = torch.zeros(
                        (batch, max_pages), dtype=torch.int32, pin_memory=True
                    )
                    lengths = torch.zeros(batch, dtype=torch.int32, pin_memory=True)
                    first = 0
                    shared = history // page
                    for group, width in enumerate(widths):
                        for row in range(first, first + width):
                            ids = (
                                torch.arange(shared, dtype=torch.int32)
                                + group * max_pages
                                + 1
                            )
                            table[row, :shared] = ids
                            table[row, shared] = 8 * max_pages + 2 * row + 1
                            if layout == "distinct":
                                table[row, 0] = 8 * max_pages + 2 * row + 2
                            elif layout == "partial":
                                table[row, 10] = 8 * max_pages + 2 * row + 2
                        lengths[first : first + width] = (
                            history - group * 21
                            if layout == "mixed_lengths"
                            else history
                        )
                        first += width
                    op.plan(table, lengths)
                    torch.cuda.synchronize()
                    if ci == 0:
                        work = _compact_graph_workspace(
                            op.prefix, work, heads, dim, (op.tail,)
                        )
                        self.assertLess(work.numel(), 128 * 1024**2)
                        self.assertEqual(
                            op.prefix._float_workspace_buffer.data_ptr(),
                            work.data_ptr(),
                        )
                        self.assertEqual(
                            op.tail._float_workspace_buffer.data_ptr(), work.data_ptr()
                        )
                        op._signature = None
                        op._schedule_key = None
                        op.plan(table, lengths)
                        for _ in range(3):
                            op.run(q, kv)
                        torch.cuda.synchronize()
                        with torch.cuda.graph(graph):
                            output = op.run(q, kv)
                    graph.replay()
                    torch.cuda.synchronize()
                    counts = (lengths + page) // page
                    ptr = torch.cat(
                        (torch.zeros(1, dtype=torch.int32), counts.cumsum(0).int())
                    )
                    indices = table[
                        torch.arange(max_pages)[None, :] < counts[:, None]
                    ].cuda()
                    normal.plan(
                        ptr,
                        indices,
                        lengths % page + 1,
                        heads,
                        kv_heads,
                        dim,
                        page,
                        q_data_type=dtype,
                        kv_data_type=dtype,
                    )
                    reference = normal.run(q, kv)
                    torch.testing.assert_close(output, reference, atol=0.004, rtol=0.02)
                    # Independent FP32 softmax reference for first/last valid query.
                    for row in (0, first - 1):
                        data = kv[table[row, : counts[row]].long()]
                        data = data.permute(1, 2, 0, 3, 4).reshape(2, kv_heads, -1, dim)
                        data = data[:, :, : int(lengths[row]) + 1].float()
                        k = data[0].repeat_interleave(heads // kv_heads, 0)
                        v = data[1].repeat_interleave(heads // kv_heads, 0)
                        weights = torch.einsum(
                            "hd,hld->hl", q[row].float(), k
                        ) / math.sqrt(dim)
                        expected = torch.einsum("hl,hld->hd", weights.softmax(-1), v)
                        torch.testing.assert_close(
                            output[row].float(), expected, atol=0.004, rtol=0.02
                        )
                    if layout == "distinct" or history < 128:
                        self.assertEqual(op.shared_tokens, 0)
                    else:
                        self.assertGreater(op.shared_tokens, 0)

    def test_cached_graph_schedule_refreshes_page_ids_and_replans_lengths(self):
        torch.manual_seed(20261030)
        batch, page, dim = 128, 16, 128
        work = torch.empty(128 * 1024**2, dtype=torch.uint8, device="cuda")
        op = BeamSearchCascadeAttention(
            batch, 16, 8, dim, page, 2, 32, torch.bfloat16, True, work
        )
        table = torch.empty(batch, 32, dtype=torch.int32, pin_memory=True)
        table[:, :31] = torch.arange(31, dtype=torch.int32)
        table[:, 31] = torch.arange(batch, dtype=torch.int32) + 64
        lengths = torch.full((batch,), 501, dtype=torch.int32, pin_memory=True)
        q = torch.randn(batch, 16, dim, dtype=torch.bfloat16, device="cuda")
        kv = torch.randn(512, 2, 8, page, dim, dtype=q.dtype, device="cuda")
        op.plan(table, lengths)
        for _ in range(3):
            op.run(q, kv)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = op.run(q, kv)
        graph.replay()
        torch.cuda.synchronize()
        initial = result.clone()
        reference = BatchDecodeWithPagedKVCacheWrapper(torch.zeros_like(work), "HND")
        with (
            patch.object(op.prefix, "plan", wraps=op.prefix.plan) as prefix_plan,
            patch.object(op.tail, "plan", wraps=op.tail.plan) as tail_plan,
        ):
            # Identical offsets/lengths, completely different prefix and tail pages.
            # The cached schedule must see both new index arrays on graph replay.
            table.add_(256)
            for changed_lengths in (False, True):
                if changed_lengths:
                    lengths.add_(1)
                op.plan(table, lengths)
                self.assertEqual(prefix_plan.call_count, int(changed_lengths))
                self.assertEqual(tail_plan.call_count, int(changed_lengths))
                graph.replay()
                torch.cuda.synchronize()
                reference.plan(
                    torch.arange(batch + 1, dtype=torch.int32) * 32,
                    table.flatten().cuda(),
                    lengths % page + 1,
                    16,
                    8,
                    dim,
                    page,
                    q_data_type=q.dtype,
                    kv_data_type=q.dtype,
                )
                torch.testing.assert_close(
                    result, reference.run(q, kv), atol=0.004, rtol=0.02
                )
                self.assertGreater(
                    (result.float() - initial.float()).abs().max().item(), 0.1
                )

    def test_rejects_page_capacity_overflow(self):
        work = torch.empty(128 * 1024**2, dtype=torch.uint8, device="cuda")
        op = BeamSearchCascadeAttention(
            32, 8, 8, 128, 16, 2, 8, torch.bfloat16, True, work
        )
        with self.assertRaisesRegex(ValueError, "captured capacity"):
            op.plan(
                torch.zeros(32, 8, dtype=torch.int32),
                torch.full((32,), 128, dtype=torch.int32),
            )


if __name__ == "__main__":
    unittest.main()
