"""Exact SM90 decode tail, eager validity and Graph replay contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.ple import (
    ple_conv_triton,
    ple_decode_fusion_triton,
    ple_gather_triton,
)
from rtp_llm.models_py.modules.qwen4_exp.ple.ple import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLELayer,
)
from rtp_llm.models_py.modules.qwen4_exp.ple.ple_decode_plan_triton import (
    try_plan_decode_cache,
)


class TailFixture:
    """Inject real-shaped post-gate BF16 tensors into production decode_step."""

    def __init__(self, batch):
        kw = {"device": "cuda", "dtype": torch.bfloat16}
        self.gated = torch.randn(batch, 1, 10240, **kw)
        self.normed = torch.randn(batch, 1, 10240, **kw)
        self.initial = torch.randn(batch, 9, 10272, **kw)[..., :10240]
        self.conv_weight = torch.randn(10240, 1, 4, **kw) * 0.1
        self.conv_dilation = 3
        self.hc_hidden_size = 10240
        self.norm_key = self.norm_query = self.norm_conv = None
        self.norm_eps = 1e-6

    def _project_key_value(self, *_args, **_kwargs):
        return None, None

    def _gate_projected_values(self, *_args, **_kwargs):
        return self.gated, self.normed

    def call(self):
        with patch.object(
            ple_decode_fusion_triton, "try_ple_decode_fusion", return_value=None
        ):
            return Qwen4ExpPLELayer.decode_step(self, self.gated, None, self.initial)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PLEDecodeFusionTest(unittest.TestCase):
    def setUp(self):
        if torch.version.hip or torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("SM90 qualification")
        torch.manual_seed(31914)

    def assert_exact(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertEqual(a.shape, b.shape)
            self.assertEqual(a.dtype, b.dtype)
            self.assertTrue(
                torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
            )

    def test_tail_output_and_all_nine_state_rows(self):
        for batch in (1, 16, 32, 64, 65, 128, 256):
            with self.subTest(batch=batch):
                fixture = TailFixture(batch)
                before = fixture.initial.clone()
                with patch.object(ple_conv_triton, "try_ple_conv", return_value=None):
                    expected = fixture.call()
                with patch.object(
                    ple_conv_triton, "try_ple_conv", wraps=ple_conv_triton.try_ple_conv
                ) as launch:
                    actual = fixture.call()
                    self.assertEqual(launch.call_count, 1)
                # Independently confirm this geometry qualifies; a fallback
                # inside decode_step must not satisfy the parity assertion.
                direct = ple_conv_triton.try_ple_conv(
                    fixture.gated,
                    fixture.normed,
                    fixture.initial,
                    fixture.conv_weight,
                    3,
                    roll_state=True,
                )
                self.assertIsNotNone(direct)
                self.assert_exact(actual, expected)
                self.assert_exact(direct, expected)
                self.assert_exact((fixture.initial,), (before,))

    def test_tail_graph_reads_new_values(self):
        fixture = TailFixture(3)
        fixture.call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = fixture.call()
        for _ in range(2):
            fixture.initial.normal_()
            fixture.normed.normal_()
            fixture.gated.normal_()
            with patch.object(ple_conv_triton, "try_ple_conv", return_value=None):
                expected = fixture.call()
            graph.replay()
            self.assert_exact(actual, expected)

    def test_tail_unqualified_device_keeps_original_route(self):
        fixture = TailFixture(3)
        with patch.object(ple_conv_triton, "try_ple_conv", return_value=None):
            expected = fixture.call()
        with patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)):
            self.assertIsNone(
                ple_conv_triton.try_ple_conv(
                    fixture.gated,
                    fixture.normed,
                    fixture.initial,
                    fixture.conv_weight,
                    3,
                    roll_state=True,
                )
            )
            actual = fixture.call()
        self.assert_exact(actual, expected)

    def test_packed_projection_residual_and_graph_weight_views(self):
        kw = {"device": "cuda", "dtype": torch.bfloat16}
        layer = Qwen4ExpPLELayer(
            SimpleNamespace(ngram_size=3),
            torch.randn(10240, 2560, **kw) * 0.02,
            torch.randn(2560, 2560, **kw) * 0.02,
            torch.randn(10240, 1, 4, **kw) * 0.1,
            *(torch.randn(10240, **kw) * 0.1 for _ in range(3)),
            hc_mult=4, hidden_size=2560, conv_kernel_size=4, norm_eps=1e-6,
        )
        self.assertTrue(layer.pack_decode_projections())
        self.assertEqual(
            layer.key_proj.untyped_storage().data_ptr(),
            layer.value_proj.untyped_storage().data_ptr(),
        )

        def reference(hidden, history, embeddings):
            with patch.object(layer, "_decode_kv_weight", None), patch.object(
                ple_decode_fusion_triton, "try_ple_decode_fusion", return_value=None
            ):
                output, state = layer.decode_step(
                    hidden, None, history, embeddings=embeddings
                )
            return hidden + output, state

        def actual(hidden, history, embeddings, *, reuse=False, read_blocks=None):
            return layer.decode_step(
                hidden, None, history, embeddings=embeddings, add_residual=True,
                reuse_conv_buffer=reuse,
                conv_read_blocks=read_blocks,
            )

        fused = ple_decode_fusion_triton.try_ple_decode_fusion

        def require_fusion(*args, **kwargs):
            result = fused(*args, **kwargs)
            self.assertIsNotNone(result)
            return result

        for batch in (1, 3, 16, 32, 64, 65, 128):
            with self.subTest(batch=batch):
                hidden = torch.randn(batch, 1, 10240, **kw)
                history = torch.randn(batch, 9, 10272, **kw)[..., :10240]
                embeddings = torch.randn(batch, 1, 2560, **kw)
                before = history.clone()
                expected = reference(hidden, history, embeddings)
                with patch.object(
                    ple_decode_fusion_triton, "try_ple_decode_fusion",
                    side_effect=require_fusion,
                ) as launch:
                    result = actual(hidden, history, embeddings)
                self.assertEqual(launch.call_count, 1)
                self.assert_exact(result, expected)
                if batch >= 16:
                    scratch = history.contiguous()
                    result = actual(hidden, scratch, embeddings, reuse=True)
                    self.assertIs(result[1], scratch)
                    self.assert_exact(result, expected)
                self.assert_exact((history,), (before,))

        for small_batch in (1, 4, 8, 15):
            with self.subTest(paged_batch=small_batch):
                pool = torch.randn(19, 9, 10240, **kw)
                read = torch.arange(small_batch, device=pool.device) // 2
                small_hidden = torch.randn(small_batch, 1, 10240, **kw)
                small_embeddings = torch.randn(small_batch, 1, 2560, **kw)
                torch.cuda.synchronize()
                with patch.object(
                    ple_decode_fusion_triton, "try_ple_decode_fusion",
                    side_effect=require_fusion,
                ):
                    actual(small_hidden, pool, small_embeddings, read_blocks=read)
                graph = torch.cuda.CUDAGraph()
                torch.cuda.synchronize()
                with torch.cuda.graph(graph):
                    paged = actual(small_hidden, pool, small_embeddings, read_blocks=read)
                for _ in range(2):
                    read.add_(5).remainder_(pool.shape[0])
                    pool.normal_()
                    saved = pool.clone()
                    expected = reference(small_hidden, pool.index_select(0, read), small_embeddings)
                    graph.replay()
                    self.assert_exact(paged, expected)
                    self.assert_exact((pool,), (saved,))
                with patch.object(
                    ple_decode_fusion_triton, "try_ple_decode_fusion", return_value=None
                ):
                    self.assert_exact(
                        actual(small_hidden, pool, small_embeddings, read_blocks=read),
                        expected,
                    )

        pool = torch.randn(2 * batch + 1, 9, 10240, **kw)
        read_ids = torch.arange(batch, device=pool.device) // 2
        write_ids = torch.arange(batch, device=pool.device) + batch
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            scratch = pool.index_select(0, read_ids)
            output, rolled = actual(hidden, scratch, embeddings, reuse=True)
            self.assertIs(rolled, scratch)
            pool.index_copy_(0, write_ids, rolled)
        for replay in range(2):
            # The model loader publishes these views for in-place weight updates.
            layer.key_proj[:256].normal_(std=0.02)
            layer.value_proj[:256].normal_(std=0.02)
            hidden.normal_()
            embeddings.normal_()
            pool.normal_()
            # Exercise refreshed page IDs, shared read pages and distinct writes.
            read_ids.add_(replay)
            write_ids.sub_(replay)
            expected_output, expected_rolled = reference(
                hidden, pool.index_select(0, read_ids), embeddings
            )
            expected_pool = pool.clone()
            expected_pool.index_copy_(0, write_ids, expected_rolled)
            graph.replay()
            self.assert_exact((output, pool), (expected_output, expected_pool))

    def test_paged_conv_graph_switches_between_direct_and_staged_writes(self):
        for batch in (16, 64, 128):
            with self.subTest(batch=batch):
                self._check_paged_conv_graph(batch)

    def _check_paged_conv_graph(self, batch):
        kw = {"device": "cuda", "dtype": torch.bfloat16}
        hidden = torch.randn(batch, 1, 10240, **kw)
        key = torch.randn_like(hidden)
        value = torch.randn(batch, 1, 2560, **kw)
        norms = tuple(torch.randn(10240, **kw) * 0.1 for _ in range(3))
        weight = torch.randn(10240, 1, 4, **kw) * 0.1
        pool = torch.randn(2 * batch + 1, 9, 10240, **kw)
        initial_context = torch.arange(pool.shape[0] * 2, device="cuda").view(-1, 2)
        context = initial_context.clone()
        lengths = torch.full((batch,), 65, device="cuda", dtype=torch.int32)
        ids = torch.arange(batch, device="cuda", dtype=torch.int64)
        reads = list(range(1, batch + 1))
        writes = list(range(batch + 1, 2 * batch + 1))
        table = torch.tensor(list(zip(reads, writes)), device="cuda", dtype=torch.int32)
        ctx_table = table.clone()
        mode = torch.empty(batch, device="cuda", dtype=torch.int32)
        fuse = ple_decode_fusion_triton.try_ple_decode_fusion

        def forward():
            read, write, ctx_write, history = try_plan_decode_cache(
                lengths, ids, table, ctx_table, pool, context, 64,
                graph_mode=True,
            )
            output, state = fuse(
                key, value, hidden, *norms, pool, weight, 1e-6, 3, add_residual=True,
                history_read_blocks=read, history_write_blocks=write,
                history_write_mode=mode, context_write=(context, ctx_write, history),
            )
            self.assertIsNone(state)
            return output

        forward()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = forward()

        scenarios = (
            (reads, writes, 65, False),
            ([reads[0], *reads[:-1]], writes, 65, False),  # shared read pages
            (reads, writes, 64, False),  # same-page updates, one owner per page
            (reads, [reads[1], reads[0], *writes[2:]], 65, True),  # page swap
            (reads, [reads[-1], *writes[1:]], 65, True),  # direct and staged in one batch
            (reads, [writes[1], *writes[1:]], 65, True),  # duplicate, no read hazard
            ([0, *reads[1:]], [0, 0, *writes[2:]], 65, True),  # dummy alias
            (reads, writes, 65, False),
        )
        for src, dst, length, staged in scenarios:
            with self.subTest(src=src[:2], dst=dst[:2], length=length):
                lengths.fill_(length)
                table.copy_(torch.tensor(
                    list(zip(src, dst)), device="cuda", dtype=torch.int32,
                ))
                ctx_table.copy_(table)
                if length == 64:
                    dst = src
                pool.normal_()
                context.copy_(initial_context)
                read_ids = torch.tensor(src, device="cuda")
                gathered = pool[read_ids]
                expected, rolled = fuse(
                    key, value, hidden, *norms, gathered, weight, 1e-6, 3,
                    add_residual=True, reuse_history=True,
                )
                expected_pool = pool.clone()
                expected_context = initial_context.clone()
                next_context = torch.stack((initial_context[read_ids, 1], ids), dim=1)
                # For colliding destinations only the final request commits.
                for row, page in enumerate(dst):
                    expected_pool[page].copy_(rolled[row])
                    expected_context[page].copy_(next_context[row])

                graph.replay()
                self.assert_exact(
                    (output, pool, context), (expected, expected_pool, expected_context)
                )
                self.assertEqual(bool((mode != 1).any().item()), staged)

    def test_inline_bf16_hash_gather_graph_matches_separate_kernels(self):
        eos = 151643
        shard_size = 256
        shards = [
            torch.randn(shard_size, 160, dtype=torch.bfloat16).pin_memory()
            for _ in range(4)
        ]
        embedding = Qwen4ExpNGramEmbedding(
            shards,
            torch.full((16,), 127, device="cuda", dtype=torch.int64),
            torch.arange(16, device="cuda", dtype=torch.int64) * 128,
            torch.tensor(
                [8895235927375, -6951273546288326311, 1753814622540031003],
                device="cuda", dtype=torch.int64,
            ),
            ngram_size=3, eos_token_id=eos, shard_indices=[0, 2, 5, 7],
            total_shards=8, distributed_reduce=True, inline_cpu_uva_graph=True,
        )
        with patch.object(embedding, "_reduce_rows", side_effect=lambda x: x.flatten(-2)):
            for batch in (16, 64, 128):
                with self.subTest(batch=batch), embedding.lookup_context(True):
                    history = torch.randint(eos, (batch, 3), device="cuda")
                    embedding(history, 1)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with patch.object(embedding, "_gather_local", side_effect=AssertionError):
                        with torch.cuda.graph(graph):
                            actual = embedding(history, 1)
                    for column in range(3):
                        history.random_(0, eos)
                        history[0, column] = eos
                        with patch.object(
                            ple_gather_triton, "try_gather_hashed_decode", return_value=None,
                        ):
                            expected = embedding(history, 1)
                        graph.replay()
                        self.assert_exact((actual,), (expected,))
                    del graph
        embedding.close()

    def test_cache_plan_cross_page_independent_tables(self):
        lengths = torch.tensor([63, 64, 65, 128], device="cuda", dtype=torch.int32)
        ids = torch.tensor([19, 23, 29, 31], device="cuda", dtype=torch.int32)
        s = torch.tensor(
            [[5, 0, 8, 0, 9, 0], [2, 0, 3, 0, 4, 0],
             [1, 0, 6, 0, 7, 0], [8, 0, 5, 0, 1, 0]],
            device="cuda", dtype=torch.int32,
        )[:, ::2]
        c = torch.tensor(
            [[11, 17, 19], [13, 18, 20], [21, 7, 9], [6, 10, 3]],
            device="cuda", dtype=torch.int32,
        )
        state_pool = torch.randn(11, 9, 10240, device="cuda", dtype=torch.bfloat16)
        context_pool = torch.arange(44, device="cuda", dtype=torch.int64).view(22, 2)
        before = state_pool.clone(), context_pool.clone()

        def plan(sequence_lengths=lengths):
            return try_plan_decode_cache(
                sequence_lengths, ids, s, c, state_pool, context_pool, 64
            )

        actual = plan()
        self.assertIsNotNone(actual)
        self.assertEqual(actual.state_read.tolist(), [5, 2, 6, 5])
        self.assertEqual(actual.state_write.tolist(), [5, 3, 6, 1])
        self.assertEqual(actual.context_write.tolist(), [11, 18, 7, 3])
        self.assertEqual(actual.history.tolist(), [[22, 23, 19], [26, 27, 23],
                                                   [14, 15, 29], [20, 21, 31]])
        pinned_lengths = lengths.cpu().pin_memory()
        host_plan = plan(pinned_lengths)
        self.assertIsNotNone(host_plan)
        self.assert_exact(host_plan, actual)
        self.assertIsNone(plan(lengths.cpu()))
        # Row 1 reads page 0 but writes page 1. Reject invalid writes before
        # even the valid request can consume this mixed-batch plan.
        for table, pool_rows in ((s, 11), (c, 22)):
            saved = int(table[1, 1].item())
            for bad_id in (0, -1, pool_rows):
                table[1, 1] = bad_id
                self.assertIsNone(plan())
            table[1, 1] = saved
        for bad_length in (0, -1, -(1 << 63), (1 << 63) - 1):
            invalid = lengths.long()
            invalid[-1] = bad_length
            self.assertIsNone(plan(invalid))
            self.assertIsNone(plan(invalid.cpu().pin_memory()))
        self.assert_exact((state_pool, context_pool), before)

    def test_graph_cache_plan_replay_updates_pages_and_state(self):
        def ints(values, dtype=torch.int32):
            return torch.tensor(values, device="cuda", dtype=dtype)

        # The two tags own different page IDs, shapes and strides.
        s = torch.zeros((4, 1028), device="cuda", dtype=torch.int32)[:, ::2]
        c = torch.zeros((4, 515), device="cuda", dtype=torch.int32)
        for row, page in enumerate((0, 127, 511, 0)):
            s[row, page : page + 2] = ints([2 * row + 1, 2 * row + 2])
            c[row, page : page + 2] = ints([2 * row + 11, 2 * row + 12])
        lengths_plus_one = ints([64, 8193, 32769, 0])
        ids = ints([19, 23, 29, 31], torch.int64)
        state = torch.randn(11, 9, 10240, device="cuda", dtype=torch.bfloat16)
        context = torch.arange(200, 244, device="cuda", dtype=torch.int64).view(22, 2)
        reference_state, reference_context = state.clone(), context.clone()
        current = torch.randn(4, 1, 10240, device="cuda", dtype=torch.bfloat16)

        def resolve(table, page, pool):
            index = table.gather(1, page.clamp(0, table.shape[1] - 1)[:, None])[:, 0]
            valid = (page >= 0) & (page < table.shape[1]) & (index > 0)
            return torch.where(valid & (index < pool.shape[0]), index, 0).long()

        def reference():
            raw = lengths_plus_one.long() - 1
            read_page = torch.div(raw - 1, 64, rounding_mode="floor")
            write_page = torch.div(raw, 64, rounding_mode="floor")
            sr = resolve(s, read_page, reference_state)
            sw = resolve(s, write_page, reference_state)
            cr = resolve(c, read_page, reference_context)
            cw = resolve(c, write_page, reference_context)
            history = torch.cat((reference_context.index_select(0, cr), ids[:, None]), 1)
            return sr, sw, cw, history

        def fused(lengths=lengths_plus_one):
            return try_plan_decode_cache(
                lengths, ids, s, c, state, context, 64, graph_mode=True
            )

        def update(pools, plan):
            state_pool, context_pool = pools
            sr, sw, cw, history = plan
            # Exercise the production gather and same two state destinations.
            rolled = torch.cat((state_pool.index_select(0, sr)[:, 1:], current), 1)
            state_pool.index_copy_(0, sw, rolled)
            context_pool.index_copy_(0, cw, history[:, -2:])

        self.assertIsNone(fused(lengths_plus_one.cpu().pin_memory()))
        self.assertIsNone(fused(lengths_plus_one.long()))
        warmup = fused()
        self.assertIsNotNone(warmup)
        update((state, context), warmup)
        state.copy_(reference_state)
        context.copy_(reference_context)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = fused()
            self.assertIsNotNone(actual)
            update((state, context), actual)
        state.copy_(reference_state)
        context.copy_(reference_context)

        replays = ([64, 8193, 32769, 0], [65, 8194, 32770, 1], [65, 8193, 32769, 2])
        for step, lengths in enumerate(replays):
            lengths_plus_one.copy_(ints(lengths))
            ids.add_(1)
            current.normal_()
            if step == 2:
                # Invalid destinations/source independently map to the dummy
                # row; the other tag and the other direction stay live.
                s[0, 1] = state.shape[0]
                s[1, 128] = 9
                c[2, 511] = -1
            expected = reference()
            update((reference_state, reference_context), expected)
            graph.replay()
            self.assert_exact(actual, expected)
            self.assert_exact((state, context), (reference_state, reference_context))


if __name__ == "__main__":
    unittest.main()
