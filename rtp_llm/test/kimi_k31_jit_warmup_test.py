"""Real CUDA checks for K3.1 scratch warmup and request-time JIT reuse."""

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch


class KimiK31JitWarmupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("K3.1 JIT regression requires a CUDA GPU")
        if torch.cuda.get_device_capability()[0] != 10:
            raise RuntimeError("This native MQA regression requires SM10")
        cls.temporary = tempfile.TemporaryDirectory(prefix="k31-jit-warmup-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        for name, leaf in (
            ("DG_JIT_CACHE_DIR", "native"),
            ("TRITON_CACHE_DIR", "triton"),
        ):
            previous = os.environ.get(name)
            os.environ[name] = str(cls.root / leaf)
            cls.addClassCleanup(cls.restore_environment, name, previous)
        from rtp_llm.models_py.modules.base.cuda.fp8_indexer_score import (
            NativeFp8IndexerScore,
        )
        from rtp_llm.models_py.modules.kimi_k3 import dsa_jit_warmup

        cls.warmup = dsa_jit_warmup
        cls.score = NativeFp8IndexerScore()
        cls.device = torch.device("cuda")

    @staticmethod
    def restore_environment(name, value):
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    def native_keys(self):
        return {
            (str(p.relative_to(self.root)), hashlib.sha256(p.read_bytes()).hexdigest())
            for p in (self.root / "native").rglob("*.cubin")
        }

    def test_00_native_mqa_cold_warmup_and_changed_request_shapes(self):
        self.assertFalse(self.native_keys(), "native test cache must start cold")
        self.warmup._warmup_native_score(self.score, self.device, 4)
        prepared = self.native_keys()
        names = [path for path, _ in prepared]
        for family in (
            "sm100_mqa_logits.",
            "sm100_paged_mqa_logits.",
            "sm100_paged_mqa_logits_metadata.",
        ):
            self.assertTrue(any(family in name for name in names), family)
        for rows, columns in ((3, 129), (7, 513)):
            query = torch.full(
                (rows, 32, 128), 0.5, device=self.device, dtype=torch.float8_e4m3fn
            )
            keys = torch.ones(
                (columns, 128), device=self.device, dtype=torch.float8_e4m3fn
            )
            scales = torch.ones(columns, device=self.device)
            weights = torch.ones((rows, 32), device=self.device)
            starts = torch.arange(rows, device=self.device, dtype=torch.int32)
            ends = torch.full_like(starts, columns - 1)
            positions = torch.arange(columns, device=self.device, dtype=torch.int32)
            result = self.score.dense(
                query, keys, scales, weights, starts, ends, positions
            )
            expected = torch.full((rows, columns), 2048.0, device=self.device)
            valid = (positions[None] >= starts[:, None]) & (
                positions[None] < ends[:, None]
            )
            expected.masked_fill_(~valid, -float("inf"))
            torch.testing.assert_close(result, expected, rtol=0, atol=0)

        for batch, queries, length in (
            (1, 1, 65),
            (3, 2, 129),
            (2, 3, 257),
            (3, 4, 513),
        ):
            pages = (length + 63) // 64
            columns = pages * 64
            cache = torch.empty(
                (1 + batch * pages, 64, 1, 132), device=self.device, dtype=torch.uint8
            )
            raw = cache.view(cache.shape[0], -1)
            raw[:, : 64 * 128].copy_(
                torch.ones(
                    (raw.shape[0], 64 * 128),
                    device=self.device,
                    dtype=torch.float8_e4m3fn,
                ).view(torch.uint8)
            )
            raw[:, 64 * 128 :].view(torch.float32).fill_(1)
            table = torch.arange(
                1, 1 + batch * pages, device=self.device, dtype=torch.int32
            ).view(batch, pages)
            context = torch.full(
                (batch, queries), length, device=self.device, dtype=torch.int32
            )
            context.sub_(queries - 1).add_(
                torch.arange(queries, device=self.device, dtype=torch.int32)
            )
            metadata = self.score.prepare_paged_metadata(
                context,
                num_sms=torch.cuda.get_device_properties(
                    self.device
                ).multi_processor_count,
            )
            query = torch.full(
                (batch, queries, 32, 128),
                0.5,
                device=self.device,
                dtype=torch.float8_e4m3fn,
            )
            positions = torch.arange(columns, device=self.device, dtype=torch.int32)
            result = self.score.paged(
                query,
                cache,
                torch.ones((batch * queries, 32), device=self.device),
                context,
                table,
                metadata,
                columns,
                positions,
            )
            expected = torch.full(
                (batch * queries, columns), 2048.0, device=self.device
            )
            expected.masked_fill_(
                positions[None] >= context.reshape(-1, 1), -float("inf")
            )
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
        torch.cuda.synchronize()
        self.assertEqual(
            self.native_keys(), prepared, "requests introduced native JIT keys"
        )

    def test_pack_restore_and_sanitize_reuse_across_shapes(self):
        import triton
        from rtp_llm.models_py.kernels.cuda.fp8_indexer_cache import (
            pack_indexer_pages64,
        )
        from rtp_llm.models_py.kernels.cuda.indexer_topk_keys import (
            sanitize_indexer_scores,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_page_rr_cache_gather import (
            MlaPageRRCacheGather,
            _restore_mla_page_rr_prefix,
        )

        # SFA byte transport and Indexer FP8 use different codecs.
        layouts = (
            (torch.bfloat16, 576, False),
            (torch.uint8, 656, False),
            (torch.uint8, 132, True),
        )
        for dtype, width, indexer_fp8 in layouts:
            seed = torch.empty((3, 64, width), device=self.device, dtype=dtype)
            self.warmup._warmup_prefix(seed, 128, 2, 0, indexer_fp8=indexer_fp8)
        self.warmup._warmup_native64(64, self.device)
        for starts in (None, torch.zeros(4, device=self.device, dtype=torch.int32)):
            sanitize_indexer_scores(
                torch.zeros((4, 64), device=self.device),
                torch.arange(64, device=self.device, dtype=torch.int32),
                starts=starts,
                ends=torch.full((4,), 64, device=self.device, dtype=torch.int32),
            )
        misses = []
        with triton.knobs.compilation.scope():

            def observe(*, cache_hit, src, **_):
                if not cache_hit:
                    misses.append(src.fn.__name__)

            triton.knobs.compilation.listener = observe
            for dtype, width, indexer_fp8 in layouts:
                for batch, length, columns in ((1, 1, 3), (3, 129, 5), (2, 257, 7)):
                    blocks = 1 + batch * columns
                    cache = torch.randint(
                        0,
                        16,
                        (blocks, 64, width),
                        device=self.device,
                        dtype=torch.int32,
                    ).to(dtype)
                    canonical = cache
                    if indexer_fp8:
                        flat = cache.view(blocks, -1)
                        canonical = torch.cat(
                            (
                                flat[:, : 64 * 128].view(blocks, 64, 128),
                                flat[:, 64 * 128 :].view(blocks, 64, 4),
                            ),
                            -1,
                        )
                    storage = torch.zeros(
                        (batch, columns + 2), device=self.device, dtype=torch.int32
                    )
                    table = storage[:, :columns]
                    table.copy_(
                        torch.arange(
                            1, blocks, device=self.device, dtype=torch.int32
                        ).view(batch, columns)
                    )
                    adapter = MlaPageRRCacheGather(128, 2, 0, 64)
                    descriptor = adapter.build_prefix_chunk_descriptor(
                        tuple(range(batch)),
                        (0,) * batch,
                        (length,) * batch,
                        feature_width=width,
                    )
                    for allow_zero in (False, True):
                        payload = adapter._pack_prefix_chunk(
                            cache,
                            table,
                            descriptor,
                            indexer_fp8=indexer_fp8,
                            allow_reserved_block_zero=allow_zero,
                        )
                        gathered = torch.zeros(
                            (2, *payload.shape), device=self.device, dtype=dtype
                        )
                        gathered[0].copy_(payload)
                        restored = _restore_mla_page_rr_prefix(
                            gathered, descriptor, indexer_fp8=indexer_fp8
                        )
                        expected = torch.zeros(
                            (batch, length, width), device=self.device, dtype=dtype
                        )
                        for position in range(length):
                            if position // 128 % 2 == 0:
                                column = position // 256 * 2 + position % 128 // 64
                                expected[:, position] = canonical[
                                    table[:, column].long(), position % 64
                                ]
                        expected = expected.reshape(-1, width)
                        if indexer_fp8:
                            expected = torch.cat(
                                (
                                    expected[:, :128].flatten(),
                                    expected[:, 128:].flatten(),
                                )
                            ).view_as(restored)
                        torch.testing.assert_close(restored, expected, rtol=0, atol=0)
            for batch, start, pages, columns in (
                (1, 0, 1, 3),
                (3, 64, 3, 7),
                (2, 192, 5, 9),
            ):
                blocks = 1 + batch * columns
                cache = torch.randint(
                    0, 256, (blocks, 64, 132), device=self.device, dtype=torch.uint8
                )
                table = torch.arange(
                    1, blocks, device=self.device, dtype=torch.int32
                ).view(batch, columns)
                output = torch.empty(
                    (1 + batch * pages, 64, 1, 132),
                    device=self.device,
                    dtype=torch.uint8,
                )
                pack_indexer_pages64(cache, table, start, output)
                expected = cache[
                    table[:, start // 64 : start // 64 + pages].long()
                ].reshape(-1, 64, 1, 132)
                self.assertTrue(torch.equal(output[1:], expected))
                self.assertEqual(output[0].count_nonzero().item(), 0)
            for rows, columns in ((3, 129), (7, 513)):
                positions = torch.arange(columns, device=self.device, dtype=torch.int32)
                for bounded_start in (False, True):
                    scores = torch.ones((rows, columns + 3), device=self.device)[
                        :, :columns
                    ]
                    ends = torch.full(
                        (rows,), columns - 1, device=self.device, dtype=torch.int32
                    )
                    starts = torch.ones_like(ends) if bounded_start else None
                    sanitize_indexer_scores(scores, positions, starts=starts, ends=ends)
                    expected = torch.ones_like(scores)
                    expected[:, -1] = -float("inf")
                    if bounded_start:
                        expected[:, 0] = -float("inf")
                    torch.testing.assert_close(scores, expected, rtol=0, atol=0)
            torch.cuda.synchronize()
        self.assertEqual(misses, [], "request geometry introduced Triton JIT keys")

    def test_target_verify_and_native_draft_metadata_reuse(self):
        import triton
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.page_rr_mla_metadata import (
            PageRRMlaMetadata,
        )

        for draft in (False, True):
            self.warmup._warmup_decode_metadata(
                64, 128, 2, 0, self.device, 4, draft=draft
            )
        misses = []
        with triton.knobs.compilation.scope():

            def observe(*, cache_hit, src, **_):
                if not cache_hit:
                    misses.append(src.fn.__name__)

            triton.knobs.compilation.listener = observe
            for batch, queries in ((3, 2), (2, 3), (3, 4)):
                for draft in (False, True):
                    for mask in (False, True) if draft else (False,):
                        inputs = SimpleNamespace(
                            input_lengths=torch.ones(
                                batch, device=self.device, dtype=torch.int32
                            ),
                            physical_token_count=batch * queries,
                            is_target_verify=not draft,
                            is_mtp_draft_update=draft,
                            is_cuda_graph=False,
                            prefix_lengths_device=torch.ones(
                                batch, device=self.device, dtype=torch.int32
                            ),
                            sequence_lengths_plus_1_device=torch.full(
                                (batch,), 2, device=self.device, dtype=torch.int32
                            ),
                            kv_cache_kernel_block_id_device=torch.ones(
                                (batch, 7), device=self.device, dtype=torch.int32
                            ),
                            cu_seqlens_device=torch.arange(
                                0,
                                (batch + 1) * queries,
                                queries,
                                device=self.device,
                                dtype=torch.int32,
                            ),
                            valid_token_mask=(
                                torch.ones(
                                    batch * queries,
                                    device=self.device,
                                    dtype=torch.bool,
                                )
                                if mask
                                else None
                            ),
                        )
                        for compact in (False, True):
                            metadata = PageRRMlaMetadata(128, 64, 2, 0)
                            metadata.prepare_decode(inputs, compact_queries=compact)
                            expected = torch.arange(
                                1, queries + 1, device=self.device, dtype=torch.int32
                            ).repeat(batch)
                            torch.testing.assert_close(
                                metadata.positions_d, expected, rtol=0, atol=0
                            )
                            torch.testing.assert_close(
                                metadata.local_causal_lens.reshape(-1),
                                expected + 1,
                                rtol=0,
                                atol=0,
                            )
            torch.cuda.synchronize()
        self.assertEqual(misses, [], "verify/draft update introduced metadata JIT keys")

    def test_grouped_norm_reuses_kernel_across_token_counts(self):
        import triton
        from rtp_llm.models_py.triton_kernels.common.layernorm_gated import (
            grouped_q_kv_rmsnorm,
        )

        qw = torch.rand(1536, device=self.device, dtype=torch.bfloat16) + 0.5
        kw = torch.rand(512, device=self.device, dtype=torch.bfloat16) + 0.5

        def run(rows, packed):
            # Match the strided views of QKV + 96 * 128 output-gate features.
            source = torch.randn(
                (rows, 14400), device=self.device, dtype=torch.bfloat16
            )
            q, kv, suffix = source[:, :2112].split((1536, 512, 64), dim=-1)
            actual = grouped_q_kv_rmsnorm(
                q, kv, qw, kw, 1e-5, kv_suffix=suffix if packed else None
            )
            for value, x, weight in zip(actual, (q, kv), (qw, kw)):
                reference = (
                    x.float()
                    * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
                    * weight.float()
                ).bfloat16()
                ulp = (
                    value.view(torch.int16).int() - reference.view(torch.int16).int()
                ).abs()
                self.assertLessEqual(int(ulp.max()), 1)
            if packed:
                storage = actual[1].as_strided((rows, 576), (576, 1))
                torch.testing.assert_close(storage[:, 512:], suffix, rtol=0, atol=0)

        for packed in (False, True):
            run(128, packed)
        misses = []
        with triton.knobs.compilation.scope():

            def observe(*, cache_hit, src, **_):
                if not cache_hit:
                    misses.append(src.fn.__name__)

            triton.knobs.compilation.listener = observe
            for rows in (129, 542, 8192):
                for packed in (False, True):
                    run(rows, packed)
            torch.cuda.synchronize()
        self.assertEqual(misses, [], "token count introduced grouped Norm JIT keys")

    def test_prefill_warmup_prepares_grouped_norm_layouts(self):
        import triton
        from rtp_llm.models_py.triton_kernels.common.layernorm_gated import (
            grouped_q_kv_rmsnorm,
        )

        attention = SimpleNamespace(
            _qkv_norm=grouped_q_kv_rmsnorm,
            prefill_query_cp=True,
            q_rank=1536,
            kv_rank=512,
            suffix_dim=64,
            heads=96,
            v_dim=128,
            q_norm=SimpleNamespace(
                weight=torch.ones(1536, device=self.device, dtype=torch.bfloat16),
                variance_epsilon=1e-6,
            ),
            kv_norm=SimpleNamespace(
                weight=torch.ones(512, device=self.device, dtype=torch.bfloat16)
            ),
        )
        model = SimpleNamespace(
            kv_cache=object(), layers=[SimpleNamespace(attention=attention)]
        )
        for dtype, width in ((torch.bfloat16, 14400), (torch.float8_e4m3fn, 14464)):
            with self.subTest(dtype=dtype):
                attention.input = SimpleNamespace(
                    weight=torch.empty((width, 7168), device="meta", dtype=dtype)
                )
                self.warmup.warmup_kimi_k31_dsa_jit(
                    model, SimpleNamespace(is_decode_role=False)
                )
                source = torch.ones(
                    (129, width), device=self.device, dtype=torch.bfloat16
                )
                misses = []
                with triton.knobs.compilation.scope():

                    def observe(*, cache_hit, src, **_):
                        if not cache_hit:
                            misses.append(src.fn.__name__)

                    triton.knobs.compilation.listener = observe
                    for rows in (1, 34, 58, 129):
                        q, kv, suffix = source[:rows, :2112].split(
                            (1536, 512, 64), dim=-1
                        )
                        for packed in (False, True):
                            actual = attention._qkv_norm(
                                q,
                                kv,
                                attention.q_norm.weight,
                                attention.kv_norm.weight,
                                attention.q_norm.variance_epsilon,
                                kv_suffix=suffix if packed else None,
                            )
                            for output in actual:
                                self.assertTrue(
                                    torch.equal(output, torch.ones_like(output))
                                )
                    torch.cuda.synchronize()
                self.assertEqual(
                    misses, [], "Prefill warmup missed grouped Norm layouts"
                )

    def test_prefill_slot_map_reuses_kernel_across_request_geometry(self):
        import triton
        from rtp_llm.models_py.kernels.cuda.dsa_kv_chunk import map_dsa_chunk_slots

        def run(rows, start, width, request_offset):
            requests = torch.arange(
                rows + request_offset, device=self.device, dtype=torch.int32
            ).remainder_(2)[request_offset:]
            slices = torch.tensor(
                [[start, start + width, 0], [start + 17, start + 17 + width, width]],
                device=self.device,
                dtype=torch.int32,
            )
            indices = (
                torch.arange(2048, device=self.device, dtype=torch.int32) - 3
            ).repeat(rows, 1)
            output = torch.empty_like(indices)
            params = SimpleNamespace(
                is_prefill=True,
                query_requests=requests,
                page_size=128,
                shard_size=8,
                shard_rank=1,
            )
            block = SimpleNamespace(start=start, width=width, slices=slices)
            map_dsa_chunk_slots(indices, params, block, output)
            begin, end, offset = slices[requests.long()].unbind(1)
            valid = (indices >= begin[:, None]) & (indices < end[:, None])
            expected = torch.where(
                valid, indices - begin[:, None] + offset[:, None], -1
            )
            torch.testing.assert_close(output, expected, rtol=0, atol=0)

        for offset in (0, 1, 2):
            run(16, 0, 64, offset)
        misses = []
        with triton.knobs.compilation.scope():

            def observe(*, cache_hit, src, **_):
                if not cache_hit:
                    misses.append(src.fn.__name__)

            triton.knobs.compilation.listener = observe
            for rows in (34, 58, 176, 924, 1030, 1252, 2118, 3596, 5836, 8192):
                for offset in (0, 1, 2):
                    run(rows, rows // 3, rows + 17, offset)
            torch.cuda.synchronize()
        self.assertEqual(misses, [], "Prefill geometry introduced slot-mapping JIT")

    def test_decode_slot_map_preserves_chunk_and_owner_bounds(self):
        from rtp_llm.models_py.kernels.cuda.dsa_kv_chunk import map_dsa_chunk_slots

        indices = torch.tensor(
            [[4, 5, 6, 7, 12, 13, 14, 15, 20, -1]],
            device=self.device,
            dtype=torch.int32,
        ).repeat(2, 1)
        params = SimpleNamespace(
            is_prefill=False,
            batch_indice_d=torch.arange(2, device=self.device, dtype=torch.int32),
            page_size=4,
            kernel_page_size=4,
            shard_size=2,
            shard_rank=1,
        )
        block = SimpleNamespace(start=4, width=4, slices=None)
        table = torch.tensor(
            [[0, 7, 9], [0, 11, 13]], device=self.device, dtype=torch.int32
        )
        for block_table, selected in (
            (None, [[0, 1, 2, 3], [4, 5, 6, 7]]),
            (table, [[28, 29, 30, 31], [44, 45, 46, 47]]),
        ):
            output = torch.empty_like(indices)
            map_dsa_chunk_slots(indices, params, block, output, block_table=block_table)
            expected = torch.full_like(indices, -1)
            expected[:, 4:8] = torch.tensor(
                selected, device=self.device, dtype=torch.int32
            )
            torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_fused_absorbed_query_reuses_kernel_across_token_counts(self):
        import triton
        from rtp_llm.models_py.kernels.cuda.dsa_buffer_ops import absorbed_query

        weight = torch.arange(96 * 128 * 512, device=self.device).remainder_(4)
        weight = weight.reshape(96, 128, 512).to(torch.bfloat16).div_(8)

        def run(rows):
            query = torch.full(
                (rows, 96, 192), 0.5, device=self.device, dtype=torch.bfloat16
            )
            query[..., 128:] = (
                torch.arange(64, device=self.device, dtype=query.dtype) / 64
            )
            output = absorbed_query(query, weight, 512, 128, fuse_suffix=True)
            expected = torch.zeros_like(output)
            # Each NoPE dot is 128 * 0.5 * ((column % 4) / 8), exactly representable.
            expected[:, :96, :512] = (
                torch.arange(512, device=self.device).remainder_(4) * 8
            )
            expected[:, :96, 512:] = query[..., 128:]
            torch.testing.assert_close(output, expected, rtol=0, atol=0)

        run(128)
        misses = []
        with triton.knobs.compilation.scope():

            def observe(*, cache_hit, src, **_):
                if not cache_hit:
                    misses.append(src.fn.__name__)

            triton.knobs.compilation.listener = observe
            for rows in (34, 58, 176, 924):
                run(rows)
            torch.cuda.synchronize()
        self.assertEqual(misses, [], "token count introduced fused sparse-query JIT")

    def test_model_warmup_does_not_change_live_kv(self):
        import triton
        from rtp_llm.models_py.modules.base.cuda.fp8_indexer_op import Fp8IndexerOp

        op = SimpleNamespace(native_score=self.score)
        op.cache_view = MethodType(Fp8IndexerOp.cache_view, op)
        for dtype, width in (
            (torch.bfloat16, 576),
            (torch.float8_e4m3fn, 656),
            (torch.uint8, 656),
        ):
            with self.subTest(sfa_dtype=dtype):
                caches = [
                    SimpleNamespace(
                        tag=tag,
                        seq_size_per_block=128,
                        kv_cache_base=torch.empty(
                            (17, 128, features), device=self.device, dtype=storage_dtype
                        ),
                    )
                    for tag, features, storage_dtype in (
                        ("default", width, dtype),
                        ("indexer_kv", 132, torch.uint8),
                    )
                ]
                for cache in caches:
                    cache.kv_cache_base.view(torch.uint8).random_(0, 256)
                model = SimpleNamespace(
                    layers=[
                        SimpleNamespace(
                            index=0,
                            attention=SimpleNamespace(
                                kv_rank=512,
                                suffix_dim=64,
                                indexer=SimpleNamespace(indexer_op=op),
                            ),
                        )
                    ],
                    kv_cache=SimpleNamespace(
                        get_layer_cache_groups=lambda _: caches,
                        get_seq_size_per_block=lambda _: 4096,
                    ),
                    parallelism_config=SimpleNamespace(tp_size=8, tp_rank=1),
                    config=SimpleNamespace(gen_num_per_cycle=3),
                )
                before = [
                    cache.kv_cache_base.view(torch.uint8).clone() for cache in caches
                ]
                for decode in (False, True):
                    self.warmup.warmup_kimi_k31_dsa_jit(
                        model, SimpleNamespace(is_decode_role=decode)
                    )
                sfa = caches[0].kv_cache_base
                if dtype != torch.bfloat16:
                    sfa = sfa.view(torch.uint8)
                misses = []
                with triton.knobs.compilation.scope():

                    def observe(*, cache_hit, src, **_):
                        if not cache_hit:
                            misses.append(src.fn.__name__)

                    triton.knobs.compilation.listener = observe
                    for view, indexer_fp8 in (
                        (sfa, False),
                        (op.cache_view(caches[1], 128), True),
                    ):
                        self.warmup._warmup_prefix(
                            view, 4096, 8, 1, indexer_fp8=indexer_fp8
                        )
                    torch.cuda.synchronize()
                self.assertEqual(misses, [], "model warmup missed a cache codec")
                for cache, expected in zip(caches, before):
                    self.assertTrue(
                        torch.equal(cache.kv_cache_base.view(torch.uint8), expected)
                    )


if __name__ == "__main__":
    unittest.main()
