"""Exercise the production MTP logits wrapper and graph schedule refresh."""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main, skipUnless
from unittest.mock import patch

os.environ.setdefault(
    "DG_JIT_CACHE_DIR",
    str(Path(os.environ.get("TEST_TMPDIR", "/tmp")) / "paged_mqa_jit"),
)

import torch

from rtp_llm.models_py.modules.base.cuda.indexer_op import IndexerOp
from rtp_llm.models_py.modules.base.cuda.paged_mqa import paged_mqa_context_lens
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
    SparseMlaImpl,
    SparseMlaOp,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


@skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "SM10x CUDA required",
)
class PagedMqaTest(TestCase):
    def setUp(self):
        torch.manual_seed(921)
        self.env = patch.dict(os.environ, {"RTP_LLM_INDEXER_NATIVE_MTP": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def fixture(self, batch, next_n, draft=False, heads=32):
        blocks, page, dim = 20, 64, 128
        count = batch * blocks + 1
        cache = torch.empty((count, page, dim + 4), device="cuda", dtype=torch.uint8)
        raw = cache.view(count, -1)
        keys = torch.randn((count, page, dim), device="cuda").to(torch.float8_e4m3fn)
        raw[:, : page * dim].copy_(keys.view(torch.uint8).reshape(count, -1))
        scales = torch.rand((count, page), device="cuda") * 0.05 + 0.01
        raw[:, page * dim :].copy_(scales.view(torch.uint8))
        table = (torch.randperm(count - 1, device="cuda", dtype=torch.int32) + 1).view(
            batch, blocks
        )
        # Include a zero-length graph-padding row and a page-boundary crossing.
        prefix = (
            torch.arange(batch, device="cuda", dtype=torch.int32) * 59 + 63
        ) % 1000
        lengths = prefix[:, None] + torch.arange(
            1, next_n + 1, device="cuda", dtype=torch.int32
        )
        if batch > 1:
            lengths[-1].zero_()
        inputs = SimpleNamespace(
            is_prefill=True,
            is_target_verify=not draft,
            is_draft_extend=draft,
            kv_cache_kernel_block_id_device=table,
            kv_cache_block_id_device=torch.zeros_like(table),
        )
        params = SimpleNamespace(
            expanded_seq_lens=lengths.flatten(),
            kvlen_d=lengths[:, -1].contiguous(),
            schedule_metadata=None,
        )
        owner = SimpleNamespace(fmha_params=params, seq_size_per_block=page)
        q = torch.randn((batch * next_n, heads, dim), device="cuda").to(
            torch.float8_e4m3fn
        )
        weights = torch.rand((batch * next_n, heads), device="cuda") / heads
        op = IndexerOp(heads, dim, 512, 64)

        def topk(scores, lens, out, workspace, k, width):
            op.last_logits = scores
            rtp_llm_ops.topk_v3_tie_break(
                scores, torch.zeros_like(lens), lens, out, k, width
            )

        op._paged_topk_op = topk
        return op, owner, inputs, SimpleNamespace(kv_scale_base=cache), q, weights

    def call(self, op, owner, inputs, cache, q, weights):
        return op._get_topk_paged(q, weights, cache, owner.fmha_params, inputs)

    def refresh(self, owner, inputs, forbid=False):
        SparseMlaImpl._refresh_paged_mqa_schedule_metadata(owner, inputs, forbid)

    def test_native_and_flattened_wrapper(self):
        for batch, next_n, draft, heads in [
            (1, 4, False, 32),
            (3, 3, True, 32),
            (16, 4, False, 32),
            (32, 4, False, 32),
            (64, 4, False, 32),
            (3, 5, True, 64),
            (3, 1, False, 32),
        ]:
            with self.subTest(batch=batch, next_n=next_n, draft=draft, heads=heads):
                args = self.fixture(batch, next_n, draft, heads)
                op, owner, inputs = args[:3]
                with patch.dict(os.environ, {"RTP_LLM_INDEXER_NATIVE_MTP": "0"}):
                    self.refresh(owner, inputs)
                    ref_topk = self.call(*args)
                    ref_logits = op.last_logits
                self.refresh(owner, inputs)
                actual = self.call(*args)
                valid = (
                    torch.arange(ref_logits.shape[1], device="cuda")[None, :]
                    < owner.fmha_params.expanded_seq_lens[:, None]
                )
                self.assertTrue(torch.equal(ref_logits[valid], op.last_logits[valid]))
                self.assertTrue(
                    torch.equal(ref_topk.sort().values, actual.sort().values)
                )
                self.assertEqual(
                    tuple(paged_mqa_context_lens(inputs, owner.fmha_params).shape),
                    (batch, next_n),
                )

    def test_graph_replay_refreshes_lengths_and_cache_groups(self):
        args = self.fixture(3, 4)
        op, owner, inputs = args[:3]
        self.refresh(owner, inputs)
        schedule_ptr = owner.fmha_params.schedule_metadata.data_ptr()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.call(*args)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_topk = self.call(*args)
        captured_logits = op.last_logits
        for step in range(3):
            inputs.kv_cache_kernel_block_id_device.copy_(
                inputs.kv_cache_kernel_block_id_device.roll(1, 1)
            )
            owner.fmha_params.expanded_seq_lens.add_(1)
            self.refresh(owner, inputs, True)
            self.assertEqual(
                owner.fmha_params.schedule_metadata.data_ptr(), schedule_ptr
            )
            graph.replay()
            with patch.dict(os.environ, {"RTP_LLM_INDEXER_NATIVE_MTP": "0"}):
                ref_owner = SimpleNamespace(
                    fmha_params=SimpleNamespace(
                        expanded_seq_lens=owner.fmha_params.expanded_seq_lens,
                        kvlen_d=owner.fmha_params.kvlen_d,
                        schedule_metadata=None,
                    ),
                    seq_size_per_block=64,
                )
                self.refresh(ref_owner, inputs)
                ref = self.call(op, ref_owner, *args[2:])
            valid = (
                torch.arange(captured_logits.shape[1], device="cuda")[None, :]
                < owner.fmha_params.expanded_seq_lens[:, None]
            )
            self.assertTrue(torch.equal(captured_logits[valid], op.last_logits[valid]))
            self.assertTrue(torch.equal(captured_topk.sort().values, ref.sort().values))

    def test_single_token_decode_and_empty_queries(self):
        args = self.fixture(3, 1)
        op, owner, inputs, cache, q, weights = args
        inputs.is_prefill = inputs.is_target_verify = False
        self.assertEqual(
            tuple(paged_mqa_context_lens(inputs, owner.fmha_params).shape), (3, 1)
        )
        self.refresh(owner, inputs)
        self.call(*args)
        inputs.is_target_verify = True
        owner.fmha_params.expanded_seq_lens = torch.empty(
            0, dtype=torch.int32, device="cuda"
        )
        out = self.call(op, owner, inputs, cache, q[:0], weights[:0])
        self.assertEqual(tuple(out.shape), (0, 512))

    def test_invalid_request_layout_fails_before_kernel(self):
        _, owner, inputs, *_ = self.fixture(3, 4)
        owner.fmha_params.expanded_seq_lens = owner.fmha_params.expanded_seq_lens[:-1]
        with self.assertRaisesRegex(ValueError, "equal rows"):
            paged_mqa_context_lens(inputs, owner.fmha_params)

    def test_schedule_replay_requires_initialized_storage(self):
        _, owner, inputs, *_ = self.fixture(3, 4)
        with self.assertRaisesRegex(RuntimeError, "not captured"):
            self.refresh(owner, inputs, True)
        self.refresh(owner, inputs)
        owner.fmha_params.schedule_metadata = torch.empty(
            1, dtype=torch.int32, device="cuda"
        )
        with self.assertRaisesRegex(RuntimeError, "shape changed"):
            self.refresh(owner, inputs, True)

    def test_physical_index_tiles_match_at_dispatch_boundary(self):
        for tokens, topk in [
            *((tokens, 2048) for tokens in (4, 64, 127, 128, 129, 256)),
            *((128, topk) for topk in (256, 384, 512, 768)),
        ]:
            batch = (tokens + 3) // 4
            op = SparseMlaOp(64, 512, 64, 192, 64, 1.0, topk)
            op.block_table = torch.randint(
                1, 1000, (batch, 20), device="cuda", dtype=torch.int32
            )
            op.block_table[:, 3] = 0
            op.mla_params = SimpleNamespace(
                batch_indice_d=torch.arange(tokens, device="cuda", dtype=torch.int32)
                // 4
            )
            indices = torch.randint(
                -7, 1400, (tokens, topk), device="cuda", dtype=torch.int32
            )
            with patch.dict(os.environ, {"RTP_LLM_DSA_WIDE_INDEX_MAP": "0"}):
                reference = op._convert_topk_indices_to_global(indices)
            with patch.dict(os.environ, {"RTP_LLM_DSA_WIDE_INDEX_MAP": "1"}):
                actual = op._convert_topk_indices_to_global(indices)
            self.assertTrue(torch.equal(reference, actual))


if __name__ == "__main__":
    main()
