"""MTP draft prefill graph padding-tail replay regression.

Component regression for the fused prefill QKV+bias+RoPE+paged-cache writer:
``add_fusedQKV_bias_transpose_prefill_kernel_v1`` and
``add_fusedQKV_bias_transpose_prefill_kernel`` in
``rtp_llm/models_py/bindings/rocm/kernels/fused_rope_kvcache_kernel.cu``.

MTP draft prefill CUDA graphs capture a fixed lane count per request (the
graph key is ``batch * q_len``), so a replay launches the captured token
capacity while the packed layout holds fewer live tokens: the batch shrinks,
or requests fill only part of their lanes. The graph runner refreshes only
the live prefix of the graph-owned ``padding_offset`` buffer, and that single
buffer is shared by every prefill graph instance. Tail rows therefore keep the
slot mapping of the previous, larger replay, resolve ``batch_idx`` past the
live batch, and read freed page-table rows. Production symptom:
WARP_ILLEGAL_ADDRESS in
``add_fusedQKV_bias_transpose_prefill_kernel_v1<BF16, BF16,
/*PREFIX_PROMPT=*/true, /*USE_PAGED_FMHA=*/false, RopeStyle::Base>``
after an MTP draft prefill graph replay with a shrunk batch.

This regression depends on the kernel-side live-token guard from the same
change: rows at or past ``cu_seqlens[batch_size]`` are no-ops. On graph
replays the runner keeps ``cu_seqlens[batch_size]`` at the live packed token
count (the captured tail entries are plateaued at that total), while eager
calls keep ``cu_seqlens[batch_size] == token_num`` and are unchanged.

The scenario mirrors production buffer ownership: a batch-2 x 4-lane graph
and a batch-1 x 4-lane graph share one padding buffer. Replays cover a fully
live control batch (guard inactive), partially filled lanes, the batch shrink
onto the smaller graph whose stale tail maps past the live batch, and the
batch shrink on the captured graph itself (plateaued cu_seqlens tail). Live
rows are checked byte-exactly against an eager reference on cloned cache
bytes, like ``decode_rope_block64_repro_test``.

This test needs a GPU and a built ``rtp_kernel`` extension; it is written for
the CI/H20 GPU pools registered in this package's BUILD.
"""

import unittest
from types import SimpleNamespace

import torch
from libth_transformer_config import AttentionConfigs, RopeStyle

from rtp_llm.ops.compute_ops import get_typemeta
from rtp_llm.ops.fused_rope_kvcache_op import (
    FusedRopeKVCachePrefillOpQKVOut,
    _get_fused_rope_kvcache,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class PrefillRopePaddingTailReplayReproTest(unittest.TestCase):
    SEED = 20261030
    HEAD_NUM = 2
    KV_HEAD_NUM = 1
    HEAD_DIM = 128
    HIDDEN = (HEAD_NUM + 2 * KV_HEAD_NUM) * HEAD_DIM
    TOKENS_PER_BLOCK = 16
    LANES = 4  # per-request lane capacity captured by the graph
    PREFIX = 3  # captured max prefix prompt length
    MAX_SEQ_LEN = 32768  # RoPE cache sizing only
    PAGES = 40

    def setUp(self):
        self.step = 0

    def config(self):
        cfg = AttentionConfigs()
        cfg.head_num, cfg.kv_head_num, cfg.size_per_head = self.HEAD_NUM, self.KV_HEAD_NUM, self.HEAD_DIM
        cfg.kernel_tokens_per_block = self.TOKENS_PER_BLOCK
        cfg.max_seq_len = self.MAX_SEQ_LEN
        cfg.need_rope_kv_cache, cfg.use_logn_attn = True, False
        cfg.rope_config.style = RopeStyle.Base
        cfg.rope_config.index_factor, cfg.rope_config.dim = 1, self.HEAD_DIM
        cfg.rope_config.base, cfg.rope_config.scale = 10000000.0, 1.0
        return cfg

    def make_graph(self, batch, table):
        """Capture one prefill graph with a fully live batch.

        The op class keeps ``use_paged_fmha=False`` while the prefix makes the
        kernel take the PREFIX_PROMPT + store_cache path, which is the crashed
        production instantiation. All metadata lives in graph-owned device
        buffers; the padding buffer is shared with the other graph instance.
        """
        op = FusedRopeKVCachePrefillOpQKVOut(self.config())
        cu = torch.arange(batch + 1, dtype=torch.int32, device="cuda") * self.LANES
        prefix = torch.full((batch,), self.PREFIX, dtype=torch.int32, device="cuda")
        lengths = torch.full((batch,), self.LANES, dtype=torch.int32, device="cuda")
        attn_inputs = SimpleNamespace(
            input_lengths=lengths,
            prefix_lengths=prefix,
            sequence_lengths=torch.full((batch,), self.PREFIX + self.LANES, dtype=torch.int32, device="cuda"),
            padding_offset=self.padding,
            cu_seqlens_device=cu,
            cu_kv_seqlens_device=cu,
            kv_cache_kernel_block_id_device=table,
            combo_position_ids=None,
            context_parallel_info=None,
            context_total_kv_length=self.PREFIX + self.LANES,
            dtype=get_typemeta(torch.zeros([1], dtype=torch.bfloat16)),
        )
        params = op.prepare(attn_inputs)
        torch.manual_seed(self.SEED + 100 * batch)
        qkv = torch.randn(batch * self.LANES, self.HIDDEN, device="cuda", dtype=torch.bfloat16)
        for _ in range(2):
            op.forward(qkv, self.cache, params)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = op.forward(qkv, self.cache, params)
        return SimpleNamespace(
            cu=cu,
            prefix=prefix,
            table=table,
            qkv=qkv,
            graph=graph,
            output=output,
            batch=batch,
        )

    def refresh(self, rec, lengths, prefix):
        """Refresh replay metadata the way the runner does for a shrunk batch.

        The live cu_seqlens prefix is copied and the captured tail entries are
        plateaued at the live packed total, so ``cu_seqlens[batch_size]``
        stays the live token count for the captured batch index. Only the live
        prefix of the shared padding buffer is rebuilt; the tail deliberately
        keeps the previous replay's slot mapping (the bug scenario).
        """
        self.step += 1
        cum = [0]
        for length in lengths:
            cum.append(cum[-1] + length)
        live = cum[-1]
        plateau = [live] * (rec.batch + 1 - len(cum))
        rec.cu.copy_(torch.tensor(cum + plateau, dtype=torch.int32, device="cuda"))
        rec.prefix[: len(prefix)].copy_(torch.tensor(prefix, dtype=torch.int32, device="cuda"))
        offsets = []
        for batch_idx, length in enumerate(lengths):
            offsets.extend([batch_idx * self.LANES - cum[batch_idx]] * length)
        self.padding[:live].copy_(torch.tensor(offsets, dtype=torch.int32, device="cuda"))
        torch.manual_seed(self.SEED + 1000 * self.step)
        rows = torch.randn(live, self.HIDDEN, device="cuda", dtype=torch.bfloat16)
        rec.qkv[:live].copy_(rows)
        return live, rows

    def reference(self, lengths, prefix, rows, pool, table):
        """Eager forward on the live rows only, on a clone of the pool.

        The reference keeps the captured per-request stride (max_seq_len) and
        prefix maximum so its live rows run the exact same kernel math as the
        replay. Eager calls hold ``cu_seqlens[batch_size] == token_num``, so
        the kernel-side live-token guard never triggers here.
        """
        cum = [0]
        for length in lengths:
            cum.append(cum[-1] + length)
        params = SimpleNamespace(
            cu_seqlens=torch.tensor(cum, dtype=torch.int32, device="cuda"),
            max_seq_len=self.LANES,
            kv_cache_offset=_get_fused_rope_kvcache().convert_offset_to_block_array(
                table[: len(lengths)].contiguous()
            ),
            kv_cache_offset_h=None,
            padding_offset=self.padding[: rows.shape[0]].clone(),
            position_ids=None,
            prefix_lengths=torch.tensor(prefix, dtype=torch.int32, device="cuda"),
            max_prefix_length=self.PREFIX,
        )
        cache = SimpleNamespace(kv_cache_base=pool, kv_scale_base=None)
        output = self.reference_op.forward(rows.clone(), cache, params)
        torch.cuda.synchronize()
        return output

    def replay_and_check(self, rec, lengths, prefix):
        live, rows = self.refresh(rec, lengths, prefix)
        expected_pool = self.pool.clone()
        expected = self.reference(lengths, prefix, rows, expected_pool, rec.table)
        rec.graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(rec.output[:live], expected, rtol=0, atol=0)
        self.assertTrue(
            torch.equal(self.pool.view(torch.uint8), expected_pool.view(torch.uint8))
        )

    def test_padding_tail_replays(self):
        torch.manual_seed(self.SEED)
        self.pool = torch.zeros(
            self.PAGES,
            2,
            self.KV_HEAD_NUM,
            self.TOKENS_PER_BLOCK,
            self.HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )
        self.cache = SimpleNamespace(kv_cache_base=self.pool, kv_scale_base=None)
        # One shared padding buffer, like the runner's single graph-owned
        # padding_offset allocation shared by every prefill graph instance.
        self.padding = torch.zeros(2 * self.LANES, dtype=torch.int32, device="cuda")
        self.reference_op = FusedRopeKVCachePrefillOpQKVOut(self.config())

        graph_a = self.make_graph(2, torch.tensor([[10], [20]], dtype=torch.int32, device="cuda"))
        # The smaller graph must capture on a clean padding buffer.
        self.padding.zero_()
        graph_b = self.make_graph(1, torch.tensor([[30]], dtype=torch.int32, device="cuda"))

        scenarios = [
            # Fully live batch: the guard never triggers, results unchanged.
            ("full_batch_control", graph_a, [4, 4], [3, 3]),
            # Partial lanes: the stale tail maps inside the captured batch and
            # writes dead KV slots (wrong-KV mode of the bug).
            ("partial_lanes_leading_short", graph_a, [4, 1], [3, 3]),
            ("partial_lanes_trailing_short", graph_a, [1, 4], [3, 2]),
            # Batch shrink onto the smaller graph: the stale tail kept from the
            # batch-2 replay resolves batch_idx past the live batch and the
            # page table (WARP_ILLEGAL_ADDRESS mode of the bug).
            ("batch_shrink_smaller_graph", graph_b, [1], [1]),
            ("full_batch_after_shrink", graph_b, [4], [3]),
            # Batch shrink on the captured graph itself: cu_seqlens tail is
            # plateaued at the live total, the guard must honor it.
            ("batch_shrink_captured_graph", graph_a, [1], [1]),
        ]
        for name, rec, lengths, prefix in scenarios:
            with self.subTest(name=name):
                self.replay_and_check(rec, lengths, prefix)


if __name__ == "__main__":
    unittest.main()
