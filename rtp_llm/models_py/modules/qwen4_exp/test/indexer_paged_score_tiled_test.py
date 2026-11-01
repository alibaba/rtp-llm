"""Batch-matrix correctness tests for the tiled tensor-core paged score kernel.

Covers the `_qsa_paged_score_tiled_kernel` dispatch in
``qsa_paged_indexer_score`` against the scalar kernel's exact contract
(``sum_h weight * relu(dot_h)``, -inf outside context or for missing blocks)
across batch sizes, row counts, head counts, weight signs, and pool layouts.
"""

import importlib.util
import os
import sys
import unittest

import torch

import rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_paged_score as _installed
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_paged_score import (
    qsa_paged_indexer_score as _installed_score,
)

_DEVICE = "cuda"

_OVERRIDE = os.environ.get("QSA_SCORE_MODULE_OVERRIDE")
if _OVERRIDE:
    _spec = importlib.util.spec_from_file_location(
        "rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_paged_score", _OVERRIDE
    )
    _candidate = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _candidate
    _spec.loader.exec_module(_candidate)
    score = _candidate.qsa_paged_indexer_score
    tiled_supported = getattr(_candidate, "_score_tiled_supported", None)
else:
    _candidate = None
    score = _installed_score
    tiled_supported = getattr(_installed, "_score_tiled_supported", None)


def _make_inputs(batch, next_n, heads, dim, block_size, pool_blocks, ctx_len,
                 *, weight_mode="uniform", seed=0):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(
        batch, next_n, heads, dim, device=_DEVICE, dtype=torch.bfloat16
    )
    if weight_mode == "uniform":
        weight = torch.full(
            (batch * next_n, heads), 1.0 / dim**0.5,
            dtype=torch.float32, device=_DEVICE,
        )
    elif weight_mode == "positive":
        weight = torch.rand(
            batch * next_n, heads, dtype=torch.float32, device=_DEVICE
        ) + 0.25
    else:
        weight = (torch.rand(
            batch * next_n, heads, dtype=torch.float32, device=_DEVICE
        ) - 0.5) * 2.0
    pool = torch.randn(
        pool_blocks * block_size, dim, device=_DEVICE, dtype=torch.bfloat16
    )
    pages = torch.randperm(pool_blocks - 1, generator=gen)[:pool_blocks - 1] + 1
    # leave logical block 2 unmapped (physical 0) to exercise the sentinel
    pages = torch.cat([pages[:2], torch.zeros(1, dtype=torch.long), pages[2:]])
    table = pages[: max(4, pool_blocks // 2)].to(torch.int32).to(_DEVICE)
    table = table.unsqueeze(0).expand(batch, -1).contiguous()
    lengths = torch.randint(
        0, ctx_len + 1, (batch, next_n), generator=gen, dtype=torch.int32
    ).to(_DEVICE)
    return q, weight, pool, table, lengths


def _run(fn, q, weight, pool, table, lengths, block_size):
    return fn(
        q, weight, pool, table, lengths,
        block_size=block_size,
        max_ctx_len=int(table.shape[1]) * block_size,
        validate_block_table=False,
    )


class TiledScoreBatchMatrixTest(unittest.TestCase):
    def _check(self, got, expected, context):
        torch.testing.assert_close(
            got, expected, atol=2e-3, rtol=2e-3,
            msg=lambda m: f"{context}: {m}",
        )

    def test_matches_scalar_kernel_batch_matrix(self):
        if tiled_supported is None:
            self.skipTest("tiled kernel not present")
        for batch, next_n, heads, dim in (
            (1, 2, 4, 128),
            (1, 4, 4, 128),
            (2, 8, 4, 128),
            (4, 64, 4, 128),
            (1, 128, 4, 128),
            (1, 2052, 4, 128),
            (2, 3, 2, 64),
            (1, 16, 8, 256),
            (3, 5, 1, 32),
        ):
            for mode in ("uniform", "positive", "mixed"):
                with self.subTest(batch=batch, next_n=next_n, heads=heads,
                                  dim=dim, mode=mode):
                    q, weight, pool, table, lengths = _make_inputs(
                        batch, next_n, heads, dim, 64, 40, 2000,
                        weight_mode=mode, seed=hash((batch, next_n, heads, dim)) % 1000,
                    )
                    self.assertTrue(
                        tiled_supported(q, pool, next_n, heads),
                        "batch-matrix case must reach the tiled kernel",
                    )
                    got = _run(score, q, weight, pool, table, lengths, 64)
                    ref = _run(
                        _installed_score, q, weight, pool, table, lengths, 64
                    )
                    self._check(got, ref, f"B={batch} n={next_n} H={heads} D={dim} w={mode}")

    def test_torch_reference_independent(self):
        batch, next_n, heads, dim, block_size = 2, 32, 4, 128, 64
        q, weight, pool, table, lengths = _make_inputs(
            batch, next_n, heads, dim, block_size, 40, 2000, seed=7
        )
        got = _run(score, q, weight, pool, table, lengths, block_size)
        m, width = got.shape
        ref = torch.full_like(got, float("-inf"))
        for b in range(batch):
            for n in range(next_n):
                row = b * next_n + n
                t = int(lengths[b, n])
                for col in range(0, min(t, width)):
                    bi, ei = col // block_size, col % block_size
                    phys = int(table[b, bi])
                    if phys <= 0:
                        continue
                    key = pool[phys * block_size + ei].float()
                    acc = 0.0
                    for h in range(heads):
                        dot = float(q[b, n, h].float() @ key)
                        acc += float(weight[row, h]) * max(dot, 0.0)
                    ref[row, col] = acc
        self._check(got, ref, "torch reference")

    def test_single_row_decode_stays_on_scalar_kernel(self):
        if tiled_supported is None:
            self.skipTest("tiled kernel not present")
        q, weight, pool, table, lengths = _make_inputs(
            8, 1, 4, 128, 64, 40, 2000, seed=5
        )
        self.assertFalse(
            tiled_supported(q, pool, 1, 4),
            "single-row decode must stay on the scalar kernel",
        )
        got = _run(score, q, weight, pool, table, lengths, 64)
        ref = _run(_installed_score, q, weight, pool, table, lengths, 64)
        torch.testing.assert_close(got, ref, atol=0.0, rtol=0.0)

    def test_unmapped_block_and_boundary_lengths_are_negative_infinity(self):
        q, weight, pool, table, lengths = _make_inputs(
            2, 8, 4, 128, 64, 40, 2000, seed=3
        )
        got = _run(score, q, weight, pool, table, lengths, 64)
        ref = _run(_installed_score, q, weight, pool, table, lengths, 64)
        # logical block 2 (columns 128..191) is unmapped -> -inf everywhere
        self.assertTrue(torch.isinf(got[:, 128:192]).all())
        self._check(got, ref, "unmapped block")

    def test_graph_replay_matches_eager(self):
        batch, next_n, heads, dim, block_size = 1, 4, 4, 128, 64
        q, weight, pool, table, lengths = _make_inputs(
            batch, next_n, heads, dim, block_size, 40, 1500, seed=11
        )
        eager = _run(score, q, weight, pool, table, lengths, block_size)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replay = _run(score, q, weight, pool, table, lengths, block_size)
        q2, weight2, pool2, table2, lengths2 = _make_inputs(
            batch, next_n, heads, dim, block_size, 40, 1500, seed=99
        )
        q.copy_(q2)
        weight.copy_(weight2)
        pool.copy_(pool2)
        table.copy_(table2)
        lengths.copy_(lengths2)
        graph.replay()
        torch.cuda.synchronize()
        expected = _run(_installed_score, q, weight, pool, table, lengths, block_size)
        self._check(replay, expected, "graph replay")


if __name__ == "__main__":
    unittest.main()
