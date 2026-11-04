"""PageRR Decode bounds preserve exact selection and exclude stale Graph tails."""

import unittest

import numpy as np
import torch

from rtp_llm.models_py.kernels.cuda.dsa_decode_compact import (
    restore_visible_decode_scores as restore,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops

select = rtp_llm_ops.dsa_select_topk_candidates


def verify(values, positions, output_scores, output_positions, k):
    for r in range(len(values)):
        good = np.isfinite(values[r])
        v = values[r, good]
        p = positions[good]
        order = np.lexsort((p, -np.where(v == 0, 0, v)))[:k]
        want = p[order]
        scores = v[order]
        mask = output_positions[r] >= 0
        actual = output_positions[r, mask]
        ai = np.argsort(actual)
        wi = np.argsort(want)
        assert np.array_equal(actual[ai], want[wi]), ("position set", r)
        assert np.array_equal(output_scores[r, mask][ai], scores[wi]), ("scores", r)
        assert (
            np.isneginf(output_scores[r, ~mask]).all()
            and (output_positions[r, ~mask] == -1).all()
        )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/Triton")
class DsaDecodeCompactTest(unittest.TestCase):
    def test_owner_layout_bounds_and_nonfinite(self):
        rng = np.random.default_rng(310104)
        for start in [0, 2048]:
            for kind in ["finite", "ties", "nonfinite"]:
                shards, page, rows, width = 16, 2048, 4, 4096
                capacity = shards * width
                base = start * shards
                values = rng.normal(size=(rows, capacity)).astype(np.float32)
                if kind == "ties":
                    values.fill(0)
                    values[:, ::2] = -0.0
                if kind == "nonfinite":
                    values[:, ::5] = np.nan
                    values[:, 1::7] = np.inf
                    values[:, 2::9] = -np.inf
                owner = (np.arange(width) + start)[:, None]
                mapping = (
                    (
                        ((owner // page) * shards + np.arange(shards)) * page
                        + owner % page
                    )
                    .T.reshape(-1)
                    .astype(np.int32)
                )
                received = torch.from_numpy(
                    values[:, mapping - base]
                    .reshape(rows, shards, width)
                    .transpose(1, 0, 2)
                    .copy()
                    .reshape(rows * shards, width)
                ).cuda()
                for rank in [0, 8, 15]:
                    qpos = torch.full(
                        (rows * shards - 1,), -1, dtype=torch.int32, device="cuda"
                    )
                    counts = [0, 128, 2049, 40000]
                    for r, n in enumerate(counts):
                        i = rank * rows + r
                        if i < len(qpos):
                            qpos[i] = base + n - 1 if n else -1
                    complete = torch.full((rows, capacity), float("inf"), device="cuda")
                    ends = torch.empty(rows, dtype=torch.int32, device="cuda")
                    restore(
                        received,
                        qpos,
                        complete,
                        ends,
                        shard_rank=rank,
                        shards=shards,
                        page=page,
                        start=start,
                    )
                    observed = ends.cpu().numpy()
                    expected = np.array(counts, dtype=np.int32)
                    if rank == 15:
                        expected[-1] = 0
                    assert np.array_equal(observed, expected)
                    restored = complete.cpu().numpy()
                    for r, n in enumerate(expected):
                        assert np.array_equal(
                            restored[r, :n], values[r, :n], equal_nan=True
                        )
                        assert np.isposinf(
                            restored[r, n:]
                        ).all(), "Capacity tail was written"
                    logical_positions = torch.arange(
                        base, base + capacity, device="cuda", dtype=torch.int32
                    )
                    reference_positions = torch.from_numpy(mapping).cuda()
                    original = (
                        received.view(shards, rows, width)
                        .transpose(0, 1)
                        .reshape(rows, capacity)
                        .clone()
                    )
                    for r, n in enumerate(expected):
                        original[r, reference_positions >= base + int(n)] = -float(
                            "inf"
                        )
                    for k in [512, 1024, 2048]:
                        oscores = torch.empty((rows, k), device="cuda")
                        opos = torch.empty((rows, k), device="cuda", dtype=torch.int32)
                        select(
                            complete, logical_positions, None, ends, oscores, opos, k
                        )
                        bounded = np.full_like(values, -np.inf)
                        for r, n in enumerate(expected):
                            bounded[r, :n] = values[r, :n]
                        verify(
                            bounded,
                            np.arange(base, base + capacity, dtype=np.int32),
                            oscores.cpu().numpy(),
                            opos.cpu().numpy(),
                            k,
                        )
                        ref_scores = torch.empty_like(oscores)
                        ref_pos = torch.empty_like(opos)
                        select(
                            original,
                            reference_positions,
                            None,
                            None,
                            ref_scores,
                            ref_pos,
                            k,
                        )
                        assert torch.equal(
                            opos.sort(-1).values, ref_pos.sort(-1).values
                        )

    def test_graph_bounds_ignore_stale_tail(self):
        shards, page, rows, width = 16, 2048, 4, 6144
        capacity = shards * width
        mapping = (
            (
                (
                    (np.arange(width) // page)[None, :] * shards
                    + np.arange(shards)[:, None]
                )
                * page
                + np.arange(width)[None, :] % page
            )
            .reshape(-1)
            .astype(np.int32)
        )
        received = torch.from_numpy(
            np.broadcast_to(mapping.astype(np.float32), (rows, capacity))
            .copy()
            .reshape(rows, shards, width)
            .transpose(1, 0, 2)
            .copy()
            .reshape(rows * shards, width)
        ).cuda()
        qpos = torch.zeros(rows * shards, dtype=torch.int32, device="cuda")
        complete = torch.full((rows, capacity), float("inf"), device="cuda")
        ends = torch.empty(rows, device="cuda", dtype=torch.int32)
        positions = torch.arange(capacity, device="cuda", dtype=torch.int32)
        scores = torch.empty(rows, 2048, device="cuda")
        indices = torch.empty(rows, 2048, device="cuda", dtype=torch.int32)
        restore(
            received,
            qpos,
            complete,
            ends,
            shard_rank=0,
            shards=shards,
            page=page,
            start=0,
        )
        select(complete, positions, None, ends, scores, indices, 2048)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            restore(
                received,
                qpos,
                complete,
                ends,
                shard_rank=0,
                shards=shards,
                page=page,
                start=0,
            )
            select(complete, positions, None, ends, scores, indices, 2048)
        for n in [65539, 128, 0, 2048, 2049, 65540, 1]:
            qpos[:rows] = n - 1
            g.replay()
            torch.cuda.synchronize()
            assert (ends == n).all().item()
            for row in range(rows):
                assert torch.equal(
                    indices[row][indices[row] >= 0].sort().values,
                    torch.arange(max(0, n - 2048), n, device="cuda", dtype=torch.int32),
                )
        del g


if __name__ == "__main__":
    unittest.main()
