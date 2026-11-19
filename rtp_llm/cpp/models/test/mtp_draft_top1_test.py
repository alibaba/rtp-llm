"""Exact TP-sharded greedy selection, including native heads and CUDA graphs."""

import datetime
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.cpp.models.test.native_lm_head_test import _extension, indexes, make_head


def worker(rank, rendezvous, point_mass):
    os.environ["RTP_LLM_MTP_POINT_MASS"] = "1" if point_mass else "0"
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=120),
    )
    gathers = []

    def unexpected(*_):
        raise AssertionError("unexpected collective")

    def gather(recv, mode, send, inplace):
        assert mode == 0 and len(recv) == 1 and not send
        dest = recv[0].view(-1)
        size = dest.numel() // 2
        gathers.append((dest.dtype, dest.numel()))
        dist.all_gather_into_tensor(dest, dest[rank * size : (rank + 1) * size].clone())

    _extension.register_comm_ops(unexpected, unexpected, gather, unexpected, unexpected)
    graph = None
    try:
        torch.manual_seed(1697)
        for dtype in (torch.bfloat16, torch.float32):
            for cols, vocab in ((17, 34), (4099, 8191), (17, 9)):
                mapping = torch.arange(
                    vocab - 1, -1, -1, device="cuda", dtype=torch.int64
                )
                head = make_head(
                    torch.zeros(cols, 1, device="cuda", dtype=torch.bfloat16),
                    tp_size=2,
                    tp_rank=rank,
                    vocab_size=vocab,
                    d2t_map=mapping,
                    mtp_draft=True,
                )
                for rows in (0, 1, 9):
                    full = torch.randn(rows, cols * 2, device="cuda", dtype=dtype)
                    if rows == 9:
                        full[0].fill_(-float("inf"))
                        full[1].zero_()
                        full[1, 0] = -0.0
                        full[2, 1] = full[2, vocab - 1] = float("inf")
                        full[3, 2] = full[3, vocab - 1] = float("nan")
                        full[4, vocab - 1] = float("nan")
                        full[5].fill_(float("nan"))
                        full[6].fill_(-float("inf"))
                        full[6, vocab - 1] = -1
                        full[7].zero_()
                        full[7, vocab - 1] = torch.finfo(dtype).tiny / 2
                        full[8].fill_(-torch.finfo(dtype).tiny)
                        full[8, vocab - 1] = -torch.finfo(dtype).tiny / 2
                    if vocab < full.size(1):
                        full[:, vocab:] = float("nan")
                    local = full[:, rank * cols : (rank + 1) * cols].contiguous()
                    expected = full[:, :vocab].argmax(-1, keepdim=True)
                    before = len(gathers)
                    result = head(local, indexes(rows), sample_local_logits=True)
                    assert torch.equal(result["tokens"], mapping[expected])
                    if point_mass:
                        assert result["probs"] is None
                    else:
                        assert torch.equal(
                            result["probs"],
                            torch.zeros(rows, vocab, device="cuda").scatter_(
                                -1, expected, 1
                            ),
                        )
                    assert result["tokens"].dtype == torch.int64
                    if rows:
                        assert gathers[before:] == [(torch.int64, 2 * rows)]
                    else:
                        assert len(gathers) == before
                # Capture on a nondefault stream; replace the input every replay.
                local = torch.zeros(9, cols, device="cuda", dtype=dtype)
                selected = indexes(9)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        head(
                            local, selected, sample_local_logits=True, fp32_probs=False
                        )
                torch.cuda.current_stream().wait_stream(stream)
                dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    result = head(
                        local, selected, sample_local_logits=True, fp32_probs=False
                    )
                for iteration in range(32):
                    full = torch.randn(9, 2 * cols, device="cuda", dtype=dtype)
                    full[iteration % 9, iteration % vocab] = float("nan")
                    local.copy_(full[:, rank * cols : (rank + 1) * cols])
                    graph.replay()
                    expected = full[:, :vocab].argmax(-1, keepdim=True)
                    assert torch.equal(result["tokens"], mapping[expected])
                    if point_mass:
                        assert result["probs"] is None
                    else:
                        assert torch.equal(
                            result["probs"],
                            torch.zeros(9, vocab, device="cuda", dtype=dtype).scatter_(
                                -1, expected, 1
                            ),
                        )
                graph.reset()
                del head
        # Compare against actual native GEMM output, with both precision policies.
        for policy in ("0", "1"):
            cols, width, rows = 257, 128, 9
            weights = torch.randn(2 * cols, width, device="cuda", dtype=torch.bfloat16)
            hidden = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
            selected = indexes(rows)
            for compact in (False, True):
                vocab = cols * 2 if compact else cols * 2 - 3
                for enabled in ("0", "1", None):
                    with patch.dict(
                        os.environ,
                        {"RTP_LLM_BF16_LOGITS": policy},
                    ):
                        if enabled is None:
                            os.environ.pop("RTP_LLM_MTP_SHARDED_TOP1", None)
                        else:
                            os.environ["RTP_LLM_MTP_SHARDED_TOP1"] = enabled
                        head = make_head(
                            weights[rank * cols : (rank + 1) * cols],
                            tp_size=2,
                            tp_rank=rank,
                            vocab_size=vocab,
                            mtp_draft=True,
                        )
                    result = head(hidden, selected, compact=compact)
                    if enabled == "0":
                        expected = result["logits"].argmax(-1, keepdim=True)
                        expected_probs = torch.zeros_like(result["logits"]).scatter_(
                            -1, expected, 1
                        )
                    else:
                        assert result["logits"] is None
                        assert torch.equal(result["tokens"], expected)
                        if point_mass:
                            assert result["probs"] is None
                        else:
                            assert torch.equal(result["probs"], expected_probs)
                        assert gathers[-1] == (torch.int64, 2 * rows)
                        all_result = head(hidden, selected, need_all_logits=True)
                        assert all_result["tokens"] is None
                        assert all_result["all_logits"] is not None
                del head
            # Check the target and compact padded fallback under the default.
            with patch.dict(os.environ):
                os.environ.pop("RTP_LLM_MTP_SHARDED_TOP1", None)
                target = make_head(
                    weights[rank * cols : (rank + 1) * cols], tp_size=2, tp_rank=rank
                )
                padded = make_head(
                    weights[rank * cols : (rank + 1) * cols],
                    tp_size=2,
                    tp_rank=rank,
                    vocab_size=2 * cols - 3,
                    mtp_draft=True,
                )
            for head in (target, padded):
                result = head(hidden, selected, compact=True)
                assert result["tokens"] is None
                assert result["logits"].shape == (rows, 2 * cols)
            del target, padded
        print(
            f"rank {rank}: point_mass={point_mass}, exact top1 passed; 192 changing-input graph replays",
            flush=True,
        )
        dist.barrier()
    finally:
        if graph is not None:
            graph.reset()
        _extension.clear_comm_ops()
        dist.destroy_process_group()


class MtpDraftTop1Test(unittest.TestCase):
    def test_tp2_exact_selection_and_native_head(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        for point_mass in (False, True):
            with self.subTest(
                point_mass=point_mass
            ), tempfile.TemporaryDirectory() as directory:
                mp.spawn(
                    worker,
                    args=(str(Path(directory, "rendezvous")), point_mass),
                    nprocs=2,
                    join=True,
                )


if __name__ == "__main__":
    unittest.main()
