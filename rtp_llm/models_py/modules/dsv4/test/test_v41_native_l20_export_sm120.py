"""Actual maintained L20 runtime snapshots through CP4/EP4/Engram stages."""

import socket
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
import torch.multiprocessing as mp

from rtp_llm.models.dsv41_kv_cache import dsv41_page_layouts
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_worker_sm120 import checkpoint
from rtp_llm.models_py.modules.dsv4.native_l20_export import snapshot_native_l20_frame
from rtp_llm.models_py.modules.dsv4.test.test_v41_cep_stage_execution_sm120 import (
    run_rank,
)


def capture_checked(block, hidden, spans, rows):
    attention = block.attn
    common = attention._prefill_meta_shared
    shared = attention._shared_attention
    sources = [hidden, block.ffn_hc.pre_mix_out, shared["topk"][20], rows]
    saved = [value.clone() for value in sources]
    frame = snapshot_native_l20_frame(block, hidden, spans, rows)
    fields = [value.clone() for value in frame.__dict__.values()]
    bad_rows = rows.clone()
    bad_rows[0, 1] = 255
    cp = attention._cp_ctx
    variants = [
        ("owner", block, "layer_id", 19),
        ("producer", attention, "is_kv_source", False),
        ("layout", attention, "_cache_layouts", dsv41_page_layouts(1)),
        ("stale", attention, "_prefill_meta_shared", None),
        ("shape", attention, "_prefill_meta_shared", common._replace(seqlen_full=999)),
        ("premix", block.ffn_hc, "pre_mix_out", None),
        ("cp-size", attention, "_cp_ctx", replace(cp, cp_size=2)),
        ("cp-shard", attention, "_cp_ctx", replace(cp, kv_cache_sharded=True)),
        ("cp-rank", attention, "_cp_ctx", replace(cp, cp_rank=(cp.cp_rank + 1) % 4)),
    ]
    for _, obj, field, bad in variants:
        with patch.object(obj, field, bad):
            try:
                snapshot_native_l20_frame(block, hidden, spans, rows)
            except (ValueError, NotImplementedError):
                pass
            else:
                raise AssertionError(f"invalid L20 export admitted: {field}")
    for variant in ("rows", "spans", "sparse", "config", "topk"):
        with ExitStack() as stack:
            expected_rows = bad_rows if variant == "rows" else rows
            expected_spans = spans
            if variant == "spans":
                expected_spans = [replace(spans[0], count=257), *spans[1:]]
            elif variant == "sparse":
                stack.enter_context(
                    patch.dict(shared, {"candidates": frame.candidate_blocks})
                )
            elif variant == "config":
                stack.enter_context(
                    patch.dict(attention.v41_config, {"candidate_block_size": 16})
                )
            elif variant == "topk":
                stack.enter_context(
                    patch.dict(shared["topk"], {20: saved[2][:, :511].contiguous()})
                )
            try:
                snapshot_native_l20_frame(block, hidden, expected_spans, expected_rows)
            except (ValueError, NotImplementedError):
                pass
            else:
                raise AssertionError(f"invalid L20 export admitted: {variant}")
    for value, before in zip(sources, saved):
        torch.testing.assert_close(value, before, rtol=0, atol=0)
        value.zero_()
    for value, before in zip(frame.__dict__.values(), fields):
        torch.testing.assert_close(value, before, rtol=0, atol=0)
    for value, before in zip(sources, saved):
        value.copy_(before)
    return frame


class V41NativeL20ExportSM120Test(unittest.TestCase):
    def test_runtime_export_with_cp4_ep4_engram_owner_controls(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 8)
        for rank in range(8):
            self.assertEqual(torch.cuda.get_device_capability(rank), (12, 0))
        with tempfile.TemporaryDirectory() as root, socket.socket() as sock:
            torch.manual_seed(41700)
            config = checkpoint(Path(root), full_schedule=True, hidden_size=5120)
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.close()
            mp.spawn(
                run_rank, args=(port, config, capture_checked), nprocs=8, join=True
            )


if __name__ == "__main__":
    unittest.main()
