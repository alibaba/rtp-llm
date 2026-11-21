"""CPU: logical packed-cache slots, fixed request scope and comparison protocol."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules import v41_cache_model_trace as trace
from rtp_llm.models_py.modules.dsv4.fp8._cp_slot_mapping import cp_kv_slot_mapping


class CacheModelProtocolTest(unittest.TestCase):
    def test_logical_rows_are_invariant_to_physical_page_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for i, rank in enumerate((0, 1)):
                summaries = []
                for j, blocks in enumerate(([1, 3], [4, 2])):
                    run = str(2 * i + j) * 32
                    recorder = trace.Trace(root, run, "hit", 512)
                    cp = SimpleNamespace(cp_rank=rank)
                    table = torch.tensor([blocks])

                    def slots(region, pos, req):
                        return cp_kv_slot_mapping(pos, table, req, 256, 128, 2, 2, rank)

                    # Last two bytes of each physical block are padding.
                    raw = torch.full((5, 514), 99, dtype=torch.uint8)
                    pool = raw.as_strided((5, 128, 4), (514, 4, 1))
                    for pos in trace.SLOTS:
                        ix = slots(None, torch.tensor([pos]), torch.tensor([0])).item()
                        if ix >= 0:
                            pool[ix // 128, ix % 128].fill_(pos % 251)
                    module = SimpleNamespace(
                        _source_pool=lambda region: pool,
                        _global_region=lambda: "csa_kv",
                        _slots=slots,
                    )
                    recorder.pool("reused_before_write", module, cp)
                    fn = root / f"v41_cache_model_{run}_hit_{rank}.jsonl"
                    summaries.append(fn.read_text())
                self.assertEqual(*summaries)
                paths.append(json.loads(summaries[0].splitlines()[0])["positions"])
            self.assertEqual(
                sorted(paths[0] + paths[1]), [p for p in trace.SLOTS if p < 512]
            )

    def test_draft_three_layer_packed_bytes_reassemble_across_cp_boundary(self):
        block = torch.arange(3 * 528, dtype=torch.int32).to(torch.uint8).view(1, 3, 528)
        # A 3-row block splits through the middle logical KV row.
        for slot in (0, 1, 2):
            span = block.flatten(1)
            # The native codec requires 16-byte alignment. Split at 784 bytes.
            # Pad each physical slice to the actual fixed width.
            raw = torch.nn.functional.pad(span, (0, 16)).view(2, 800)
            chunks = {}
            for rank in (0, 1):
                chunks.update(trace.chunks(raw[rank : rank + 1], slot, 3, rank, True))
            self.assertEqual(chunks, trace.chunks(block, slot, 3, 0, False))

    def test_forward_scope_marker_is_exact_and_restored(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            TEST_TARGET=trace.TARGET,
            RTP_V41_CACHE_TRACE="1",
            TEST_UNDECLARED_OUTPUTS_DIR=tmp,
        ):
            root = Path(tmp)
            self.assertIsNone(trace.run_fixed(0, trace.active))
            marker = root / "v41_cache_trace_stage"
            marker.write_text("f" * 32 + " hit\n")
            self.assertEqual(trace.run_fixed(512, lambda: trace.active().phase), "hit")
            with self.assertRaises(ValueError):
                trace.run_fixed(0, trace.active)
            self.assertIsNone(trace.active())
            marker.write_text("../untrusted hit\n")
            with self.assertRaises(ValueError):
                trace.run_fixed(512, trace.active)

    def test_complete_both_ranks_and_execution_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = "a" * 32
            with self.assertRaisesRegex(ValueError, "Missing model trace"):
                trace.finalize(root, run)
            self.assertFalse((root / "v41_cache_model.json").exists())
            for phase in trace.PHASES:
                prefix = 512 if phase == "hit" else 0
                for rank in range(2):
                    s = trace.Trace(root, run, phase, prefix)
                    token_positions = list(range(prefix + rank, 783, 2))
                    for stage in (
                        "embedding",
                        *(f"layer_{j}" for j in range(40)),
                        "target_export",
                        "draft_import",
                    ):
                        # Only synthetic metadata for the wide hand-off; use
                        # actual snapshot/digest in the separate routing test.
                        s.emit(
                            rank,
                            stage,
                            token_positions,
                            torch.ones(len(token_positions), 1, dtype=torch.bfloat16),
                        )
                    for point in (
                        "after_write",
                        "readback",
                        *(("reused_before_write",) if prefix else ()),
                    ):
                        p = list(trace.SLOTS)
                        if point == "reused_before_write":
                            p = [x for x in p if x < 512]
                        if point != "readback":
                            p = [x for x in p if x // 256 % 2 == rank]
                        for region in ("main", "index"):
                            s.emit(
                                rank,
                                f"layer_2.{point}.{region}",
                                p,
                                torch.ones(len(p), 4, dtype=torch.uint8),
                            )
                    file = root / f"v41_cache_model_{run}_{phase}_{rank}.jsonl"
                    rows = [json.loads(line) for line in file.read_text().splitlines()]
                    for row in rows:
                        if row["stage"] in ("target_export", "draft_import"):
                            row["row_shape"] = [15360]
                    file.write_text("".join(json.dumps(row) + "\n" for row in rows))
                pieces = {str(i): "a" * 64 for i in range(33)}
                read = {str(p): pieces for p in trace.DRAFT_POS if p >= 655}
                for rank in range(2):
                    out = (
                        root / f"v41_cache_model_{run}_{phase}_{rank}_draft_write.jsonl"
                    )
                    rows = {
                        p: {i: h for i, h in v.items() if int(i) % 2 == rank}
                        for p, v in read.items()
                    }
                    out.write_text(
                        "".join(
                            json.dumps(
                                {
                                    "rank": rank,
                                    "layer": lid,
                                    "rows": rows,
                                    "sliced": True,
                                }
                            )
                            + "\n"
                            for lid in range(3)
                        )
                    )
                (root / f"v41_cache_model_{run}_{phase}_draft_read.json").write_text(
                    json.dumps(
                        {
                            "path": "graph",
                            "prefix": 783,
                            "visible_start": 655,
                            "layers": {str(lid): read for lid in range(3)},
                        }
                    )
                )
            old = trace.finalize(root, run)
            new = copy.deepcopy(old)
            for stage in ("layer_12", "layer_2"):
                new["phases"]["hit"]["ranks"]["0"][stage]["sha256"] = "0" * 64
            result = trace.compare(old, new)
            self.assertEqual(result["by_phase"]["hit"]["stage"], "layer_2")
            self.assertTrue(
                result["within_run"]["old"]["target_equals_draft_import"]["hit"]
            )
            # A changed rank-one row order must win over a rank-zero value
            # mismatch at this stage: the CP layouts cannot be compared.
            new["phases"]["hit"]["ranks"]["1"]["layer_2"]["positions"].reverse()
            self.assertEqual(
                trace.compare(old, new)["by_phase"]["hit"]["status"],
                "incomparable_layout",
            )
            del new["phases"]["hit"]["ranks"]["0"]["draft_import"]
            with self.assertRaisesRegex(ValueError, "stages"):
                trace.compare(old, new)


if __name__ == "__main__":
    unittest.main()
