"""CPU contracts for the CP4 fixed-tile projected source stream."""

import ast
import bisect
import random
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F


def _load():
    path = Path(__file__).resolve().parents[1] / "attention_v41.py"
    names = {
        "_prefill_x_chunk_rows",
        "_prefill_x_group_plan",
        "_prefill_projected_x_groups",
        "_prefill_x_tile_plan",
        "_start_prefill_x_gather_async",
        "_wait_prefill_x_gather",
    }
    constants = {"_PRODUCE_GLOBAL_TILE_ROWS", "_CP_PROJECTED_TRANSPORT_MAX_BYTES"}
    nodes = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name in names
        or isinstance(node, ast.ClassDef)
        and node.name == "_V41AsyncXGather"
        or isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in constants
            for target in node.targets
        )
    ]
    module = types.ModuleType("cp_stream_cpu")
    module.torch = torch
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
        module.__dict__,
    )
    return module


STREAM = _load()
OPS = "rtp_llm.ops.compute_ops"


@contextmanager
def _native_gemm():
    previous = sys.modules.get(OPS)
    fake = types.ModuleType(OPS)
    calls = []

    def packed(x, weights, out):
        rows = len(x)
        calls.append((rows, len(out)))
        for column, weight in enumerate(weights):
            width = weight.shape[0]
            out[:rows, column * width : (column + 1) * width] = F.linear(
                x.float(), weight.float()
            )

    fake.rtp_llm_ops = types.SimpleNamespace(cublas_gemm_bf16_bf16_fp32_packed=packed)
    sys.modules[OPS] = fake
    try:
        yield calls
    finally:
        if previous is None:
            sys.modules.pop(OPS, None)
        else:
            sys.modules[OPS] = previous


def _case(lengths):
    """Build CP packed rows without using the stream planner."""
    chunks = tuple(2 * ((length + 7) // 8) for length in lengths)
    parts = [[] for _ in range(4)]
    base = 0
    for length, chunk in zip(lengths, chunks):
        padded = torch.full((chunk * 4,), -1, dtype=torch.long)
        padded[:length] = torch.arange(base, base + length)
        halves = padded.reshape(8, chunk // 2)
        for rank in range(4):
            parts[rank].extend((halves[rank], halves[7 - rank]))
        base += length
    packed = [torch.cat(part) for part in parts]
    restore = torch.empty(base, dtype=torch.long)
    all_packed = torch.cat(packed)
    valid = all_packed >= 0
    restore[all_packed[valid]] = torch.arange(len(all_packed))[valid]
    raw = torch.arange(base, dtype=torch.long)
    values = torch.stack([(raw // (128**column)) % 128 for column in range(4)], 1)
    expected = values.bfloat16()
    local = [
        expected[rows.clamp_min(0)].masked_fill((rows < 0)[:, None], -999)
        for rows in packed
    ]
    cp = types.SimpleNamespace(
        cp_size=4,
        chunk_lengths_per_req=chunks,
        input_lengths_global_host=tuple(lengths),
        chunk_length=sum(chunks),
        padded_seq_len=4 * sum(chunks),
        seq_len_full=base,
        unpad_restore=restore,
    )
    return cp, local, expected


class CPStreamPlanCPU(unittest.TestCase):
    def test_zigzag_and_bounded_groups(self):
        rng = random.Random(20261031)
        cases = [
            (0,),
            (1,),
            (2,),
            (3,),
            (0, 1, 2, 3, 8, 9),
            (127,),
            (128,),
            (129,),
            (32767,),
            (32768,),
            (32769,),
            (65537,),
            (1048576,),
            (2049,) * 64,
            (1,) * 128,
            (2,) * 1024,
            (2049,) * 128,
            (65537,) * 128,
            (1, 3, 127, 2049, 65537),
            (0, 1, 7, 8, 9, 127, 128, 129, 2049, 65537) * 102 + (0, 1, 7, 8),
            (1048576,) + (2,) * 1023,
            (65537,) * 1024,
        ]
        cases += [
            tuple(
                rng.choice((1, 7, 8, 9, 127, 129, 2048, 2049, 65537))
                for _ in range(rng.randrange(1, 65))
            )
            for _ in range(300)
        ]
        for lengths in cases:
            chunks = tuple(2 * ((length + 7) // 8) for length in lengths)
            cp = types.SimpleNamespace(
                cp_size=4,
                chunk_lengths_per_req=chunks,
                input_lengths_global_host=lengths,
                chunk_length=sum(chunks),
                padded_seq_len=4 * sum(chunks),
                seq_len_full=sum(lengths),
            )
            x = types.SimpleNamespace(
                ndim=2,
                dtype=torch.bfloat16,
                shape=(sum(chunks), 4),
                is_contiguous=lambda: True,
            )
            self.assertEqual(STREAM._prefill_x_chunk_rows(cp, 1024), 8192)
            self.assertEqual(STREAM._prefill_x_chunk_rows(cp, 512), 16384)
            for width in (4, 8):
                with patch.object(
                    STREAM, "_CP_PROJECTED_TRANSPORT_MAX_BYTES", 5 * width * 4 * 8192
                ):
                    limit = STREAM._prefill_x_chunk_rows(cp, width)
                    packets = list(
                        STREAM._prefill_x_group_plan(x, cp, output_width=width)
                    )
                cursor = 0
                for pieces, counts, start, end in packets:
                    self.assertEqual(start, cursor)
                    self.assertGreater(end, start)
                    self.assertLessEqual(end - start, 32768)
                    self.assertLessEqual(max(counts), limit)
                    self.assertEqual(sum(counts), end - start)
                    self.assertEqual(sum(rows for _, _, rows, _ in pieces), end - start)
                    offsets = [0] * 4
                    row_offset = 0
                    for owner, source, rows, destination in pieces:
                        self.assertLessEqual(rows, limit)
                        self.assertEqual(destination, start + row_offset)
                        row_offset += rows
                        offsets[owner] += rows
                    self.assertEqual(tuple(offsets), counts)
                    cursor = end
                self.assertEqual(cursor, sum(lengths))
            ends = []
            bases = []
            total = local_total = 0
            for length, chunk in zip(lengths, chunks):
                total += length
                ends.append(total)
                bases.append(local_total)
                local_total += chunk
            cursor = 0
            for owner, source, rows in STREAM._prefill_x_tile_plan(cp, 8192):
                self.assertGreater(rows, 0)
                self.assertLessEqual(rows, 8192)
                for global_row in (cursor, cursor + rows - 1):
                    request = bisect.bisect_right(ends, global_row)
                    request_start = ends[request - 1] if request else 0
                    half = chunks[request] // 2
                    position = global_row - request_start
                    segment = position // half
                    self.assertEqual(owner, min(segment, 7 - segment))
                    self.assertEqual(
                        source + global_row - cursor,
                        bases[request]
                        + (half if segment >= 4 else 0)
                        + position % half,
                    )
                cursor += rows
            self.assertEqual(cursor, sum(lengths))

    def test_four_rank_exchange_and_reuse(self):
        for lengths in (
            (5, 6, 65, 7, 2),
            (127, 128, 129),
            (1, 3, 17, 33),
            (1025,),
            (1, 127, 128, 2049, 32769),
            (32769,),
        ):
            cp, local, raw = _case(lengths)
            for weights in (
                (torch.eye(4).bfloat16(),),
                (torch.eye(4).bfloat16(), (torch.eye(4) / 8).bfloat16()),
            ):
                width = 4 * len(weights)
                expected = torch.cat(
                    [F.linear(raw.float(), weight.float()) for weight in weights], 1
                )
                with patch.object(
                    STREAM, "_CP_PROJECTED_TRANSPORT_MAX_BYTES", 5 * width * 4 * 8
                ), patch.object(STREAM, "_PRODUCE_GLOBAL_TILE_ROWS", 32):
                    packets = list(
                        STREAM._prefill_x_group_plan(local[0], cp, output_width=width)
                    )
                    for rank in range(4):
                        cp.cp_rank = rank
                        pending_packets = iter(packets)

                        def begin(send, unused_cp, tile, buffer, projection_weights=()):
                            owner, source, rows = tile
                            self.assertIs(unused_cp, cp)
                            self.assertEqual(source, 0)
                            self.assertFalse(projection_weights)
                            pieces, counts, _, _ = next(pending_packets)
                            packet_rows = max(counts)
                            owners = [i for i, count in enumerate(counts) if count]
                            self.assertEqual(
                                owner, owners[0] if len(owners) == 1 else None
                            )
                            self.assertEqual(
                                rows,
                                packet_rows if owner is not None else 4 * packet_rows,
                            )
                            packet = []
                            for peer in range(4):
                                part = torch.zeros((packet_rows, width))
                                offset = 0
                                for segment_owner, segment_source, count, _ in pieces:
                                    if segment_owner == peer:
                                        part[offset : offset + count] = torch.cat(
                                            [
                                                F.linear(
                                                    local[peer][
                                                        segment_source : segment_source
                                                        + count
                                                    ].float(),
                                                    weight.float(),
                                                )
                                                for weight in weights
                                            ],
                                            1,
                                        )
                                        offset += count
                                self.assertEqual(offset, counts[peer])
                                packet.append(part)
                            torch.testing.assert_close(
                                send, packet[rank], rtol=0, atol=0
                            )
                            buffer.copy_(
                                packet[owner]
                                if owner is not None
                                else torch.cat(packet)
                            )
                            return types.SimpleNamespace(gathered=buffer)

                        with patch.object(
                            STREAM, "_start_prefill_x_gather_async", begin
                        ), patch.object(
                            STREAM,
                            "_wait_prefill_x_gather",
                            lambda handle: handle.gathered,
                        ), _native_gemm() as calls:
                            generator = STREAM._prefill_projected_x_groups(
                                local[rank], cp, iter(packets), weights
                            )
                            tiles = [
                                (start, value.clone()) for start, value in generator
                            ]
                        self.assertEqual(
                            calls,
                            [
                                (counts[rank], max(counts))
                                for _, counts, _, _ in packets
                                if counts[rank]
                            ],
                        )
                        self.assertEqual(
                            [start for start, _ in tiles],
                            list(range(0, len(raw), 32)),
                        )
                        torch.testing.assert_close(
                            torch.cat([value for _, value in tiles]),
                            expected,
                            rtol=0,
                            atol=0,
                        )


if __name__ == "__main__":
    unittest.main()
