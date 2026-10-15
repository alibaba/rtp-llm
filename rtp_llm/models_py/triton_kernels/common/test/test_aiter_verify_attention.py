"""ROCm regression tests for the fixed-shape AITER verify-attention kernel.

These are correctness tests, not kernel timing tests.  They keep every fixture
in the production BF16/vectorized paged-KV layout and build an independent
FP32 causal reference from decoded physical pages.

Coverage spans the Qwen3.5 family's per-rank GQA geometry at non-wasting TP
splits: 27B (24Q/4KV) at TP1/2/4, 35B-A3B (16Q/2KV) at TP1/2, and
397B-A17B (128Q/8KV) at TP1/2/4.
"""

import math
import random
import unittest
from dataclasses import dataclass

import torch
from rtp_llm.models_py.triton_kernels.common.aiter_verify_attention import (
    VerifyAttentionWorkspace,
    is_supported_device,
    supports_shape,
)

HEAD_DIM, PAGE_SIZE = 256, 16
FP32_ATOL, FP32_RTOL = 1e-3, 2e-2
BF16_ATOL, BF16_RTOL = 1e-2, 2e-2

# (label, per-rank Q heads, per-rank KV heads) across the family's TP splits.
FAMILY_GEOMETRIES = (
    ("qwen35-27b-tp1", 24, 4),
    ("qwen35-27b-tp2", 12, 2),
    ("qwen35-27b-tp4", 6, 1),
    ("qwen35-35b-a3b-tp1", 16, 2),
    ("qwen35-35b-a3b-tp2", 8, 1),
    ("qwen35-397b-a17b-tp1", 128, 8),
    ("qwen35-397b-a17b-tp2", 64, 4),
    ("qwen35-397b-a17b-tp4", 32, 2),
)

# Geometry exercised by the CUDA-graph replay suites.  Graph mechanics
# (scratch reset, counter hand-off, address stability) are geometry-independent,
# so only the original baseline and the TP4 target replay under graphs.
GRAPH_GEOMETRIES = (
    "qwen35-27b-tp2",
    "qwen35-397b-a17b-tp4",
)


@dataclass(frozen=True)
class Geometry:
    label: str
    q_heads: int
    kv_heads: int

    @property
    def heads_per_kv(self) -> int:
        return self.q_heads // self.kv_heads


GEOMETRIES = tuple(Geometry(*spec) for spec in FAMILY_GEOMETRIES)
GEOMETRY_BY_LABEL = {geo.label: geo for geo in GEOMETRIES}


@dataclass
class Fixture:
    query: torch.Tensor
    kv: torch.Tensor
    block_table: torch.Tensor
    geo: Geometry


def _decode_page(page: torch.Tensor, kv_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode AITER's vectorized K/V physical page into [P, Hkv, D]."""
    lanes = 16 // page.element_size()
    raw_k, raw_v = page[0], page[1]
    key = raw_k.view(kv_heads, HEAD_DIM // lanes, PAGE_SIZE, lanes)
    key = key.permute(0, 2, 1, 3).reshape(kv_heads, PAGE_SIZE, HEAD_DIM)
    value = raw_v.view(kv_heads, PAGE_SIZE // lanes, HEAD_DIM, lanes)
    value = value.permute(0, 1, 3, 2).reshape(kv_heads, PAGE_SIZE, HEAD_DIM)
    return key.permute(1, 0, 2), value.permute(1, 0, 2)


def _fixture(
    batch: int, q_len: int, capacity: int, seed: int, device: torch.device, geo: Geometry
) -> Fixture:
    assert capacity % PAGE_SIZE == 0
    pages_per_row = capacity // PAGE_SIZE
    generator = torch.Generator(device=device).manual_seed(seed)
    query = torch.randn(
        batch * q_len,
        geo.q_heads,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    kv = torch.empty(
        batch * pages_per_row,
        2,
        geo.kv_heads,
        PAGE_SIZE,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    # Do not use identity page order: each row gets a distinct randomized physical mapping.
    page_ids = list(range(batch * pages_per_row))
    random.Random(seed).shuffle(page_ids)
    block_table = torch.tensor(page_ids, dtype=torch.int32, device=device).view(
        batch, pages_per_row
    )
    # One device-side fill avoids thousands of tiny GPU launches; the randomized
    # block table still ensures each logical row sees non-contiguous physical pages.
    kv.normal_(generator=generator)
    return Fixture(query=query, kv=kv, block_table=block_table, geo=geo)


def _refresh_fixture(fixture: Fixture, seed: int) -> None:
    """Mutate live query/KV and remap the stable table storage before graph replay."""
    generator = torch.Generator(device=fixture.query.device).manual_seed(seed)
    fixture.query.normal_(generator=generator)
    fixture.kv.normal_(generator=generator)
    ids = list(range(fixture.kv.shape[0]))
    random.Random(seed).shuffle(ids)
    fixture.block_table.copy_(
        torch.tensor(ids, dtype=torch.int32, device=fixture.query.device).view_as(
            fixture.block_table
        )
    )


def _reference(
    fixture: Fixture, lengths: torch.Tensor, q_len: int, *, bf16_math: bool = False
) -> torch.Tensor:
    """Independent causal attention over each row's live page prefix.

    For L=0, the production contract is a neutral all-zero row.  The BF16
    path is only used for the L=8 historical near-zero comparison; it is not a
    relaxed FP32 correctness gate for normal lengths.
    """
    geo = fixture.geo
    outputs = []
    repeat = geo.q_heads // geo.kv_heads
    for row, length in enumerate(lengths.cpu().tolist()):
        if length == 0:
            outputs.append(
                torch.zeros(
                    q_len,
                    geo.q_heads,
                    HEAD_DIM,
                    dtype=torch.float32,
                    device=fixture.query.device,
                )
            )
            continue
        page_count = math.ceil(length / PAGE_SIZE)
        keys, values = [], []
        for page_id in fixture.block_table[row, :page_count].cpu().tolist():
            key, value = _decode_page(fixture.kv[int(page_id)], geo.kv_heads)
            keys.append(key)
            values.append(value)
        key = torch.cat(keys, dim=0)[:length]
        value = torch.cat(values, dim=0)[:length]
        query = fixture.query[row * q_len : (row + 1) * q_len]
        if bf16_math:
            q = query.transpose(0, 1)
            k = key.transpose(0, 1).repeat_interleave(repeat, dim=0)
            v = value.transpose(0, 1).repeat_interleave(repeat, dim=0)
        else:
            q = query.float().transpose(0, 1)
            k = key.float().transpose(0, 1).repeat_interleave(repeat, dim=0)
            v = value.float().transpose(0, 1).repeat_interleave(repeat, dim=0)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(HEAD_DIM)
        q_positions = torch.arange(q_len, device=q.device) + length - q_len
        k_positions = torch.arange(length, device=q.device)
        scores.masked_fill_(
            k_positions[None, None, :] > q_positions[None, :, None], float("-inf")
        )
        outputs.append(
            torch.matmul(torch.softmax(scores, dim=-1), v)
            .transpose(0, 1)
            .reshape(q_len, geo.q_heads, HEAD_DIM)
            .float()
        )
    return torch.cat(outputs, dim=0)


class TestAiterVerifyAttention(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("ROCm GPU is unavailable")
        self.device = torch.device("cuda")
        if not is_supported_device(self.device):
            raise unittest.SkipTest("requires the gfx942 AITER verify-attention device")

    def _workspace(self, batch: int, q_len: int, geo: Geometry) -> VerifyAttentionWorkspace:
        return VerifyAttentionWorkspace(
            batch, q_len, geo.q_heads, geo.kv_heads, HEAD_DIM, PAGE_SIZE, self.device
        )

    def _assert_reference(
        self, actual: torch.Tensor, fixture: Fixture, lengths: torch.Tensor, q_len: int
    ) -> None:
        geo = fixture.geo
        self.assertEqual(
            tuple(actual.shape),
            (fixture.block_table.shape[0] * q_len, geo.q_heads, HEAD_DIM),
        )
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(actual.is_contiguous())
        fp32 = _reference(fixture, lengths, q_len)
        # L=8 has a known BF16 near-zero cancellation boundary in the old AITER
        # baseline.  Keep the normal FP32 gate for all other rows, and compare
        # only L=8 rows against a BF16 arithmetic baseline rather than weakening
        # the full test tolerance.
        for row, length in enumerate(lengths.cpu().tolist()):
            got = actual[row * q_len : (row + 1) * q_len].float()
            expected = fp32[row * q_len : (row + 1) * q_len]
            if length == 8:
                # Slice the actual row's Q/BT; passing lengths[row:] alone would
                # accidentally re-reference row zero in a mixed batch.
                one_row = Fixture(
                    query=fixture.query[row * q_len : (row + 1) * q_len],
                    kv=fixture.kv,
                    block_table=fixture.block_table[row : row + 1],
                    geo=geo,
                )
                bf16_expected = _reference(
                    one_row, lengths[row : row + 1], q_len, bf16_math=True
                )
                torch.testing.assert_close(
                    got, bf16_expected, atol=BF16_ATOL, rtol=BF16_RTOL
                )
            else:
                torch.testing.assert_close(
                    got, expected, atol=FP32_ATOL, rtol=FP32_RTOL
                )

    def test_support_gate(self) -> None:
        for geo in GEOMETRIES:
            with self.subTest(geometry=geo.label):
                for batch in (1, 2, 4, 8, 16, 32):
                    for q_len in (5, 6, 7, 8):
                        self.assertTrue(
                            supports_shape(
                                batch,
                                q_len,
                                geo.q_heads,
                                geo.kv_heads,
                                HEAD_DIM,
                                PAGE_SIZE,
                                torch.bfloat16,
                            )
                        )
        geo = GEOMETRY_BY_LABEL["qwen35-27b-tp2"]
        self.assertFalse(
            supports_shape(33, 8, geo.q_heads, geo.kv_heads, 256, 16, torch.bfloat16)
        )
        self.assertFalse(
            supports_shape(1, 4, geo.q_heads, geo.kv_heads, 256, 16, torch.bfloat16)
        )
        self.assertFalse(
            supports_shape(1, 8, geo.q_heads, geo.kv_heads, 128, 16, torch.bfloat16)
        )
        self.assertFalse(
            supports_shape(1, 8, geo.q_heads, geo.kv_heads, 256, 32, torch.bfloat16)
        )
        self.assertFalse(
            supports_shape(1, 8, geo.q_heads, geo.kv_heads, 256, 16, torch.float16)
        )
        # Q heads must split evenly across KV heads.
        self.assertFalse(supports_shape(1, 8, 13, 2, 256, 16, torch.bfloat16))
        self.assertFalse(supports_shape(1, 8, 4, 8, 256, 16, torch.bfloat16))

    def test_family_geometries_q5_to_q8_and_all_batch_branches(self) -> None:
        # Covers every family geometry, every supported B branch and Q=5/6/7/8
        # with a randomized page map.
        for geo in GEOMETRIES:
            for batch, q_len in ((1, 5), (2, 6), (4, 7), (8, 8), (16, 5), (32, 8)):
                with self.subTest(geometry=geo.label, batch=batch, q_len=q_len):
                    fixture = _fixture(
                        batch,
                        q_len,
                        4096,
                        1000 + batch * 10 + q_len + geo.q_heads,
                        self.device,
                        geo,
                    )
                    lengths = torch.full(
                        (batch,), 4096, dtype=torch.int32, device=self.device
                    )
                    workspace = self._workspace(batch, q_len, geo)
                    actual = workspace.forward(
                        fixture.query, fixture.kv, fixture.block_table, lengths
                    )
                    torch.cuda.synchronize()
                    self._assert_reference(actual, fixture, lengths, q_len)
                    self.assertEqual(torch.count_nonzero(workspace.counters).item(), 0)

    def test_graph_growth_shrink_padding_and_scratch_reset(self) -> None:
        for label in GRAPH_GEOMETRIES:
            geo = GEOMETRY_BY_LABEL[label]
            with self.subTest(geometry=label):
                batch, q_len = 8, 8
                fixture = _fixture(batch, q_len, 8192, 20260914, self.device, geo)
                lengths = torch.full(
                    (batch,), 4096, dtype=torch.int32, device=self.device
                )
                workspace = self._workspace(batch, q_len, geo)
                query_ptr, kv_ptr, table_ptr, lens_ptr = (
                    fixture.query.data_ptr(),
                    fixture.kv.data_ptr(),
                    fixture.block_table.data_ptr(),
                    lengths.data_ptr(),
                )
                workspace.forward(
                    fixture.query, fixture.kv, fixture.block_table, lengths
                )
                torch.cuda.synchronize()
                self.assertEqual(torch.count_nonzero(workspace.counters).item(), 0)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = workspace.forward(
                        fixture.query, fixture.kv, fixture.block_table, lengths
                    )
                torch.cuda.synchronize()
                phases = (
                    ("4k", [4096] * batch),
                    ("8k", [8192] * batch),
                    ("mixed", [0, 8, 511, 512, 513, 4095, 4097, 8192]),
                    ("4k_after_shrink", [4096] * batch),
                )
                for index, (name, live_lengths) in enumerate(phases):
                    with self.subTest(geometry=label, phase=name):
                        _refresh_fixture(fixture, 20261000 + index)
                        lengths.copy_(
                            torch.tensor(
                                live_lengths, dtype=torch.int32, device=self.device
                            )
                        )
                        self.assertEqual(
                            (
                                fixture.query.data_ptr(),
                                fixture.kv.data_ptr(),
                                fixture.block_table.data_ptr(),
                                lengths.data_ptr(),
                            ),
                            (query_ptr, kv_ptr, table_ptr, lens_ptr),
                        )
                        graph.replay()
                        torch.cuda.synchronize()
                        self._assert_reference(captured, fixture, lengths, q_len)
                        self.assertEqual(
                            torch.count_nonzero(workspace.counters).item(), 0
                        )
                        # Replays use the same scratch/counters.  A stale counter
                        # or partial buffer is visible when the identical phase is
                        # replayed repeatedly.
                        for _ in range(3):
                            graph.replay()
                        torch.cuda.synchronize()
                        self._assert_reference(captured, fixture, lengths, q_len)
                        self.assertEqual(
                            torch.count_nonzero(workspace.counters).item(), 0
                        )

    def test_workspaces_are_independent(self) -> None:
        for label in GRAPH_GEOMETRIES:
            geo = GEOMETRY_BY_LABEL[label]
            with self.subTest(geometry=label):
                fixture = _fixture(2, 7, 8192, 20260977, self.device, geo)
                lengths = torch.tensor(
                    [4096, 4097], dtype=torch.int32, device=self.device
                )
                first = self._workspace(2, 7, geo)
                second = self._workspace(2, 7, geo)
                first_out = first.forward(
                    fixture.query, fixture.kv, fixture.block_table, lengths
                )
                second_out = second.forward(
                    fixture.query, fixture.kv, fixture.block_table, lengths
                )
                torch.cuda.synchronize()
                self.assertNotEqual(first_out.data_ptr(), second_out.data_ptr())
                self.assertNotEqual(first.output.data_ptr(), second.output.data_ptr())
                self.assertNotEqual(first.exp_sums.data_ptr(), second.exp_sums.data_ptr())
                self.assertNotEqual(first.partial.data_ptr(), second.partial.data_ptr())
                self.assertEqual(torch.count_nonzero(first.counters).item(), 0)
                self.assertEqual(torch.count_nonzero(second.counters).item(), 0)
                self._assert_reference(first_out, fixture, lengths, 7)
                self._assert_reference(second_out, fixture, lengths, 7)
                # Reusing the first workspace after the second one must not share
                # scratch state.
                again = first.forward(
                    fixture.query, fixture.kv, fixture.block_table, lengths
                )
                torch.cuda.synchronize()
                self._assert_reference(again, fixture, lengths, 7)
                self.assertEqual(torch.count_nonzero(first.counters).item(), 0)

    def test_graph_q5_to_q7_tail_tiles(self) -> None:
        """Exercise Q5/Q6/Q7's partial final q_tile under CUDA graph replay."""
        for label in GRAPH_GEOMETRIES:
            geo = GEOMETRY_BY_LABEL[label]
            for batch, q_len, live_lengths in (
                (1, 5, [4097]),
                (2, 6, [0, 4097]),
                (1, 7, [4097]),
            ):
                with self.subTest(geometry=label, batch=batch, q_len=q_len):
                    fixture = _fixture(
                        batch, q_len, 8192, 20261100 + q_len, self.device, geo
                    )
                    lengths = torch.full(
                        (batch,), 4096, dtype=torch.int32, device=self.device
                    )
                    workspace = self._workspace(batch, q_len, geo)
                    workspace.forward(
                        fixture.query, fixture.kv, fixture.block_table, lengths
                    )
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = workspace.forward(
                            fixture.query, fixture.kv, fixture.block_table, lengths
                        )
                    _refresh_fixture(fixture, 20261200 + q_len)
                    lengths.copy_(
                        torch.tensor(
                            live_lengths, dtype=torch.int32, device=self.device
                        )
                    )
                    for _ in range(3):
                        graph.replay()
                    torch.cuda.synchronize()
                    self._assert_reference(captured, fixture, lengths, q_len)
                    self.assertEqual(torch.count_nonzero(workspace.counters).item(), 0)


if __name__ == "__main__":
    unittest.main()
