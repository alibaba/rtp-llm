"""Ragged SWA payload, causal indices and real CUDA allocation regressions.

The large fixtures reproduce workspace geometry under a controlled allocator
budget, not a complete model's resident memory. They deliberately use only
pre-existing production APIs so the baseline fails at the CUDA allocation.
"""

import gc
import itertools
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv41.attn_type import SWA_KV
from rtp_llm.models_py.modules.dsv41.flash_mla import flash_mla_sparse_fwd
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_prefill_kv_workspace as workspace
from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import AttentionV41FP8

ENTRIES, SPAN, PAGE_BYTES = 136, 512, 72192
SMALL_LENGTHS = (1, 7, 127, 128, 129, 257)
SMALL_PREFIXES = (0, 127, 128, 129, 511, 1025)

# Numeric-only ragged fixtures, including original reused-prefix geometry.
RAGGED_53 = (
    (
        1443,
        26399,
        622,
        227,
        4599,
        633,
        2795,
        2983,
        1522,
        730,
        1350,
        178,
        863,
        1566,
        2187,
        674,
        1199,
        704,
        2173,
        2172,
        628,
        628,
        1022,
        632,
        774,
        149,
        631,
        5962,
        5962,
        1553,
        5961,
        5962,
        5965,
        1737,
        6030,
        1318,
        1344,
        30100,
        635,
        619,
        7477,
        12803,
        1346,
        633,
        129,
        17485,
        622,
        1875,
        631,
        323,
        117,
        1822,
        673780,
    ),
    (
        1024,
        3072,
        0,
        512,
        0,
        0,
        1024,
        1536,
        1024,
        1024,
        1024,
        1024,
        311808,
        334336,
        165376,
        13312,
        54272,
        1536,
        17920,
        528384,
        0,
        0,
        0,
        0,
        3072,
        512,
        0,
        5120,
        5120,
        5120,
        5120,
        5120,
        5120,
        8192,
        48640,
        93696,
        131072,
        3072,
        0,
        0,
        0,
        0,
        0,
        0,
        512,
        4096,
        0,
        1024,
        1024,
        0,
        0,
        0,
        0,
    ),
)
RAGGED_78 = (
    (
        216376,
        1845,
        433,
        382,
        2450,
        1375,
        510,
        495,
        475,
        473,
        469,
        477,
        515,
        472,
        696,
        4294,
        1213,
        876,
        571,
        4971,
        203,
        318,
        497,
        523,
        540,
        527,
        403,
        507,
        500,
        521,
        516,
        502,
        2254,
        1706,
        750,
        14408,
        5142,
        176,
        1297,
        3346,
        5480,
        539,
        1856,
        33990,
        623,
        627,
        542,
        517,
        696,
        324,
        474,
        2290,
        1290,
        625,
        786,
        366,
        7261,
        28255,
        495,
        491,
        521,
        16835,
        480,
        512,
        483,
        504,
        506,
        468,
        541,
        1816,
        691,
        610,
        1327,
        281,
        203,
        3559,
        1335,
        418147,
    ),
    (
        0,
        536064,
        121344,
        1536,
        2560,
        78848,
        22528,
        22528,
        22528,
        22528,
        22528,
        22528,
        22528,
        22528,
        1536,
        84992,
        355840,
        158208,
        4096,
        9216,
        512,
        1536,
        22528,
        22528,
        22528,
        22528,
        1024,
        22528,
        22528,
        22528,
        22528,
        22528,
        92160,
        1536,
        23040,
        22016,
        5120,
        4608,
        135168,
        113152,
        173056,
        25600,
        5632,
        83968,
        0,
        0,
        1024,
        0,
        0,
        0,
        512,
        0,
        0,
        0,
        12800,
        0,
        0,
        0,
        22528,
        22528,
        22528,
        0,
        22528,
        22528,
        22528,
        22528,
        22528,
        22528,
        22528,
        144384,
        161792,
        132096,
        557568,
        1024,
        512,
        96256,
        155648,
        512,
    ),
)


def reference_codec(values):
    """CPU MXFP8 oracle: no production quantizer/decoder is used."""
    groups = values.cpu().float().reshape(-1, 16, 32)
    exponents = (groups.abs().amax(-1).clamp_min(1e-4) / 448).log2().ceil()
    scales = exponents.exp2()
    payload = (groups / scales[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    encoded = torch.cat(
        (payload.view(torch.uint8).reshape(-1, 512), (exponents + 127).byte()), 1
    )
    decoded = (payload.float() * scales[..., None]).reshape(-1, 512).bfloat16()
    return encoded, decoded


def row_values(request, positions):
    positions = torch.as_tensor(positions, dtype=torch.int64)
    columns = torch.arange(512)
    return (
        ((positions[:, None] * 17 + columns * 13 + request * 29) % 211 - 105) / 64
    ).bfloat16()


def make_fixture(
    lengths=SMALL_LENGTHS,
    prefixes=SMALL_PREFIXES,
    *,
    rank=0,
    cp_size=1,
    memory_geometry=False,
    query_meta=True
):
    """Bind real GPU pools/metadata; cache handle only supplies pool layout.

    Memory-only cases use a masked page table because no cache write is
    performed. The tested cost is fresh/prefix workspace storage, independent
    of model weights or a complete KV cache. Content tests bind real pages.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    batch = len(lengths)
    counts = [(p + n + SPAN - 1) // SPAN for p, n in zip(prefixes, lengths)]
    table = torch.zeros((batch, max(counts)), dtype=torch.int32)
    next_page = 1
    for request, count in enumerate(counts):
        if not memory_geometry:
            table[request, :count] = torch.arange(next_page, next_page + count)
            next_page += count
    raw_cpu = torch.zeros((next_page, PAGE_BYTES), dtype=torch.uint8)
    reference_parts, fresh_parts = [], []
    for request, (n, p) in enumerate(zip(lengths, prefixes)):
        if memory_geometry:
            continue
        history = list(range(max(0, p - 127), p))
        encoded, decoded = reference_codec(row_values(request, history))
        for row, position in enumerate(history):
            page = int(table[request, position // SPAN])
            start = (position % ENTRIES) * 528
            raw_cpu[page, start : start + 528] = encoded[row]
        fresh = row_values(request, range(p, p + n))
        reference_parts.append(torch.cat((decoded, fresh)))
        fresh_parts.append(fresh)
    local_bytes = PAGE_BYTES // cp_size
    raw = (
        raw_cpu[:, rank * local_bytes : (rank + 1) * local_bytes]
        .contiguous()
        .to(device)
    )
    cu = [0] + list(itertools.accumulate(lengths))
    positions, requests, local_cu = [], [], [0]
    for request, (n, p) in enumerate(zip(lengths, prefixes)):
        if cp_size == 1:
            relative = range(n)
        else:
            half = (n + 2 * cp_size - 1) // (2 * cp_size)
            relative = [
                min(i, n - 1)
                for start in (rank * half, (2 * cp_size - 1 - rank) * half)
                for i in range(start, start + half)
            ]
        positions.extend(p + i for i in relative)
        requests.extend([request] * len(relative))
        local_cu.append(len(positions))

    def tensor(values, dtype=torch.int32):
        return torch.tensor(values, dtype=dtype, device=device)

    cp = SimpleNamespace(
        cp_size=cp_size,
        cp_rank=rank,
        kv_cache_sharded=cp_size > 1,
        input_lengths_global_host=tuple(lengths),
        prefix_lengths_host=tuple(prefixes),
        input_lengths_global=tensor(lengths),
        prefix_lengths=tensor(prefixes),
        cu_seqlens_global=tensor(cu),
        seq_len_full=sum(lengths),
        global_positions=tensor(positions, torch.int64),
        swa_replay_start=None,
        swa_replay_starts_host=None,
    )
    owner = AttentionV41FP8.__new__(AttentionV41FP8)
    torch.nn.Module.__init__(owner)
    owner.layer_id, owner.compress_ratio = 0, 0
    owner.head_dim, owner.window_size = 512, 128
    owner.swa_bounded_replay, owner.is_kv_source = False, False
    owner._shared_attention = {}
    owner._cp_ctx = cp
    owner._pool_spec = {SWA_KV: (torch.uint8, 528)}
    owner._kv_cache = SimpleNamespace(
        get_entries_per_block=lambda tag: ENTRIES,
        get_seq_size_per_block=lambda tag: SPAN,
        get_layer_cache=lambda *args: SimpleNamespace(kv_cache_base=raw),
    )
    owner._block_tables_by_type = {SWA_KV: table.to(device)}
    input_lengths = tensor([b - a for a, b in zip(local_cu, local_cu[1:])])
    metadata = owner._build_swa_prefill_meta_varlen(
        seqlen=len(positions),
        device=device,
        any_cont=any(prefixes),
        batch_size=batch,
        cu_seqlens=tensor(local_cu),
        input_lengths=input_lengths,
        prefix_lengths=cp.prefix_lengths,
        position_ids=cp.global_positions,
        req_id_per_token=tensor(requests),
        host_swa_table=table,
        swa_query_meta=query_meta,
    )
    common = SimpleNamespace(
        swa_meta=metadata,
        cp_ctx=cp,
        cp_on=cp_size > 1,
        any_cont=any(prefixes),
        batch_size=batch,
        input_lengths=input_lengths,
        prefix_lengths=cp.prefix_lengths,
    )
    fresh = (
        torch.zeros((sum(lengths), 512), dtype=torch.bfloat16, device=device)
        if memory_geometry
        else torch.cat(fresh_parts).to(device)
    )
    return SimpleNamespace(
        owner=owner,
        common=common,
        qkv=SimpleNamespace(kv_full=fresh),
        raw=raw,
        raw_cpu=raw_cpu,
        table=table,
        lengths=tuple(lengths),
        prefixes=tuple(prefixes),
        expected=reference_parts,
        positions=positions,
        requests=requests,
    )


def assert_packed(test, case, packed):
    sizes = [n + min(p, 127) for n, p in zip(case.lengths, case.prefixes)]
    test.assertEqual(tuple(packed.shape), (sum(sizes), 512))
    test.assertEqual(packed.untyped_storage().nbytes(), sum(sizes) * 1024)
    torch.testing.assert_close(packed.cpu(), torch.cat(case.expected), rtol=0, atol=0)
    offsets = [0] + list(itertools.accumulate(sizes))[:-1]
    meta = case.common.swa_meta
    test.assertEqual(tuple(meta.packed_row_lengths), tuple(sizes))
    test.assertEqual(meta.packed_row_offsets.cpu().tolist(), offsets)
    expected_slots = [
        offset + min(p, 127) + i
        for offset, n, p in zip(offsets, case.lengths, case.prefixes)
        for i in range(n)
    ]
    test.assertEqual(meta.slot_in_flat.cpu().tolist(), expected_slots)
    indices = meta.combined_indices.cpu().reshape(len(case.positions), -1)
    expected_indices = torch.full_like(indices, -1)
    expected_lengths = []
    for row, (request, position) in enumerate(zip(case.requests, case.positions)):
        start = max(0, case.prefixes[request] - 127)
        visible = list(range(max(0, position - 127), position + 1))
        expected_indices[row, : len(visible)] = torch.tensor(
            [offsets[request] + p - start for p in visible], dtype=indices.dtype
        )
        expected_lengths.append(len(visible))
    torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)
    test.assertEqual(meta.combined_lens.cpu().tolist(), expected_lengths)


def assert_attention(test, case, packed):
    """Real FlashMLA output vs per-request dense attention without index reuse."""
    generator = torch.Generator().manual_seed(947)
    query = (
        torch.randn(len(case.positions), 2, 512, generator=generator) / 8
    ).bfloat16()
    sink = torch.tensor([-0.4, 0.2])
    scale = 1 / math.sqrt(512)
    expected = []
    for row, (request, position) in enumerate(zip(case.requests, case.positions)):
        history_start = max(0, case.prefixes[request] - 127)
        lo, hi = max(0, position - 127) - history_start, position + 1 - history_start
        keys = case.expected[request][lo:hi].float()
        scores = torch.mm(query[row].float(), keys.T) * scale
        probs = torch.cat((scores, sink[:, None]), dim=1).softmax(-1)[:, :-1]
        expected.append(torch.mm(probs, keys))
    meta = case.common.swa_meta
    actual, _, _ = flash_mla_sparse_fwd(
        query.to(packed.device),
        packed.unsqueeze(1),
        meta.combined_indices.unsqueeze(1),
        scale,
        attn_sink=sink.to(packed.device),
        topk_length=meta.combined_lens,
    )
    torch.testing.assert_close(
        actual.cpu().float(), torch.stack(expected), rtol=0.02, atol=0.004
    )


def expected_cache_after(case):
    expected = case.raw_cpu.clone()
    offset = 0
    for request, (n, p) in enumerate(zip(case.lengths, case.prefixes)):
        encoded, _ = reference_codec(case.qkv.kv_full[offset : offset + n])
        for i in range(n):
            position = p + i
            if position + ENTRIES < min((position // SPAN + 1) * SPAN, p + n):
                continue
            page = int(case.table[request, position // SPAN])
            start = (position % ENTRIES) * 528
            expected[page, start : start + 528] = encoded[i]
        offset += n
    return expected


def write_measurement(label, values):
    directory = Path(
        os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", tempfile.gettempdir())
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / (label + ".json")).write_text(json.dumps(values, indent=2) + "\n")
    print(label, json.dumps(values, sort_keys=True), flush=True)


class PackedSwaCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("This target requires a CUDA GPU")

    @torch.no_grad()
    def test_prefix_fresh_ring_wrap_indices_and_attention(self):
        case = make_fixture()
        packed = case.owner._swa_prefill_concat(case.qkv, case.common)
        assert_packed(self, case, packed)
        assert_attention(self, case, packed)
        before = case.raw.clone()
        case.owner._prefill_write_swa_fp8_paged(case.common, case.qkv.kv_full)
        self.assertFalse(torch.equal(before, case.raw))
        torch.testing.assert_close(
            case.raw.cpu(), expected_cache_after(case), rtol=0, atol=0
        )
        # A completed prefix read must survive the subsequent ring overwrite.
        assert_packed(self, case, packed)

    @torch.no_grad()
    def test_compressed_workspace_views_and_global_merge(self):
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                case = make_fixture()
                case.owner.compress_ratio = ratio
                rows, starts = case.owner._swa_prefill_workspace(case.qkv, case.common)
                self.assertIsInstance(rows, workspace.SwaRows)
                self.assertEqual(starts, [max(0, p - 127) for p in case.prefixes])
                assert_packed(self, case, rows.flat)
                globals_by_req = [
                    (
                        torch.full(
                            (i + 1, 512), i / 8, device="cuda", dtype=torch.bfloat16
                        ),
                        None,
                    )
                    for i in range(len(case.lengths))
                ]
                combined = workspace.combine_kv({}, globals_by_req, rows)
                expected = torch.cat(
                    [
                        value
                        for pair, swa in zip(globals_by_req, case.expected)
                        for value in (pair[0].cpu(), swa)
                    ]
                )
                torch.testing.assert_close(combined.cpu(), expected, rtol=0, atol=0)

    @torch.no_grad()
    def test_legacy_rectangular_metadata_remains_compatible(self):
        case = make_fixture(query_meta=False)
        self.assertIsNone(case.common.swa_meta.packed_row_offsets)
        output = case.owner._swa_prefill_concat(case.qkv, case.common)
        self.assertEqual(output.ndim, 3)
        for request, expected in enumerate(case.expected):
            torch.testing.assert_close(
                output[request, : len(expected)].cpu(), expected, rtol=0, atol=0
            )

    @torch.no_grad()
    def _gpu_memory_case(self, label, fixture):
        device = torch.cuda.current_device()
        get_fraction = getattr(torch.cuda, "get_per_process_memory_fraction", None)
        old_fraction = get_fraction(device) if get_fraction is not None else None
        values = {
            "scope": "real production SWA geometry under controlled CUDA allocator budget; not full-model memory",
            "allocator_budget_bytes": 8 * 1024**3 if old_fraction is not None else None,
            "device": torch.cuda.get_device_name(device),
            "cp_size": 1,
            "head_dim": 512,
            "dtype": "bfloat16",
        }
        case = result = warm = None
        try:
            if old_fraction is not None:
                total = torch.cuda.get_device_properties(device).total_memory
                torch.cuda.set_per_process_memory_fraction(
                    min(old_fraction, 8 * 1024**3 / total), device
                )
            warm = make_fixture()
            warm_result = warm.owner._swa_prefill_concat(warm.qkv, warm.common)
            torch.cuda.synchronize()
            del warm_result
            warm = None
            gc.collect()
            lengths, prefixes = fixture
            case = make_fixture(lengths, prefixes, memory_geometry=True)
            sizes = [n + min(p, 127) for n, p in zip(lengths, prefixes)]
            payload = sum(sizes) * 1024
            values.update(
                batch=len(lengths),
                non_cache_tokens=sum(lengths),
                payload_bytes=payload,
                rectangular_bytes=len(lengths) * max(sizes) * 1024,
            )
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            result = case.owner._swa_prefill_concat(case.qkv, case.common)
            torch.cuda.synchronize()
            values.update(
                storage_bytes=result.untyped_storage().nbytes(),
                peak_increment_bytes=torch.cuda.max_memory_allocated(device) - before,
                allocator_oom=False,
            )
            write_measurement(label, values)
            self.assertEqual(values["storage_bytes"], payload)
            self.assertGreaterEqual(values["peak_increment_bytes"], payload)
            self.assertLessEqual(values["peak_increment_bytes"], payload + 64 * 1024**2)
        except torch.OutOfMemoryError as error:
            values.update(allocator_oom=True, exception=str(error))
            write_measurement(label, values)
            raise
        finally:
            case = result = warm = None
            gc.collect()
            if old_fraction is not None:
                torch.cuda.set_per_process_memory_fraction(old_fraction, device)

    def test_gpu_memory_ragged_53(self):
        self._gpu_memory_case("swa_ragged_53_memory", RAGGED_53)

    def test_gpu_memory_ragged_78(self):
        self._gpu_memory_case("swa_ragged_78_memory", RAGGED_78)


if __name__ == "__main__":
    unittest.main()
