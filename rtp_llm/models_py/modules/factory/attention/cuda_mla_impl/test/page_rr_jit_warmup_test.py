"""Check Page-RR JIT reuse against the current owner-local table contract."""

import contextlib
import unittest
from itertools import accumulate
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.page_rr_mla_metadata import (
    PageRRMlaMetadata,
)
from rtp_llm.models_py.triton_kernels.mla import page_rr

PAGE, KERNEL = 4096, 128


def source_table(batch, pages, tp, rank):
    """Enumerate owned logical pages independently of production slot helpers."""
    owned = list(range(rank, pages, tp))
    columns = len(owned) * (PAGE // KERNEL)
    host = torch.arange(1, batch * columns + 1, dtype=torch.int32).reshape(
        batch, columns
    )
    storage = torch.zeros((batch, columns + 5), dtype=torch.int32, device="cuda")
    table = storage[:, :columns]
    table.copy_(host)
    maps = []
    for request in range(batch):
        slots = {}
        for local_page, logical_page in enumerate(owned):
            for subpage in range(PAGE // KERNEL):
                physical = int(host[request, local_page * (PAGE // KERNEL) + subpage])
                for offset in range(KERNEL):
                    position = logical_page * PAGE + subpage * KERNEL + offset
                    slots[position] = physical * KERNEL + offset
        maps.append(slots)
    return table, maps


@contextlib.contextmanager
def compile_attempts(kernel):
    """Count JIT lookups, including disk hits, rather than only compilation."""
    from triton import knobs

    kernel.device_caches.clear()
    attempts, previous = [], knobs.runtime.jit_cache_hook

    def listen(**event):
        if event["fn"].jit_function is kernel:
            attempts.append(event["key"])
        if previous is not None:
            return previous(**event)
        return None

    knobs.runtime.jit_cache_hook = listen
    try:
        yield attempts
    finally:
        knobs.runtime.jit_cache_hook = previous


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/Triton")
class PageRRJitWarmupTest(unittest.TestCase):
    def test_prefill_request_shapes_reuse_jit(self):
        with compile_attempts(page_rr._prepare_prefill_metadata) as attempts:
            for lengths, pages in (([7], 2), ([0, 2, 7], 4)):
                batch, total = len(lengths), sum(lengths)
                table, maps = source_table(batch, pages, 2, 1)
                prefixes = [PAGE - 2 + 3 * r for r in range(batch)]
                padding = [
                    r * max(lengths) + j - row
                    for row, (r, j) in enumerate(
                        (r, j) for r, n in enumerate(lengths) for j in range(n)
                    )
                ]
                inputs = SimpleNamespace(
                    input_lengths=torch.tensor(lengths, dtype=torch.int32),
                    prefix_lengths=torch.tensor(prefixes, dtype=torch.int32),
                    input_lengths_device=torch.tensor(
                        lengths, dtype=torch.int32, device="cuda"
                    ),
                    prefix_lengths_device=torch.tensor(
                        prefixes, dtype=torch.int32, device="cuda"
                    ),
                    cu_seqlens_device=torch.tensor(
                        list(accumulate([0] + lengths)),
                        dtype=torch.int32,
                        device="cuda",
                    ),
                    kv_cache_kernel_block_id_device=table,
                    logical_token_count=total - 1,
                    physical_token_count=total,
                    padding_offset=torch.tensor(
                        padding, dtype=torch.int32, device="cuda"
                    ),
                )
                metadata = PageRRMlaMetadata(PAGE, KERNEL, 2, 1)
                metadata.prepare_prefill(inputs)
                expected = [
                    maps[r].get(prefixes[r] + j, -1)
                    for r, n in enumerate(lengths)
                    for j in range(n)
                ]
                expected[-1] = -1
                self.assertEqual(metadata.slot_mapping.cpu().tolist(), expected)
        self.assertEqual(len(attempts), 1, attempts)

    def test_decode_request_shapes_reuse_jit(self):
        for mtp in (False, True):
            with compile_attempts(page_rr._prepare_decode_metadata) as attempts:
                for batch, query, pages in ((1, 1, 2), (3, 4 if mtp else 1, 4)):
                    table, maps = source_table(batch, pages, 2, 1)
                    lengths = (
                        ([0] + [query] * (batch - 1))
                        if mtp and batch > 1
                        else [query] * batch
                    )
                    rows = batch * query
                    starts = [PAGE - 1 + r for r in range(batch)]
                    base = torch.tensor(starts, dtype=torch.int32, device="cuda")
                    inputs = SimpleNamespace(
                        input_lengths=torch.tensor(lengths, dtype=torch.int32),
                        physical_token_count=rows,
                        is_target_verify=False,
                        is_mtp_draft_update=mtp,
                        is_cuda_graph=False,
                        prefix_lengths_device=base,
                        sequence_lengths_plus_1_device=base + 1,
                        kv_cache_kernel_block_id_device=table,
                        cu_seqlens_device=torch.tensor(
                            list(accumulate([0] + lengths)),
                            dtype=torch.int32,
                            device="cuda",
                        ),
                        # Zero-length request slots are suppressed by cu_seqlens
                        # without compacting subsequent requests into their rows.
                        valid_token_mask=torch.ones(
                            rows, dtype=torch.bool, device="cuda"
                        ),
                    )
                    metadata = PageRRMlaMetadata(PAGE, KERNEL, 2, 1)
                    metadata.prepare_decode(inputs)
                    positions = [
                        starts[r] + j if j < n else 0
                        for r, n in enumerate(lengths)
                        for j in range(query)
                    ]
                    slots = [
                        maps[r].get(starts[r] + j, -1) if j < n else -1
                        for r, n in enumerate(lengths)
                        for j in range(query)
                    ]
                    self.assertEqual(
                        metadata.positions_d.cpu().tolist(),
                        positions,
                    )
                    self.assertEqual(
                        metadata.slot_mapping.cpu().tolist(),
                        slots,
                    )
            self.assertEqual(len(attempts), 1, attempts)

    def test_compact_target_verify_shapes_reuse_jit(self):
        query = 4
        with compile_attempts(page_rr._prepare_decode_metadata) as attempts:
            for batch, pages in ((1, 2), (3, 4)):
                table, maps = source_table(batch, pages, 2, 1)
                starts = [PAGE - 1 + r for r in range(batch)]
                base = torch.tensor(starts, dtype=torch.int32, device="cuda")
                inputs = SimpleNamespace(
                    input_lengths=torch.full((batch,), query, dtype=torch.int32),
                    physical_token_count=batch * query,
                    is_target_verify=True,
                    is_mtp_draft_update=False,
                    is_cuda_graph=False,
                    prefix_lengths_device=base,
                    kv_cache_kernel_block_id_device=table,
                )
                metadata = PageRRMlaMetadata(PAGE, KERNEL, 2, 1)
                for increment in (0, 1):
                    if increment:
                        base.add_(1)
                    metadata.prepare_decode(
                        inputs, forbid_realloc=bool(increment), compact_queries=True
                    )
                    positions = [
                        [start + increment + j for j in range(query)]
                        for start in starts
                    ]
                    # Count independently enumerated owner-local keys, including
                    # the query that straddles the first rank-1 ownership page.
                    lengths = [
                        [sum(key <= position for key in maps[r]) for position in row]
                        for r, row in enumerate(positions)
                    ]
                    self.assertEqual(metadata.local_causal_lens.shape, (batch, query))
                    self.assertEqual(metadata.local_causal_lens.dtype, torch.int32)
                    self.assertEqual(metadata.local_causal_lens.cpu().tolist(), lengths)
                    self.assertEqual(metadata.local_seq_lens.shape, (batch,))
                    self.assertEqual(metadata.local_seq_lens.dtype, torch.int32)
                    self.assertTrue(metadata.local_seq_lens.is_contiguous())
                    self.assertEqual(
                        metadata.local_seq_lens.cpu().tolist(),
                        [max(row) for row in lengths],
                    )
                    self.assertEqual(metadata.global_causal_lens.shape, (batch,))
                    self.assertEqual(metadata.global_causal_lens.dtype, torch.int32)
                    self.assertTrue(metadata.global_causal_lens.is_contiguous())
                    self.assertEqual(
                        metadata.global_causal_lens.cpu().tolist(),
                        [row[-1] + 1 for row in positions],
                    )
                    self.assertEqual(metadata.query_block_tables.shape, table.shape)
                    self.assertEqual(metadata.query_block_tables.dtype, torch.int32)
                    self.assertTrue(metadata.query_block_tables.is_contiguous())
                    self.assertEqual(
                        metadata.query_block_tables.cpu().tolist(), table.cpu().tolist()
                    )
                    self.assertEqual(
                        metadata.positions_d.cpu().tolist(),
                        [position for row in positions for position in row],
                    )
                    self.assertEqual(
                        metadata.slot_mapping.cpu().tolist(),
                        [
                            maps[r].get(position, -1)
                            for r, row in enumerate(positions)
                            for position in row
                        ],
                    )
        self.assertEqual(len(attempts), 1, attempts)


if __name__ == "__main__":
    unittest.main()
