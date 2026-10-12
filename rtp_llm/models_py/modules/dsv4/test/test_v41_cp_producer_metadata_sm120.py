"""Donor CP integers with native unsharded 256-token producer layouts."""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models.dsv41_kv_cache import dsv41_page_layouts
from rtp_llm.models_py.modules.dsv4 import _cp_metadata_triton as cp_metadata
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_producer_metadata as producer
from rtp_llm.models_py.modules.dsv4.fp8._v41_prefill_metadata import try_slot_mapping


def inputs(prefixes, sizes, device):
    starts = torch.tensor(prefixes, dtype=torch.int64, device=device)
    lengths = torch.tensor(sizes, dtype=torch.int64, device=device)
    positions = torch.cat(
        [torch.arange(p, p + n, device=device) for p, n in zip(prefixes, sizes)]
    )
    requests = torch.repeat_interleave(torch.arange(len(sizes), device=device), lengths)
    cp = SimpleNamespace(
        cp_size=4,
        cp_rank=0,
        kv_cache_sharded=False,
        input_lengths_global_host=sizes,
        prefix_lengths_host=prefixes,
    )
    return cp, positions, requests, starts, lengths


def native_slots(prefixes, sizes, ratio, device, *, cuda=False):
    layout = dsv41_page_layouts(0)
    tags = ("csa_kv" if ratio == 2 else "hca_kv", "indexer_kv", "csa_state")
    columns = (max(p + n for p, n in zip(prefixes, sizes)) + 255) // 256
    tables = {}
    for index, tag in enumerate(tags):
        width = 1 if tag == "csa_state" else columns
        full = torch.arange(
            1 + index * 10000,
            1 + index * 10000 + len(sizes) * (width + 2),
            device=device,
            dtype=torch.int32,
        ).reshape(len(sizes), width + 2)
        tables[tag] = full[:, 1 : width + 1]

    def oracle(tag, pos, req, state_end=None):
        entries = layout[tag].entries
        state = tag == "csa_state"
        if state:
            pages = tables[tag][req, 0].long()
            valid = torch.ones_like(pos, dtype=torch.bool)
            if state_end is not None:
                end = state_end[req]
                valid = (pos + entries >= end) & (pos < end)
            offset = pos.remainder(entries)
        else:
            pages = tables[tag][req, pos // 256].long()
            offset = pos.remainder(256) // ratio
            valid = (pos + 1).remainder(ratio) == 0
        return torch.where(valid, pages * entries + offset, -1)

    def actual(tag, pos, req, state_end=None):
        if not cuda:
            return oracle(tag, pos, req, state_end)
        return try_slot_mapping(
            pos,
            req,
            tables[tag],
            layout[tag].entries,
            256,
            ratio,
            state=tag == "csa_state",
            seq_ends=state_end,
            single_block_per_request=tag == "csa_state",
        )

    return tags, actual, oracle


class V41CPProducerMetadataSM120Test(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
        "SM120 CP dispatch metadata requires a GPU lease",
    )
    def test_sm120_forward_metadata_preserves_cp4_zigzag_padding_and_requests(self):
        for sizes in ([5], [1, 0, 255, 256, 257], [1] * 64):
            prefixes = [255 + 17 * i for i in range(len(sizes))]
            chunks = [2 * ((n + 7) // 8) for n in sizes]
            total = sum(chunks)
            for rank in range(4):
                padding, restore = [], []
                shuffle, relative, positions, requests = [], [], [], []
                unpad, local_start, padded_start = [], 0, 0
                for request, (prefix, size, chunk) in enumerate(
                    zip(prefixes, sizes, chunks)
                ):
                    padded = chunk * 4
                    half = chunk // 2
                    inverse = [0] * padded
                    for peer in range(4):
                        order = [
                            *range(peer * half, (peer + 1) * half),
                            *range(padded - (peer + 1) * half, padded - peer * half),
                        ]
                        for index, position in enumerate(order):
                            inverse[position] = peer * total + local_start + index
                        if peer == rank:
                            shuffle.extend(order)
                            relative.extend(padded_start + p for p in order)
                            positions.extend(
                                prefix + min(p, max(size - 1, 0)) for p in order
                            )
                            requests.extend([request] * chunk)
                    padding.extend([1] * size + [0] * (padded - size))
                    restore.extend(inverse)
                    unpad.extend(inverse[:size])
                    local_start += chunk
                    padded_start += padded

                def cuda(values, dtype=torch.int32):
                    return torch.tensor(values, dtype=dtype, device="cuda")

                actual = cp_metadata.try_build_cp_forward_metadata(
                    cuda(sizes),
                    cuda(chunks),
                    cuda(prefixes),
                    cuda(padding),
                    cuda(restore),
                    cuda(shuffle),
                    cp_size=4,
                    cp_rank=rank,
                    chunk_length=total,
                    seq_len_full=sum(sizes),
                )
                self.assertIsNotNone(actual)
                expected = (
                    cuda(relative, torch.int64),
                    cuda(positions, torch.int64),
                    cuda(requests),
                    cuda([bool(padding[p]) for p in relative], torch.bool),
                    cuda(unpad, torch.int64),
                    cuda([0, *torch.tensor(sizes).cumsum(0).tolist()]),
                    cuda(prefixes, torch.int64),
                )
                for value, reference in zip(actual, expected):
                    torch.testing.assert_close(value, reference, rtol=0, atol=0)

    def test_cpu_metadata_builders_keep_existing_fallback(self):
        lengths = torch.tensor([1, 0, 257])
        self.assertIsNone(
            cp_metadata.try_build_cp_full_prefill_positions(
                lengths, torch.tensor([0, 8, 255]), total_tokens=258
            )
        )
        self.assertIsNone(
            cp_metadata.try_build_cp_restore_indices(
                lengths,
                cp_size=4,
                owner_block_size=256,
                total_tokens=258,
                total_local_kv=512,
            )
        )
        cp, pos, req, starts, lens = inputs([0, 255], [3, 2], "cpu")
        tags, slots, _ = native_slots([0, 255], [3, 2], 2, "cpu")
        self.assertIsNone(
            producer.prepare_raw(cp, pos, req, starts, lens, 2, slots, tags)
        )

    def verify_metadata(self, prefixes, sizes, ratio, device, *, raw):
        cp, pos, req, starts, lens = inputs(prefixes, sizes, device)
        tags, slots, oracle = native_slots(
            prefixes, sizes, ratio, device, cuda=device == "cuda"
        )
        tiles = [(0, 0, n) for n in sizes]
        meta = (
            producer.prepare_raw(cp, pos, req, starts, lens, ratio, slots, tags)
            if raw
            else producer.prepare(cp, pos, req, starts, lens, ratio, tiles, slots, tags)
        )
        self.assertIsNotNone(meta)
        selected = torch.nonzero((pos + 1).remainder(ratio) == 0).flatten()
        ends = starts + lens
        torch.testing.assert_close(meta.positions, pos[selected], rtol=0, atol=0)
        torch.testing.assert_close(meta.requests, req[selected], rtol=0, atol=0)
        torch.testing.assert_close(meta.seq_ends, ends, rtol=0, atol=0)
        for tag, actual in zip(tags[:2], (meta.main_slots, meta.index_slots)):
            torch.testing.assert_close(
                actual, oracle(tag, pos[selected], req[selected]), rtol=0, atol=0
            )
        if ratio == 2:
            torch.testing.assert_close(
                meta.state_slots, oracle(tags[2], pos, req, ends), rtol=0, atol=0
            )
        else:
            self.assertIsNone(meta.state_slots)
        for start, (end, first, stop, phase) in meta.segments.items():
            indices, p, r, main, index, state = meta.tile(start, end)
            logical = selected[(selected >= start) & (selected < end)]
            torch.testing.assert_close(indices, logical - start, rtol=0, atol=0)
            torch.testing.assert_close(p, pos[logical], rtol=0, atol=0)
            torch.testing.assert_close(r, req[logical], rtol=0, atol=0)
            torch.testing.assert_close(main, oracle(tags[0], p, r), rtol=0, atol=0)
            torch.testing.assert_close(index, oracle(tags[1], p, r), rtol=0, atol=0)
            if ratio == 2:
                torch.testing.assert_close(
                    state,
                    oracle(tags[2], pos[start:end], req[start:end], ends),
                    rtol=0,
                    atol=0,
                )
        return meta

    def test_cpu_owner_segments_keep_odd_prefix_and_native_ring_slots(self):
        for ratio in (1, 2):
            self.verify_metadata(
                [0, 255, 256, 511], [3, 2, 5, 4], ratio, "cpu", raw=False
            )

    def test_cpu_invalid_owner_segments_and_topology_use_fallback(self):
        cp, pos, req, starts, lens = inputs([0, 255], [3, 2], "cpu")
        tags, slots, _ = native_slots([0, 255], [3, 2], 2, "cpu")
        for tiles in ([(0, 0, 4), (0, 0, 1)], [(0, 0, 3)], []):
            self.assertIsNone(
                producer.prepare(cp, pos, req, starts, lens, 2, tiles, slots, tags)
            )
        cp.cp_size = 2
        self.assertIsNone(
            producer.prepare(
                cp, pos, req, starts, lens, 2, [(0, 0, 3), (0, 0, 2)], slots, tags
            )
        )

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
        "SM120 metadata requires a GPU lease",
    )
    def test_sm120_full_positions_preserve_empty_requests_prefixes_and_batch_bound(
        self,
    ):
        for prefixes, sizes in (
            ([255], [257]),
            ([0, 100, 255, 256, 511], [1, 0, 255, 256, 257]),
            (list(range(64)), [1] * 64),
        ):
            _, pos, req, starts, lens = inputs(prefixes, sizes, "cuda")
            actual = cp_metadata.try_build_cp_full_prefill_positions(
                lens, starts, total_tokens=sum(sizes)
            )
            self.assertIsNotNone(actual)
            for a, b in zip(actual, (pos, req)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        many = torch.ones(65, device="cuda", dtype=torch.int64)
        self.assertIsNone(
            cp_metadata.try_build_cp_full_prefill_positions(many, many, total_tokens=65)
        )

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
        "SM120 metadata requires a GPU lease",
    )
    def test_sm120_cp_restore_matches_independent_request_page_map(self):
        for sizes in ([257], [1, 0, 255, 256, 257]):
            for block in (128, 256):
                local = sum(((n + 4 * block - 1) // (4 * block)) * block for n in sizes)
                lens = torch.tensor(sizes, device="cuda", dtype=torch.int64)
                actual = cp_metadata.try_build_cp_restore_indices(
                    lens,
                    cp_size=4,
                    owner_block_size=block,
                    total_tokens=sum(sizes),
                    total_local_kv=local,
                )
                reference, start = [], 0
                for n in sizes:
                    reference.extend(
                        (p // block % 4) * local
                        + start
                        + (p // (4 * block)) * block
                        + p % block
                        for p in range(n)
                    )
                    start += ((n + 4 * block - 1) // (4 * block)) * block
                torch.testing.assert_close(
                    actual.cpu(), torch.tensor(reference), rtol=0, atol=0
                )

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
        "SM120 producer metadata requires a GPU lease",
    )
    def test_sm120_raw_tiles_keep_native_tagged_slots_across_requests_and_pages(self):
        for ratio in (1, 2):
            self.verify_metadata(
                [0, 255, 1, 253, 256, 511],
                [3, 2, 513, 32761, 1, 257],
                ratio,
                "cuda",
                raw=True,
            )
        empty = self.verify_metadata([0, 256], [1, 1], 2, "cuda", raw=True)
        self.assertEqual(empty.positions.numel(), 0)


if __name__ == "__main__":
    unittest.main()
