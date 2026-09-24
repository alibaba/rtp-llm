from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv4.fp8.decode._fused_prepare_meta_triton import (
    fused_phase2b_pool_slot_mapping,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.pool_slot_mapping import (
    compute_kv_pool_slot_mapping,
)
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
    CSA_KV,
    HCA_KV,
    INDEXER_KV,
    SWA_KV,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for Triton parity")
class FusedPoolSlotMappingContractTest(unittest.TestCase):
    def _meta(
        self,
        tag: str,
        q_len: int,
        block_table: torch.Tensor | None = None,
        output: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        if block_table is None:
            block_table = torch.tensor([[11, 999]], dtype=torch.int32, device="cuda")
        if output is None:
            output = torch.full((q_len,), -777, dtype=torch.int64, device="cuda")
        return SimpleNamespace(
            q_len_per_req=q_len,
            pool_block_tables={tag: block_table},
            pool_write_slot_mappings={tag: output},
        )

    def _assert_parity(
        self,
        *,
        tag: str,
        start: int,
        q_len: int,
        entries: int,
        raw_tokens_per_block: int,
        ratio: int,
    ) -> torch.Tensor:
        meta = self._meta(tag, q_len)
        start_pos = torch.tensor([start], dtype=torch.int32, device="cuda")
        entries_by_tag = {tag: entries}
        raw_tokens_by_tag = {tag: raw_tokens_per_block}
        fused_phase2b_pool_slot_mapping(
            meta,
            start_pos,
            bs=1,
            paged_pool_entries_per_block=entries_by_tag,
            paged_pool_tokens_per_block=raw_tokens_by_tag,
        )

        positions = start_pos.view(1, 1) + torch.arange(
            q_len, dtype=torch.int32, device="cuda"
        ).view(1, q_len)
        if ratio == 1:
            pool_positions = positions.reshape(-1)
            pool_tokens_per_block = raw_tokens_per_block
        else:
            plus_one = positions + 1
            on_boundary = plus_one % ratio == 0
            compressed = plus_one // ratio - 1
            pool_positions = torch.where(
                on_boundary, compressed, torch.full_like(compressed, -1)
            ).reshape(-1)
            pool_tokens_per_block = raw_tokens_per_block // ratio
        eager = compute_kv_pool_slot_mapping(
            meta.pool_block_tables[tag],
            pool_positions,
            pool_entries_per_block=entries,
            pool_tokens_per_block=pool_tokens_per_block,
            ring_entries=entries,
        )
        torch.testing.assert_close(meta.pool_write_slot_mappings[tag], eager)
        return meta.pool_write_slot_mappings[tag]

    def test_eager_fused_parity_at_valid_last_row_and_oob(self) -> None:
        cases = (
            # Last valid SWA position 511, first OOB position 512.
            dict(tag=SWA_KV, start=511, q_len=2, entries=128, ratio=1),
            # CSA/indexer: p=511 -> compressed 127 valid; p=515 -> 128 OOB.
            dict(tag=CSA_KV, start=511, q_len=5, entries=64, ratio=4),
            dict(tag=INDEXER_KV, start=511, q_len=5, entries=64, ratio=4),
            # HCA: p=511 -> compressed 3 valid; p=639 -> 4 OOB.
            dict(tag=HCA_KV, start=511, q_len=129, entries=2, ratio=128),
        )
        for case in cases:
            with self.subTest(tag=case["tag"]):
                self._assert_parity(raw_tokens_per_block=256, **case)

    def test_oob_never_aliases_positive_tail_column(self) -> None:
        for tag, start, q_len, entries, ratio in (
            (SWA_KV, 512, 1, 128, 1),
            (CSA_KV, 515, 1, 64, 4),
            (INDEXER_KV, 515, 1, 64, 4),
            (HCA_KV, 639, 1, 2, 128),
        ):
            with self.subTest(tag=tag):
                actual = self._assert_parity(
                    tag=tag,
                    start=start,
                    q_len=q_len,
                    entries=entries,
                    raw_tokens_per_block=256,
                    ratio=ratio,
                )
                self.assertEqual(actual.tolist(), [-1])

    def test_launcher_rejects_zero_batch_before_zero_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid fused slot shape"):
            fused_phase2b_pool_slot_mapping(
                self._meta(SWA_KV, 1),
                torch.empty((0,), dtype=torch.int32, device="cuda"),
                0,
                {SWA_KV: 128},
                {SWA_KV: 256},
            )

    def test_launcher_rejects_missing_pair_and_bad_start(self) -> None:
        start = torch.tensor([0], dtype=torch.int32, device="cuda")
        meta = self._meta(SWA_KV, 1)
        del meta.pool_write_slot_mappings[SWA_KV]
        with self.assertRaisesRegex(ValueError, "requires both"):
            fused_phase2b_pool_slot_mapping(meta, start, 1, {SWA_KV: 128}, {SWA_KV: 256})

        bad_starts = {
            "dtype": start.to(torch.int64),
            "device": start.cpu(),
            "rank": start.view(1, 1),
            "stride": torch.zeros((4,), dtype=torch.int32, device="cuda")[::2],
        }
        for name, bad_start in bad_starts.items():
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "start_pos"
            ):
                fused_phase2b_pool_slot_mapping(
                    self._meta(SWA_KV, 1),
                    bad_start,
                    1,
                    {SWA_KV: 128},
                    {SWA_KV: 256},
                )
        with self.assertRaisesRegex(ValueError, "start_pos"):
            fused_phase2b_pool_slot_mapping(
                self._meta(SWA_KV, 1), start, 2, {SWA_KV: 128}, {SWA_KV: 256}
            )

    def test_launcher_rejects_block_table_contract_violations(self) -> None:
        bad_tables = {
            "rank": torch.ones((2,), dtype=torch.int32, device="cuda"),
            "dtype": torch.ones((1, 2), dtype=torch.int64, device="cuda"),
            "device": torch.ones((1, 2), dtype=torch.int32),
            "capacity": torch.ones((1, 2), dtype=torch.int32, device="cuda"),
            "stride": torch.ones((1, 4), dtype=torch.int32, device="cuda").as_strided(
                (1, 2), (4, 2)
            ),
        }
        for name, block_table in bad_tables.items():
            case_bs = 2 if name == "capacity" else 1
            case_start = torch.zeros(
                (case_bs,), dtype=torch.int32, device="cuda"
            )
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "block table"
            ):
                fused_phase2b_pool_slot_mapping(
                    self._meta(SWA_KV, 1, block_table=block_table),
                    case_start,
                    case_bs,
                    {SWA_KV: 128},
                    {SWA_KV: 256},
                )

    def test_launcher_rejects_output_contract_and_geometry(self) -> None:
        start = torch.tensor([0], dtype=torch.int32, device="cuda")
        bad_outputs = {
            "dtype": torch.empty((2,), dtype=torch.int32, device="cuda"),
            "device": torch.empty((2,), dtype=torch.int64),
            "capacity": torch.empty((0,), dtype=torch.int64, device="cuda"),
            "stride": torch.empty((4,), dtype=torch.int64, device="cuda")[::2],
        }
        for name, output in bad_outputs.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "output"):
                fused_phase2b_pool_slot_mapping(
                    self._meta(SWA_KV, 2, output=output),
                    start,
                    1,
                    {SWA_KV: 128},
                    {SWA_KV: 256},
                )

        with self.assertRaisesRegex(ValueError, "geometry"):
            fused_phase2b_pool_slot_mapping(
                self._meta(SWA_KV, 1), start, 1, {SWA_KV: 0}, {SWA_KV: 256}
            )

        for name, entries, tokens in (
            ("entries", {}, {SWA_KV: 256}),
            ("tokens", {SWA_KV: 128}, {}),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, rf"missing .*tag={SWA_KV}"
            ):
                fused_phase2b_pool_slot_mapping(
                    self._meta(SWA_KV, 1), start, 1, entries, tokens
                )


if __name__ == "__main__":
    unittest.main()
