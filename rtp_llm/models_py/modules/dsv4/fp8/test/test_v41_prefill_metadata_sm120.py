"""Upstream prefill metadata/workspace qualification on required SM120 hardware."""

import unittest

import torch

from rtp_llm.models.dsv41_kv_cache import dsv41_page_layouts
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_metadata as metadata
from rtp_llm.models_py.modules.dsv4.fp8.test import (
    test_v41_prefill_kv_workspace,
    test_v41_prefill_metadata,
)


class TaggedPrefillSlots(unittest.TestCase):
    def test_unsharded_global_and_full_index_page_slots(self):
        positions = [-1, 0, 1, 127, 128, 254, 255, 256, 257, 510, 511, 512, 513, 768]
        requests = [0] * len(positions) + [1] * len(positions) + [-1, 2]
        positions = positions * 2 + [0, 0]
        table_rows = [[3, 0, 7], [2, 5, 1]]
        for dtype in (torch.int32, torch.int64):
            pos = torch.tensor(positions, device="cuda", dtype=dtype)
            req = torch.tensor(requests, device="cuda", dtype=dtype)
            table = torch.tensor(table_rows, device="cuda", dtype=dtype)
            for ratio in (1, 2):
                main_tag = "hca_kv" if ratio == 1 else "csa_kv"
                for tag in (main_tag, "indexer_kv"):
                    with self.subTest(dtype=dtype, ratio=ratio, tag=tag):
                        entries = dsv41_page_layouts(0)[tag].entries
                        expected = []
                        for position, request in zip(positions, requests):
                            page, offset = divmod(max(position, 0), 256)
                            block = (
                                table_rows[request][page]
                                if 0 <= request < len(table_rows) and page < 3
                                else 0
                            )
                            expected.append(
                                block * entries + offset // ratio
                                if position >= 0
                                and (position + 1) % ratio == 0
                                and block > 0
                                else -1
                            )
                        actual = metadata.try_slot_mapping(
                            pos, req, table, entries, 256, ratio, cp_size=1, cp_rank=0
                        )
                        self.assertEqual(actual.cpu().tolist(), expected)
                        if ratio == 2 and tag == "indexer_kv":
                            # Physical index pages stay 256 entries even for CSA.
                            self.assertEqual(entries, 256)
                            self.assertEqual(actual[2].item(), 3 * 256)

    def test_persistent_state_uses_one_block_with_generation_slack(self):
        ends = [512, 519]
        blocks = [3, 7]
        positions = [position for end in ends for position in range(248, end)]
        requests = [
            request for request, end in enumerate(ends) for _ in range(248, end)
        ]
        for gen in (0, 1, 7):
            with self.subTest(gen=gen):
                entries = dsv41_page_layouts(gen)["csa_state"].entries
                retained = [
                    {
                        position
                        for begin in range(0, end, 256)
                        for position in range(
                            max(begin, min(begin + 256, end) - entries),
                            min(begin + 256, end),
                        )
                    }
                    for end in ends
                ]
                expected = [
                    (
                        blocks[request] * entries + position % entries
                        if position in retained[request]
                        else -1
                    )
                    for position, request in zip(positions, requests)
                ]
                actual = metadata.try_slot_mapping(
                    torch.tensor(positions, device="cuda"),
                    torch.tensor(requests, device="cuda"),
                    torch.tensor(blocks, device="cuda")[:, None],
                    entries,
                    256,
                    2,
                    state=True,
                    seq_ends=torch.tensor(ends, device="cuda"),
                )
                self.assertEqual(actual.cpu().tolist(), expected)


def load_tests(loader, tests, pattern):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("Prefill metadata qualification requires a real SM120 GPU")
    tests.addTests(loader.loadTestsFromModule(test_v41_prefill_metadata))
    tests.addTests(loader.loadTestsFromModule(test_v41_prefill_kv_workspace))
    return tests


if __name__ == "__main__":
    unittest.main()
