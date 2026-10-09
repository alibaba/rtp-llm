"""Upstream decode cache production with SM120's 256-token tagged pages."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models.dsv41_kv_cache import dsv41_page_layouts
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_decode_indexer as indexer
from rtp_llm.models_py.modules.dsv4.fp8.test import test_v41_decode_global as donor
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_compressor_precision import (
    compressor_bf16,
)
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import CSA_STATE, INDEXER_KV


class DecodeGlobalSM120(unittest.TestCase):
    def assert_rounded_result(self, old, new, before):
        donor._assert_result(self, old, new, before)
        # Accepted state remains raw FP32, including zero signs; BF16 rounding
        # belongs only to the normalization input, never the state ring.
        self.assertTrue(
            torch.equal(
                old.owner._test_pools[CSA_STATE][1:].view(torch.uint8),
                new.owner._test_pools[CSA_STATE][1:].view(torch.uint8),
            )
        )

    @torch.inference_mode()
    def test_fused_bf16_rounding_native_pages_and_norm_dtypes(self):
        torch.manual_seed(411209)
        for ratio in (1, 2):
            for dtype in (torch.float32, torch.bfloat16):
                for gen in (0, 1):
                    for span in range(1, gen + 2):
                        with (
                            self.subTest(ratio=ratio, dtype=dtype, gen=gen, span=span),
                            patch.object(donor, "_KV_TPB", 256),
                            compressor_bf16(True, fused=True),
                        ):
                            old = donor._case(
                                "cuda",
                                ratio,
                                2,
                                span,
                                norm_dtype=dtype,
                                ring_entries=dsv41_page_layouts(gen)[CSA_STATE].entries,
                            )
                            new = donor._clone_case(old)
                            for starts in (
                                [126, 254],
                                [127, 255],
                                [128, 256],
                                [129, 257],
                            ):
                                for case in (old, new):
                                    donor._set_starts(case, starts)
                                value = torch.randn_like(old.x)
                                old.x.copy_(value)
                                new.x.copy_(value)
                                before = donor._snapshot(old)
                                donor._reference(old)
                                donor._candidate(new)
                                self.assert_rounded_result(old, new, before)

    @torch.inference_mode()
    def test_fused_bf16_rounding_unallocated_and_out_of_table_rows(self):
        for ratio in (1, 2):
            with (
                self.subTest(ratio=ratio),
                patch.object(donor, "_KV_TPB", 256),
                compressor_bf16(True, fused=True),
            ):
                old = donor._case(
                    "cuda",
                    ratio,
                    4,
                    2,
                    norm_dtype=torch.bfloat16,
                    ring_entries=dsv41_page_layouts(1)[CSA_STATE].entries,
                )
                donor._set_starts(old, [127, 255, 0, 511])
                for table in old.owner._block_tables_by_type.values():
                    table[0, 0] = 0
                    table[1, min(1, table.shape[1] - 1)] = -1
                    table[2].zero_()
                new = donor._clone_case(old)
                before = donor._snapshot(old)
                donor._reference(old)
                donor._candidate(new)
                self.assert_rounded_result(old, new, before)

    @torch.inference_mode()
    def test_fused_bf16_rounding_changed_input_graph_replay(self):
        torch.manual_seed(411210)
        for ratio in (1, 2):
            with (
                self.subTest(ratio=ratio),
                patch.object(donor, "_KV_TPB", 256),
                compressor_bf16(True, fused=True),
            ):
                old = donor._case(
                    "cuda",
                    ratio,
                    2,
                    2,
                    norm_dtype=torch.bfloat16,
                    ring_entries=dsv41_page_layouts(1)[CSA_STATE].entries,
                )
                new = donor._clone_case(old)
                initial = donor._snapshot(old)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        donor._candidate(new)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    donor._candidate(new)
                for region, pool in initial.items():
                    new.owner._test_pools[region].copy_(pool)
                for round_id, starts in enumerate(([127, 255], [128, 256], [129, 257])):
                    before = donor._snapshot(old)
                    for case in (old, new):
                        donor._set_starts(case, starts)
                        if round_id == 1:
                            for table in case.owner._block_tables_by_type.values():
                                table[1, -1] = -1
                    value = torch.randn_like(old.x)
                    old.x.copy_(value)
                    new.x.copy_(value)
                    donor._reference(old)
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assert_rounded_result(old, new, before)
                del graph

    @torch.inference_mode()
    def test_native_page_and_generation_layouts(self):
        for ratio in (1, 2):
            for gen in (0, 1):
                for span in range(1, gen + 2):
                    with self.subTest(ratio=ratio, gen=gen, span=span), patch.object(
                        donor, "_KV_TPB", 256
                    ):
                        layouts = dsv41_page_layouts(gen)
                        old = donor._case(
                            "cuda",
                            ratio,
                            2,
                            span,
                            ring_entries=layouts[CSA_STATE].entries,
                        )
                        new = donor._clone_case(old)
                        for tag in (old.owner._global_region(), INDEXER_KV):
                            self.assertEqual(
                                old.owner._test_pools[tag].shape[1],
                                layouts[tag].entries,
                            )
                        for starts in ([126, 254], [127, 255], [128, 256], [129, 257]):
                            for case in (old, new):
                                donor._set_starts(case, starts)
                            new.x.normal_()
                            old.x.copy_(new.x)
                            before = donor._snapshot(old)
                            donor._reference(old)
                            donor._candidate(new)
                            donor._assert_result(self, old, new, before)

    def test_sm100_paged_scorer_retains_sm120_fallback(self):
        for block in (64, 128, 256):
            self.assertFalse(indexer.is_supported(torch.device("cuda"), block))


def load_tests(loader, tests, pattern):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("Decode cache qualification requires a real SM120 GPU")
    tests.addTests(loader.loadTestsFromModule(donor))
    return tests


if __name__ == "__main__":
    unittest.main()
