"""Upstream decode cache production with SM120's 256-token tagged pages."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models.dsv41_kv_cache import dsv41_page_layouts
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_decode_indexer as indexer
from rtp_llm.models_py.modules.dsv4.fp8.test import test_v41_decode_global as donor
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import CSA_STATE, INDEXER_KV


class DecodeGlobalSM120(unittest.TestCase):
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
