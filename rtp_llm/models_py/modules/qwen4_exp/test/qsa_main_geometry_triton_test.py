"""One-launch Triton verdict for the main paged-cache geometry.

Production timelines showed the head of every layer's QSA window is the
19-launch aten verdict chain of ``_validate_main_paged_geometry_uncached``
(required-column math, the arange/le/ge mask pair, three ``any`` reductions,
the bitwise verdict combination and ``_assert_async``): ~122us of host
dispatch per layer for 22us of GPU work, and it cannot be shared across
layers because every QSA layer owns its own main KV pool (the step memo's
cache fingerprint never matches cross-layer).

This test locks the replacement contract:

* the single Triton reduction's verdict matches the legacy aten chain
  bit-for-bit across valid and adversarial geometries (including the
  whole-table out-of-pool term that covers stale columns);
* the production (non-strict, non-graph) path enqueues exactly two
  device kernels per layer (the reduction plus the async assert);
* oversized/unsupported shapes fall back to the legacy aten chain with the
  same verdict;
* no new compiled variants appear for new table shapes after warm-up (the
  fixed-lane reduction has no shape-valued constexpr).
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

_DEV = "cuda"


def _legacy_verdict(table, bases, lengths, pool_blocks, page):
    """The replaced async aten chain, verbatim, for bit-exact comparison."""
    visible_ends = bases + lengths
    required_columns = (visible_ends + page - 1) // page
    covered = required_columns <= int(table.shape[1])
    columns = torch.arange(
        int(table.shape[1]), device=table.device
    ).unsqueeze(0)
    required = columns < required_columns.unsqueeze(1)
    invalid_blocks = (table <= 0) | (table >= pool_blocks)
    verdict = (
        torch.any(~covered)
        | torch.any(table >= pool_blocks)
        | torch.any(required & invalid_blocks)
    )
    return bool((~verdict).item())


class _Case:
    def __init__(self, tag, bases, lengths, width, pool, page, rows, valid):
        self.tag = tag
        self.bases = bases
        self.lengths = lengths
        self.width = width
        self.pool = pool
        self.page = page
        self.rows = rows
        self.valid = valid

    def tensors(self, device=_DEV, dtype=torch.int32):
        bases = torch.tensor(self.bases, dtype=dtype, device=device)
        lengths = torch.tensor(self.lengths, dtype=dtype, device=device)
        table = torch.zeros(
            (len(self.rows), self.width), dtype=torch.int32, device=device
        )
        for r, row in enumerate(self.rows):
            table[r, : len(row)] = torch.tensor(
                row, dtype=torch.int32, device=device
            )
        return table, bases, lengths


_CASES = [
    _Case(
        "valid_two_requests",
        bases=[0, 128], lengths=[512, 256], width=8, pool=64, page=128,
        rows=[[1, 2, 3, 4, 0, 0, 0, 0], [5, 6, 7, 9, 0, 0, 0, 0]],
        valid=True,
    ),
    _Case(
        "uncovered_required_columns",
        bases=[0], lengths=[1152], width=8, pool=64, page=128,
        rows=[[1, 2, 3, 4, 5, 6, 7, 8]],
        valid=False,
    ),
    _Case(
        "required_column_unallocated",
        bases=[0], lengths=[384], width=8, pool=64, page=128,
        rows=[[1, 2, 0, 0, 0, 0, 0, 0]],
        valid=False,
    ),
    _Case(
        "required_column_negative",
        bases=[0], lengths=[384], width=8, pool=64, page=128,
        rows=[[1, 2, -3, 0, 0, 0, 0, 0]],
        valid=False,
    ),
    _Case(
        "stale_column_out_of_pool",
        bases=[0], lengths=[256], width=8, pool=64, page=128,
        rows=[[1, 2, 0, 0, 0, 0, 0, 70]],
        valid=False,
    ),
    _Case(
        "pool_boundary_inclusive",
        bases=[0], lengths=[256], width=4, pool=64, page=128,
        rows=[[1, 63, 0, 0]],
        valid=True,
    ),
    _Case(
        "pool_boundary_exceeded",
        bases=[0], lengths=[256], width=4, pool=64, page=128,
        rows=[[1, 64, 0, 0]],
        valid=False,
    ),
    _Case(
        "mixed_batch_one_bad_row",
        bases=[0, 0], lengths=[256, 256], width=4, pool=64, page=128,
        rows=[[1, 2, 0, 0], [3, 0, 0, 0]],
        valid=False,
    ),
    _Case(
        "production_width_55",
        bases=[5248], lengths=[1440], width=55, pool=4096, page=128,
        rows=[[((i % 4000) + 1) for i in range(55)]],
        valid=True,
    ),
    _Case(
        "page_64",
        bases=[128], lengths=[192], width=6, pool=32, page=64,
        rows=[[1, 2, 3, 4, 5, 0]],
        valid=True,
    ),
    _Case(
        "page_64_uncovered",
        bases=[0], lengths=[448], width=6, pool=32, page=64,
        rows=[[1, 2, 3, 4, 5, 6, 7][:6]],
        valid=False,
    ),
]


class MainGeometryVerdictTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_verdict_matches_legacy_chain(self):
        from rtp_llm.models_py.modules.qwen4_exp.qsa.qsa_validation_triton import (
            main_geometry_supported,
            main_geometry_verdict,
        )

        for case in _CASES:
            with self.subTest(case.tag):
                table, bases, lengths = case.tensors()
                self.assertTrue(
                    main_geometry_supported(table, bases, lengths), case.tag
                )
                got = bool(
                    main_geometry_verdict(
                        table,
                        bases,
                        lengths,
                        pool_blocks=case.pool,
                        page_size=case.page,
                    ).item()
                )
                expected = _legacy_verdict(
                    table, bases, lengths, case.pool, case.page
                )
                self.assertEqual(expected, case.valid, case.tag)
                self.assertEqual(got, expected, case.tag)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_oversized_tables_fall_back_to_the_torch_chain(self):
        from rtp_llm.models_py.modules.qwen4_exp.qsa.qsa_validation_triton import (
            main_geometry_supported,
        )

        # 2 x 4096 exceeds the one-program bound; the verdict must still be
        # correct through the aten fallback inside the production path.
        table = torch.zeros((2, 4096), dtype=torch.int32, device=_DEV)
        table[0, 0] = 1
        table[0, 1] = 2
        table[1, 0] = 1
        table[1, 1] = 2
        bases = torch.tensor([0, 0], dtype=torch.int32, device=_DEV)
        lengths = torch.tensor([256, 256], dtype=torch.int32, device=_DEV)
        self.assertFalse(main_geometry_supported(table, bases, lengths))

        from rtp_llm.models_py.modules.qwen4_exp.qsa import fmha_validation

        module = SimpleNamespace(
            attn_configs=SimpleNamespace(kernel_tokens_per_block=128),
            kv_head_num=2,
            head_dim=128,
        )
        cache = torch.zeros(64, 2 * 2 * 128 * 128, dtype=torch.bfloat16, device=_DEV)
        with patch.dict(os.environ, {"RTP_LLM_QSA_STRICT_VALIDATION": "0"}):
            with patch("torch._assert_async") as assert_async:
                fmha_validation._validate_main_paged_geometry_uncached(
                    module,
                    cache,
                    table,
                    bases,
                    lengths,
                    table.device,
                )
            self.assertEqual(assert_async.call_count, 1)
            verdict = assert_async.call_args[0][0]
            self.assertEqual(
                bool(verdict.item()),
                _legacy_verdict(table, bases, lengths, 64, 128),
            )


def _production_fixture():
    from rtp_llm.models_py.modules.qwen4_exp.qsa import fmha_validation

    module = SimpleNamespace(
        attn_configs=SimpleNamespace(kernel_tokens_per_block=128),
        kv_head_num=2,
        head_dim=128,
    )
    cache = torch.zeros(64, 2 * 2 * 128 * 128, dtype=torch.bfloat16, device=_DEV)
    table = torch.zeros((1, 55), dtype=torch.int32, device=_DEV)
    table[0, :53] = torch.arange(1, 54, device=_DEV, dtype=torch.int32)
    bases = torch.tensor([5248], dtype=torch.int32, device=_DEV)
    lengths = torch.tensor([1440], dtype=torch.int32, device=_DEV)
    return fmha_validation, module, cache, table, bases, lengths


class MainGeometryLaunchCountTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_production_path_enqueues_two_device_kernels(self):
        import tempfile
        import json
        from torch.profiler import ProfilerActivity, profile

        fmha_validation, module, cache, table, bases, lengths = _production_fixture()
        with patch.dict(os.environ, {"RTP_LLM_QSA_STRICT_VALIDATION": "0"}):
            # Warm the Triton compile outside the profile window.
            fmha_validation._validate_main_paged_geometry_uncached(
                module, cache, table, bases, lengths, table.device
            )
            torch.cuda.synchronize()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                fmha_validation._validate_main_paged_geometry_uncached(
                    module, cache, table, bases, lengths, table.device
                )
                torch.cuda.synchronize()
        path = os.path.join(
            tempfile.mkdtemp(prefix="qsa_main_geometry_triton_"), "t.json"
        )
        prof.export_chrome_trace(path)
        with open(path) as handle:
            evs = json.load(handle)["traceEvents"]
        kernels = [e["name"] for e in evs if e.get("cat") == "kernel"]
        self.assertEqual(
            len(kernels),
            2,
            f"the fused verdict must enqueue exactly the reduction and the "
            f"async assert, got {len(kernels)}: {kernels}",
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_no_new_variants_across_new_table_shapes(self):
        from rtp_llm.models_py.modules.qwen4_exp.qsa.qsa_validation_triton import (
            _main_geometry_verdict_kernel,
            main_geometry_verdict,
        )

        def variant_total(fn):
            caches = getattr(fn, "device_caches", None) or {}
            total = 0
            for cache in caches.values():
                entry = cache[0] if isinstance(cache, tuple) else cache
                total += len(entry)
            return total

        def run(width, base, length):
            table = torch.zeros((1, width), dtype=torch.int32, device=_DEV)
            need = min((base + length + 127) // 128, width)
            table[0, :need] = torch.arange(1, need + 1, device=_DEV, dtype=torch.int32)
            bases = torch.tensor([base], dtype=torch.int32, device=_DEV)
            lengths = torch.tensor([length], dtype=torch.int32, device=_DEV)
            main_geometry_verdict(
                table, bases, lengths, pool_blocks=4096, page_size=128
            )

        for width, base, length in ((55, 5248, 1440), (256, 0, 32768), (7, 0, 896)):
            run(width, base, length)
        before = variant_total(_main_geometry_verdict_kernel)
        for width, base, length in ((63, 0, 8064), (130, 512, 15872), (33, 0, 4096)):
            run(width, base, length)
        self.assertEqual(
            variant_total(_main_geometry_verdict_kernel),
            before,
            "table shapes must not add compiled variants (fixed-block loop)",
        )


if __name__ == "__main__":
    unittest.main()
