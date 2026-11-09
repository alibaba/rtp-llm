"""The PLE page projection must not recompile per residual length.

Serving timelines (warm5-warm7, 49-layer steps) show a ~190ms ``qwen4.ple.apply``
giant window once per capture window plus a ~60ms step-tail GPU gap: the PLE
page projection kernels ``_page_amax``/``_page_quant`` declare their element
counts ``ROWS``/``PADDED`` as ``tl.constexpr``, and ``quantize_pages`` passes
the residual chunk's row count. Every new chunk length — one per distinct
request length — compiles a fresh variant pair (trace evidence: 123.4ms
``_page_quant`` compile + 59.7ms and 7.2ms ``_page_amax`` compiles inside one
window, with Triton's lazy-load/event-query signature).

The structural constants (K, PAGE, TILES, REDUCE, BLOCK) are fixed by the
layer geometry, so demoting ROWS/PADDED to runtime arguments keeps one variant
per structural class.

This test locks the contract:

* a new row count inside a warmed structural class (same page size and width)
  compiles no new variant for either kernel;
* the padded geometry still holds: PADDED rows are zero-filled and the scale
  count follows the page count (guards the runtime-argument plumbing).
"""

import unittest

import torch

from rtp_llm.models_py.modules.qwen4_exp.ple import ple_page_projection


def _variant_total(fn):
    caches = getattr(fn, "device_caches", None) or {}
    total = 0
    for cache in caches.values():
        entry = cache[0] if isinstance(cache, tuple) else cache
        total += len(entry)
    return total


class PlePageProjectionJitTest(unittest.TestCase):
    KS = (0.03,)
    VS = (0.07,)
    WIDTH = 2560
    PAGE = 128

    @classmethod
    def _quantize(cls, rows, *, row_scales=False):
        x = torch.randn(rows, cls.WIDTH, device="cuda", dtype=torch.bfloat16)
        x[::7] *= 300.0
        ks = torch.tensor(list(cls.KS), device="cuda")
        vs = torch.tensor(list(cls.VS), device="cuda")
        return ple_page_projection.quantize_pages(
            x, ks, vs, page_size=cls.PAGE, row_scales=row_scales
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_new_rows_compile_no_new_variant(self):
        torch.manual_seed(17)
        # Warm both kernels for several row counts in the structural class
        # (width 2560, page 128), with and without row scales. Cover Triton's
        # integer value classes for the runtime ROWS arg (multiple of 16 vs
        # odd) crossed with the ROW_SCALES constexpr, and note PADDED is
        # always a multiple of the page size.
        self._quantize(128 * 10)
        self._quantize(128 * 11, row_scales=True)
        self._quantize(128 * 10 + 37)
        self._quantize(128 * 11 + 37, row_scales=True)
        torch.cuda.synchronize()
        before_amax = _variant_total(ple_page_projection._page_amax)
        before_quant = _variant_total(ple_page_projection._page_quant)
        # Probe: brand-new row counts — full pages, remainder pages, single page.
        self._quantize(128 * 12 - 1, row_scales=True)
        self._quantize(129)
        self._quantize(128 * 3 + 64)
        self._quantize(128 * 9 + 3, row_scales=True)
        torch.cuda.synchronize()
        self.assertEqual(
            _variant_total(ple_page_projection._page_amax),
            before_amax,
            "a new row count must not compile a new _page_amax variant",
        )
        self.assertEqual(
            _variant_total(ple_page_projection._page_quant),
            before_quant,
            "a new row count must not compile a new _page_quant variant",
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_padded_geometry_holds_across_row_counts(self):
        torch.manual_seed(19)
        for rows, row_scales in (
            (128 * 4, False),
            (128 * 4 + 37, False),
            (129, True),
            (128 * 11 - 1, True),
        ):
            with self.subTest(rows=rows, row_scales=row_scales):
                quantized, scales = self._quantize(rows, row_scales=row_scales)
                pages = -(-rows // self.PAGE)
                padded = pages * self.PAGE
                self.assertEqual(quantized.shape, (padded, self.WIDTH))
                # scales[0] is per-page; the pair members are per-page or
                # per-row depending on row_scales.
                self.assertEqual(scales[0].numel(), pages)
                if row_scales:
                    self.assertEqual(scales[1].shape, (padded, 1))
                    self.assertEqual(scales[2].shape, (padded, 1))
                else:
                    self.assertEqual(scales[1].numel(), pages)
                    self.assertEqual(scales[2].numel(), pages)
                # Padding rows beyond the residual must stay zero.
                self.assertEqual(
                    int(
                        torch.count_nonzero(quantized[rows:].view(torch.uint8)).item()
                    ),
                    0,
                )


if __name__ == "__main__":
    unittest.main()
