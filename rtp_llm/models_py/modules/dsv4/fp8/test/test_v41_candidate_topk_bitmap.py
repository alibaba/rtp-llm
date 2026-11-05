"""R53: byte-exactness of the fused candidate TopK on the build_bitmap=True path.

Compares the fused path (try_select_v2 -> candidates, then build_bitmap_kernel ->
flags) against the production torch.topk + store_bitmap path. The bitmaps and the
candidate SETS must be bit-identical (the candidates array order may differ — all
consumers are order-independent). Covers ties at the k cutoff, -inf/NaN rows, and
+inf newest-block pins.

Run on SM120 with DSV41_CANDIDATE_TOPK_V2=1.
"""

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path

import torch

_FP8 = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# The candidates module does `from ._v41_candidate_topk_v2 import ...` (relative),
# so register a minimal package shell for the fp8 dir.
def _load_candidates():
    pkg = "rtp_llm.models_py.modules.dsv4.fp8"
    for mod in [
        "rtp_llm",
        "rtp_llm.models_py",
        "rtp_llm.models_py.modules",
        "rtp_llm.models_py.modules.dsv4",
        pkg,
    ]:
        if mod not in sys.modules:
            shell = types.ModuleType(mod)
            shell.__path__ = [str(_FP8.parents[0])] if mod.endswith("fp8") else []
            sys.modules[mod] = shell
    shell = sys.modules[pkg]
    shell.__path__ = [str(_FP8)]
    v2 = _load(pkg + "._v41_candidate_topk_v2", _FP8 / "_v41_candidate_topk_v2.py")
    setattr(sys.modules[pkg], "_v41_candidate_topk_v2", v2)
    cand = _load(pkg + "._v41_prefill_candidates", _FP8 / "_v41_prefill_candidates.py")
    return v2, cand


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(
    os.environ.get("DSV41_CANDIDATE_TOPK_V2", "0") == "1", "v2 flag off"
)
class TestCandidateTopkBitmapByteExact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v2, cls.cand = _load_candidates()
        assert cls.v2.warmup(torch.device("cuda", 0)), "v2 warmup failed"

    def _make_scores(self, rows, nblocks, seed, ties=False, pins=True):
        g = torch.Generator(device="cuda").manual_seed(seed)
        scores = torch.randn(
            rows, nblocks, device="cuda", dtype=torch.float32, generator=g
        )
        if ties:
            # Force exact ties at the cutoff region.
            cutoff_vals = scores[:, -1].clone()
            scores[:, :8] = cutoff_vals.unsqueeze(1).expand(rows, 8)
        if pins:
            scores[:, 0] = float("inf")  # newest-block pin
        # some -inf (masked) and NaN rows/cols
        scores[:, nblocks // 2 :] = float("-inf")  # half masked
        if rows > 2:
            scores[1, :] = float("-inf")  # fully-masked row
            scores[2, 5] = float("nan")  # a NaN entry
        return scores

    def test_byte_exact_bitmap_and_set(self):
        v2, cand = self.v2, self.cand
        rows, nblocks, count = 96, 4096, 2048
        words = (nblocks + 31) // 32
        for seed, ties in [(0, False), (1, True)]:
            scores = self._make_scores(rows, nblocks, seed, ties=ties)
            # Reference: torch.topk + store_bitmap
            out_ref = torch.empty((rows, count), dtype=torch.int32, device="cuda")
            flags_ref = torch.zeros((rows, words), dtype=torch.int32, device="cuda")
            values, indices = scores.topk(count, dim=-1, largest=True, sorted=True)
            cand._prefill_candidate_store_bitmap_kernel[(rows,)](
                values,
                indices,
                out_ref,
                flags_ref,
                count,
                out_ref.shape[1],
                out_ref.stride(0),
                words,
                flags_ref.stride(0),
                True,
                words,
            )
            # Fused: try_select_v2 + build_bitmap
            out_f = torch.empty((rows, count), dtype=torch.int32, device="cuda")
            flags_f = torch.zeros((rows, words), dtype=torch.int32, device="cuda")
            sel = v2.try_select(scores, out_f, k=count)
            self.assertIsNotNone(sel, "fused kernel fell back")
            cand._prefill_candidate_build_bitmap_kernel[(rows,)](
                out_f,
                flags_f,
                count,
                out_f.stride(0),
                nblocks,
                words,
                flags_f.stride(0),
                words,
            )
            # Bitmaps must be bit-identical (order-independent).
            self.assertTrue(
                torch.equal(flags_ref, flags_f),
                f"bitmap mismatch ties={ties}: {(flags_ref != flags_f).sum()} words differ",
            )
            # Candidate SETS must match (sorted), order may differ.
            self.assertTrue(
                torch.equal(out_ref.sort(dim=-1).values, out_f.sort(dim=-1).values),
                f"candidate set mismatch ties={ties}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
