"""Host tests for the SM90 wo_a scale unpack.

``unpack_ue8m0_int32_scale`` reinterprets the int32-packed UE8M0 activation scale
``fused_inv_rope_fp8_quant`` emits as the fp32 per-token scale SM90's
``fp8_gemm_nt`` takes.  It is plain integer/float tensor algebra, so it is covered
here without CUDA or DeepGEMM -- the module keeps its ``deep_gemm`` import inside
``wo_a_grouped_gemm`` for exactly this reason.

Three properties matter and none of them is obvious from reading the shifts:

* the byte extraction is correct for words whose top byte is ``>= 0x80``, where the
  arithmetic right shift sign-extends and only the ``& 0xFF`` saves it;
* the trailing bytes of the last word are dropped, since the packer rounds the word
  count up;
* a packed tensor whose word count does not match the requested block count is
  rejected rather than silently yielding scales for the wrong K-blocks -- the case
  that would arise if either packer's grouping changed.

``prepare_wo_a_grouped`` and the GEMM itself are in ``wo_a_sm90_gemm_test.py``:
the former needs a ``float8_e8m0fnu`` tensor, whose CPU conversion support varies
by torch build, and the latter needs DeepGEMM.
"""

import os
import sys
import types
import unittest

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _stub_package(name: str, path: str) -> None:
    """Bind a package name to its directory without running its ``__init__``.

    Same device as ``test_dsv4_kernel_jit_warmup.py``: ``models_py.modules``'s
    ``__init__`` reaches the C++ ops module and the device layer, which a host test
    of tensor algebra has no reason to need and which pins a torch floor the
    function under test does not.
    """
    module = types.ModuleType(name)
    module.__path__ = [path]
    sys.modules.setdefault(name, module)


_MODULES = os.path.join(_REPO, "rtp_llm", "models_py", "modules")
_stub_package("rtp_llm", os.path.join(_REPO, "rtp_llm"))
_stub_package("rtp_llm.models_py", os.path.join(_REPO, "rtp_llm", "models_py"))
_stub_package("rtp_llm.models_py.modules", _MODULES)
_stub_package("rtp_llm.models_py.modules.dsv4", os.path.join(_MODULES, "dsv4"))
_stub_package(
    "rtp_llm.models_py.modules.dsv4.fp8", os.path.join(_MODULES, "dsv4", "fp8")
)

from rtp_llm.models_py.modules.dsv4.fp8._wo_a_sm90 import (
    _UE8M0_BIAS,
    unpack_ue8m0_int32_scale,
)


def _pack(byte_rows: list[list[int]]) -> torch.Tensor:
    """Pack rows of UE8M0 bytes into int32 words, least-significant byte first.

    Mirrors the packer's own layout; the trailing bytes of a short final word are
    zero-filled, which is what the ``k_blocks`` trim exists to discard.
    """
    words = []
    for row in byte_rows:
        padded = list(row) + [0] * ((-len(row)) % 4)
        row_words = []
        for i in range(0, len(padded), 4):
            b0, b1, b2, b3 = padded[i : i + 4]
            # Assemble as unsigned, then reinterpret into int32's range so a top
            # byte >= 0x80 lands as a negative word, exactly as the kernel writes.
            word = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
            if word >= 1 << 31:
                word -= 1 << 32
            row_words.append(word)
        words.append(row_words)
    return torch.tensor(words, dtype=torch.int32)


class UnpackUe8m0Int32ScaleTest(unittest.TestCase):
    def test_matches_two_to_the_biased_exponent(self):
        rows = [
            [127, 128, 126, 120],  # 1.0, 2.0, 0.5, 2**-7
            [100, 130, 127, 135],
        ]
        packed = _pack(rows)
        got = unpack_ue8m0_int32_scale(packed, 4)
        want = torch.tensor(
            [[2.0 ** (b - _UE8M0_BIAS) for b in row] for row in rows],
            dtype=torch.float32,
        )
        self.assertEqual(got.shape, (2, 4))
        self.assertTrue(torch.equal(got, want), f"{got} != {want}")

    def test_top_byte_above_0x7f_makes_the_word_negative(self):
        """The case an unsigned assumption would get wrong."""
        rows = [[127, 127, 127, 0x80]]  # top byte 128 -> word is negative
        packed = _pack(rows)
        self.assertLess(int(packed[0, 0]), 0, "test setup: word should be negative")
        got = unpack_ue8m0_int32_scale(packed, 4)
        self.assertTrue(
            torch.equal(
                got, torch.tensor([[1.0, 1.0, 1.0, 2.0]], dtype=torch.float32)
            ),
            got,
        )

    def test_every_byte_position_survives_a_negative_word(self):
        rows = [[0x81, 0x82, 0x83, 0x84]]
        packed = _pack(rows)
        self.assertLess(int(packed[0, 0]), 0)
        got = unpack_ue8m0_int32_scale(packed, 4)
        want = torch.tensor(
            [[2.0 ** (b - _UE8M0_BIAS) for b in rows[0]]], dtype=torch.float32
        )
        self.assertTrue(torch.equal(got, want), f"{got} != {want}")

    def test_trailing_bytes_of_the_last_word_are_trimmed(self):
        """``K/128`` need not be a multiple of four; the packer rounds up."""
        for k_blocks in (1, 2, 3, 5, 6, 7):
            with self.subTest(k_blocks=k_blocks):
                exps = [127 + i for i in range(k_blocks)]
                packed = _pack([exps])
                self.assertEqual(packed.shape[-1], -(-k_blocks // 4))
                got = unpack_ue8m0_int32_scale(packed, k_blocks)
                want = torch.tensor(
                    [[2.0 ** (b - _UE8M0_BIAS) for b in exps]], dtype=torch.float32
                )
                self.assertEqual(got.shape, (1, k_blocks))
                self.assertTrue(torch.equal(got, want), f"{got} != {want}")

    def test_word_count_mismatch_is_rejected(self):
        """A packer grouping change must fail here, not produce wrong scales."""
        packed = _pack([[127] * 8])  # 2 words
        with self.assertRaisesRegex(AssertionError, "cannot hold"):
            unpack_ue8m0_int32_scale(packed, 16)
        with self.assertRaisesRegex(AssertionError, "cannot hold"):
            unpack_ue8m0_int32_scale(packed, 4)

    def test_bytes_per_word_is_honoured(self):
        """Guards the assumption the two packers only coincide at head_dim 512."""
        packed = _pack([[127, 128, 126, 120]])
        with self.assertRaisesRegex(AssertionError, "cannot hold"):
            unpack_ue8m0_int32_scale(packed, 4, bytes_per_word=2)
        got = unpack_ue8m0_int32_scale(packed, 2, bytes_per_word=2)
        self.assertTrue(
            torch.equal(got, torch.tensor([[1.0, 2.0]], dtype=torch.float32)), got
        )

    def test_leading_dims_are_preserved(self):
        """Production passes ``[M, G, K/512]``; only the last dim is unpacked."""
        rows = [[127, 128, 126, 120], [121, 122, 123, 124], [125, 126, 127, 128]]
        packed = _pack(rows).reshape(3, 1, 1)
        got = unpack_ue8m0_int32_scale(packed, 4)
        self.assertEqual(got.shape, (3, 1, 4))
        want = torch.tensor(
            [[[2.0 ** (b - _UE8M0_BIAS) for b in row]] for row in rows],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(got, want), f"{got} != {want}")

    def test_non_contiguous_input(self):
        """The reason the implementation shifts instead of ``view(torch.uint8)``.

        Production hands this an MN-major scale whose last dim is not the fastest
        axis, and a bit-view would need contiguity. An earlier version of this case
        used ``.expand(3, 1, 1)`` on an already-(3,1,1) tensor -- a no-op that left
        the property it was named for untested.
        """
        rows = [[127, 128, 126, 120], [121, 122, 123, 124], [125, 126, 127, 128]]
        # [M, words] -> transpose to [words, M]: last dim stride is now M, not 1.
        packed = _pack(rows)  # [3, 1]
        wide = torch.cat([packed, packed + 1], dim=1)  # [3, 2], contiguous
        view = wide.transpose(0, 1)  # [2, 3], stride (1, 2)
        self.assertFalse(view.is_contiguous(), "test setup: need a strided view")

        got = unpack_ue8m0_int32_scale(view, 12, bytes_per_word=4)
        self.assertEqual(got.shape, (2, 12))
        # Same answer as unpacking the contiguous copy, which is the claim.
        want = unpack_ue8m0_int32_scale(view.contiguous(), 12, bytes_per_word=4)
        self.assertTrue(torch.equal(got, want), f"{got} != {want}")


if __name__ == "__main__":
    unittest.main()
