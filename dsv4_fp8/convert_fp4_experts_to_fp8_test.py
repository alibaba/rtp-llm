"""Tests for the FP4 -> FP8 routed-expert converter.

Two levels, and the second is the one that matters:

* ``convert_expert`` in isolation -- the 5-tuple contract, exactness on a block
  whose dynamic range fits e4m3's normal range, and the reported span;
* the script end to end over a synthetic one-shard checkpoint, by subprocess.

The end-to-end case exists because a previous revision returned five values from
``convert_expert`` while the caller unpacked four. Nothing here executed the script,
so it passed review twice and would have raised ``ValueError`` on the first expert
tensor of a real run -- after having already written ``config.json`` with
``expert_dtype: fp8``. A pure-Python arity mismatch on the only code path that
matters is exactly what a subprocess test catches and a unit test of the helper
alone does not.

Everything runs on CPU: the converter picks its device from
``torch.cuda.is_available()`` and the shapes here are one 128x128 block.
"""

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

import convert_fp4_experts_to_fp8 as conv

_SCRIPT = os.path.join(_THIS, "convert_fp4_experts_to_fp8.py")

# One block: the smallest legal shape (both dims a multiple of FP8_BLOCK).
_N = 128
_K = 128
_K_PACKED = _K // 2
_K_GROUPS = _K // conv.FP4_GROUP  # 4 UE8M0 bytes per row

# e2m1 code -> value, from the converter's own table.
_CODE_HALF = 1  # 0.5
_CODE_ONE = 2  # 1.0
_CODE_THREE = 5  # 3.0
_CODE_SIX = 7  # 6.0


def _pack_codes(codes: np.ndarray) -> np.ndarray:
    """``[N, K]`` nibble codes -> ``[N, K/2]`` bytes, low half-byte first."""
    low = codes[:, 0::2].astype(np.uint8)
    high = codes[:, 1::2].astype(np.uint8)
    return (low | (high << 4)).astype(np.uint8)


def _write_safetensors(path: str, tensors: dict, metadata: dict | None = None):
    """Minimal writer: ``{name: (dtype_str, shape, bytes)}`` in insertion order."""
    header: dict = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    cursor = 0
    for name, (dtype, shape, payload) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [cursor, cursor + len(payload)],
        }
        cursor += len(payload)
    blob = json.dumps(header).encode()
    pad = (-len(blob)) % 8
    blob += b" " * pad
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        for _name, (_dtype, _shape, payload) in tensors.items():
            handle.write(payload)


def _expert_pair(codes: np.ndarray, group_exponents: np.ndarray):
    """(weight bytes, scale bytes) for one routed expert tensor."""
    return _pack_codes(codes).tobytes(), group_exponents.astype(np.uint8).tobytes()


def _uniform_codes(code: int = _CODE_THREE) -> np.ndarray:
    return np.full((_N, _K), code, dtype=np.uint8)


def _flat_exponents(biased: int = 127) -> np.ndarray:
    return np.full((_N, _K_GROUPS), biased, dtype=np.uint8)


class ConvertExpertTest(unittest.TestCase):
    """The helper's contract, including the arity the caller depends on."""

    def _convert(self, codes, exponents, verify=True):
        weight_bytes, scale_bytes = _expert_pair(codes, exponents)
        return conv.convert_expert(
            np.frombuffer(weight_bytes, dtype=np.uint8).copy(),
            np.frombuffer(scale_bytes, dtype=np.uint8).copy(),
            _N,
            _K_PACKED,
            "cpu",
            verify=verify,
        )

    def test_returns_five_values(self):
        """The arity the caller unpacks. A 4-tuple here is a crash in convert_file."""
        result = self._convert(_uniform_codes(), _flat_exponents())
        self.assertEqual(len(result), 5)

    def test_output_byte_lengths_match_the_declared_shapes(self):
        weight_out, scale_out, _exact, _rel, _span = self._convert(
            _uniform_codes(), _flat_exponents()
        )
        self.assertEqual(len(weight_out), _N * _K)  # e4m3, one byte per element
        self.assertEqual(
            len(scale_out), (_N // conv.FP8_BLOCK) * (_K // conv.FP8_BLOCK)
        )

    def test_exact_for_a_block_inside_the_normal_range(self):
        # Values 0.5 .. 6 under one group exponent: a dynamic range of 12, far
        # inside the 2^14 the rewrite is exact for.
        codes = _uniform_codes()
        codes[:, 0::4] = _CODE_HALF
        codes[:, 1::4] = _CODE_ONE
        codes[:, 2::4] = _CODE_THREE
        codes[:, 3::4] = _CODE_SIX
        _w, _s, exact, rel, span = self._convert(codes, _flat_exponents())
        self.assertEqual(exact, 1.0)
        self.assertEqual(rel, 0.0)
        self.assertAlmostEqual(span, 12.0, places=5)

    def test_verify_off_reports_nothing(self):
        _w, _s, exact, rel, span = self._convert(
            _uniform_codes(), _flat_exponents(), verify=False
        )
        self.assertIsNone(exact)
        self.assertIsNone(rel)
        self.assertIsNone(span)

    def test_inexact_when_the_block_spans_past_the_normal_range(self):
        """A block wide enough to push its small values out of e4m3.

        Group 0 gets 6.0 at 2^20 and the rest 0.5 at 2^0, so the block spans
        12 * 2^20 = 2^23.6. The block scale aims the max at (2^7, 2^8], which puts
        the small end below e4m3's smallest subnormal and flushes it to zero.
        """
        codes = _uniform_codes(_CODE_HALF)
        codes[:, : conv.FP4_GROUP] = _CODE_SIX
        exponents = _flat_exponents()
        exponents[:, 0] = 127 + 20
        _w, _s, exact, rel, span = self._convert(codes, exponents)
        self.assertLess(exact, 1.0)
        self.assertGreater(rel, 0.0)
        self.assertGreater(span, 2.0**14)

    def test_rejects_shapes_that_are_not_block_multiples(self):
        with self.assertRaisesRegex(ValueError, "not a multiple"):
            conv.convert_expert(
                np.zeros(64 * 32, dtype=np.uint8),
                np.zeros(64 * 2, dtype=np.uint8),
                64,
                32,
                "cpu",
            )


class ConvertScriptEndToEndTest(unittest.TestCase):
    """Runs the script. The arity bug lived on this path and only on this path."""

    def _make_checkpoint(self, root: str, *, wide_block: bool) -> None:
        os.makedirs(root, exist_ok=True)
        if wide_block:
            codes = _uniform_codes(_CODE_HALF)
            codes[:, : conv.FP4_GROUP] = _CODE_SIX
            exponents = _flat_exponents()
            exponents[:, 0] = 127 + 20
        else:
            codes = _uniform_codes()
            exponents = _flat_exponents()
        weight_bytes, scale_bytes = _expert_pair(codes, exponents)

        # One routed expert plus a tensor the converter must copy untouched.
        other = np.arange(_N, dtype=np.float32).tobytes()
        shard = "model-00001-of-00001.safetensors"
        _write_safetensors(
            os.path.join(root, shard),
            {
                "layers.0.ffn.experts.0.w1.weight": ("I8", [_N, _K_PACKED], weight_bytes),
                "layers.0.ffn.experts.0.w1.scale": (
                    "F8_E8M0",
                    [_N, _K_GROUPS],
                    scale_bytes,
                ),
                "layers.0.attn_norm.weight": ("F32", [_N], other),
            },
            metadata={"format": "pt"},
        )
        with open(os.path.join(root, "config.json"), "w") as handle:
            json.dump({"model_type": "deepseek_v4", "num_hidden_layers": 1}, handle)
        with open(os.path.join(root, "model.safetensors.index.json"), "w") as handle:
            json.dump(
                {
                    "metadata": {"total_size": 1},
                    "weight_map": {
                        "layers.0.ffn.experts.0.w1.weight": shard,
                        "layers.0.ffn.experts.0.w1.scale": shard,
                        "layers.0.attn_norm.weight": shard,
                    },
                },
                handle,
            )

    @staticmethod
    def _read_json(path: str):
        with open(path) as handle:
            return json.load(handle)

    def _run(self, src: str, dst: str, *extra: str):
        return subprocess.run(
            [sys.executable, _SCRIPT, "--src", src, "--dst", dst, *extra],
            capture_output=True,
            text=True,
            timeout=600,
        )

    def test_converts_and_rewrites_the_side_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = os.path.join(tmp, "fp4"), os.path.join(tmp, "fp8")
            self._make_checkpoint(src, wide_block=False)
            proc = self._run(src, dst)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

            header, _start = conv.read_header(
                os.path.join(dst, "model-00001-of-00001.safetensors")
            )
            self.assertEqual(
                header["layers.0.ffn.experts.0.w1.weight"]["dtype"], "F8_E4M3"
            )
            self.assertEqual(
                header["layers.0.ffn.experts.0.w1.weight"]["shape"], [_N, _K]
            )
            self.assertEqual(
                header["layers.0.ffn.experts.0.w1.scale"]["shape"],
                [_N // conv.FP8_BLOCK, _K // conv.FP8_BLOCK],
            )
            # Untouched tensors keep their dtype and shape.
            self.assertEqual(header["layers.0.attn_norm.weight"]["dtype"], "F32")

            config = self._read_json(os.path.join(dst, "config.json"))
            self.assertEqual(config["expert_dtype"], "fp8")

            index = self._read_json(
                os.path.join(dst, "model.safetensors.index.json")
            )
            # Recomputed from the output, not the copied FP4 figure.
            self.assertNotEqual(index["metadata"]["total_size"], 1)
            self.assertGreater(index["metadata"]["total_size"], _N * _K)

            self.assertIn("exactness:", proc.stdout)
            self.assertIn("largest in-block dynamic range:", proc.stdout)

    def test_fails_and_names_the_tensor_when_a_block_is_too_wide(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = os.path.join(tmp, "fp4"), os.path.join(tmp, "fp8")
            self._make_checkpoint(src, wide_block=True)
            proc = self._run(src, dst)
            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            self.assertIn("FAILED", proc.stdout)
            self.assertIn("layers.0.ffn.experts.0.w1", proc.stdout)
            self.assertIn("block_span=", proc.stdout)

    def test_no_verify_skips_the_check_and_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = os.path.join(tmp, "fp4"), os.path.join(tmp, "fp8")
            self._make_checkpoint(src, wide_block=True)
            proc = self._run(src, dst, "--no-verify")
            # Same input as the failing case above: without the check it passes.
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("--no-verify", proc.stdout)

    def test_limit_does_not_claim_the_output_is_converted(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = os.path.join(tmp, "fp4"), os.path.join(tmp, "fp8")
            self._make_checkpoint(src, wide_block=False)
            proc = self._run(src, dst, "--limit", "1")
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            config = self._read_json(os.path.join(dst, "config.json"))
            self.assertNotIn("expert_dtype", config)
            index = self._read_json(
                os.path.join(dst, "model.safetensors.index.json")
            )
            self.assertEqual(index["metadata"]["total_size"], 1)


if __name__ == "__main__":
    unittest.main()
