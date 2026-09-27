"""Arithmetic oracle for the per-tag PD cache-store wire geometry.

Compressed MLA aligns (kernel / ratio) * 584 payload bytes to a 576-byte stride,
so (page / kernel) * aligned_stride depends on kernel size. Indexer and state/SWA
geometry do not. Descriptor constants are read from the production source.
This oracle does not execute SpecBuilder; the companion production-chain
regression is //rtp_llm/models/test:pd_transfer_geometry_production_test.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "rtp_llm/models/dsv4_kv_cache.py").is_file()
)


def _dsv4_kv_cache_constants() -> dict:
    tree = ast.parse((ROOT / "rtp_llm/models/dsv4_kv_cache.py").read_text())
    wanted = {
        "DSV4_FP8_KV_ENTRY_BYTES",
        "DSV4_FP8_INDEXER_ENTRY_BYTES",
        "DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES",
        "CSA_LAYER_COMPRESS_RATIO",
        "HCA_LAYER_COMPRESS_RATIO",
        "DSV4_TOKENS_PER_BLOCK",
    }
    out = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in wanted
            and isinstance(node.value, ast.Constant)
        ):
            out[node.targets[0].id] = node.value.value
    return out


C = _dsv4_kv_cache_constants()
PAGE = C["DSV4_TOKENS_PER_BLOCK"]  # 256
KV_ENTRY = C["DSV4_FP8_KV_ENTRY_BYTES"]  # 584
INDEXER_ENTRY = C["DSV4_FP8_INDEXER_ENTRY_BYTES"]  # 132
ALIGN = C["DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES"]  # 576
CSA_RATIO = C["CSA_LAYER_COMPRESS_RATIO"]  # 4
HCA_RATIO = C["HCA_LAYER_COMPRESS_RATIO"]  # 128


def _align_up(n: int, a: int) -> int:
    return ((n + a - 1) // a) * a


def _validate_kernel_ratio(kernel_block: int, ratio: int) -> None:
    """Mirror of the production validity rules (CacheConfigCreator requires
    page % kernel == 0; the desc validator requires kernel > 0 and
    kernel % ratio == 0).  The oracle must reject invalid shapes, not return
    a silent zero."""
    if kernel_block <= 0 or ratio <= 0:
        raise ValueError("kernel_block and ratio must be positive")
    if PAGE % kernel_block != 0:
        raise ValueError(f"kernel {kernel_block} must divide page {PAGE}")
    if kernel_block % ratio != 0:
        raise ValueError(f"kernel {kernel_block} must be a multiple of ratio {ratio}")


def compressed_kv_block_bytes(kernel_block: int, ratio: int) -> int:
    """Per-PAGE transfer bytes of an fp8 compressed KV pool (csa_kv/hca_kv).

    One page holds ``page/kernel`` kernel blocks; each kernel block carries
    ``kernel/ratio`` entries of ``KV_ENTRY`` bytes, padded up to the 576 B
    FlashMLA stride.
    """
    _validate_kernel_ratio(kernel_block, ratio)
    entries = kernel_block // ratio
    per_kernel = _align_up(entries * KV_ENTRY, ALIGN)
    return (PAGE // kernel_block) * per_kernel


def indexer_kv_block_bytes(kernel_block: int) -> int:
    """indexer_kv: 132 B entries, compression ratio 4 (same as CSA), no 576 B
    alignment -> the kernel block factors out and the pool is kernel-block
    independent. (An earlier draft of this oracle wrongly used ratio 1; the
    production regression pd_transfer_geometry_production_test is the
    authoritative check.)"""
    _validate_kernel_ratio(kernel_block, CSA_RATIO)
    entries = kernel_block // CSA_RATIO
    return (PAGE // kernel_block) * entries * INDEXER_ENTRY


class PdTransferGeometryTest(unittest.TestCase):
    def test_constants_from_product_source(self):
        self.assertEqual(C["DSV4_FP8_KV_ENTRY_BYTES"], 584)
        self.assertEqual(C["DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES"], 576)
        self.assertEqual(C["CSA_LAYER_COMPRESS_RATIO"], 4)
        self.assertEqual(C["HCA_LAYER_COMPRESS_RATIO"], 128)
        self.assertEqual(C["DSV4_TOKENS_PER_BLOCK"], 256)

    def test_r6_ground_truth_prefill_kernel128(self):
        # Expected published extents with kernel size 128.
        self.assertEqual(compressed_kv_block_bytes(128, CSA_RATIO), 38016)
        self.assertEqual(compressed_kv_block_bytes(128, HCA_RATIO), 2304)

    def test_r6_ground_truth_decode_kernel256(self):
        # Expected receive extents with kernel size 256.
        self.assertEqual(compressed_kv_block_bytes(256, CSA_RATIO), 37440)
        self.assertEqual(compressed_kv_block_bytes(256, HCA_RATIO), 1728)

    def test_indexer_kv_is_kernel_block_independent(self):
        # Without alignment, both kernel sizes produce the same 8448-byte block.
        self.assertEqual(indexer_kv_block_bytes(128), indexer_kv_block_bytes(256))
        self.assertEqual(indexer_kv_block_bytes(128), 32 * INDEXER_ENTRY * 2)  # 8448

    def test_mismatched_kernel_blocks_disagree(self):
        # A kernel128 publisher and kernel256 receiver have different aligned extents.
        self.assertNotEqual(
            compressed_kv_block_bytes(128, CSA_RATIO),
            compressed_kv_block_bytes(256, CSA_RATIO),
        )
        self.assertNotEqual(
            compressed_kv_block_bytes(128, HCA_RATIO),
            compressed_kv_block_bytes(256, HCA_RATIO),
        )

    def test_invalid_inputs_rejected(self):
        # kernel64 with the HCA ratio (128) must not silently yield a
        # zero-entry block; invalid kernels raise.
        with self.assertRaises(ValueError):
            compressed_kv_block_bytes(64, HCA_RATIO)
        with self.assertRaises(ValueError):
            compressed_kv_block_bytes(0, CSA_RATIO)
        with self.assertRaises(ValueError):
            compressed_kv_block_bytes(100, CSA_RATIO)  # 256 % 100 != 0
        with self.assertRaises(ValueError):
            indexer_kv_block_bytes(0)

    def test_kernel_block_legal_set(self):
        # CacheConfigCreator requires page % kernel == 0; the desc validator
        # requires kernel % ratio == 0.  Legal DSv4 fp8 kernel set for page
        # 256 and max ratio 128.
        legal = [k for k in (32, 64, 128, 256) if PAGE % k == 0 and k % HCA_RATIO == 0]
        self.assertEqual(legal, [128, 256])

    def test_pairing_contract_prefill_cp2pp2_x_decode_dp4ep4(self):
        # The proxy (and target CEP4PP2) prefill recipe pins kernel 128; the
        # decode side MUST also run kernel 128 for the wire geometry to match.
        prefill = {
            tag: compressed_kv_block_bytes(128, ratio)
            for tag, ratio in (("csa_kv", 4), ("hca_kv", 128))
        }
        prefill["indexer_kv"] = indexer_kv_block_bytes(128)
        decode = {
            tag: compressed_kv_block_bytes(128, ratio)
            for tag, ratio in (("csa_kv", 4), ("hca_kv", 128))
        }
        decode["indexer_kv"] = indexer_kv_block_bytes(128)
        self.assertEqual(prefill, decode)
        self.assertEqual(prefill, {"csa_kv": 38016, "hca_kv": 2304, "indexer_kv": 8448})


if __name__ == "__main__":
    unittest.main()
