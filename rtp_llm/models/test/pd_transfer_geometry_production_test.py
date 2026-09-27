"""Verify PD transfer geometry through production descriptors and C++ cache specs.

The test-only binding exposes per-tag block sizes and payload extents. Kernel
sizes 128 and 256 intentionally produce different aligned compressed-MLA wire
sizes; state and SWA geometry must remain kernel-independent. Invalid shapes
must raise the specific pybind-translated RuntimeError in an isolated process.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

# Import order matters: rtp_llm.models.dsv4_kv_cache pulls rtp_llm.ops, which
# imports torch first; torch's loader makes the bundled torch libs (incl.
# libtorch_nvshmem.so) resolvable for the subsequently dlopen'd test binding.
# isort: off
from rtp_llm.models.dsv4_kv_cache import build_dsv4_kv_cache_spec_descs
from rtp_llm.cpp.cache.test.libcache_config_creator_py_test import dsv4_block_geometry

# isort: on

# The real DeepSeek-V4-Flash-0731 checkpoint's compress_ratios (46 entries:
# 43 model layers + 3 MTP-layer entries; production passes the full list and
# only the first num_layers=43 are consumed). The model layers contain ratio
# 4 x21, 128 x20, and two SWA-only layers.
REAL_COMPRESS_RATIOS = [
    0,
    0,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    0,
    0,
    0,
]
NUM_LAYERS = 43


def _real_ratios():
    return REAL_COMPRESS_RATIOS[:NUM_LAYERS]


def _build_descs():
    return build_dsv4_kv_cache_spec_descs(
        layer_num=NUM_LAYERS,
        layer_compress_ratios=REAL_COMPRESS_RATIOS,
        fp8_kv=True,
        head_dim=576,
        indexer_head_dim=128,
        fixed_pool_use_host_memory=False,
    )


class ProductionGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.descs = _build_descs()
        cls.g128 = dsv4_block_geometry(cls.descs, 256, 128, 0)
        cls.g256 = dsv4_block_geometry(cls.descs, 256, 256, 0)

    def test_r6_ground_truth_kernel128(self):
        self.assertEqual(self.g128["csa_kv"]["block_size_bytes"], 38016)
        self.assertEqual(self.g128["hca_kv"]["block_size_bytes"], 2304)
        self.assertEqual(self.g128["indexer_kv"]["block_size_bytes"], 8448)

    def test_r6_ground_truth_kernel256(self):
        self.assertEqual(self.g256["csa_kv"]["block_size_bytes"], 37440)
        self.assertEqual(self.g256["hca_kv"]["block_size_bytes"], 1728)
        self.assertEqual(self.g256["indexer_kv"]["block_size_bytes"], 8448)

    def test_mismatched_kernels_disagree_exactly_like_r6(self):
        # The pairing contract is kernel-block PARITY: a kernel128 publisher
        # and a kernel256 receiver must be detected as incompatible, exactly
        # and rejects mismatched transfer extents.
        self.assertNotEqual(
            self.g128["csa_kv"]["block_size_bytes"],
            self.g256["csa_kv"]["block_size_bytes"],
        )
        self.assertNotEqual(
            self.g128["hca_kv"]["block_size_bytes"],
            self.g256["hca_kv"]["block_size_bytes"],
        )
        self.assertEqual(
            self.g128["indexer_kv"]["block_size_bytes"],
            self.g256["indexer_kv"]["block_size_bytes"],
        )

    def test_padding_is_explicit(self):
        # The 576B alignment padding is exactly what made the geometry
        # kernel-dependent: stride must exceed raw payload for the fp8
        # compressed pools.
        for tag in ("csa_kv", "hca_kv"):
            self.assertGreater(
                self.g128[tag]["block_size_bytes"],
                self.g128[tag]["block_payload_bytes"],
                f"{tag}: padding must be visible",
            )

    def test_state_and_swa_tags_kernel_independent(self):
        # Window/ring-derived pools must stay
        # kernel-block independent through the real builder.
        for tag in ("indexer_state", "csa_state", "hca_state", "swa_kv"):
            self.assertIn(tag, self.g128)
            self.assertEqual(
                self.g128[tag]["block_size_bytes"],
                self.g256[tag]["block_size_bytes"],
                f"{tag} must not depend on kernel block",
            )

    def test_layer_coverage_matches_model(self):
        # Every tag's layer set must match the ratio assignment of the real
        # 43-layer stack.
        ratios = _real_ratios()
        want = {
            "csa_kv": [i for i, r in enumerate(ratios) if r == 4],
            "hca_kv": [i for i, r in enumerate(ratios) if r == 128],
            "swa_kv": list(range(43)),
        }
        for tag, layers in want.items():
            self.assertEqual(sorted(self.g128[tag]["layers"]), layers, tag)


def _expect_production_reject(seq_size: int, kernel: int) -> None:
    """Invalid geometry must be rejected by the production builder.

    A valid control proves the import/binding chain works. The invalid child
    must enter the builder and fail with its specific geometry RuntimeError,
    not merely a nonzero loader, assertion, or process exit.
    """
    import_cmd = (
        "from rtp_llm.models.dsv4_kv_cache import build_dsv4_kv_cache_spec_descs;"
        "from rtp_llm.cpp.cache.test.libcache_config_creator_py_test import dsv4_block_geometry;"
    )
    build = (
        "d=build_dsv4_kv_cache_spec_descs(layer_num=4,layer_compress_ratios=[4,128,4,0],"
        "fp8_kv=True,head_dim=576,indexer_head_dim=128,fixed_pool_use_host_memory=False);"
    )
    # The bazel test process resolves libtorch_nvshmem.so through its own
    # launcher environment; a bare subprocess does not inherit the resolver.
    # Give the child the torch lib dir computed from the WORKING parent import.
    import torch  # noqa: E402  (parent import is known-good inside the test)

    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    child_env = dict(os.environ)
    child_env["LD_LIBRARY_PATH"] = (
        torch_lib + os.pathsep + child_env.get("LD_LIBRARY_PATH", "")
    )
    control = subprocess.run(
        [sys.executable, "-c", import_cmd + build + "dsv4_block_geometry(d,256,128,0)"],
        capture_output=True,
        env=child_env,
        timeout=60,
    )
    if control.returncode != 0:
        raise AssertionError(
            "control subprocess (valid geometry) failed rc=%d: %s"
            % (control.returncode, control.stderr[-300:])
        )
    code = (
        import_cmd
        + build
        + "print('ENTERED_PRODUCTION_CALL', flush=True);"
        + f"dsv4_block_geometry(d,{seq_size},{kernel},0)"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, env=child_env, timeout=60
    )
    if r.returncode == 0:
        raise AssertionError(
            f"production accepted invalid geometry seq={seq_size} kernel={kernel}"
        )
    if b"ENTERED_PRODUCTION_CALL" not in r.stdout:
        raise AssertionError(
            f"subprocess never reached the production call (rc={r.returncode}): "
            f"{r.stderr[-300:]}"
        )
    # The production rejection is a pybind-translated RuntimeError from
    # OpaqueKVCacheSpec (myAssert), not a signal. Require the SPECIFIC
    # geometry diagnostic plus the production frame, so an unrelated
    # exception cannot pass.
    diag = r.stderr.decode(errors="replace")
    expected_diag = (
        "kernel_tokens_per_block is 0"
        if kernel == 0
        else f"must divide kernel block {kernel}"
    )
    has_geometry_diag = "RuntimeError:" in diag and expected_diag in diag
    has_production_frame = "OpaqueKVCacheSpec" in diag
    if not (r.returncode == 1 and has_geometry_diag and has_production_frame):
        raise AssertionError(
            f"rejection is not the production geometry error: rc={r.returncode}, "
            f"stderr tail: {diag[-300:]}"
        )


class ProductionGeometryInvalidInputTest(unittest.TestCase):
    def test_kernel_zero_rejected(self):
        _expect_production_reject(256, 0)

    def test_kernel_not_dividing_page_rejected(self):
        _expect_production_reject(256, 100)

    def test_kernel64_with_hca_ratio128_rejected(self):
        # 64 % 128 != 0: production desc validation must reject, not return
        # zero-entry geometry.
        _expect_production_reject(256, 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
