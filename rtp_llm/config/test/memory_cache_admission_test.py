import os
import unittest
from unittest.mock import patch

from rtp_llm.config.memory_cache_admission import (
    MIB,
    HostMemorySnapshot,
    validate_memory_cache_capacity,
)


class MemoryCacheAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.snapshot = HostMemorySnapshot(1024 * 1024 * MIB, 900 * 1024 * MIB)

    def test_aggregates_per_rank_capacity(self):
        validate_memory_cache_capacity(True, 100 * 1024, 4, self.snapshot)
        with self.assertRaisesRegex(ValueError, "aggregate_mb=614400"):
            validate_memory_cache_capacity(True, 100 * 1024, 6, self.snapshot)

    def test_boundary_is_allowed_with_adaptive_reserve(self):
        snapshot = HostMemorySnapshot(200 * MIB, 200 * MIB)
        validate_memory_cache_capacity(True, 50, 2, snapshot)

    def test_disabled_cache_and_check_skip_validation(self):
        validate_memory_cache_capacity(False, 0, 0, self.snapshot)
        with patch.dict(
            os.environ, {"RTP_LLM_MEMORY_CACHE_ADMISSION_CHECK": "0"}, clear=False
        ):
            validate_memory_cache_capacity(True, -1, 0, self.snapshot)

    def test_cgroup_headroom_and_error_diagnostics(self):
        snapshot = HostMemorySnapshot(1024 * MIB, 900 * MIB, 600 * MIB, 400 * MIB)
        with patch.dict(
            os.environ, {"RTP_LLM_MEMORY_CACHE_HOST_RESERVE_MB": "100"}, clear=False
        ):
            with self.assertRaisesRegex(
                ValueError,
                "per_rank_mb=60, local_world_size=2, aggregate_mb=120.*cgroup_limit_mb=600",
            ):
                validate_memory_cache_capacity(True, 60, 2, snapshot)

    def test_adaptive_reserve_uses_cgroup_limited_effective_total(self):
        snapshot = HostMemorySnapshot(
            4 * 1024 * 1024 * MIB,
            3 * 1024 * 1024 * MIB,
            64 * 1024 * MIB,
            0,
        )
        validate_memory_cache_capacity(True, 10 * 1024, 2, snapshot)


if __name__ == "__main__":
    unittest.main()
