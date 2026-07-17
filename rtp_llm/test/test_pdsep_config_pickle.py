"""
Integration tests for PDSepConfig and RuntimeConfig pickle serialization.

These tests verify the backward-compatible pickle support added in Task 4 Phase A:
1. PDSepConfig.prefill_stop_stream_wait_timeout_ms field exists and is read/write accessible
2. PDSepConfig pickle round-trip with all 27 fields preserves values
3. Legacy 20-item PDSepConfig state (pre-batch-timeout era) can be deserialized,
   with missing fields using their default values
4. Legacy 13-item RuntimeConfig state (pre-specify_gpu_arch era) can be deserialized,
   with missing fields using their default values

Running prerequisites:
- The rtp_llm C++ extension (.so) must be compiled and importable as rtp_llm.ops
- Run from the repository root:
    python -m pytest rtp_llm/test/test_pdsep_config_pickle.py
  or:
    python -m unittest rtp_llm.test.test_pdsep_config_pickle

PDSepConfig __getstate__ field layout (27 items):
  0:  role_type                          (RoleType)
  1:  cache_store_rdma_mode              (bool)
  2:  cache_store_listen_port            (int64)
  3:  cache_store_connect_port           (int64)
  4:  cache_store_rdma_listen_port       (int64)
  5:  cache_store_rdma_connect_port      (int64)
  6:  remote_rpc_server_port             (int64)
  7:  prefill_retry_times                (int64)
  8:  prefill_retry_timeout_ms           (int64)
  9:  prefill_max_wait_timeout_ms        (int64)
  10: decode_retry_times                 (int64)
  11: decode_retry_timeout_ms            (int64)
  12: decode_retry_interval_ms           (int64)
  13: decode_polling_kv_cache_step_ms    (int64)
  14: decode_polling_call_prefill_ms     (int64)
  15: rdma_connect_retry_times           (int64)
  16: load_cache_timeout_ms              (int64)
  17: max_rpc_timeout_ms                 (int64)
  18: worker_port_offset                 (int64)
  19: decode_entrance                    (bool)
  -- minimum 20 (legacy baseline) --
  20: batch_dispatch_timeout_ms          (int64)   [if size >= 23]
  21: batch_prepare_timeout_ms           (int64)   [if size >= 23]
  22: batch_load_timeout_ms              (int64)   [if size >= 23]
  23: prefill_enqueue_pool_size          (int64)   [if size >= 26]
  24: prefill_worker_lambda_pool_size    (int64)   [if size >= 26]
  25: prefill_slot_pool_size             (int64)   [if size >= 26]
  26: prefill_stop_stream_wait_timeout_ms (int64)  [if size >= 27]

RuntimeConfig __getstate__ field layout (14 items):
  0:  max_generate_batch_size       (int64)
  1:  max_block_size_per_item       (int64)
  2:  reserve_runtime_mem_mb        (int64)
  3:  warm_up                       (bool)
  4:  warm_up_with_loss             (bool)
  5:  use_batch_decode_scheduler    (bool)
  6:  use_gather_batch_scheduler    (bool)
  7:  batch_decode_scheduler_config (BatchDecodeSchedulerConfig)
  8:  fifo_scheduler_config         (FIFOSchedulerConfig)
  9:  model_name                    (string)
  10: worker_grpc_addrs            (list[string])
  11: worker_addrs                 (list[string])
  12: all_worker_grpc_addrs        (list[string])
  -- minimum 13 (legacy baseline) --
  13: specify_gpu_arch              (string)  [if size >= 14]
"""

import pickle
import unittest

from rtp_llm.ops import PDSepConfig, RuntimeConfig


def _unpickle_from_state(cls, state_tuple):
    """Reconstruct an object from a state tuple via the full pickle protocol.

    This simulates loading a pickled object whose __getstate__ produced a shorter
    tuple (legacy format). A carrier object is used whose __reduce__ delegates to
    the target class's factory function and __setstate__, exactly mirroring what
    Python's pickle module does during unpickling.

    Args:
        cls: The pybind11 class (e.g., PDSepConfig, RuntimeConfig).
        state_tuple: The legacy state tuple (shorter than the current __getstate__ output).

    Returns:
        A new instance of cls populated from state_tuple, with missing fields
        using their C++ default values.
    """
    template = cls()
    reduce_info = template.__reduce_ex__(2)
    factory = reduce_info[0]  # typically copyreg.__newobj__
    args = reduce_info[1]  # typically (cls,)

    class _Carrier:
        """Throwaway object whose __reduce__ triggers the target class's pickle path."""

        def __reduce__(self):
            return (factory, args, state_tuple)

    return pickle.loads(pickle.dumps(_Carrier()))


# ---------------------------------------------------------------------------
# PDSepConfig pickle tests
# ---------------------------------------------------------------------------
class TestPDSepConfigPickle(unittest.TestCase):
    """Tests for PDSepConfig pickle serialization and backward compatibility."""

    # -- A1: field existence and read/write --

    def test_prefill_stop_stream_wait_timeout_ms_exists(self):
        """prefill_stop_stream_wait_timeout_ms field exists, has default 2000, and is writable."""
        cfg = PDSepConfig()
        self.assertTrue(hasattr(cfg, "prefill_stop_stream_wait_timeout_ms"))
        self.assertEqual(cfg.prefill_stop_stream_wait_timeout_ms, 2000)

        cfg.prefill_stop_stream_wait_timeout_ms = 12345
        self.assertEqual(cfg.prefill_stop_stream_wait_timeout_ms, 12345)

    # -- A2: full 27-item round-trip --

    def test_pickle_roundtrip_27_fields(self):
        """Full 27-item pickle round-trip preserves all field values."""
        cfg = PDSepConfig()
        cfg.prefill_stop_stream_wait_timeout_ms = 9999
        cfg.prefill_slot_pool_size = 42
        cfg.batch_dispatch_timeout_ms = 7777
        cfg.decode_entrance = True
        cfg.load_cache_timeout_ms = 123456

        data = pickle.dumps(cfg)
        restored = pickle.loads(data)

        self.assertEqual(restored.prefill_stop_stream_wait_timeout_ms, 9999)
        self.assertEqual(restored.prefill_slot_pool_size, 42)
        self.assertEqual(restored.batch_dispatch_timeout_ms, 7777)
        self.assertTrue(restored.decode_entrance)
        self.assertEqual(restored.load_cache_timeout_ms, 123456)

    def test_pickle_state_has_27_items(self):
        """__getstate__ produces exactly 27 items."""
        cfg = PDSepConfig()
        state = cfg.__reduce_ex__(2)[2]
        self.assertEqual(len(state), 27)

    # -- A2: legacy 20-item deserialization --

    def test_unpickle_legacy_20_items(self):
        """Legacy 20-item tuple (pre-batch-timeout) deserializes; missing fields use defaults."""
        cfg = PDSepConfig()
        state = cfg.__reduce_ex__(2)[2]

        # Truncate to 20 items (the original baseline before batch timeout fields)
        legacy_state = tuple(state[:20])

        restored = _unpickle_from_state(PDSepConfig, legacy_state)

        # Fields 0-19 are preserved (all defaults since we used a fresh PDSepConfig)
        self.assertFalse(restored.decode_entrance)  # field 19

        # Fields 20-22 (batch timeouts) should use defaults
        self.assertEqual(restored.batch_dispatch_timeout_ms, 60000)
        self.assertEqual(restored.batch_prepare_timeout_ms, 10000)
        self.assertEqual(restored.batch_load_timeout_ms, 10000)

        # Fields 23-25 (prefill pools) should use defaults
        self.assertEqual(restored.prefill_enqueue_pool_size, 0)
        self.assertEqual(restored.prefill_worker_lambda_pool_size, 0)
        self.assertEqual(restored.prefill_slot_pool_size, 0)

        # Field 26 (prefill_stop_stream_wait_timeout_ms) should use default
        self.assertEqual(restored.prefill_stop_stream_wait_timeout_ms, 2000)

    def test_unpickle_legacy_20_items_with_custom_values(self):
        """Legacy 20-item tuple with custom values preserves them; missing fields get defaults."""
        cfg = PDSepConfig()
        state = cfg.__reduce_ex__(2)[2]

        # Modify some fields in the 20-item legacy state
        legacy_state = list(state[:20])
        legacy_state[16] = 99999  # load_cache_timeout_ms
        legacy_state[19] = True  # decode_entrance

        restored = _unpickle_from_state(PDSepConfig, tuple(legacy_state))

        # Modified fields should be preserved
        self.assertEqual(restored.load_cache_timeout_ms, 99999)
        self.assertTrue(restored.decode_entrance)

        # Missing fields should use defaults
        self.assertEqual(restored.batch_dispatch_timeout_ms, 60000)
        self.assertEqual(restored.prefill_slot_pool_size, 0)
        self.assertEqual(restored.prefill_stop_stream_wait_timeout_ms, 2000)

    # -- Additional boundary tests --

    def test_unpickle_legacy_23_items(self):
        """23-item tuple (with batch timeouts, without prefill pools) deserializes correctly."""
        cfg = PDSepConfig()
        state = cfg.__reduce_ex__(2)[2]

        legacy_state = tuple(state[:23])
        restored = _unpickle_from_state(PDSepConfig, legacy_state)

        # Fields 20-22 should be preserved
        self.assertEqual(restored.batch_dispatch_timeout_ms, 60000)
        self.assertEqual(restored.batch_prepare_timeout_ms, 10000)
        self.assertEqual(restored.batch_load_timeout_ms, 10000)

        # Fields 23-26 should use defaults
        self.assertEqual(restored.prefill_enqueue_pool_size, 0)
        self.assertEqual(restored.prefill_worker_lambda_pool_size, 0)
        self.assertEqual(restored.prefill_slot_pool_size, 0)
        self.assertEqual(restored.prefill_stop_stream_wait_timeout_ms, 2000)

    def test_unpickle_legacy_26_items(self):
        """26-item tuple (with prefill pools, without stop_stream_wait) deserializes correctly."""
        cfg = PDSepConfig()
        state = cfg.__reduce_ex__(2)[2]

        legacy_state = tuple(state[:26])
        restored = _unpickle_from_state(PDSepConfig, legacy_state)

        # Field 25 should be preserved
        self.assertEqual(restored.prefill_slot_pool_size, 0)

        # Field 26 should use default
        self.assertEqual(restored.prefill_stop_stream_wait_timeout_ms, 2000)

    def test_unpickle_too_short_raises(self):
        """Tuple with fewer than 20 items raises an error."""
        cfg = PDSepConfig()
        state = cfg.__reduce_ex__(2)[2]

        # 19 items - below the minimum of 20
        short_state = tuple(state[:19])
        with self.assertRaises(Exception):
            _unpickle_from_state(PDSepConfig, short_state)


# ---------------------------------------------------------------------------
# RuntimeConfig pickle tests
# ---------------------------------------------------------------------------
class TestRuntimeConfigPickle(unittest.TestCase):
    """Tests for RuntimeConfig pickle serialization and backward compatibility."""

    def test_pickle_roundtrip_14_fields(self):
        """Full 14-item pickle round-trip preserves all field values."""
        cfg = RuntimeConfig()
        cfg.model_name = "test_model"
        cfg.specify_gpu_arch = "H20"
        cfg.warm_up = True

        data = pickle.dumps(cfg)
        restored = pickle.loads(data)

        self.assertEqual(restored.model_name, "test_model")
        self.assertEqual(restored.specify_gpu_arch, "H20")
        self.assertTrue(restored.warm_up)

    def test_pickle_state_has_14_items(self):
        """__getstate__ produces exactly 14 items."""
        cfg = RuntimeConfig()
        state = cfg.__reduce_ex__(2)[2]
        self.assertEqual(len(state), 14)

    # -- A3: legacy 13-item deserialization --

    def test_unpickle_legacy_13_items(self):
        """Legacy 13-item tuple (pre-specify_gpu_arch) deserializes; missing field uses default."""
        cfg = RuntimeConfig()
        state = cfg.__reduce_ex__(2)[2]

        # Truncate to 13 items (before specify_gpu_arch was added)
        legacy_state = tuple(state[:13])

        restored = _unpickle_from_state(RuntimeConfig, legacy_state)

        # Fields 0-12 should be preserved
        self.assertEqual(restored.model_name, "")

        # Field 13 (specify_gpu_arch) should use default (empty string)
        self.assertEqual(restored.specify_gpu_arch, "")

    def test_unpickle_legacy_13_items_with_custom_values(self):
        """Legacy 13-item tuple with custom values preserves them; specify_gpu_arch gets default."""
        cfg = RuntimeConfig()
        state = cfg.__reduce_ex__(2)[2]

        # Modify some fields in the 13-item legacy state
        legacy_state = list(state[:13])
        legacy_state[9] = "my_custom_model"  # model_name
        legacy_state[0] = 128  # max_generate_batch_size

        restored = _unpickle_from_state(RuntimeConfig, tuple(legacy_state))

        self.assertEqual(restored.max_generate_batch_size, 128)
        self.assertEqual(restored.model_name, "my_custom_model")
        # specify_gpu_arch should be the default empty string
        self.assertEqual(restored.specify_gpu_arch, "")

    def test_unpickle_too_short_raises(self):
        """Tuple with fewer than 13 items raises an error."""
        cfg = RuntimeConfig()
        state = cfg.__reduce_ex__(2)[2]

        # 12 items - below the minimum of 13
        short_state = tuple(state[:12])
        with self.assertRaises(Exception):
            _unpickle_from_state(RuntimeConfig, short_state)


if __name__ == "__main__":
    unittest.main()
