import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.utils.multicast_keeper_generation import (
    CURRENT_EPOCH_KEY,
    MulticastGenerationError,
    MulticastGenerationGuard,
    generation_guard_enabled,
)


class FakeStore:
    def __init__(self):
        self._data = {}
        self._lock = threading.Lock()

    def set(self, key, value):
        with self._lock:
            self._data[key] = value.encode()

    def get(self, key):
        with self._lock:
            if key not in self._data:
                raise RuntimeError(f"Key {key} not found")
            return self._data[key]

    def check(self, keys):
        with self._lock:
            return all(key in self._data for key in keys)

    def compare_set(self, key, expected, desired):
        with self._lock:
            current = self._data.get(key, b"")
            if current == expected.encode():
                current = desired.encode()
                self._data[key] = current
            return current


def make_config(*, enabled=True, level=3):
    return SimpleNamespace(
        runtime_config=SimpleNamespace(
            enable_sleep_mode=enabled,
            sleep_mode_level=level,
        ),
        distribute_config=SimpleNamespace(
            dist_comm_timeout=2,
            gang_timeout_min=0,
        ),
    )


class MulticastGenerationGuardTest(unittest.TestCase):
    def test_disabled_sleep_modes_do_not_create_guard_or_touch_store(self):
        store = FakeStore()
        keeper_env = {"RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER": "1"}
        cases = (
            (make_config(enabled=False, level=3), keeper_env),
            (make_config(enabled=True, level=1), keeper_env),
            (make_config(enabled=True, level=2), keeper_env),
            (make_config(enabled=True, level=3), {}),
        )
        for config, env in cases:
            with self.subTest(
                enabled=config.runtime_config.enable_sleep_mode,
                level=config.runtime_config.sleep_mode_level,
                env=env,
            ):
                self.assertFalse(generation_guard_enabled(config, env))
                self.assertIsNone(
                    MulticastGenerationGuard.from_config(
                        config,
                        store=store,
                        rank=0,
                        world_size=1,
                        env=env,
                    )
                )
        self.assertEqual({}, store._data)

    def test_level3_keeper_creates_guard_without_touching_store_until_join(self):
        store = FakeStore()
        guard = MulticastGenerationGuard.from_config(
            make_config(),
            store=store,
            rank=0,
            world_size=1,
            env={"RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER": "1"},
        )

        self.assertIsNotNone(guard)
        self.assertEqual({}, store._data)

    def test_rank_claim_does_not_wait_for_membership_or_commit_barrier(self):
        store = FakeStore()
        rank0 = MulticastGenerationGuard(
            store=store,
            rank=0,
            world_size=2,
            incarnation="rank-0",
        )
        rank1 = MulticastGenerationGuard(
            store=store,
            rank=1,
            world_size=2,
            incarnation="rank-1",
        )
        with patch.dict(os.environ, {}, clear=False):
            try:
                epoch = rank0.join()
                self.assertTrue(
                    store.check(
                        [MulticastGenerationGuard.rank_incarnation_key(epoch, 0)]
                    )
                )
                self.assertFalse(
                    store.check(
                        [MulticastGenerationGuard.rank_incarnation_key(epoch, 1)]
                    )
                )
                self.assertEqual(epoch, rank1.join())
                self.assertFalse(hasattr(rank0, "commit"))
            finally:
                rank0.stop()
                rank1.stop()

    def test_restarted_peer_poison_old_generation_before_collective_init(self):
        store = FakeStore()
        old_rank0 = MulticastGenerationGuard(
            store=store,
            rank=0,
            world_size=2,
            incarnation="old-0",
        )
        old_rank1 = MulticastGenerationGuard(
            store=store,
            rank=1,
            world_size=2,
            incarnation="old-1",
        )
        with patch.dict(os.environ, {}, clear=False):
            try:
                old_rank0.join()
                old_rank1.join()
                replacement = MulticastGenerationGuard(
                    store=store,
                    rank=1,
                    world_size=2,
                    incarnation="replacement-1",
                )

                with self.assertRaisesRegex(
                    MulticastGenerationError,
                    "stale multicast keeper rank join",
                ):
                    replacement.join()

                self.assertTrue(old_rank0._abort_event.wait(timeout=2))
                self.assertIn("replacement-1", old_rank0.abort_reason)
            finally:
                old_rank0.stop()
                old_rank1.stop()

    def test_rank_zero_rejects_and_aborts_reuse_of_active_store(self):
        store = FakeStore()
        store.set(CURRENT_EPOCH_KEY, "active-epoch")
        guard = MulticastGenerationGuard(
            store=store,
            rank=0,
            world_size=1,
        )

        with self.assertRaisesRegex(
            MulticastGenerationError,
            "already contains an active epoch",
        ):
            guard.join()
        self.assertTrue(
            store.check([MulticastGenerationGuard.abort_key("active-epoch")])
        )


if __name__ == "__main__":
    unittest.main()
