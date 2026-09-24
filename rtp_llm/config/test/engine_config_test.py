import unittest
from unittest import TestCase

from rtp_llm.config.engine_config import finalize_scheduler_config
from rtp_llm.ops import RoleType


class DummyFIFOSchedulerConfig:
    def __init__(self):
        self.max_context_batch_size = 2
        self.max_batch_tokens_size = 0
        self.prefill_chunk_size = 0
        self.prefill_chunk_batch_tokens = 0


class EngineConfigTest(TestCase):
    def _finalize(self, chunk_size=0, chunk_batch_tokens=0, **overrides):
        cfg = DummyFIFOSchedulerConfig()
        cfg.prefill_chunk_size = chunk_size
        cfg.prefill_chunk_batch_tokens = chunk_batch_tokens
        args = {
            "max_seq_len": 1024,
            "role_type": RoleType.PREFILL,
            "use_batch_decode_scheduler": False,
            "seq_size_per_block": 64,
        }
        args.update(overrides)
        finalize_scheduler_config(cfg, **args)
        return cfg

    def test_finalize_scheduler_config_disabled_by_default(self):
        # prefill_chunk_size <= 0 => chunked prefill disabled, no validation runs.
        cfg = self._finalize()

        self.assertEqual(cfg.max_batch_tokens_size, 2048)
        self.assertEqual(cfg.prefill_chunk_size, 0)

    def test_finalize_scheduler_config_rejects_chunk_size_smaller_than_one_block(self):
        with self.assertRaises(ValueError):
            self._finalize(chunk_size=17)

    def test_chunk_batch_budget_preserves_default_and_aligns_explicit_budget(self):
        self.assertEqual(self._finalize(chunk_size=128).prefill_chunk_batch_tokens, 0)
        cfg = self._finalize(chunk_size=130, chunk_batch_tokens=259)
        self.assertEqual(cfg.prefill_chunk_size, 128)
        self.assertEqual(cfg.prefill_chunk_batch_tokens, 256)

    def test_chunk_batch_budget_rejects_invalid_ranges(self):
        for chunk, batch in ((0, 256), (128, -1), (128, 64), (128, 17), (128, 2112)):
            with self.subTest(chunk=chunk, batch=batch):
                with self.assertRaisesRegex(ValueError, "prefill_chunk_batch_tokens"):
                    self._finalize(chunk_size=chunk, chunk_batch_tokens=batch)

    def test_chunk_batch_budget_is_disabled_on_decode_role(self):
        cfg = self._finalize(chunk_size=128, chunk_batch_tokens=256, role_type=RoleType.DECODE)
        self.assertEqual(cfg.prefill_chunk_size, 0)
        self.assertEqual(cfg.prefill_chunk_batch_tokens, 0)

    def test_finalize_scheduler_config_floor_aligns_chunk_size(self):
        requested_chunk_size = 130
        cfg = self._finalize(chunk_size=requested_chunk_size)

        self.assertEqual(cfg.prefill_chunk_size, 128)
        self.assertLessEqual(cfg.prefill_chunk_size, requested_chunk_size)

    def test_finalize_scheduler_config_allows_int_max_chunk_size(self):
        cfg = self._finalize(
            chunk_size=2**31 - 1,
            seq_size_per_block=1,
        )

        self.assertEqual(cfg.prefill_chunk_size, 2**31 - 1)

    def test_finalize_scheduler_config_rejects_chunk_size_above_int_max(self):
        with self.assertRaises(ValueError):
            self._finalize(
                chunk_size=2**31,
                seq_size_per_block=1,
            )

    def test_finalize_scheduler_config_disables_chunked_prefill_for_unsupported_role(
        self,
    ):
        # Roles other than PREFILL / PDFUSION never activate chunked prefill in C++; config
        # finalization should not reject their model combination just because the shared env
        # var is present, and it should silently zero prefill_chunk_size so downstream sees a
        # disabled config.
        cfg = self._finalize(
            chunk_size=64,
            role_type=RoleType.DECODE,
            use_batch_decode_scheduler=True,
        )

        self.assertEqual(cfg.max_batch_tokens_size, 2048)
        self.assertEqual(cfg.prefill_chunk_size, 0)

    def test_finalize_scheduler_config_rejects_batch_decode_scheduler(self):
        with self.assertRaisesRegex(ValueError, "use_batch_decode_scheduler=True"):
            self._finalize(
                chunk_size=64,
                use_batch_decode_scheduler=True,
            )

    def test_finalize_scheduler_config_allows_supported_roles(self):
        # Both roles execute prefill locally and share the same chunked-prefill gate.
        for role_type in (RoleType.PREFILL, RoleType.PDFUSION):
            with self.subTest(role_type=role_type):
                cfg = self._finalize(
                    chunk_size=64,
                    role_type=role_type,
                )
                self.assertEqual(cfg.prefill_chunk_size, 64)


if __name__ == "__main__":
    unittest.main()
