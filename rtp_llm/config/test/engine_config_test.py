import unittest
from unittest import TestCase

from rtp_llm.config.engine_config import finalize_scheduler_config
from rtp_llm.ops import RoleType


class DummyFIFOSchedulerConfig:
    def __init__(self):
        self.max_context_batch_size = 2
        self.max_batch_tokens_size = 0
        self.prefill_chunk_size = 0


class EngineConfigTest(TestCase):
    def _finalize(self, chunk_size=0, **overrides):
        cfg = DummyFIFOSchedulerConfig()
        cfg.prefill_chunk_size = chunk_size
        args = {
            "max_seq_len": 1024,
            "use_mla": False,
            "use_hybrid_attention": False,
            "role_type": RoleType.PREFILL,
            "use_batch_decode_scheduler": False,
            "seq_size_per_block": 64,
        }
        args.update(overrides)
        finalize_scheduler_config(cfg, **args)
        return cfg

    def test_finalize_scheduler_config_disabled_by_default(self):
        # prefill_chunk_size <= 0 => chunked prefill disabled, no validation runs.
        cfg = self._finalize(
            use_mla=True,  # would raise if chunked prefill were enabled
            use_hybrid_attention=True,  # would raise if chunked prefill were enabled
        )

        self.assertEqual(cfg.max_batch_tokens_size, 2048)
        self.assertEqual(cfg.prefill_chunk_size, 0)

    def test_finalize_scheduler_config_clamps_chunk_size_to_one_block(self):
        cfg = self._finalize(chunk_size=17)

        self.assertEqual(cfg.prefill_chunk_size, 64)

    def test_finalize_scheduler_config_rejects_unsupported_attention_modes(self):
        cases = (
            ("mla", True, False, "MLA models"),
            ("hybrid_attention", False, True, "hybrid / linear-attention models"),
        )
        for gate, use_mla, use_hybrid_attention, error_pattern in cases:
            with self.subTest(gate=gate):
                with self.assertRaisesRegex(ValueError, error_pattern):
                    self._finalize(
                        chunk_size=64,
                        use_mla=use_mla,
                        use_hybrid_attention=use_hybrid_attention,
                    )

    def test_finalize_scheduler_config_disables_chunked_prefill_for_unsupported_role(self):
        # Roles other than PREFILL / PDFUSION never activate chunked prefill in C++; config
        # finalization should not reject their model combination just because the shared env
        # var is present, and it should silently zero prefill_chunk_size so downstream sees a
        # disabled config.
        cfg = self._finalize(
            chunk_size=64,
            use_mla=True,
            use_hybrid_attention=True,
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
