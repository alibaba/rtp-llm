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
    def test_finalize_scheduler_config_disabled_by_default(self):
        # prefill_chunk_size <= 0 => chunked prefill disabled, no validation runs.
        cfg = DummyFIFOSchedulerConfig()

        finalize_scheduler_config(
            cfg,
            max_seq_len=1024,
            use_mla=True,  # would raise if chunked prefill were enabled
            use_hybrid_attention=True,  # would raise if chunked prefill were enabled
            role_type=RoleType.PREFILL,
            seq_size_per_block=1024,
        )

        self.assertEqual(cfg.max_batch_tokens_size, 2048)
        self.assertEqual(cfg.prefill_chunk_size, 0)

    def test_finalize_scheduler_config_clamps_chunk_size_to_one_block(self):
        cfg = DummyFIFOSchedulerConfig()
        cfg.prefill_chunk_size = 17

        finalize_scheduler_config(
            cfg,
            max_seq_len=1024,
            use_mla=False,
            use_hybrid_attention=False,
            role_type=RoleType.PREFILL,
            seq_size_per_block=64,
        )

        self.assertEqual(cfg.prefill_chunk_size, 64)

    def test_finalize_scheduler_config_rejects_known_unsupported_models(self):
        # Regression check: enumerate the model_types we currently expect the gate to reject,
        # annotated with WHICH flag catches each one (as set in that model's `_from_hf`).
        # If a model ever loses its flag, its _from_hf breaks first — but this table also fails
        # loudly if someone weakens the gate (e.g. drops the use_hybrid_attention branch).
        expected = {
            # model_type: (use_mla, use_hybrid_attention)
            "deepseek2":       (True, False),
            "deepseek3":       (True, False),
            "deepseek-v3-mtp": (True, False),
            "deepseek_v31":    (True, False),
            "deepseek_v32":    (True, False),
            "kimi_k2":         (True, False),
            "kimi_k25":        (True, False),
            "glm_5":           (True, False),
            "qwen3_next":      (False, True),
            "qwen3_next_mtp":  (False, True),
            "kimi_linear":     (True, True),
        }
        for model_type, (use_mla, use_hybrid_attention) in expected.items():
            with self.subTest(model_type=model_type):
                cfg = DummyFIFOSchedulerConfig()
                cfg.prefill_chunk_size = 64
                with self.assertRaises(ValueError):
                    finalize_scheduler_config(
                        cfg,
                        max_seq_len=1024,
                        use_mla=use_mla,
                        use_hybrid_attention=use_hybrid_attention,
                        role_type=RoleType.PREFILL,
                        seq_size_per_block=64,
                    )

    def test_finalize_scheduler_config_disables_chunked_prefill_for_unsupported_role(self):
        # Roles other than PREFILL / PDFUSION never activate chunked prefill in C++; config
        # finalization should not reject their model combination just because the shared env
        # var is present, and it should silently zero prefill_chunk_size so downstream sees a
        # disabled config.
        cfg = DummyFIFOSchedulerConfig()
        cfg.prefill_chunk_size = 64

        finalize_scheduler_config(
            cfg,
            max_seq_len=1024,
            use_mla=True,
            use_hybrid_attention=True,
            role_type=RoleType.DECODE,
            use_batch_decode_scheduler=True,
            seq_size_per_block=64,
        )

        self.assertEqual(cfg.max_batch_tokens_size, 2048)
        self.assertEqual(cfg.prefill_chunk_size, 0)

    def test_finalize_scheduler_config_rejects_batch_decode_scheduler(self):
        cfg = DummyFIFOSchedulerConfig()
        cfg.prefill_chunk_size = 64

        with self.assertRaisesRegex(ValueError, "use_batch_decode_scheduler=True"):
            finalize_scheduler_config(
                cfg,
                max_seq_len=1024,
                use_mla=False,
                use_hybrid_attention=False,
                role_type=RoleType.PREFILL,
                use_batch_decode_scheduler=True,
                seq_size_per_block=64,
            )

    def test_finalize_scheduler_config_allows_plain_attention_model(self):
        cfg = DummyFIFOSchedulerConfig()
        cfg.prefill_chunk_size = 64

        finalize_scheduler_config(
            cfg,
            max_seq_len=1024,
            use_mla=False,
            use_hybrid_attention=False,
            role_type=RoleType.PREFILL,
            seq_size_per_block=64,
        )
        self.assertEqual(cfg.prefill_chunk_size, 64)

    def test_finalize_scheduler_config_allows_pdfusion_role(self):
        # PDFUSION runs prefill locally too; chunked prefill applies the same as PREFILL.
        cfg = DummyFIFOSchedulerConfig()
        cfg.prefill_chunk_size = 64

        finalize_scheduler_config(
            cfg,
            max_seq_len=1024,
            use_mla=False,
            use_hybrid_attention=False,
            role_type=RoleType.PDFUSION,
            seq_size_per_block=64,
        )
        self.assertEqual(cfg.prefill_chunk_size, 64)


if __name__ == "__main__":
    unittest.main()
