import unittest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest import TestCase

from rtp_llm.config.collective_config import CollectiveConfig
from rtp_llm.config.engine_config import (
    EngineConfig,
    finalize_scheduler_config,
    resolve_pd_role_type,
)
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.ops import RoleType, VitSeparation


class DummyFIFOSchedulerConfig:
    def __init__(self):
        self.max_context_batch_size = 2
        self.max_batch_tokens_size = 0
        self.prefill_chunk_size = 0


class PdRoleResolutionTest(TestCase):
    """VIT_SEPARATION=ROLE must not turn a prefill/decode container into VIT."""

    @staticmethod
    def _config(role_type, separation):
        return SimpleNamespace(
            role_config=SimpleNamespace(role_type=role_type),
            vit_config=SimpleNamespace(vit_separation=separation),
        )

    def test_vit_container_reports_vit(self):
        self.assertEqual(
            resolve_pd_role_type(
                self._config(RoleType.VIT, VitSeparation.VIT_SEPARATION_ROLE)
            ),
            RoleType.VIT,
        )

    def test_prefill_container_keeps_its_role_under_role_separation(self):
        self.assertEqual(
            resolve_pd_role_type(
                self._config(RoleType.PREFILL, VitSeparation.VIT_SEPARATION_ROLE)
            ),
            RoleType.PREFILL,
        )

    def test_decode_container_keeps_its_role_under_role_separation(self):
        self.assertEqual(
            resolve_pd_role_type(
                self._config(RoleType.DECODE, VitSeparation.VIT_SEPARATION_ROLE)
            ),
            RoleType.DECODE,
        )

    def test_local_separation_uses_the_launcher_role(self):
        self.assertEqual(
            resolve_pd_role_type(
                self._config(RoleType.PDFUSION, VitSeparation.VIT_SEPARATION_LOCAL)
            ),
            RoleType.PDFUSION,
        )


class EngineConfigTest(TestCase):
    def test_collective_policy_is_an_immutable_snapshot(self):
        configs = PyEnvConfigs()
        configs.all_reduce_backend = "nccl"
        configs.all_reduce_large_prefill = False
        engine = EngineConfig.create(configs)
        self.assertEqual(engine.collective_config, CollectiveConfig("nccl", False))
        configs.all_reduce_backend = "auto"
        configs.all_reduce_large_prefill = None
        self.assertEqual(engine.collective_config, CollectiveConfig("nccl", False))
        with self.assertRaises(FrozenInstanceError):
            setattr(engine.collective_config, "large_prefill", True)
        self.assertIn("all_reduce_backend: nccl", engine.to_string())

    def test_collective_policy_defaults_to_auto(self):
        engine = EngineConfig.create(PyEnvConfigs())
        self.assertEqual(engine.collective_config, CollectiveConfig())

    def test_explicit_collective_topology_is_validated_on_single_rank(self):
        configs = PyEnvConfigs()
        for backend in ("sm120_ipc", "sm120_p2p"):
            with self.subTest(backend=backend):
                configs.all_reduce_backend = backend
                with self.assertRaisesRegex(ValueError, "requires single-node"):
                    EngineConfig.create(configs)

    def _finalize(self, chunk_size=0, **overrides):
        cfg = DummyFIFOSchedulerConfig()
        cfg.prefill_chunk_size = chunk_size
        args = {
            "max_seq_len": 1024,
            "role_type": RoleType.PREFILL,
            "use_batch_decode_scheduler": False,
            "seq_size_per_block": 64,
        }
        args.update(overrides)
        finalize_scheduler_config(cfg, **args)
        return cfg

    def test_finalize_scheduler_config_rejects_chunk_size_smaller_than_one_block(self):
        with self.assertRaises(ValueError):
            self._finalize(chunk_size=17)

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


if __name__ == "__main__":
    unittest.main()
