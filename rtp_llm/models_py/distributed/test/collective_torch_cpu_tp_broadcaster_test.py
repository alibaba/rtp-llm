import os
import tempfile
import unittest
from unittest.mock import patch

from rtp_llm.models_py.distributed import collective_torch as ct
from rtp_llm.ops import ParallelismConfig


class _FakeLibrtpComputeOps:
    def __init__(self, init_error=None):
        self.init_calls = []
        self.init_error = init_error

    def init_cpu_tp_broadcaster(self, tp_rank, tp_size, base_path):
        if self.init_error is not None:
            raise self.init_error
        self.init_calls.append((tp_rank, tp_size, base_path))


class TestCpuTpBroadcasterBootstrap(unittest.TestCase):
    def setUp(self):
        self._old_parallelism_config = ct._parallelism_config
        self._old_base_path = ct._cpu_tp_broadcaster_base_path
        self._old_nccl_init_port = ct._cpu_tp_broadcaster_nccl_init_port

    def tearDown(self):
        ct._parallelism_config = self._old_parallelism_config
        ct._cpu_tp_broadcaster_base_path = self._old_base_path
        ct._cpu_tp_broadcaster_nccl_init_port = self._old_nccl_init_port

    def _parallelism_config(self, tp_size: int, local_world_size: int):
        parallelism_config = ParallelismConfig()
        parallelism_config.world_rank = 0
        parallelism_config.world_size = max(tp_size, local_world_size)
        parallelism_config.local_rank = 0
        parallelism_config.local_world_size = local_world_size
        parallelism_config.tp_size = tp_size
        parallelism_config.tp_rank = 0
        parallelism_config.dp_rank = 0
        return parallelism_config

    def test_should_init_cpu_tp_broadcaster_only_for_local_tp_groups(self):
        self.assertFalse(
            ct._should_init_cpu_tp_broadcaster(
                self._parallelism_config(tp_size=1, local_world_size=8)
            )
        )
        self.assertFalse(
            ct._should_init_cpu_tp_broadcaster(
                self._parallelism_config(tp_size=6, local_world_size=8)
            )
        )
        self.assertTrue(
            ct._should_init_cpu_tp_broadcaster(
                self._parallelism_config(tp_size=4, local_world_size=8)
            )
        )

    def test_skip_cpu_tp_broadcaster_tp1_with_long_uds_path(self):
        fake_ops = _FakeLibrtpComputeOps()
        ct._parallelism_config = self._parallelism_config(tp_size=1, local_world_size=8)
        ct._cpu_tp_broadcaster_nccl_init_port = 12345
        with patch.dict(
            os.environ,
            {"RTP_LLM_CPU_TP_BROADCASTER_DIR": "/tmp/" + "x" * 200},
        ):
            ct._init_cpu_tp_broadcaster_if_needed(fake_ops)

        self.assertIsNone(ct._cpu_tp_broadcaster_base_path)
        self.assertEqual(fake_ops.init_calls, [])

    def test_skip_cpu_tp_broadcaster_cross_node_with_long_uds_path(self):
        fake_ops = _FakeLibrtpComputeOps()
        ct._parallelism_config = self._parallelism_config(tp_size=6, local_world_size=8)
        ct._cpu_tp_broadcaster_nccl_init_port = 12345
        with patch.dict(
            os.environ,
            {"RTP_LLM_CPU_TP_BROADCASTER_DIR": "/tmp/" + "x" * 200},
        ):
            ct._init_cpu_tp_broadcaster_if_needed(fake_ops)

        self.assertIsNone(ct._cpu_tp_broadcaster_base_path)
        self.assertEqual(fake_ops.init_calls, [])

    def test_init_cpu_tp_broadcaster_makes_base_path_when_enabled(self):
        fake_ops = _FakeLibrtpComputeOps()
        ct._parallelism_config = self._parallelism_config(tp_size=4, local_world_size=8)
        ct._cpu_tp_broadcaster_nccl_init_port = 12345

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {
                "RTP_LLM_CPU_TP_BROADCASTER_DIR": tmpdir,
                "RTP_LLM_CPU_TP_BROADCASTER_ID": "unit-test",
            },
        ):
            ct._init_cpu_tp_broadcaster_if_needed(fake_ops)

        self.assertEqual(len(fake_ops.init_calls), 1)
        tp_rank, tp_size, base_path = fake_ops.init_calls[0]
        self.assertEqual((tp_rank, tp_size), (0, 4))
        self.assertIn("rtp_llm_tp_unit-test_dp0", base_path)
        self.assertEqual(ct._cpu_tp_broadcaster_base_path, base_path)

    def test_init_cpu_tp_broadcaster_fallbacks_when_base_path_invalid(self):
        fake_ops = _FakeLibrtpComputeOps()
        ct._parallelism_config = self._parallelism_config(tp_size=4, local_world_size=8)
        ct._cpu_tp_broadcaster_nccl_init_port = 12345

        with patch.dict(
            os.environ,
            {"RTP_LLM_CPU_TP_BROADCASTER_DIR": "/tmp/" + "x" * 200},
        ), patch("logging.warning") as mock_warning:
            ct._init_cpu_tp_broadcaster_if_needed(fake_ops)

        self.assertIsNone(ct._cpu_tp_broadcaster_base_path)
        self.assertEqual(fake_ops.init_calls, [])
        mock_warning.assert_called_once()
        self.assertIn(
            "Failed to initialize CpuTpBroadcaster",
            mock_warning.call_args.args[0],
        )

    def test_init_cpu_tp_broadcaster_fallbacks_when_cpp_init_fails(self):
        fake_ops = _FakeLibrtpComputeOps(init_error=RuntimeError("init failed"))
        ct._parallelism_config = self._parallelism_config(tp_size=4, local_world_size=8)
        ct._cpu_tp_broadcaster_nccl_init_port = 12345

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {
                "RTP_LLM_CPU_TP_BROADCASTER_DIR": tmpdir,
                "RTP_LLM_CPU_TP_BROADCASTER_ID": "unit-test",
            },
        ), patch("logging.warning") as mock_warning:
            ct._init_cpu_tp_broadcaster_if_needed(fake_ops)

        self.assertIsNone(ct._cpu_tp_broadcaster_base_path)
        self.assertEqual(fake_ops.init_calls, [])
        mock_warning.assert_called_once()
        self.assertIn(
            "Failed to initialize CpuTpBroadcaster",
            mock_warning.call_args.args[0],
        )


if __name__ == "__main__":
    unittest.main()
