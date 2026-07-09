import os
import stat
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from rtp_llm.models_py.distributed import collective_torch as ct
from rtp_llm.ops import ParallelismConfig


class _FakeLibrtpComputeOps:
    def __init__(self, init_error=None):
        self.init_calls = []
        self.destroy_calls = 0
        self.init_error = init_error

    def init_cpu_tp_broadcaster(self, tp_rank, tp_size, base_path):
        if self.init_error is not None:
            raise self.init_error
        self.init_calls.append((tp_rank, tp_size, base_path))

    def destroy_cpu_tp_broadcaster(self):
        self.destroy_calls += 1


class _FakeIncompleteDestroyLibrtpComputeOps:
    def __init__(self):
        self.clear_calls = 0

    def clear_comm_ops(self):
        self.clear_calls += 1


class _FakeProcessGroup:
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


class TestCpuTpBroadcasterBootstrap(unittest.TestCase):
    def setUp(self):
        self._old_parallelism_config = ct._parallelism_config
        self._old_base_path = ct._cpu_tp_broadcaster_base_path
        self._old_nccl_init_port = ct._cpu_tp_broadcaster_nccl_init_port
        self._old_nccl_master_addr = ct._cpu_tp_broadcaster_nccl_master_addr
        self._old_initialized = ct._initialized

    def tearDown(self):
        ct._parallelism_config = self._old_parallelism_config
        ct._cpu_tp_broadcaster_base_path = self._old_base_path
        ct._cpu_tp_broadcaster_nccl_init_port = self._old_nccl_init_port
        ct._cpu_tp_broadcaster_nccl_master_addr = self._old_nccl_master_addr
        ct._initialized = self._old_initialized

    def _parallelism_config(
        self,
        tp_size: int,
        local_world_size: int,
        world_rank: int = 0,
        local_rank: int = 0,
    ):
        parallelism_config = ParallelismConfig()
        parallelism_config.world_rank = world_rank
        parallelism_config.world_size = max(tp_size, local_world_size)
        parallelism_config.local_rank = local_rank
        parallelism_config.local_world_size = local_world_size
        parallelism_config.tp_size = tp_size
        parallelism_config.tp_rank = world_rank % tp_size if tp_size > 0 else 0
        parallelism_config.dp_rank = world_rank // tp_size if tp_size > 0 else 0
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
        striped_config = self._parallelism_config(
            tp_size=2, local_world_size=4, world_rank=2, local_rank=1
        )
        self.assertFalse(ct._should_init_cpu_tp_broadcaster(striped_config))
        mismatched_config = self._parallelism_config(tp_size=2, local_world_size=4)
        mismatched_config.tp_rank = 1
        self.assertFalse(ct._should_init_cpu_tp_broadcaster(mismatched_config))

    def test_disable_env_skips_cpu_tp_broadcaster(self):
        fake_ops = _FakeLibrtpComputeOps()
        ct._parallelism_config = self._parallelism_config(tp_size=4, local_world_size=8)
        ct._cpu_tp_broadcaster_nccl_init_port = 12345

        with patch.dict(
            os.environ,
            {
                "RTP_LLM_CPU_TP_BROADCASTER_DISABLE": "true",
                "RTP_LLM_CPU_TP_BROADCASTER_DIR": "/tmp/" + "x" * 200,
            },
        ):
            ct._init_cpu_tp_broadcaster_if_needed(fake_ops)

        self.assertIsNone(ct._cpu_tp_broadcaster_base_path)
        self.assertEqual(fake_ops.init_calls, [])

    def test_make_cpu_tp_broadcaster_base_path_chmods_existing_dir(self):
        parallelism_config = self._parallelism_config(tp_size=2, local_world_size=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chmod(tmpdir, 0o755)
            with patch.dict(
                os.environ,
                {
                    "RTP_LLM_CPU_TP_BROADCASTER_DIR": tmpdir,
                    "RTP_LLM_CPU_TP_BROADCASTER_ID": "unit-test",
                },
            ):
                ct._make_cpu_tp_broadcaster_base_path(parallelism_config, 12345)

            mode = stat.S_IMODE(os.stat(tmpdir).st_mode)
            self.assertEqual(mode, 0o700)

    def test_make_cpu_tp_broadcaster_base_path_defaults_to_job_session(self):
        parallelism_config = self._parallelism_config(tp_size=2, local_world_size=4)
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {"RTP_LLM_CPU_TP_BROADCASTER_DIR": tmpdir},
        ):
            os.environ.pop("RTP_LLM_CPU_TP_BROADCASTER_ID", None)
            base_path = ct._make_cpu_tp_broadcaster_base_path(
                parallelism_config, 12345, "10.0.0.1"
            )

        self.assertIn("rtp_llm_tp_port12345_w4_tp2_", base_path)
        self.assertNotIn("ppid", base_path)

    def test_make_cpu_tp_broadcaster_base_path_distinguishes_master_addr(self):
        parallelism_config = self._parallelism_config(tp_size=2, local_world_size=4)
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {"RTP_LLM_CPU_TP_BROADCASTER_DIR": tmpdir},
        ):
            os.environ.pop("RTP_LLM_CPU_TP_BROADCASTER_ID", None)
            base_path1 = ct._make_cpu_tp_broadcaster_base_path(
                parallelism_config, 12345, "10.0.0.1"
            )
            base_path2 = ct._make_cpu_tp_broadcaster_base_path(
                parallelism_config, 12345, "10.0.0.2"
            )

        self.assertNotEqual(base_path1, base_path2)

    def test_make_cpu_tp_broadcaster_base_path_checks_highest_rank_path(self):
        parallelism_config = self._parallelism_config(tp_size=11, local_world_size=11)
        with tempfile.TemporaryDirectory() as tmpdir:
            fixed_path = os.path.join(tmpdir, "rtp_llm_tp__dp0") + "_10.sock"
            session_len = ct._UDS_SUN_PATH_LIMIT - len(os.fsencode(fixed_path))
            self.assertGreater(session_len, 0)
            with patch.dict(
                os.environ,
                {
                    "RTP_LLM_CPU_TP_BROADCASTER_DIR": tmpdir,
                    "RTP_LLM_CPU_TP_BROADCASTER_ID": "a" * session_len,
                },
            ):
                with self.assertRaisesRegex(ValueError, "UDS path too long"):
                    ct._make_cpu_tp_broadcaster_base_path(parallelism_config, 12345)

    def test_make_cpu_tp_broadcaster_base_path_rejects_symlink_dir(self):
        parallelism_config = self._parallelism_config(tp_size=2, local_world_size=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "target")
            link = os.path.join(tmpdir, "link")
            os.mkdir(target)
            os.chmod(target, 0o755)
            os.symlink(target, link)
            with patch.dict(
                os.environ,
                {
                    "RTP_LLM_CPU_TP_BROADCASTER_DIR": link,
                    "RTP_LLM_CPU_TP_BROADCASTER_ID": "unit-test",
                },
            ):
                with self.assertRaisesRegex(ValueError, "not safe"):
                    ct._make_cpu_tp_broadcaster_base_path(parallelism_config, 12345)

            mode = stat.S_IMODE(os.stat(target).st_mode)
            self.assertEqual(mode, 0o755)

    def test_init_cpu_tp_broadcaster_skips_when_tp_group_eligibility_diverges(self):
        fake_ops = _FakeLibrtpComputeOps()
        ct._parallelism_config = self._parallelism_config(tp_size=2, local_world_size=4)
        ct._cpu_tp_broadcaster_nccl_init_port = 12345
        ct._initialized = True

        def fake_all_gather_object(output, value, group):
            output[:] = [True, False]

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {
                "RTP_LLM_CPU_TP_BROADCASTER_DIR": tmpdir,
                "RTP_LLM_CPU_TP_BROADCASTER_ID": "unit-test",
            },
        ), patch(
            "rtp_llm.models_py.distributed.collective_torch.torch.distributed.is_initialized",
            return_value=True,
        ), patch(
            "rtp_llm.models_py.distributed.collective_torch._get_group",
            return_value=_FakeProcessGroup(2),
        ), patch(
            "rtp_llm.models_py.distributed.collective_torch.torch.distributed.all_gather_object",
            side_effect=fake_all_gather_object,
        ), patch(
            "logging.warning"
        ) as mock_warning:
            ct._init_cpu_tp_broadcaster_if_needed(fake_ops)

        self.assertIsNone(ct._cpu_tp_broadcaster_base_path)
        self.assertEqual(fake_ops.init_calls, [])
        self.assertIn("inconsistent eligibility", mock_warning.call_args.args[0])

    def test_init_cpu_tp_broadcaster_destroys_when_actual_init_diverges(self):
        fake_ops = _FakeLibrtpComputeOps()
        ct._parallelism_config = self._parallelism_config(tp_size=2, local_world_size=4)
        ct._cpu_tp_broadcaster_nccl_init_port = 12345
        ct._initialized = True
        gather_calls = []

        def fake_all_gather_object(output, value, group):
            gather_calls.append(value)
            output[:] = [True, True] if len(gather_calls) == 1 else [True, False]

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {
                "RTP_LLM_CPU_TP_BROADCASTER_DIR": tmpdir,
                "RTP_LLM_CPU_TP_BROADCASTER_ID": "unit-test",
            },
        ), patch(
            "rtp_llm.models_py.distributed.collective_torch.torch.distributed.is_initialized",
            return_value=True,
        ), patch(
            "rtp_llm.models_py.distributed.collective_torch._get_group",
            return_value=_FakeProcessGroup(2),
        ), patch(
            "rtp_llm.models_py.distributed.collective_torch.torch.distributed.all_gather_object",
            side_effect=fake_all_gather_object,
        ), patch(
            "logging.warning"
        ) as mock_warning:
            ct._init_cpu_tp_broadcaster_if_needed(fake_ops)

        self.assertIsNone(ct._cpu_tp_broadcaster_base_path)
        self.assertEqual(len(fake_ops.init_calls), 1)
        self.assertEqual(fake_ops.destroy_calls, 1)
        self.assertIn(
            "inconsistent initialized state",
            mock_warning.call_args.args[0],
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

    def test_destroy_distributed_environment_requires_matching_cpp_symbols(self):
        fake_ops = _FakeIncompleteDestroyLibrtpComputeOps()
        fake_arch = types.ModuleType("rtp_llm.models_py.utils.arch")
        fake_arch.is_cuda = lambda: False

        with patch.dict(
            sys.modules,
            {
                "librtp_compute_ops": fake_ops,
                "rtp_llm.models_py.utils.arch": fake_arch,
            },
        ), patch(
            "rtp_llm.models_py.distributed.collective_torch.torch.distributed.get_rank",
            return_value=0,
        ), patch(
            "rtp_llm.models_py.distributed.collective_torch.torch.distributed.is_initialized",
            return_value=False,
        ), patch(
            "rtp_llm.models_py.distributed.collective_torch.rocm_rccl.is_available_runtime",
            return_value=False,
        ):
            with self.assertRaises(AttributeError):
                ct.destroy_distributed_environment()

        self.assertEqual(fake_ops.clear_calls, 1)
        self.assertFalse(ct._initialized)
        self.assertIsNone(ct._parallelism_config)


if __name__ == "__main__":
    unittest.main()
