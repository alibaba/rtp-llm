import importlib
import os
from unittest import TestCase, main

import pytest

pytestmark = [pytest.mark.gpu(type="A10")]


class ServerArgsPyEnvConfigsTest(TestCase):
    """Test that environment variables and command line arguments are correctly set to py_env_configs structure."""


class ServerArgsSetTest(TestCase):
    def setUp(self):
        self._environ_backup = os.environ.copy()
        os.environ.clear()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._environ_backup)

    def _setup_args(self, args=None):
        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        if args is None:
            args = []
        return rtp_llm.server.server_args.server_args.setup_args(args=args)

    def test_env_vars_set_to_py_env_configs(self):
        """Test that environment variables are correctly set to py_env_configs."""
        os.environ["MODEL_TYPE"] = "qwen"
        os.environ["CHECKPOINT_PATH"] = "/path/to/checkpoint"
        os.environ["ACT_TYPE"] = "BF16"
        os.environ["TP_SIZE"] = "4"
        os.environ["DP_SIZE"] = "2"
        os.environ["WORLD_SIZE"] = "8"
        os.environ["CONCURRENCY_LIMIT"] = "64"
        os.environ["MAX_CONTEXT_BATCH_SIZE"] = "32"
        os.environ["WARM_UP"] = "1"
        os.environ["MAX_SEQ_LEN"] = "4096"

        py_env_configs = self._setup_args()

        self.assertEqual(py_env_configs.model_args.model_type, "qwen")
        self.assertEqual(py_env_configs.model_args.ckpt_path, "/path/to/checkpoint")
        self.assertEqual(py_env_configs.model_args.act_type, "BF16")
        self.assertEqual(py_env_configs.parallelism_config.tp_size, 4)
        self.assertEqual(py_env_configs.parallelism_config.dp_size, 2)
        self.assertEqual(py_env_configs.parallelism_config.world_size, 8)
        self.assertEqual(py_env_configs.concurrency_config.concurrency_limit, 64)
        self.assertEqual(
            py_env_configs.runtime_config.fifo_scheduler_config.max_context_batch_size,
            32,
        )
        self.assertEqual(py_env_configs.runtime_config.warm_up, True)

    def test_cmd_args_set_to_py_env_configs(self):
        """Test that command line arguments are correctly set to py_env_configs."""
        py_env_configs = self._setup_args(
            args=[
                "--model_type",
                "llama",
                "--checkpoint_path",
                "/path/to/llama/checkpoint",
                "--act_type",
                "FP16",
                "--tp_size",
                "8",
                "--dp_size",
                "4",
                "--world_size",
                "32",
                "--concurrency_limit",
                "128",
                "--max_context_batch_size",
                "64",
                "--warm_up",
                "0",
                "--cache_store_rdma_io_thread_count",
                "4",
                "--cache_store_rdma_worker_thread_count",
                "2",
            ]
        )

        self.assertEqual(py_env_configs.model_args.model_type, "llama")
        self.assertEqual(
            py_env_configs.model_args.ckpt_path, "/path/to/llama/checkpoint"
        )
        self.assertEqual(py_env_configs.model_args.act_type, "FP16")
        self.assertEqual(py_env_configs.parallelism_config.tp_size, 8)
        self.assertEqual(py_env_configs.parallelism_config.dp_size, 4)
        self.assertEqual(py_env_configs.parallelism_config.world_size, 32)
        self.assertEqual(py_env_configs.concurrency_config.concurrency_limit, 128)
        self.assertEqual(
            py_env_configs.runtime_config.fifo_scheduler_config.max_context_batch_size,
            64,
        )
        self.assertEqual(py_env_configs.runtime_config.warm_up, False)
        self.assertEqual(py_env_configs.cache_store_config.rdma_io_thread_count, 4)
        self.assertEqual(py_env_configs.cache_store_config.rdma_worker_thread_count, 2)

    def test_cmd_args_override_env_vars(self):
        """Test that command line arguments override environment variables."""
        os.environ["MODEL_TYPE"] = "qwen"
        os.environ["CHECKPOINT_PATH"] = "/path/to/qwen/checkpoint"
        os.environ["ACT_TYPE"] = "BF16"
        os.environ["TP_SIZE"] = "4"
        os.environ["CONCURRENCY_LIMIT"] = "32"

        py_env_configs = self._setup_args(
            args=[
                "--model_type",
                "llama",
                "--checkpoint_path",
                "/path/to/llama/checkpoint",
                "--act_type",
                "FP16",
                "--tp_size",
                "8",
                "--concurrency_limit",
                "64",
            ]
        )

        self.assertEqual(py_env_configs.model_args.model_type, "llama")
        self.assertEqual(
            py_env_configs.model_args.ckpt_path, "/path/to/llama/checkpoint"
        )
        self.assertEqual(py_env_configs.model_args.act_type, "FP16")
        self.assertEqual(py_env_configs.parallelism_config.tp_size, 8)
        self.assertEqual(py_env_configs.concurrency_config.concurrency_limit, 64)

    def test_mixed_env_and_cmd_args(self):
        """Test mixed environment variables and command line arguments."""
        os.environ["MODEL_TYPE"] = "qwen"
        os.environ["CHECKPOINT_PATH"] = "/path/to/qwen/checkpoint"
        os.environ["ACT_TYPE"] = "BF16"
        os.environ["DP_SIZE"] = "2"
        os.environ["WORLD_SIZE"] = "8"

        py_env_configs = self._setup_args(
            args=[
                "--tp_size",
                "4",
                "--concurrency_limit",
                "64",
                "--max_context_batch_size",
                "32",
            ]
        )

        self.assertEqual(py_env_configs.model_args.model_type, "qwen")
        self.assertEqual(
            py_env_configs.model_args.ckpt_path, "/path/to/qwen/checkpoint"
        )
        self.assertEqual(py_env_configs.model_args.act_type, "BF16")
        self.assertEqual(py_env_configs.parallelism_config.dp_size, 2)
        self.assertEqual(py_env_configs.parallelism_config.world_size, 8)
        self.assertEqual(py_env_configs.parallelism_config.tp_size, 4)
        self.assertEqual(py_env_configs.concurrency_config.concurrency_limit, 64)
        self.assertEqual(
            py_env_configs.runtime_config.fifo_scheduler_config.max_context_batch_size,
            32,
        )

    def test_batch_decode_scheduler_config(self):
        """Test that batch_decode_scheduler_config is correctly set."""
        py_env_configs = self._setup_args(
            args=[
                "--use_batch_decode_scheduler",
                "1",
                "--batch_decode_scheduler_batch_size",
                "16",
                "--batch_decode_scheduler_warmup_type",
                "1",
            ]
        )

        self.assertEqual(py_env_configs.runtime_config.use_batch_decode_scheduler, True)
        self.assertEqual(
            py_env_configs.runtime_config.batch_decode_scheduler_config.batch_decode_scheduler_batch_size,
            16,
        )
        self.assertEqual(
            py_env_configs.runtime_config.batch_decode_scheduler_config.batch_decode_scheduler_warmup_type,
            1,
        )

        runtime_config = py_env_configs.runtime_config
        self.assertEqual(runtime_config.use_batch_decode_scheduler, True)
        self.assertEqual(
            runtime_config.batch_decode_scheduler_config.batch_decode_scheduler_batch_size,
            16,
        )
        self.assertEqual(
            runtime_config.batch_decode_scheduler_config.batch_decode_scheduler_warmup_type,
            1,
        )

    def test_pdfusion_scheduler_mode_config(self):
        """Test that pdfusion_scheduler_mode is opt-in and decode_prefill_ratio is configurable."""
        py_env_configs = self._setup_args(args=[])
        self.assertEqual(
            py_env_configs.runtime_config.fifo_scheduler_config.pdfusion_scheduler_mode,
            "",
        )

        py_env_configs = self._setup_args(
            args=[
                "--pdfusion_scheduler_mode",
                "ratio",
                "--decode_prefill_ratio",
                "1/3",
            ]
        )
        self.assertEqual(
            py_env_configs.runtime_config.fifo_scheduler_config.pdfusion_scheduler_mode,
            "ratio",
        )
        self.assertEqual(
            py_env_configs.runtime_config.fifo_scheduler_config.decode_prefill_ratio,
            "1/3",
        )

    def test_pdfusion_scheduler_mode_rejects_unknown_value(self):
        """Test that pdfusion_scheduler_mode only accepts fixed scheduler patterns."""
        with self.assertRaises(SystemExit):
            self._setup_args(args=["--pdfusion_scheduler_mode", "ratioo"])

    def test_default_argv_path_parses_sys_argv(self):
        """Regression: the production entry (rtp_llm.start_server -> setup_args()) reads sys.argv.

        The other tests call setup_args(args=[...]) with an explicit list, so the default
        ``args=None`` code path a deployed server actually uses had no coverage. Here we drive the
        real sys.argv path: with PYTEST_CURRENT_TEST unset (the deployed condition), setup_args()
        must parse argv into the config exactly like the CLI does.
        """
        import sys

        import rtp_llm.server.server_args.server_args as server_args_mod

        argv_backup = sys.argv
        # setUp() already clears os.environ; drop PYTEST_CURRENT_TEST explicitly so the assertion
        # does not depend on setUp ordering and truly exercises the production branch.
        os.environ.pop("PYTEST_CURRENT_TEST", None)
        try:
            sys.argv = ["prog", "--model_type", "llama", "--tp_size", "8"]
            importlib.reload(server_args_mod)
            py_env_configs = server_args_mod.setup_args()  # args=None -> parse sys.argv
        finally:
            sys.argv = argv_backup

        self.assertEqual(py_env_configs.model_args.model_type, "llama")
        self.assertEqual(py_env_configs.parallelism_config.tp_size, 8)

    def test_pytest_env_guard_ignores_sys_argv(self):
        """Under pytest (PYTEST_CURRENT_TEST set) with args=None, pytest's own argv is ignored.

        This is the guard that lets the suite run under pytest without argparse choking on pytest's
        flags; assert it actually suppresses sys.argv parsing rather than silently leaking it.
        """
        import sys

        import rtp_llm.server.server_args.server_args as server_args_mod

        argv_backup = sys.argv
        os.environ["PYTEST_CURRENT_TEST"] = "server_args_test::guard (call)"
        try:
            sys.argv = ["prog", "--model_type", "should_be_ignored"]
            importlib.reload(server_args_mod)
            py_env_configs = server_args_mod.setup_args()  # args=None + guard -> args=[]
        finally:
            sys.argv = argv_backup

        self.assertNotEqual(py_env_configs.model_args.model_type, "should_be_ignored")


if __name__ == "__main__":
    main()
