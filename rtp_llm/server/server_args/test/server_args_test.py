import argparse
import importlib
import os
import sys
from unittest import TestCase, main

from rtp_llm.utils.pre_import_config import (
    configure_expandable_segments_for_warmup,
    is_start_server_entrypoint,
    str2bool,
    warmup_requested,
)


class ServerArgsPyEnvConfigsTest(TestCase):
    """Test that environment variables and command line arguments are correctly set to py_env_configs structure."""


class PreImportConfigTest(TestCase):
    def test_server_parser_reuses_pre_import_bool_parser(self):
        from rtp_llm.server.server_args.util import str2bool as server_str2bool

        self.assertIs(server_str2bool, str2bool)

    def test_identifies_supported_server_entrypoints(self):
        self.assertTrue(
            is_start_server_entrypoint(
                ["python", "-m", "rtp_llm.start_server", "--warm_up", "1"]
            )
        )
        self.assertTrue(is_start_server_entrypoint(["/tmp/start_server.py"]))
        self.assertTrue(is_start_server_entrypoint(["python", "/tmp/start_server.py"]))
        self.assertTrue(
            is_start_server_entrypoint(["python", "-u", "/tmp/start_server.py"])
        )
        self.assertFalse(is_start_server_entrypoint(["python", "other.py"]))
        self.assertFalse(
            is_start_server_entrypoint(
                ["python", "other.py", "--config", "/tmp/start_server.py"]
            )
        )

    def test_warmup_cli_overrides_environment(self):
        self.assertFalse(warmup_requested(["--warm_up", "0"], {"WARM_UP": "1"}))
        self.assertTrue(warmup_requested(["--warm_up=on"], {"WARM_UP": "0"}))

    def test_warmup_environment_and_default(self):
        self.assertFalse(warmup_requested([], {"WARM_UP": "false"}))
        self.assertTrue(warmup_requested([], {}))

    def test_rejects_invalid_warmup_value(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            warmup_requested(["--warm_up", "invalid"], {})

    def test_invalid_warmup_value_does_not_break_pre_import_configuration(self):
        env = {}
        with self.assertLogs(
            "rtp_llm.utils.pre_import_config", level="WARNING"
        ) as logs:
            enabled = configure_expandable_segments_for_warmup(
                ["--warm_up", "invalid"], env, is_rocm=False
            )

        self.assertTrue(enabled)
        self.assertEqual(env["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
        self.assertIn("using default=true", logs.output[0])

    def test_sets_allocator_default_only_for_cuda_warmup(self):
        env = {}
        self.assertTrue(
            configure_expandable_segments_for_warmup([], env, is_rocm=False)
        )
        self.assertEqual(env["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")

        env = {}
        self.assertFalse(
            configure_expandable_segments_for_warmup(
                ["--warm_up", "0"], env, is_rocm=False
            )
        )
        self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", env)

    def test_preserves_user_allocator_configuration(self):
        env = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"}
        self.assertFalse(
            configure_expandable_segments_for_warmup([], env, is_rocm=False)
        )
        self.assertEqual(env["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:False")

    def test_skips_allocator_default_on_rocm(self):
        env = {"WARM_UP": "1"}
        self.assertFalse(
            configure_expandable_segments_for_warmup([], env, is_rocm=True)
        )
        self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", env)


class ServerArgsSetTest(TestCase):
    def setUp(self):
        self._environ_backup = os.environ.copy()
        self._argv_backup = sys.argv.copy()
        os.environ.clear()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._environ_backup)
        sys.argv = self._argv_backup

    def test_env_vars_set_to_py_env_configs(self):
        """Test that environment variables are correctly set to py_env_configs."""
        # Set environment variables
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

        sys.argv = ["prog"]

        # Import and setup args
        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        py_env_configs = rtp_llm.server.server_args.server_args.setup_args()

        # Verify model_args
        self.assertEqual(py_env_configs.model_args.model_type, "qwen")
        self.assertEqual(py_env_configs.model_args.ckpt_path, "/path/to/checkpoint")
        self.assertEqual(py_env_configs.model_args.act_type, "BF16")

        # Verify parallelism_config
        self.assertEqual(py_env_configs.parallelism_config.tp_size, 4)
        self.assertEqual(py_env_configs.parallelism_config.dp_size, 2)
        self.assertEqual(py_env_configs.parallelism_config.world_size, 8)

        # Verify concurrency_config
        self.assertEqual(py_env_configs.concurrency_config.concurrency_limit, 64)

        # Verify fifo_scheduler_config
        self.assertEqual(
            py_env_configs.runtime_config.fifo_scheduler_config.max_context_batch_size,
            32,
        )

        # Verify runtime_config (warm_up is now in RuntimeConfig)
        self.assertEqual(py_env_configs.runtime_config.warm_up, True)  # bool in C++
        # Note: max_seq_len is in ModelConfig, not RuntimeConfig or EngineConfig
        # It will be set when ModelConfig is created from model_args

    def test_runtime_tuning_args_are_validated_and_exported(self):
        from rtp_llm.server.server_args.server_args import setup_args

        setup_args(
            [
                "--runtime_mem_safety_ratio",
                "0.08",
                "--runtime_mem_no_warmup_floor_mb",
                "3072",
                "--moe_runtime_mem_log",
                "true",
                "--moe_runtime_slot_log",
                "false",
                "--moe_runtime_slot_min_slots",
                "128",
                "--moe_skew_mult",
                "1.75",
                "--moe_skew_add",
                "0.2",
            ]
        )

        self.assertEqual(os.environ["RUNTIME_MEM_SAFETY_RATIO"], "0.08")
        self.assertEqual(os.environ["RUNTIME_MEM_NO_WARMUP_FLOOR_MB"], "3072")
        self.assertEqual(os.environ["MOE_RUNTIME_MEM_LOG"], "1")
        self.assertEqual(os.environ["MOE_RUNTIME_SLOT_LOG"], "0")
        self.assertEqual(os.environ["MOE_RUNTIME_SLOT_MIN_SLOTS"], "128")
        self.assertEqual(os.environ["MOE_SKEW_MULT"], "1.75")
        self.assertEqual(os.environ["MOE_SKEW_ADD"], "0.2")

    def test_runtime_tuning_args_reject_invalid_ranges(self):
        from rtp_llm.server.server_args.server_args import setup_args

        with self.assertRaises(SystemExit):
            setup_args(["--runtime_mem_safety_ratio", "1.0"])
        with self.assertRaises(SystemExit):
            setup_args(["--runtime_mem_no_warmup_floor_mb", "-1"])

    def test_cmd_args_set_to_py_env_configs(self):
        """Test that command line arguments are correctly set to py_env_configs."""
        sys.argv = [
            "prog",
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
            "--enable_flashinfer_trtllm_gen",
            "false",
            "--enable_flashinfer_trt_fmha_v2",
            "false",
            "--enable_paged_flashinfer_trt_fmha_v2",
            "false",
            "--disable_flashinfer_native",
            "true",
            # Note: max_seq_len is in ModelConfig, not ModelArgs
            # It will be set when ModelConfig is created from model_args
        ]

        # Import and setup args
        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        py_env_configs = rtp_llm.server.server_args.server_args.setup_args()

        # Verify model_args
        self.assertEqual(py_env_configs.model_args.model_type, "llama")
        self.assertEqual(
            py_env_configs.model_args.ckpt_path, "/path/to/llama/checkpoint"
        )
        self.assertEqual(py_env_configs.model_args.act_type, "FP16")

        # Verify parallelism_config
        self.assertEqual(py_env_configs.parallelism_config.tp_size, 8)
        self.assertEqual(py_env_configs.parallelism_config.dp_size, 4)
        self.assertEqual(py_env_configs.parallelism_config.world_size, 32)

        # Verify concurrency_config
        self.assertEqual(py_env_configs.concurrency_config.concurrency_limit, 128)

        # Verify fifo_scheduler_config
        self.assertEqual(
            py_env_configs.runtime_config.fifo_scheduler_config.max_context_batch_size,
            64,
        )

        # Verify runtime_config (warm_up is now in RuntimeConfig)
        self.assertEqual(py_env_configs.runtime_config.warm_up, False)  # bool in C++
        # Note: max_seq_len is in ModelConfig, not RuntimeConfig or EngineConfig
        # It will be set when ModelConfig is created from model_args

        # Verify cache_store_config
        self.assertEqual(py_env_configs.cache_store_config.rdma_io_thread_count, 4)
        self.assertEqual(py_env_configs.cache_store_config.rdma_worker_thread_count, 2)

        # Verify fmha_config
        self.assertFalse(py_env_configs.fmha_config.enable_flashinfer_trtllm_gen)
        self.assertFalse(py_env_configs.fmha_config.enable_flashinfer_trt_fmha_v2)
        self.assertFalse(py_env_configs.fmha_config.enable_paged_flashinfer_trt_fmha_v2)
        self.assertTrue(py_env_configs.fmha_config.disable_flashinfer_native)

    def test_cmd_args_override_env_vars(self):
        """Test that command line arguments override environment variables."""
        # Set environment variables
        os.environ["MODEL_TYPE"] = "qwen"
        os.environ["CHECKPOINT_PATH"] = "/path/to/qwen/checkpoint"
        os.environ["ACT_TYPE"] = "BF16"
        os.environ["TP_SIZE"] = "4"
        os.environ["CONCURRENCY_LIMIT"] = "32"

        # Set command line arguments (should override env vars)
        sys.argv = [
            "prog",
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

        # Import and setup args
        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        py_env_configs = rtp_llm.server.server_args.server_args.setup_args()

        # Verify that command line arguments override environment variables
        self.assertEqual(py_env_configs.model_args.model_type, "llama")  # Overridden
        self.assertEqual(
            py_env_configs.model_args.ckpt_path, "/path/to/llama/checkpoint"
        )  # Overridden
        self.assertEqual(py_env_configs.model_args.act_type, "FP16")  # Overridden
        self.assertEqual(py_env_configs.parallelism_config.tp_size, 8)  # Overridden
        self.assertEqual(
            py_env_configs.concurrency_config.concurrency_limit, 64
        )  # Overridden

    def test_mixed_env_and_cmd_args(self):
        """Test mixed environment variables and command line arguments."""
        # Set some environment variables
        os.environ["MODEL_TYPE"] = "qwen"
        os.environ["CHECKPOINT_PATH"] = "/path/to/qwen/checkpoint"
        os.environ["ACT_TYPE"] = "BF16"
        os.environ["DP_SIZE"] = "2"
        os.environ["WORLD_SIZE"] = "8"

        # Set some command line arguments
        sys.argv = [
            "prog",
            "--tp_size",
            "4",
            "--concurrency_limit",
            "64",
            "--max_context_batch_size",
            "32",
        ]

        # Import and setup args
        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        py_env_configs = rtp_llm.server.server_args.server_args.setup_args()

        # Verify values from environment variables
        self.assertEqual(py_env_configs.model_args.model_type, "qwen")
        self.assertEqual(
            py_env_configs.model_args.ckpt_path, "/path/to/qwen/checkpoint"
        )
        self.assertEqual(py_env_configs.model_args.act_type, "BF16")
        self.assertEqual(py_env_configs.parallelism_config.dp_size, 2)
        self.assertEqual(py_env_configs.parallelism_config.world_size, 8)

        # Verify values from command line arguments
        self.assertEqual(py_env_configs.parallelism_config.tp_size, 4)
        self.assertEqual(py_env_configs.concurrency_config.concurrency_limit, 64)
        self.assertEqual(
            py_env_configs.runtime_config.fifo_scheduler_config.max_context_batch_size,
            32,
        )

    def test_batch_decode_scheduler_config(self):
        """Test that batch_decode_scheduler_config is correctly set."""
        sys.argv = [
            "prog",
            "--use_batch_decode_scheduler",
            "1",
            "--batch_decode_scheduler_batch_size",
            "16",
            "--batch_decode_scheduler_warmup_type",
            "1",
        ]

        # Import and setup args
        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        py_env_configs = rtp_llm.server.server_args.server_args.setup_args()

        # Verify batch_decode_scheduler_config
        self.assertEqual(py_env_configs.runtime_config.use_batch_decode_scheduler, True)
        self.assertEqual(
            py_env_configs.runtime_config.batch_decode_scheduler_config.batch_decode_scheduler_batch_size,
            16,
        )
        self.assertEqual(
            py_env_configs.runtime_config.batch_decode_scheduler_config.batch_decode_scheduler_warmup_type,
            1,
        )

        # Verify it's also set in the C++ binding object
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
        sys.argv = ["prog"]

        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        py_env_configs = rtp_llm.server.server_args.server_args.setup_args()
        self.assertEqual(
            py_env_configs.runtime_config.fifo_scheduler_config.pdfusion_scheduler_mode,
            "",
        )

        sys.argv = [
            "prog",
            "--pdfusion_scheduler_mode",
            "ratio",
            "--decode_prefill_ratio",
            "1/3",
        ]
        importlib.reload(rtp_llm.server.server_args.server_args)
        py_env_configs = rtp_llm.server.server_args.server_args.setup_args()
        self.assertEqual(
            py_env_configs.runtime_config.fifo_scheduler_config.pdfusion_scheduler_mode,
            "ratio",
        )
        self.assertEqual(
            py_env_configs.runtime_config.fifo_scheduler_config.decode_prefill_ratio,
            "1/3",
        )

        sys.argv = [
            "prog",
            "--pdfusion_scheduler_mode",
            "ratio",
            "--decode_prefill_ratio",
            "0",
        ]
        importlib.reload(rtp_llm.server.server_args.server_args)
        py_env_configs = rtp_llm.server.server_args.server_args.setup_args()
        self.assertEqual(
            py_env_configs.runtime_config.fifo_scheduler_config.pdfusion_scheduler_mode,
            "ratio",
        )
        self.assertEqual(
            py_env_configs.runtime_config.fifo_scheduler_config.decode_prefill_ratio,
            "0",
        )

    def test_pdfusion_scheduler_mode_rejects_unknown_value(self):
        """Test that pdfusion_scheduler_mode only accepts fixed scheduler patterns."""
        sys.argv = ["prog", "--pdfusion_scheduler_mode", "ratioo"]

        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        with self.assertRaises(SystemExit):
            rtp_llm.server.server_args.server_args.setup_args()

    def test_gpu_batch_vit_args_parse(self):
        from rtp_llm.config.py_config_modules import PyEnvConfigs
        from rtp_llm.server.server_args.server_args import (
            EnvArgumentParser,
            init_all_group_args,
        )

        parser = EnvArgumentParser(description="test")
        cfg = PyEnvConfigs()
        parser.set_root_config(cfg)
        init_all_group_args(parser, cfg)
        parser.parse_args(["--gpu_batch_wait_ms", "500", "--gpu_max_batch_size", "8"])
        self.assertEqual(cfg.vit_config.gpu_max_batch_size, 8)
        self.assertEqual(cfg.vit_config.gpu_batch_wait_ms, 500)


if __name__ == "__main__":
    main()
