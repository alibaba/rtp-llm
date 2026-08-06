import importlib
import os
import pickle
import sys
from unittest import TestCase, main


class ServerArgsPyEnvConfigsTest(TestCase):
    """Test that environment variables and command line arguments are correctly set to py_env_configs structure."""


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
        os.environ["PREFILL_PREPARE_RESOURCE_POOL_SIZE"] = "256"
        os.environ["MAX_CONTEXT_BATCH_SIZE"] = "32"
        os.environ["MAX_BATCH_TOKENS_WITHOUT_CACHE"] = "2048"
        os.environ["WARM_UP"] = "1"
        os.environ["MAX_SEQ_LEN"] = "4096"
        os.environ["REMOTE_JIT_DIR"] = "dfs://bucket/jit/cache"
        os.environ["GRAMMAR_COMPILE_TIMEOUT_MS"] = "1500"
        os.environ["GRAMMAR_COMPILE_CONCURRENCY"] = "3"
        os.environ["GRAMMAR_COMPILE_QUEUE_SIZE"] = "16"
        os.environ["GRAMMAR_COMPILER_CACHE_BYTES"] = "4294967296"

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

        # Verify prefill thread-pool configuration
        self.assertEqual(
            py_env_configs.pd_separation_config.prefill_prepare_resource_pool_size,
            256,
        )
        restored_pd_config = pickle.loads(
            pickle.dumps(py_env_configs.pd_separation_config)
        )
        self.assertEqual(restored_pd_config.prefill_prepare_resource_pool_size, 256)

        # Verify fifo_scheduler_config
        self.assertEqual(
            py_env_configs.runtime_config.fifo_scheduler_config.max_context_batch_size,
            32,
        )
        self.assertEqual(
            py_env_configs.runtime_config.fifo_scheduler_config.max_batch_tokens_without_cache,
            2048,
        )
        restored_fifo_config = pickle.loads(
            pickle.dumps(py_env_configs.runtime_config.fifo_scheduler_config)
        )
        self.assertEqual(restored_fifo_config.max_batch_tokens_without_cache, 2048)
        fifo_config_type = type(py_env_configs.runtime_config.fifo_scheduler_config)
        legacy_fifo_config = fifo_config_type.__new__(fifo_config_type)
        legacy_fifo_config.__setstate__((32, 8192, False, 16))
        self.assertEqual(legacy_fifo_config.max_context_batch_size, 32)
        self.assertEqual(legacy_fifo_config.max_batch_tokens_size, 8192)
        self.assertEqual(legacy_fifo_config.max_inited_kv_cache_streams, 16)
        self.assertEqual(legacy_fifo_config.max_batch_tokens_without_cache, 0)

        # Verify grammar_config compile guards
        self.assertEqual(py_env_configs.grammar_config.compile_timeout_ms, 1500)
        self.assertEqual(py_env_configs.grammar_config.compile_concurrency, 3)
        self.assertEqual(py_env_configs.grammar_config.compile_queue_size, 16)
        self.assertEqual(py_env_configs.grammar_config.compiler_cache_bytes, 4294967296)
        # GrammarConfig crosses a process boundary, and these fields widened its pickle tuple.
        restored_grammar_config = pickle.loads(
            pickle.dumps(py_env_configs.grammar_config)
        )
        self.assertEqual(restored_grammar_config.compile_timeout_ms, 1500)
        self.assertEqual(restored_grammar_config.compile_concurrency, 3)
        self.assertEqual(restored_grammar_config.compile_queue_size, 16)
        self.assertEqual(restored_grammar_config.compiler_cache_bytes, 4294967296)

        # Verify runtime_config (warm_up is now in RuntimeConfig)
        self.assertEqual(py_env_configs.runtime_config.warm_up, True)  # bool in C++
        self.assertEqual(py_env_configs.runtime_config.warm_up_with_loss, False)
        self.assertEqual(py_env_configs.runtime_config.model_warm_up, True)
        # Note: max_seq_len is in ModelConfig, not RuntimeConfig or EngineConfig
        # It will be set when ModelConfig is created from model_args
        self.assertEqual(
            py_env_configs.jit_config.remote_jit_dir,
            "dfs://bucket/jit/cache",
        )

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
            "--prefill_prepare_resource_pool_size",
            "384",
            "--max_context_batch_size",
            "64",
            "--max_batch_tokens_without_cache",
            "4096",
            "--max_inited_kv_cache_streams",
            "16",
            "--warm_up",
            "0",
            "--cache_store_rdma_io_thread_count",
            "4",
            "--cache_store_rdma_worker_thread_count",
            "2",
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

        # Verify prefill thread-pool configuration
        self.assertEqual(
            py_env_configs.pd_separation_config.prefill_prepare_resource_pool_size,
            384,
        )
        # Verify fifo_scheduler_config
        self.assertEqual(
            py_env_configs.runtime_config.fifo_scheduler_config.max_context_batch_size,
            64,
        )
        self.assertEqual(
            py_env_configs.runtime_config.fifo_scheduler_config.max_batch_tokens_without_cache,
            4096,
        )
        self.assertEqual(
            py_env_configs.runtime_config.fifo_scheduler_config.max_inited_kv_cache_streams,
            16,
        )

        # Verify runtime_config (warm_up is now in RuntimeConfig)
        self.assertEqual(py_env_configs.runtime_config.warm_up, False)  # bool in C++
        self.assertEqual(py_env_configs.runtime_config.warm_up_with_loss, False)
        self.assertEqual(py_env_configs.runtime_config.model_warm_up, True)
        # Note: max_seq_len is in ModelConfig, not RuntimeConfig or EngineConfig
        # It will be set when ModelConfig is created from model_args

        # Verify cache_store_config
        self.assertEqual(py_env_configs.cache_store_config.rdma_io_thread_count, 4)
        self.assertEqual(py_env_configs.cache_store_config.rdma_worker_thread_count, 2)

    def test_model_warm_up_env_and_global_master(self):
        os.environ["WARM_UP"] = "0"
        os.environ["WARM_UP_WITH_LOSS"] = "1"
        os.environ["MODEL_WARM_UP"] = "1"
        sys.argv = ["prog"]

        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        py_env_configs = rtp_llm.server.server_args.server_args.setup_args()

        self.assertFalse(py_env_configs.runtime_config.warm_up)
        self.assertTrue(py_env_configs.runtime_config.warm_up_with_loss)
        self.assertTrue(py_env_configs.runtime_config.model_warm_up)
        self.assertEqual(os.environ["WARM_UP"], "0")
        self.assertEqual(os.environ["WARM_UP_WITH_LOSS"], "1")
        self.assertEqual(os.environ["MODEL_WARM_UP"], "1")

    def test_warm_up_with_loss_is_independent_of_model_warm_up(self):
        os.environ["WARM_UP"] = "1"
        os.environ["WARM_UP_WITH_LOSS"] = "1"
        os.environ["MODEL_WARM_UP"] = "0"
        sys.argv = ["prog"]

        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        py_env_configs = rtp_llm.server.server_args.server_args.setup_args()

        self.assertTrue(py_env_configs.runtime_config.warm_up)
        self.assertTrue(py_env_configs.runtime_config.warm_up_with_loss)
        self.assertFalse(py_env_configs.runtime_config.model_warm_up)

        restored_runtime_config = pickle.loads(
            pickle.dumps(py_env_configs.runtime_config)
        )
        self.assertTrue(restored_runtime_config.warm_up)
        self.assertTrue(restored_runtime_config.warm_up_with_loss)
        self.assertFalse(restored_runtime_config.model_warm_up)

    def test_enable_sleep_mode_arg_configures_runtime_and_weight_saver(self):
        """Sleep mode CLI flag should enable both C++ runtime config and Python weight tagging."""
        sys.argv = [
            "prog",
            "--enable-sleep-mode",
            "1",
        ]

        import rtp_llm.server.server_args.server_args
        from rtp_llm.model_loader import weight_memory_saver as wms

        importlib.reload(rtp_llm.server.server_args.server_args)
        wms._reset_for_testing()
        py_env_configs = rtp_llm.server.server_args.server_args.setup_args()

        if hasattr(py_env_configs.runtime_config, "enable_sleep_mode"):
            self.assertEqual(py_env_configs.runtime_config.enable_sleep_mode, True)
        self.assertTrue(wms.is_enabled())

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

    def test_repetition_detection_config(self):
        """Test that repetition detection args bind to PyEnvConfigs."""
        sys.argv = [
            "prog",
            "--tool_call_loop_threshold",
            "7",
            "--tool_call_loop_begin_marker",
            "<tool_call>",
            "--tool_call_loop_end_marker",
            "</tool_call>",
        ]

        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        py_env_configs = rtp_llm.server.server_args.server_args.setup_args()

        cfg = py_env_configs.repetition_detection_config
        self.assertEqual(cfg.tool_call_loop_threshold, 7)
        self.assertEqual(cfg.tool_call_loop_begin_marker, "<tool_call>")
        self.assertEqual(cfg.tool_call_loop_end_marker, "</tool_call>")


if __name__ == "__main__":
    main()
