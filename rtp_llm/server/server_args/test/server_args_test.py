import importlib
import os
import pickle
import sys
from unittest import TestCase, main
from unittest.mock import patch


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

    @staticmethod
    def _setup_args(args=None):
        from rtp_llm.server.server_args.server_args import setup_args

        # This branch's public entry reads sys.argv, not an explicit arg list.
        if args is None:
            return setup_args()
        with patch.object(sys, "argv", ["rtp-llm", *args]):
            return setup_args()

    def test_fastsafetensors_reserve_defaults_and_cli_precedence(self):
        setup_args = self._setup_args

        self.assertEqual(setup_args([]).load_config.fastsafetensors_reserve_mb, 2048)
        os.environ["RTP_FASTSAFETENSORS_RESERVE_MB"] = "512"
        self.assertEqual(setup_args([]).load_config.fastsafetensors_reserve_mb, 512)
        configs = setup_args(["--fastsafetensors_reserve_mb", "0"])
        self.assertEqual(configs.load_config.fastsafetensors_reserve_mb, 0)
        self.assertIn("fastsafetensors_reserve_mb: 0", configs.load_config.to_string())

    def test_fastsafetensors_reserve_equals_cli_overrides_environment(self):
        setup_args = self._setup_args

        os.environ["RTP_FASTSAFETENSORS_RESERVE_MB"] = "512"
        for value in (0, 128):
            for use_sys_argv in (False, True):
                with self.subTest(value=value, use_sys_argv=use_sys_argv):
                    args = [f"--fastsafetensors_reserve_mb={value}"]
                    if use_sys_argv:
                        sys.argv = ["rtp-llm", *args]
                        configs = setup_args()
                    else:
                        configs = setup_args(args)
                    self.assertEqual(
                        configs.load_config.fastsafetensors_reserve_mb, value
                    )

    def test_fastsafetensors_reserve_abbreviation_overrides_environment(self):
        setup_args = self._setup_args

        os.environ["RTP_FASTSAFETENSORS_RESERVE_MB"] = "512"
        for value in (0, 128):
            for inline in (False, True):
                with self.subTest(value=value, inline=inline):
                    args = (
                        [f"--fastsafetensors_reserve={value}"]
                        if inline
                        else ["--fastsafetensors_reserve", str(value)]
                    )
                    self.assertEqual(
                        setup_args(args).load_config.fastsafetensors_reserve_mb,
                        value,
                    )

    def test_fastsafetensors_reserve_rejects_invalid_values(self):
        setup_args = self._setup_args

        for value in ("-1", "1.5", "", "invalid"):
            with self.subTest(value=value):
                os.environ.pop("RTP_FASTSAFETENSORS_RESERVE_MB", None)
                with self.assertRaises(SystemExit):
                    setup_args(["--fastsafetensors_reserve_mb", value])
                os.environ["RTP_FASTSAFETENSORS_RESERVE_MB"] = value
                with self.assertRaises(SystemExit):
                    setup_args([])

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

    def test_warm_up_remains_enabled_by_default_with_sleep_on_or_off(self):
        from rtp_llm.model_loader import weight_memory_saver as wms
        from rtp_llm.server.server_args.server_args import setup_args

        try:
            for sleep_enabled in ("0", "1"):
                with self.subTest(sleep_enabled=sleep_enabled):
                    os.environ.pop("WARM_UP", None)
                    sys.argv = ["prog", "--enable-sleep-mode", sleep_enabled]
                    config = setup_args()
                    self.assertTrue(config.runtime_config.warm_up)
        finally:
            wms._reset_for_testing()

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

        self.assertTrue(py_env_configs.runtime_config.enable_sleep_mode)
        self.assertEqual(py_env_configs.runtime_config.sleep_mode_level, 1)
        self.assertTrue(wms.is_enabled())

    def test_sleep_level_env_validation_with_and_without_cli(self):
        for value in ("0", "3", "invalid", "1.5"):
            for args in ([], ["--enable-sleep-mode", "1"]):
                with self.subTest(value=value, args=args):
                    os.environ["SLEEP_MODE_LEVEL"] = value
                    with self.assertRaises(SystemExit):
                        self._setup_args(args)

    def test_sleep_level_binding_and_pickle_roundtrip(self):
        from rtp_llm.model_loader import weight_memory_saver as wms

        self.addCleanup(wms._reset_for_testing)
        for level in (1, 2):
            for source in ("cli", "env", "mixed", "cli_override"):
                with self.subTest(level=level, source=source):
                    wms._reset_for_testing()
                    os.environ.pop("SLEEP_MODE_LEVEL", None)
                    os.environ["ENABLE_SLEEP_MODE"] = "1"
                    args = []
                    if source in ("cli", "cli_override"):
                        args = [
                            "--enable-sleep-mode",
                            "1",
                            "--sleep-mode-level",
                            str(level),
                        ]
                        if source == "cli_override":
                            os.environ["SLEEP_MODE_LEVEL"] = "invalid"
                    else:
                        os.environ["SLEEP_MODE_LEVEL"] = str(level)
                        if source == "mixed":
                            args = ["--enable-sleep-mode", "1"]
                    config = self._setup_args(args).runtime_config
                    self.assertTrue(config.enable_sleep_mode)
                    self.assertEqual(config.sleep_mode_level, level)
                    self.assertTrue(wms.is_enabled())
                    restored = pickle.loads(pickle.dumps(config))
                    self.assertTrue(restored.enable_sleep_mode)
                    self.assertEqual(restored.sleep_mode_level, level)

    def test_runtime_config_legacy_pickle_sleep_defaults_and_field_alignment(self):
        from rtp_llm.ops import RuntimeConfig

        config = RuntimeConfig()
        config.enable_sleep_mode = True
        config.sleep_mode_level = 2
        config.model_warm_up = False
        config.use_batch_decode_scheduler = True
        config.use_gather_batch_scheduler = False
        config.model_name = "pickle-model"
        config.worker_grpc_addrs = ["127.0.0.1:18001"]
        config.worker_addrs = ["127.0.0.1:18002"]
        config.specify_gpu_arch = "sm_100"
        state = config.__getstate__()
        self.assertEqual(len(state), 16)
        # Legacy tuples lack sleep fields; the intermediate 15-field tuple
        # has sleep fields but lacks model_warm_up.
        legacy14 = state[:6] + state[7:15]
        for size, saved in (
            (13, legacy14[:13]),
            (14, legacy14),
            (15, state[:5] + state[6:]),
            (16, state),
        ):
            with self.subTest(size=size):
                restored = RuntimeConfig.__new__(RuntimeConfig)
                restored.__setstate__(saved)
                self.assertEqual(restored.enable_sleep_mode, size >= 15)
                self.assertEqual(restored.sleep_mode_level, 2 if size >= 15 else 1)
                self.assertEqual(
                    restored.model_warm_up,
                    RuntimeConfig().model_warm_up if size == 15 else False,
                )
                self.assertTrue(restored.use_batch_decode_scheduler)
                self.assertFalse(restored.use_gather_batch_scheduler)
                self.assertEqual(restored.model_name, config.model_name)
                self.assertEqual(restored.worker_grpc_addrs, config.worker_grpc_addrs)
                self.assertEqual(restored.worker_addrs, config.worker_addrs)
                self.assertEqual(
                    restored.specify_gpu_arch,
                    (
                        config.specify_gpu_arch
                        if size >= 14
                        else RuntimeConfig().specify_gpu_arch
                    ),
                )

    def _setup_args_and_reload(self):
        """Reload + setup_args with a clean weight_memory_saver, as the sleep tests do.

        Returns the weight_memory_saver module so the caller can read the switches
        the way the sleep hook path does.
        """
        import rtp_llm.server.server_args.server_args
        from rtp_llm.model_loader import weight_memory_saver as wms

        importlib.reload(rtp_llm.server.server_args.server_args)
        wms._reset_for_testing()
        rtp_llm.server.server_args.server_args.setup_args()
        return wms

    # NOTE for the four tests below: unlike --enable-sleep-mode / --sleep-mode-level,
    # --sleep_release_collective_memory has NO C++ RuntimeConfig field, so its
    # `bind_to` resolves to None and the os.environ mirror written by setup_args() is
    # the ONLY transport to the (leaf, config-less) sleep hook module. There is
    # therefore nothing to assert on py_env_configs.runtime_config for this arg.

    def test_sleep_release_collective_memory_defaults_off(self):
        """Flag absent: the collective-release switch stays off."""
        os.environ.pop("SLEEP_RELEASE_COLLECTIVE_MEMORY", None)
        sys.argv = ["prog"]

        wms = self._setup_args_and_reload()

        self.assertFalse(wms.release_collective_memory())
        self.assertEqual(os.environ["SLEEP_RELEASE_COLLECTIVE_MEMORY"], "0")

    def test_sleep_release_collective_memory_arg_enables_switch(self):
        """Both the underscored flag and its dashed alias must reach the env mirror."""
        for flag in (
            "--sleep_release_collective_memory",
            "--sleep-release-collective-memory",
        ):
            with self.subTest(flag=flag):
                os.environ.pop("SLEEP_RELEASE_COLLECTIVE_MEMORY", None)
                sys.argv = ["prog", flag, "1"]

                wms = self._setup_args_and_reload()

                self.assertTrue(wms.release_collective_memory())
                self.assertEqual(os.environ["SLEEP_RELEASE_COLLECTIVE_MEMORY"], "1")

    def test_sleep_release_collective_memory_env_only_is_honoured(self):
        """env fallback: the env var alone turns the switch on (no CLI flag).

        setup_args() unconditionally rewrites the mirror from the parsed value, so
        the env surviving as "1" also proves the fallback was actually parsed: a
        broken fallback would parse the default and stamp "0" over it.

        Note this pins the effective env name, not the ``env_name=`` keyword --
        EnvArgumentParser derives the same name from the ``--flag`` when env_name is
        omitted, so dropping the keyword here would be invisible to any test.
        """
        os.environ["SLEEP_RELEASE_COLLECTIVE_MEMORY"] = "1"
        sys.argv = ["prog"]

        wms = self._setup_args_and_reload()

        self.assertTrue(wms.release_collective_memory())
        self.assertEqual(os.environ["SLEEP_RELEASE_COLLECTIVE_MEMORY"], "1")

    def test_sleep_release_collective_memory_explicit_zero_disables(self):
        """Explicit 0 keeps the switch off, and overrides an env var asking for on."""
        sys.argv = ["prog", "--sleep_release_collective_memory", "0"]

        wms = self._setup_args_and_reload()

        self.assertFalse(wms.release_collective_memory())
        self.assertEqual(os.environ["SLEEP_RELEASE_COLLECTIVE_MEMORY"], "0")

        # Command line wins over the environment (same precedence as every other arg).
        os.environ["SLEEP_RELEASE_COLLECTIVE_MEMORY"] = "1"
        sys.argv = ["prog", "--sleep_release_collective_memory", "0"]

        wms = self._setup_args_and_reload()

        self.assertFalse(wms.release_collective_memory())
        self.assertEqual(os.environ["SLEEP_RELEASE_COLLECTIVE_MEMORY"], "0")

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
