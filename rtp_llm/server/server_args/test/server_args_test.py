import importlib
import json
import os
import sys
from unittest import TestCase, main

from rtp_llm.config.test.kv_cache_event_test_values import (
    KV_CACHE_EVENT_ENV_CASES,
)


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
        os.environ["MAX_CONTEXT_BATCH_SIZE"] = "32"
        os.environ["WARM_UP"] = "1"
        os.environ["MAX_SEQ_LEN"] = "4096"
        os.environ["FRONTEND_PRE_STOP_DRAIN_SECONDS"] = "2.5"
        os.environ["DASH_SC_GRPC_PRE_STOP_DRAIN_SECONDS"] = "9"
        os.environ["LOADER_RECYCLE_HANDLES"] = "false"
        os.environ["MOE_PURE_TP_PRESHARD"] = "false"

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

        # Verify frontend and DashSc pre-stop windows are configured independently.
        self.assertEqual(
            py_env_configs.server_config.frontend_pre_stop_drain_seconds, 2.5
        )
        self.assertEqual(
            py_env_configs.server_config.dash_sc_grpc_pre_stop_drain_seconds, 9.0
        )

        # Verify runtime_config (warm_up is now in RuntimeConfig)
        self.assertEqual(py_env_configs.runtime_config.warm_up, True)  # bool in C++

        # Verify load_config: the flag came from LOADER_RECYCLE_HANDLES=false.
        self.assertFalse(py_env_configs.load_config.loader_recycle_handles)
        # MOE_PURE_TP_PRESHARD=false must override the True default.
        self.assertFalse(py_env_configs.load_config.moe_pure_tp_preshard)
        # Note: max_seq_len is in ModelConfig, not RuntimeConfig or EngineConfig
        # It will be set when ModelConfig is created from model_args

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

        # Pins the shipped defaults: neither env nor argv sets the flags here.
        self.assertTrue(py_env_configs.load_config.loader_recycle_handles)
        self.assertTrue(py_env_configs.load_config.moe_pure_tp_preshard)
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

    def test_kv_cache_event_env_vars_bind_to_config(self):
        for env_name, _, raw_value, _ in KV_CACHE_EVENT_ENV_CASES:
            os.environ[env_name] = raw_value
        # Exercise the mixed CLI + environment path rather than argparse's
        # environment-to-argv fallback.
        sys.argv = ["prog", "--model_type", "qwen"]

        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        py_env_configs = rtp_llm.server.server_args.server_args.setup_args()

        for _, field_name, _, expected_value in KV_CACHE_EVENT_ENV_CASES:
            with self.subTest(field_name=field_name):
                self.assertEqual(
                    expected_value,
                    getattr(py_env_configs.kv_cache_config, field_name),
                )

    def test_kv_cache_event_env_rejects_unknown_publisher_type(self):
        os.environ["KV_CACHE_EVENT_PUBLISHER_TYPE"] = "KVCM"
        sys.argv = ["prog", "--model_type", "qwen"]

        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        with self.assertRaises(SystemExit):
            rtp_llm.server.server_args.server_args.setup_args()

    def test_existing_env_choice_tolerates_unknown_value_in_mixed_mode(self):
        # Pre-existing arguments keep the legacy tolerance for stale invalid
        # env values (only the new KV cache event argument is strict), so a
        # deployment upgrade cannot be broken by an old typo; the problem is
        # surfaced via an ERROR log instead.
        os.environ["PDFUSION_SCHEDULER_MODE"] = "unknown"
        sys.argv = ["prog", "--model_type", "qwen"]

        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        with self.assertLogs(level="ERROR") as logs:
            py_env_configs = rtp_llm.server.server_args.server_args.setup_args()

        self.assertEqual(
            "unknown",
            py_env_configs.runtime_config.fifo_scheduler_config.pdfusion_scheduler_mode,
        )
        self.assertTrue(
            any("PDFUSION_SCHEDULER_MODE" in message for message in logs.output)
        )

    def test_empty_env_value_is_treated_as_unset_in_mixed_mode(self):
        os.environ["KV_CACHE_EVENT_PUBLISHER_TYPE"] = ""
        sys.argv = ["prog", "--model_type", "qwen"]

        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        py_env_configs = rtp_llm.server.server_args.server_args.setup_args()

        self.assertEqual(
            "none",
            py_env_configs.kv_cache_config.kv_cache_event_publisher_type,
        )

    def test_empty_env_value_is_treated_as_unset_in_pure_env_mode(self):
        os.environ["MODEL_TYPE"] = "qwen"
        os.environ["KV_CACHE_EVENT_PUBLISHER_TYPE"] = ""
        sys.argv = ["prog"]

        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        py_env_configs = rtp_llm.server.server_args.server_args.setup_args()

        self.assertEqual(
            "none",
            py_env_configs.kv_cache_config.kv_cache_event_publisher_type,
        )

    def test_empty_env_value_still_binds_for_legacy_args_in_mixed_mode(self):
        # Empty-value-as-unset is limited to the kv_cache_event_* whitelist.
        # Pre-existing arguments keep the legacy semantics where "" is bound
        # as-is: some deployments set an env variable to an empty string as an
        # explicit "disable" switch (e.g. THINK_START_TAG="").
        os.environ["THINK_START_TAG"] = ""
        sys.argv = ["prog", "--model_type", "qwen"]

        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        py_env_configs = rtp_llm.server.server_args.server_args.setup_args()

        self.assertEqual(
            "",
            py_env_configs.generate_env_config.think_start_tag,
        )

    def test_empty_env_value_still_binds_for_legacy_args_in_pure_env_mode(self):
        os.environ["MODEL_TYPE"] = "qwen"
        os.environ["THINK_START_TAG"] = ""
        sys.argv = ["prog"]

        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        py_env_configs = rtp_llm.server.server_args.server_args.setup_args()

        self.assertEqual(
            "",
            py_env_configs.generate_env_config.think_start_tag,
        )

    def test_invalid_typed_env_value_warns_and_uses_default_in_mixed_mode(self):
        os.environ["KV_CACHE_EVENT_QUEUE_CAPACITY"] = "not-an-integer"
        sys.argv = ["prog", "--model_type", "qwen"]

        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        with self.assertLogs(level="WARNING") as logs:
            py_env_configs = rtp_llm.server.server_args.server_args.setup_args()

        self.assertEqual(
            100000,
            py_env_configs.kv_cache_config.kv_cache_event_queue_capacity,
        )
        self.assertTrue(
            any("KV_CACHE_EVENT_QUEUE_CAPACITY" in message for message in logs.output)
        )

    def test_invalid_boolean_env_value_fails_fast_in_mixed_mode(self):
        # str2bool raises ArgumentTypeError, which must fail fast so the mixed
        # CLI+env path matches the pure-env path instead of silently falling
        # back to the default value.
        os.environ["ENABLE_REMOTE_CACHE"] = "ture"
        sys.argv = ["prog", "--model_type", "qwen"]

        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        with self.assertLogs(level="ERROR") as logs:
            with self.assertRaises(SystemExit):
                rtp_llm.server.server_args.server_args.setup_args()

        self.assertTrue(
            any("ENABLE_REMOTE_CACHE" in message for message in logs.output)
        )

    def test_invalid_boolean_env_value_fails_fast_in_pure_env_mode(self):
        # The pure environment-variable path converts env values through
        # argparse itself; both paths must reject the same invalid input.
        os.environ["MODEL_TYPE"] = "qwen"
        os.environ["ENABLE_REMOTE_CACHE"] = "ture"
        sys.argv = ["prog"]

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

    def test_dash_sc_default_allows_large_requests_on_both_ends(self):
        from rtp_llm.server.server_args.grpc_group_args import (
            default_dash_sc_grpc_config_json,
        )

        config = json.loads(default_dash_sc_grpc_config_json())
        expected = 1024 * 1024 * 1024
        self.assertEqual(
            config["client_config"]["grpc.max_receive_message_length"], expected
        )
        self.assertEqual(
            config["server_config"]["grpc.max_receive_message_length"],
            64 * 1024 * 1024,
        )


class ServerArgsGrammarConfigTest(TestCase):
    """Cover every CLI-wired field on GrammarConfig (--grammar_* /
    --constrained_json_*): default values and CLI binding."""

    def setUp(self):
        environ_backup = os.environ.copy()
        argv_backup = sys.argv.copy()

        # Register restoration BEFORE mutating global state so it runs even if
        # setUp itself (or _setup) raises — a bare tearDown would be skipped on a
        # setUp failure and leave os.environ cleared for the rest of the suite.
        def _restore():
            os.environ.clear()
            os.environ.update(environ_backup)
            sys.argv = argv_backup

        self.addCleanup(_restore)

        os.environ.clear()
        sys.argv = ["prog"]

    def _setup(self):
        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        return rtp_llm.server.server_args.server_args.setup_args()

    def test_grammar_defaults(self):
        """All fields match defaults when no input is given.
        Regression guard for the wiring in init_grammar_group_args."""
        py_env_configs = self._setup()
        g = py_env_configs.grammar_config

        self.assertEqual(g.constrained_json_disable_any_whitespace, False)
        self.assertEqual(g.terminate_without_stop_token, False)
        self.assertEqual(g.num_workers, 8)
        self.assertEqual(g.compiler_cache_bytes, 512 * 1024 * 1024)

    def test_grammar_parser_defaults_override_config_initial_values(self):
        """The CLI declaration is the source of truth for grammar defaults."""
        from rtp_llm.config.py_config_modules import PyEnvConfigs
        from rtp_llm.server.server_args.grammar_group_args import (
            init_grammar_group_args,
        )
        from rtp_llm.server.server_args.server_args import EnvArgumentParser

        cfgs = PyEnvConfigs()
        g = cfgs.grammar_config
        g.constrained_json_disable_any_whitespace = True
        g.terminate_without_stop_token = True
        g.num_workers = 17
        g.compiler_cache_bytes = 1

        parser = EnvArgumentParser()
        parser.set_root_config(cfgs)
        init_grammar_group_args(parser, g)
        parser.parse_args([])

        self.assertEqual(g.constrained_json_disable_any_whitespace, False)
        self.assertEqual(g.terminate_without_stop_token, False)
        self.assertEqual(g.num_workers, 8)
        self.assertEqual(g.compiler_cache_bytes, 512 * 1024 * 1024)

    def test_grammar_cmd_args(self):
        """Every CLI flag binds to the right config field, with correct types."""
        sys.argv = [
            "prog",
            "--constrained_json_disable_any_whitespace",
            "1",
            "--grammar_terminate_without_stop_token",
            "1",
            "--grammar_num_workers",
            "7",
            "--grammar_compiler_cache_bytes",
            "67108864",
        ]

        cfgs = self._setup()
        g = cfgs.grammar_config
        self.assertEqual(g.constrained_json_disable_any_whitespace, True)
        self.assertEqual(g.terminate_without_stop_token, True)
        self.assertEqual(g.num_workers, 7)
        self.assertEqual(g.compiler_cache_bytes, 67108864)


if __name__ == "__main__":
    main()
