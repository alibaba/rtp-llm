import argparse
import contextlib
import importlib
import io
import json
import os
import sys
from unittest import TestCase, main

from rtp_llm.config.test.kv_cache_event_test_values import (
    EVENT_STATE_OFFSET,
    KV_CACHE_EVENT_DEFAULTS,
    KV_CACHE_EVENT_ENV_CASES,
    KV_CACHE_EVENT_FIELDS,
    KV_CACHE_EVENT_VALIDATION_CASES,
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
        os.environ["CP_FORCE_SINGLE_PREFILL"] = "0"
        os.environ["WARM_UP"] = "1"
        os.environ["MAX_SEQ_LEN"] = "4096"
        os.environ["FRONTEND_PRE_STOP_DRAIN_SECONDS"] = "2.5"
        os.environ["DASH_SC_GRPC_PRE_STOP_DRAIN_SECONDS"] = "9"
        os.environ["LOADER_RECYCLE_HANDLES"] = "false"
        os.environ["MOE_PURE_TP_PRESHARD"] = "true"

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
        self.assertEqual(
            py_env_configs.runtime_config.fifo_scheduler_config.cp_force_single_prefill,
            False,
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
        # MOE_PURE_TP_PRESHARD=true explicitly enables the opt-in path.
        self.assertTrue(py_env_configs.load_config.moe_pure_tp_preshard)
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
            "--cp_force_single_prefill",
            "false",
            "--max_inited_kv_cache_streams",
            "16",
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
        self.assertEqual(
            py_env_configs.runtime_config.fifo_scheduler_config.cp_force_single_prefill,
            False,
        )
        self.assertEqual(
            py_env_configs.runtime_config.fifo_scheduler_config.max_inited_kv_cache_streams,
            16,
        )

        # Verify runtime_config (warm_up is now in RuntimeConfig)
        self.assertEqual(py_env_configs.runtime_config.warm_up, False)  # bool in C++

        # Pins the shipped defaults: neither env nor argv sets the flags here.
        self.assertTrue(py_env_configs.load_config.loader_recycle_handles)
        self.assertFalse(py_env_configs.load_config.moe_pure_tp_preshard)
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

    def test_equals_style_cmd_arg_overrides_env_var(self):
        os.environ["KV_CACHE_EVENT_QUEUE_CAPACITY"] = "111"
        sys.argv = [
            "prog",
            "--model_type=qwen",
            "--kv_cache_event_queue_capacity=222",
        ]

        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        py_env_configs = rtp_llm.server.server_args.server_args.setup_args()

        self.assertEqual(
            222,
            py_env_configs.kv_cache_config.kv_cache_event_queue_capacity,
        )

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
        for case in KV_CACHE_EVENT_ENV_CASES:
            os.environ[case.env_name] = case.raw_value
        # Exercise the mixed CLI + environment path rather than argparse's
        # environment-to-argv fallback.
        sys.argv = ["prog", "--model_type", "qwen"]

        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        py_env_configs = rtp_llm.server.server_args.server_args.setup_args()

        state = py_env_configs.kv_cache_config.__getstate__()
        for index, case in enumerate(KV_CACHE_EVENT_ENV_CASES):
            with self.subTest(field_name=case.field_name):
                self.assertEqual(
                    case.expected_value,
                    getattr(py_env_configs.kv_cache_config, case.field_name),
                )
                self.assertEqual(case.expected_value, state[EVENT_STATE_OFFSET + index])
                self.assertNotIn(
                    case.field_name, py_env_configs.kv_cache_config.__dict__
                )

    def test_kv_cache_event_env_rejects_unknown_publisher_type(self):
        os.environ["KV_CACHE_EVENT_PUBLISHER_TYPE"] = "KVCM"
        sys.argv = ["prog", "--model_type", "qwen"]

        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            rtp_llm.server.server_args.server_args.setup_args()
        self.assertIn("KV_CACHE_EVENT_PUBLISHER_TYPE", stderr.getvalue())

    def test_kv_cache_event_pure_env_rejects_unknown_publisher_type(self):
        os.environ["MODEL_TYPE"] = "qwen"
        os.environ["KV_CACHE_EVENT_PUBLISHER_TYPE"] = "KVCM"
        sys.argv = ["prog"]

        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            rtp_llm.server.server_args.server_args.setup_args()
        self.assertIn("KV_CACHE_EVENT_PUBLISHER_TYPE", stderr.getvalue())

    def test_kv_cache_event_env_accepts_supported_publisher_types(self):
        for publisher_type in ("log", "kvcm"):
            with self.subTest(publisher_type=publisher_type):
                os.environ.clear()
                os.environ["KV_CACHE_EVENT_PUBLISHER_TYPE"] = publisher_type
                if publisher_type == "kvcm":
                    os.environ["KV_CACHE_EVENT_MANAGER_ENDPOINT"] = (
                        "http://kvcm-meta:56020"
                    )
                    os.environ["KV_CACHE_EVENT_INSTANCE_ID"] = "test-instance"
                    os.environ["KV_CACHE_EVENT_HOST_IP_PORT"] = "127.0.0.1:18000"
                sys.argv = ["prog", "--model_type", "qwen"]

                import rtp_llm.server.server_args.server_args

                importlib.reload(rtp_llm.server.server_args.server_args)
                configs = rtp_llm.server.server_args.server_args.setup_args()
                self.assertEqual(
                    publisher_type,
                    configs.kv_cache_config.kv_cache_event_publisher_type,
                )

    def test_kvcm_requires_identity_and_endpoint(self):
        os.environ["KV_CACHE_EVENT_PUBLISHER_TYPE"] = "kvcm"
        sys.argv = ["prog", "--model_type", "qwen"]

        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            rtp_llm.server.server_args.server_args.setup_args()
        error = stderr.getvalue()
        self.assertIn("KV_CACHE_EVENT_MANAGER_ENDPOINT", error)
        self.assertIn("KV_CACHE_EVENT_INSTANCE_ID", error)
        self.assertIn("KV_CACHE_EVENT_HOST_IP_PORT", error)

    def test_kvcm_requires_a_resolved_instance_group(self):
        os.environ.update(
            {
                "KV_CACHE_EVENT_PUBLISHER_TYPE": "kvcm",
                "KV_CACHE_EVENT_MANAGER_ENDPOINT": "http://kvcm-meta:56020",
                "KV_CACHE_EVENT_INSTANCE_ID": "test-instance",
                "KV_CACHE_EVENT_HOST_IP_PORT": "127.0.0.1:18000",
                "RECO_INSTANCE_GROUP": "",
            }
        )
        sys.argv = ["prog", "--model_type", "qwen"]

        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            rtp_llm.server.server_args.server_args.setup_args()
        self.assertIn("RECO_INSTANCE_GROUP", stderr.getvalue())

    def test_kvcm_placeholder_group_fallback_is_visible(self):
        os.environ.update(
            {
                "KV_CACHE_EVENT_PUBLISHER_TYPE": "kvcm",
                "KV_CACHE_EVENT_MANAGER_ENDPOINT": "http://kvcm-meta:56020",
                "KV_CACHE_EVENT_INSTANCE_ID": "test-instance",
                "KV_CACHE_EVENT_HOST_IP_PORT": "127.0.0.1:18000",
            }
        )
        sys.argv = ["prog", "--model_type", "qwen"]

        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        with self.assertLogs(level="WARNING") as logs:
            configs = rtp_llm.server.server_args.server_args.setup_args()

        self.assertEqual("default", configs.kv_cache_config.reco_instance_group)
        self.assertTrue(
            any(
                "placeholder reco instance group 'default'" in line
                for line in logs.output
            )
        )

    def test_kvcm_rejects_malformed_endpoint_and_host_before_engine_start(self):
        cases = (
            ("KV_CACHE_EVENT_MANAGER_ENDPOINT", "ftp://kvcm-meta:56020"),
            ("KV_CACHE_EVENT_MANAGER_ENDPOINT", "http://host:"),
            ("KV_CACHE_EVENT_MANAGER_ENDPOINT", "http://host:65536"),
            ("KV_CACHE_EVENT_MANAGER_ENDPOINT", "http://host:１２３"),
            ("KV_CACHE_EVENT_MANAGER_ENDPOINT", "http://[hostname]:56020"),
            ("KV_CACHE_EVENT_MANAGER_ENDPOINT", "http://ho%73t:56020"),
            ("KV_CACHE_EVENT_MANAGER_ENDPOINT", "http://user@host"),
            ("KV_CACHE_EVENT_MANAGER_ENDPOINT", "http://host?"),
            ("KV_CACHE_EVENT_MANAGER_ENDPOINT", "http://host/path%2"),
            ("KV_CACHE_EVENT_MANAGER_ENDPOINT", "http://host/path\\child"),
            ("KV_CACHE_EVENT_MANAGER_ENDPOINT", "http://host/路径"),
            ("KV_CACHE_EVENT_MANAGER_ENDPOINT", "http://host/path with-space"),
            ("KV_CACHE_EVENT_HOST_IP_PORT", "host:0"),
            ("KV_CACHE_EVENT_HOST_IP_PORT", "host:１２３"),
            ("KV_CACHE_EVENT_HOST_IP_PORT", "[::1]:18000"),
            ("KV_CACHE_EVENT_HOST_IP_PORT", "host%31"),
            ("KV_CACHE_EVENT_HOST_IP_PORT", "höst"),
            ("KV_CACHE_EVENT_HOST_IP_PORT", "host/path"),
            ("KV_CACHE_EVENT_INSTANCE_GROUP", "instance group"),
            ("KV_CACHE_EVENT_INSTANCE_ID", "instance id"),
            ("KV_CACHE_EVENT_INSTANCE_ID", "实例"),
        )
        for env_name, raw_value in cases:
            with self.subTest(env_name=env_name, raw_value=raw_value):
                os.environ.clear()
                os.environ.update(
                    {
                        "KV_CACHE_EVENT_PUBLISHER_TYPE": "kvcm",
                        "KV_CACHE_EVENT_MANAGER_ENDPOINT": "http://kvcm-meta:56020",
                        "KV_CACHE_EVENT_INSTANCE_ID": "test-instance",
                        "KV_CACHE_EVENT_HOST_IP_PORT": "127.0.0.1:18000",
                    }
                )
                os.environ[env_name] = raw_value
                sys.argv = ["prog", "--model_type", "qwen"]

                import rtp_llm.server.server_args.server_args

                importlib.reload(rtp_llm.server.server_args.server_args)
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                    rtp_llm.server.server_args.server_args.setup_args()
                self.assertIn(env_name, stderr.getvalue())

    def test_kvcm_python_validation_matches_shared_cpp_corpus(self):
        from rtp_llm.server.server_args.kv_cache_group_args import (
            _valid_host_ip_port,
            _valid_kvcm_identity,
            _valid_manager_endpoint,
        )

        validators = {
            "endpoint": _valid_manager_endpoint,
            "host": _valid_host_ip_port,
            "identity": _valid_kvcm_identity,
        }
        for case in KV_CACHE_EVENT_VALIDATION_CASES:
            with self.subTest(target=case.target, value=case.value):
                self.assertEqual(
                    case.expected_valid,
                    validators[case.target](case.value),
                )

    def test_kvcm_accepts_supported_endpoint_and_host_forms(self):
        for endpoint, host_ip_port in (
            ("http://kvcm-meta:56020", "kv-worker"),
            ("https://kvcm-meta.example/base", "10.0.0.8:18000"),
            ("https://[2001:db8::1]:56020/base%20path", "worker.example:443"),
        ):
            with self.subTest(endpoint=endpoint, host_ip_port=host_ip_port):
                os.environ.clear()
                os.environ.update(
                    {
                        "KV_CACHE_EVENT_PUBLISHER_TYPE": "kvcm",
                        "KV_CACHE_EVENT_MANAGER_ENDPOINT": endpoint,
                        "KV_CACHE_EVENT_INSTANCE_GROUP": "test-group",
                        "KV_CACHE_EVENT_INSTANCE_ID": "test-instance",
                        "KV_CACHE_EVENT_HOST_IP_PORT": host_ip_port,
                    }
                )
                sys.argv = ["prog", "--model_type", "qwen"]

                import rtp_llm.server.server_args.server_args

                importlib.reload(rtp_llm.server.server_args.server_args)
                configs = rtp_llm.server.server_args.server_args.setup_args()
                self.assertEqual(
                    endpoint,
                    configs.kv_cache_config.kv_cache_event_manager_endpoint,
                )
                self.assertEqual(
                    host_ip_port,
                    configs.kv_cache_config.kv_cache_event_host_ip_port,
                )

    def test_kv_cache_event_numeric_bounds_fail_fast(self):
        from rtp_llm.server.server_args.kv_cache_group_args import (
            KV_CACHE_EVENT_MAX_QUEUE_CAPACITY,
            KV_CACHE_EVENT_MAX_REPORT_BATCH_SIZE,
            KV_CACHE_EVENT_MAX_SNAPSHOT_BYTES,
            KV_CACHE_EVENT_MAX_SNAPSHOT_KEYS,
        )

        for env_name, raw_value in (
            ("KV_CACHE_EVENT_QUEUE_CAPACITY", "0"),
            (
                "KV_CACHE_EVENT_QUEUE_CAPACITY",
                str(KV_CACHE_EVENT_MAX_QUEUE_CAPACITY + 1),
            ),
            ("KV_CACHE_EVENT_REPORT_BATCH_SIZE", "-1"),
            (
                "KV_CACHE_EVENT_REPORT_BATCH_SIZE",
                str(KV_CACHE_EVENT_MAX_REPORT_BATCH_SIZE + 1),
            ),
            (
                "KV_CACHE_EVENT_SNAPSHOT_MAX_KEYS",
                str(KV_CACHE_EVENT_MAX_SNAPSHOT_KEYS + 1),
            ),
            (
                "KV_CACHE_EVENT_SNAPSHOT_MAX_BYTES",
                str(KV_CACHE_EVENT_MAX_SNAPSHOT_BYTES + 1),
            ),
            ("KV_CACHE_EVENT_LOG_MAX_KEYS", "-1"),
        ):
            with self.subTest(env_name=env_name):
                os.environ.clear()
                os.environ[env_name] = raw_value
                sys.argv = ["prog", "--model_type", "qwen"]

                import rtp_llm.server.server_args.server_args

                importlib.reload(rtp_llm.server.server_args.server_args)
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                    rtp_llm.server.server_args.server_args.setup_args()
                self.assertIn(env_name, stderr.getvalue())

    def test_kv_cache_event_resource_ceilings_are_accepted(self):
        from rtp_llm.server.server_args.kv_cache_group_args import (
            KV_CACHE_EVENT_MAX_QUEUE_CAPACITY,
            KV_CACHE_EVENT_MAX_REPORT_BATCH_SIZE,
            KV_CACHE_EVENT_MAX_SNAPSHOT_BYTES,
            KV_CACHE_EVENT_MAX_SNAPSHOT_KEYS,
        )

        limits = (
            (
                "KV_CACHE_EVENT_QUEUE_CAPACITY",
                "kv_cache_event_queue_capacity",
                KV_CACHE_EVENT_MAX_QUEUE_CAPACITY,
            ),
            (
                "KV_CACHE_EVENT_REPORT_BATCH_SIZE",
                "kv_cache_event_report_batch_size",
                KV_CACHE_EVENT_MAX_REPORT_BATCH_SIZE,
            ),
            (
                "KV_CACHE_EVENT_SNAPSHOT_MAX_KEYS",
                "kv_cache_event_snapshot_max_keys",
                KV_CACHE_EVENT_MAX_SNAPSHOT_KEYS,
            ),
            (
                "KV_CACHE_EVENT_SNAPSHOT_MAX_BYTES",
                "kv_cache_event_snapshot_max_bytes",
                KV_CACHE_EVENT_MAX_SNAPSHOT_BYTES,
            ),
        )
        for env_name, _, ceiling in limits:
            os.environ[env_name] = str(ceiling)
        sys.argv = ["prog", "--model_type", "qwen"]

        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        configs = rtp_llm.server.server_args.server_args.setup_args()
        for _, field_name, ceiling in limits:
            self.assertEqual(ceiling, getattr(configs.kv_cache_config, field_name))

    def test_kv_cache_event_env_declarations_are_complete(self):
        from rtp_llm.config.py_config_modules import PyEnvConfigs
        from rtp_llm.server.server_args.server_args import (
            EnvArgumentParser,
            init_all_group_args,
        )

        parser = EnvArgumentParser(description="test")
        configs = PyEnvConfigs()
        parser.set_root_config(configs)
        init_all_group_args(parser, configs)
        self.assertEqual(
            set(KV_CACHE_EVENT_FIELDS),
            {
                dest
                for dest in parser.get_env_mappings()
                if dest.startswith("kv_cache_event_")
            },
        )
        for field in KV_CACHE_EVENT_FIELDS:
            with self.subTest(field=field):
                self.assertTrue(parser.get_env_semantics(field).empty_as_unset)
                self.assertTrue(parser.get_env_semantics(field).strict_config_binding)
                self.assertEqual(
                    KV_CACHE_EVENT_DEFAULTS[field], parser.get_argument_default(field)
                )
        self.assertTrue(
            parser.get_env_semantics("kv_cache_event_publisher_type").strict_choice
        )
        string_fields = {
            "kv_cache_event_publisher_type",
            "kv_cache_event_manager_endpoint",
            "kv_cache_event_instance_group",
            "kv_cache_event_instance_id",
            "kv_cache_event_host_ip_port",
        }
        for field in KV_CACHE_EVENT_FIELDS:
            with self.subTest(field=field, semantic="emit_string_from_env"):
                self.assertEqual(
                    field in string_fields,
                    parser.get_env_semantics(field).emit_string_from_env,
                )
        self.assertTrue(
            parser.get_env_semantics("reco_instance_group").emit_string_from_env
        )

    def test_bounded_integer_converters_accept_exact_boundaries(self):
        from rtp_llm.server.server_args.util import (
            bounded_int,
            non_negative_int64,
            positive_int32,
        )

        positive_int64 = bounded_int(1, 2**63 - 1)
        self.assertEqual(1, positive_int32("1"))
        self.assertEqual(2**31 - 1, positive_int32(str(2**31 - 1)))
        self.assertEqual(1, positive_int64("1"))
        self.assertEqual(2**63 - 1, positive_int64(str(2**63 - 1)))
        self.assertEqual(0, non_negative_int64("0"))
        self.assertEqual(2**63 - 1, non_negative_int64(str(2**63 - 1)))

    def test_env_argument_generator_does_not_hide_invalid_bounded_values(self):
        from rtp_llm.server.server_args.generate_args_from_env_clean import (
            generate_args_list,
            read_env_value,
        )
        from rtp_llm.server.server_args.util import bounded_int

        positive_int64 = bounded_int(1, 2**63 - 1)
        os.environ["BOUNDED_VALUE"] = "0"
        with self.assertRaisesRegex(
            argparse.ArgumentTypeError, "BOUNDED_VALUE.*--bounded_value"
        ):
            read_env_value("BOUNDED_VALUE", 100, positive_int64, "--bounded_value")

        os.environ.clear()
        os.environ["KV_CACHE_EVENT_QUEUE_CAPACITY"] = "0"
        with self.assertRaisesRegex(
            argparse.ArgumentTypeError,
            "KV_CACHE_EVENT_QUEUE_CAPACITY.*--kv_cache_event_queue_capacity",
        ):
            generate_args_list(only_env_vars=True)

    def test_env_argument_generator_uses_one_boolean_conversion_path(self):
        from rtp_llm.server.server_args.generate_args_from_env_clean import (
            generate_args_list,
        )

        for raw_value, expected in (("true", "1"), ("false", "0")):
            with self.subTest(raw_value=raw_value):
                os.environ.clear()
                os.environ["REUSE_CACHE"] = raw_value
                generated = generate_args_list(only_env_vars=True)
                option_index = generated.index("--reuse_cache")
                self.assertEqual(expected, generated[option_index + 1])

        os.environ.clear()
        os.environ["REUSE_CACHE"] = "not-a-boolean"
        with self.assertRaisesRegex(
            argparse.ArgumentTypeError, "REUSE_CACHE.*--reuse_cache"
        ):
            generate_args_list(only_env_vars=True)

    def test_env_argument_generator_rejects_unrepresentable_string_whitespace(self):
        from rtp_llm.server.server_args.generate_args_from_env_clean import (
            generate_args_list,
        )

        os.environ["KV_CACHE_EVENT_INSTANCE_ID"] = "instance with spaces"
        with self.assertRaisesRegex(
            argparse.ArgumentTypeError,
            "KV_CACHE_EVENT_INSTANCE_ID.*--kv_cache_event_instance_id",
        ):
            generate_args_list(only_env_vars=True)

    def test_env_argument_generator_rejects_unsafe_command_fragment_tokens(self):
        from rtp_llm.server.server_args.generate_args_from_env_clean import (
            generate_args_list,
        )

        for value in (
            "instance*glob",
            "$(command)",
            "-looks-like-an-option",
            "quoted'value",
        ):
            with self.subTest(value=value):
                os.environ.clear()
                os.environ["KV_CACHE_EVENT_INSTANCE_ID"] = value
                with self.assertRaisesRegex(
                    argparse.ArgumentTypeError,
                    "KV_CACHE_EVENT_INSTANCE_ID.*--kv_cache_event_instance_id",
                ):
                    generate_args_list(only_env_vars=True)

    def test_argument_help_preserves_declared_choices(self):
        from rtp_llm.server.server_args.server_args import EnvArgumentParser

        parser = EnvArgumentParser(add_help=False)
        parser.add_argument("--mode", type=str, choices=("none", "log", "kvcm"))

        help_text = parser.format_help()
        self.assertIn("--mode {none,log,kvcm}", help_text)

    def test_env_argument_generator_honors_event_empty_as_unset(self):
        from rtp_llm.server.server_args.generate_args_from_env_clean import (
            generate_args_list,
        )

        os.environ["KV_CACHE_EVENT_QUEUE_CAPACITY"] = ""
        os.environ["KV_CACHE_EVENT_PUBLISHER_TYPE"] = ""

        generated = generate_args_list(only_env_vars=True)

        self.assertNotIn("--kv_cache_event_queue_capacity", generated)
        self.assertNotIn("--kv_cache_event_publisher_type", generated)

    def test_env_argument_generator_does_not_reinterpret_explicit_empty_strings(self):
        from rtp_llm.server.server_args.generate_args_from_env_clean import (
            generate_args_list,
        )

        os.environ["RECO_INSTANCE_GROUP"] = ""
        with self.assertRaisesRegex(
            argparse.ArgumentTypeError,
            "RECO_INSTANCE_GROUP.*--reco_instance_group.*empty string",
        ):
            generate_args_list(only_env_vars=True)

    def test_env_argument_generator_preserves_explicit_kvcm_strings(self):
        from rtp_llm.config.py_config_modules import PyEnvConfigs
        from rtp_llm.server.server_args.generate_args_from_env_clean import (
            generate_args_list,
        )
        from rtp_llm.server.server_args.server_args import (
            EnvArgumentParser,
            init_all_group_args,
        )

        # Keep the legacy behavior for implicit string defaults: the generated
        # command line should contain only strings explicitly supplied by the
        # deployment environment.
        self.assertNotIn(
            "--kv_cache_event_publisher_type", generate_args_list(only_env_vars=False)
        )

        expected = {
            "--kv_cache_event_publisher_type": "kvcm",
            "--kv_cache_event_manager_endpoint": "http://kvcm-meta:56020",
            "--kv_cache_event_instance_group": "test-group",
            "--kv_cache_event_instance_id": "test-instance",
            "--kv_cache_event_host_ip_port": "127.0.0.1:18000",
            "--kv_cache_event_queue_capacity": "17",
        }
        os.environ.update(
            {
                option.removeprefix("--").upper(): value
                for option, value in expected.items()
            }
        )
        os.environ["MODEL_TYPE"] = "qwen"

        generated = generate_args_list(only_env_vars=True)
        self.assertNotIn("--model_type", generated)
        for option, value in expected.items():
            with self.subTest(option=option):
                option_index = generated.index(option)
                self.assertEqual(value, generated[option_index + 1])

        # Validate the actual deployment hand-off: generated argv must remain
        # sufficient after the source environment has gone away.
        os.environ.clear()
        parser = EnvArgumentParser(description="test")
        configs = PyEnvConfigs()
        parser.set_root_config(configs)
        init_all_group_args(parser, configs)
        parser.parse_args(generated)

        event_config = configs.kv_cache_config
        self.assertEqual("kvcm", event_config.kv_cache_event_publisher_type)
        self.assertEqual(
            "http://kvcm-meta:56020",
            event_config.kv_cache_event_manager_endpoint,
        )
        self.assertEqual("test-group", event_config.kv_cache_event_instance_group)
        self.assertEqual("test-instance", event_config.kv_cache_event_instance_id)
        self.assertEqual("127.0.0.1:18000", event_config.kv_cache_event_host_ip_port)
        self.assertEqual(17, event_config.kv_cache_event_queue_capacity)

    def test_env_argument_generator_preserves_reco_group_fallback(self):
        from rtp_llm.config.py_config_modules import PyEnvConfigs
        from rtp_llm.server.server_args.generate_args_from_env_clean import (
            generate_args_list,
        )
        from rtp_llm.server.server_args.server_args import (
            EnvArgumentParser,
            init_all_group_args,
        )

        os.environ.update(
            {
                "KV_CACHE_EVENT_PUBLISHER_TYPE": "kvcm",
                "KV_CACHE_EVENT_MANAGER_ENDPOINT": "http://kvcm-meta:56020",
                "RECO_INSTANCE_GROUP": "shared-group",
                "KV_CACHE_EVENT_INSTANCE_ID": "test-instance",
                "KV_CACHE_EVENT_HOST_IP_PORT": "127.0.0.1:18000",
            }
        )

        generated = generate_args_list(only_env_vars=True)
        self.assertNotIn("--kv_cache_event_instance_group", generated)
        reco_index = generated.index("--reco_instance_group")
        self.assertEqual("shared-group", generated[reco_index + 1])

        # The generated argv must preserve the fallback input even after the
        # deployment environment has been cleared.
        os.environ.clear()
        parser = EnvArgumentParser(description="test")
        configs = PyEnvConfigs()
        parser.set_root_config(configs)
        init_all_group_args(parser, configs)
        parser.parse_args(generated)

        event_config = configs.kv_cache_config
        self.assertEqual("", event_config.kv_cache_event_instance_group)
        self.assertEqual("shared-group", event_config.reco_instance_group)

    def test_cli_value_that_looks_like_option_does_not_hide_environment_value(self):
        from rtp_llm.server.server_args.server_args import EnvArgumentParser

        parser = EnvArgumentParser()
        parser.add_argument("--payload", nargs=argparse.REMAINDER)
        parser.add_argument("--enabled", type=str, default="default")
        os.environ["ENABLED"] = "from-env"

        parsed = parser.parse_args(["--payload", "--enabled"])

        self.assertEqual(["--enabled"], parsed.payload)
        self.assertEqual("from-env", parsed.enabled)

    def test_explicit_option_spellings_override_environment_values(self):
        from rtp_llm.server.server_args.server_args import EnvArgumentParser

        parser = EnvArgumentParser()
        parser.add_argument("-c", "--count", type=int, default=0)
        parser.add_argument("--long-option-name", default="default")

        for argv, expected_count, expected_long in (
            (["--count=7"], 7, "from-env"),
            (["--long-opt", "from-cli"], 11, "from-cli"),
            (["-c", "13"], 13, "from-env"),
            (["-c17"], 17, "from-env"),
        ):
            with self.subTest(argv=argv):
                os.environ["COUNT"] = "11"
                os.environ["LONG_OPTION_NAME"] = "from-env"
                parsed = parser.parse_args(argv)
                self.assertEqual(expected_count, parsed.count)
                self.assertEqual(expected_long, parsed.long_option_name)

    def test_argparse_optional_tuple_contract_used_by_cli_precedence_scan(self):
        from rtp_llm.server.server_args.server_args import EnvArgumentParser

        parser = EnvArgumentParser(add_help=False)
        action = parser.add_argument("--value")

        attached = parser._parse_optional("--value=from-cli")
        separate = parser._parse_optional("--value")

        self.assertIsNotNone(attached)
        self.assertIs(action, attached[0])
        self.assertEqual("--value", attached[1])
        self.assertEqual("from-cli", attached[-1])
        self.assertIsNotNone(separate)
        self.assertIs(action, separate[0])
        self.assertEqual("--value", separate[1])
        self.assertIsNone(separate[-1])

    def test_combined_short_options_all_override_environment_values(self):
        from rtp_llm.server.server_args.server_args import EnvArgumentParser

        parser = EnvArgumentParser(add_help=False)
        parser.add_argument("-a", action="store_true")
        parser.add_argument("-b", action="store_true")
        parser.add_argument("-c")
        os.environ.update({"A": "0", "B": "0", "C": "from-env"})

        parsed = parser.parse_args(["-abcfrom-cli"])

        self.assertIs(True, parsed.a)
        self.assertIs(True, parsed.b)
        self.assertEqual("from-cli", parsed.c)

    def test_multi_value_options_do_not_hide_following_explicit_options(self):
        from rtp_llm.server.server_args.server_args import EnvArgumentParser

        for nargs, argv, expected_items in (
            (2, ["--items", "a", "b", "--enabled", "from-cli"], ["a", "b"]),
            ("*", ["--items", "a", "b", "--enabled", "from-cli"], ["a", "b"]),
            ("+", ["--items", "a", "b", "--enabled", "from-cli"], ["a", "b"]),
        ):
            with self.subTest(nargs=nargs):
                parser = EnvArgumentParser()
                parser.add_argument("--items", nargs=nargs)
                parser.add_argument("--enabled", default="default")
                os.environ["ENABLED"] = "from-env"

                parsed = parser.parse_args(argv)

                self.assertEqual(expected_items, parsed.items)
                self.assertEqual("from-cli", parsed.enabled)

    def test_option_value_that_matches_registered_flag_is_owned_by_argparse(self):
        from rtp_llm.server.server_args.server_args import EnvArgumentParser

        parser = EnvArgumentParser()
        parser.add_argument("--name")
        parser.add_argument("--enabled", default="default")
        os.environ["ENABLED"] = "from-env"

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--name", "--enabled"])

    def test_kvcm_validation_runs_for_direct_parser_callers(self):
        from rtp_llm.config.py_config_modules import PyEnvConfigs
        from rtp_llm.server.server_args.server_args import (
            EnvArgumentParser,
            init_all_group_args,
        )

        parser = EnvArgumentParser()
        configs = PyEnvConfigs()
        parser.set_root_config(configs)
        init_all_group_args(parser, configs)

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--kv_cache_event_publisher_type", "kvcm"])

    def test_subparser_parse_known_path_applies_bindings_and_validators(self):
        from types import SimpleNamespace

        from rtp_llm.server.server_args.server_args import EnvArgumentParser

        parser = EnvArgumentParser()
        subparsers = parser.add_subparsers(dest="command", required=True)
        serve = subparsers.add_parser("serve")
        target = SimpleNamespace(count=0)
        group = serve.add_argument_group("test")
        group.add_argument("--count", type=int, bind_to=(target, "count"))
        validated = []
        serve.register_post_parse_validator(lambda: validated.append(target.count))

        parsed, unknown = parser.parse_known_args(
            ["serve", "--count", "7", "--future-option"]
        )

        self.assertEqual("serve", parsed.command)
        self.assertEqual(7, parsed.count)
        self.assertEqual(7, target.count)
        self.assertEqual([7], validated)
        self.assertEqual(["--future-option"], unknown)

    def test_parse_args_rejects_unknown_before_any_parser_side_effects(self):
        from types import SimpleNamespace

        from rtp_llm.server.server_args.server_args import EnvArgumentParser

        parser = EnvArgumentParser()
        root_target = SimpleNamespace(count=0)
        root_group = parser.add_argument_group("root")
        root_group.add_argument(
            "--root-count", type=int, bind_to=(root_target, "count")
        )
        root_validated = []
        parser.register_post_parse_validator(
            lambda: root_validated.append(root_target.count)
        )

        subparsers = parser.add_subparsers(dest="command", required=True)
        serve = subparsers.add_parser("serve")
        child_target = SimpleNamespace(count=0)
        child_group = serve.add_argument_group("child")
        child_group.add_argument(
            "--child-count", type=int, bind_to=(child_target, "count")
        )
        child_validated = []
        serve.register_post_parse_validator(
            lambda: child_validated.append(child_target.count)
        )

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--root-count",
                    "3",
                    "serve",
                    "--child-count",
                    "7",
                    "--future-option",
                ]
            )

        self.assertIn("unrecognized arguments: --future-option", stderr.getvalue())
        self.assertEqual(0, root_target.count)
        self.assertEqual(0, child_target.count)
        self.assertEqual([], root_validated)
        self.assertEqual([], child_validated)

    def test_tuple_bindings_and_validators_do_not_require_root_config(self):
        from rtp_llm.config.py_config_modules import PyEnvConfigs
        from rtp_llm.server.server_args.kv_cache_group_args import (
            init_kv_cache_group_args,
        )
        from rtp_llm.server.server_args.server_args import EnvArgumentParser

        parser = EnvArgumentParser()
        config = PyEnvConfigs().kv_cache_config
        init_kv_cache_group_args(parser, config)

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--kv_cache_event_publisher_type", "kvcm"])

    def test_invalid_binding_declaration_fails_instead_of_using_stale_config(self):
        from rtp_llm.server.server_args.server_args import EnvArgumentParser

        parser = EnvArgumentParser()
        group = parser.add_argument_group("test")
        target = object()
        group.add_argument(
            "--count",
            type=int,
            bind_to=(target, "count"),
            strict_config_binding=True,
        )

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            parser.parse_args(["--count", "7"])
        self.assertIn("failed to bind argument count", stderr.getvalue())

    def test_legacy_binding_failure_remains_non_fatal(self):
        from rtp_llm.server.server_args.server_args import EnvArgumentParser

        parser = EnvArgumentParser()
        group = parser.add_argument_group("test")
        group.add_argument("--count", type=int, bind_to=(object(), "count"))

        with self.assertLogs(level="WARNING") as logs:
            parsed = parser.parse_args(["--count", "7"])

        self.assertEqual(7, parsed.count)
        self.assertTrue(any("count" in message for message in logs.output))

    def test_all_current_config_bindings_accept_their_declared_defaults(self):
        from rtp_llm.config.py_config_modules import PyEnvConfigs
        from rtp_llm.server.server_args.server_args import (
            EnvArgumentParser,
            init_all_group_args,
        )

        parser = EnvArgumentParser()
        configs = PyEnvConfigs()
        parser.set_root_config(configs)
        init_all_group_args(parser, configs)

        # Applying every non-None default exercises all tuple and dotted
        # binding declarations against the actual pybind config objects.
        parser.parse_args([])

    def test_env_mappings_are_isolated_per_parser(self):
        from rtp_llm.server.server_args.server_args import EnvArgumentParser

        first = EnvArgumentParser(env_prefix="first")
        first.add_argument("--only_first")
        second = EnvArgumentParser(env_prefix="second")
        second.add_argument("--only_second")

        self.assertEqual("FIRST_ONLY_FIRST", first.get_env_mappings()["only_first"])
        self.assertNotIn("only_second", first.get_env_mappings())
        self.assertEqual("SECOND_ONLY_SECOND", second.get_env_mappings()["only_second"])
        self.assertNotIn("only_first", second.get_env_mappings())

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

    def test_existing_env_choice_tolerates_unknown_value_in_pure_env_mode(self):
        os.environ["PDFUSION_SCHEDULER_MODE"] = "unknown"
        sys.argv = ["prog"]

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

    def test_invalid_kv_cache_event_integer_fails_fast_in_mixed_mode(self):
        os.environ["KV_CACHE_EVENT_QUEUE_CAPACITY"] = "not-an-integer"
        sys.argv = ["prog", "--model_type", "qwen"]

        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            rtp_llm.server.server_args.server_args.setup_args()
        self.assertIn("KV_CACHE_EVENT_QUEUE_CAPACITY", stderr.getvalue())

    def test_invalid_kv_cache_event_integer_fails_fast_in_pure_env_mode(self):
        os.environ["MODEL_TYPE"] = "qwen"
        os.environ["KV_CACHE_EVENT_QUEUE_CAPACITY"] = "not-an-integer"
        sys.argv = ["prog"]

        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            rtp_llm.server.server_args.server_args.setup_args()
        self.assertIn("KV_CACHE_EVENT_QUEUE_CAPACITY", stderr.getvalue())

    def test_kv_cache_event_int32_settings_reject_overflow(self):
        for env_name in (
            "KV_CACHE_EVENT_FLUSH_INTERVAL_MS",
            "KV_CACHE_EVENT_HEARTBEAT_INTERVAL_MS",
            "KV_CACHE_EVENT_REQUEST_TIMEOUT_MS",
            "KV_CACHE_EVENT_SNAPSHOT_TIMEOUT_MS",
            "KV_CACHE_EVENT_RETRY_INTERVAL_MS",
            "KV_CACHE_EVENT_SNAPSHOT_INTERVAL_MS",
        ):
            with self.subTest(env_name=env_name):
                os.environ.clear()
                os.environ[env_name] = str(2**31)
                sys.argv = ["prog", "--model_type", "qwen"]

                import rtp_llm.server.server_args.server_args

                importlib.reload(rtp_llm.server.server_args.server_args)
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                    rtp_llm.server.server_args.server_args.setup_args()
                self.assertIn(env_name, stderr.getvalue())

    def test_kv_cache_event_int64_settings_reject_overflow(self):
        for env_name in (
            "KV_CACHE_EVENT_QUEUE_CAPACITY",
            "KV_CACHE_EVENT_REPORT_BATCH_SIZE",
            "KV_CACHE_EVENT_SNAPSHOT_MAX_KEYS",
            "KV_CACHE_EVENT_SNAPSHOT_MAX_BYTES",
            "KV_CACHE_EVENT_LOG_MAX_KEYS",
        ):
            with self.subTest(env_name=env_name):
                os.environ.clear()
                os.environ[env_name] = str(2**63)
                sys.argv = ["prog", "--model_type", "qwen"]

                import rtp_llm.server.server_args.server_args

                importlib.reload(rtp_llm.server.server_args.server_args)
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                    rtp_llm.server.server_args.server_args.setup_args()
                self.assertIn(env_name, stderr.getvalue())

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
