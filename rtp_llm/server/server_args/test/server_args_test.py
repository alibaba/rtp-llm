import importlib
import json
import os
import pickle
import sys
import tempfile
from unittest import TestCase, main
from unittest.mock import patch

from rtp_llm.ops import HWKernelConfig
from rtp_llm.utils.backend_registry import (
    register_backend_hook,
    reset_backend_registrations,
)


class ServerArgsPyEnvConfigsTest(TestCase):
    """Test that environment variables and command line arguments are correctly set to py_env_configs structure."""

    def test_internal_backend_registers_moe_choice_before_parser_initialization(self):
        from rtp_llm.server.server_args import server_args

        loaded = False

        def load_backend():
            nonlocal loaded
            if not loaded:
                register_backend_hook(
                    "moe_strategy_choices",
                    lambda parser: next(
                        action
                        for action in parser._actions
                        if "--moe_strategy" in action.option_strings
                    ).choices.append("external_test_strategy"),
                )
                loaded = True
            return True

        reset_backend_registrations()
        try:
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(
                    server_args,
                    "ensure_backend_entrypoint_loaded",
                    side_effect=load_backend,
                ),
                patch(
                    "rtp_llm.utils.backend_registry.ensure_backend_entrypoint_loaded",
                    return_value=True,
                ),
            ):
                first_configs = server_args.setup_args(
                    ["--moe_strategy", "external_test_strategy"]
                )
                second_configs = server_args.setup_args(
                    ["--moe_strategy", "external_test_strategy"]
                )
            self.assertEqual(
                first_configs.moe_config.moe_strategy, "external_test_strategy"
            )
            self.assertEqual(
                second_configs.moe_config.moe_strategy, "external_test_strategy"
            )
        finally:
            reset_backend_registrations()


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
        os.environ["CP_FORCE_SINGLE_PREFILL"] = "0"
        os.environ["WARM_UP"] = "1"
        os.environ["MAX_SEQ_LEN"] = "4096"
        os.environ["REMOTE_JIT_DIR"] = "dfs://bucket/jit/cache"
        os.environ["FRONTEND_PRE_STOP_DRAIN_SECONDS"] = "2.5"
        os.environ["DASH_SC_GRPC_PRE_STOP_DRAIN_SECONDS"] = "9"
        os.environ["LOADER_RECYCLE_HANDLES"] = "false"
        os.environ["MOE_PURE_TP_PRESHARD"] = "true"
        os.environ["MM_IMAGE_MAX_FILE_SIZE_KB"] = "2048"
        os.environ["MM_VIDEO_MAX_FILE_SIZE_KB"] = "4096"
        os.environ["THINK_MODE"] = "adaptive"
        os.environ["DISABLE_FLASHINFER_HYBRID_PREFILL"] = "1"
        os.environ["ENABLE_PREFILL_CUDA_GRAPH"] = "1"
        os.environ["PREFILL_CUDA_GRAPH_MAX_REQUESTS"] = "4"
        os.environ["PREFILL_CUDA_GRAPH_CAPTURE_CONFIG"] = "64,128,256"

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
        self.assertEqual(
            py_env_configs.runtime_config.fifo_scheduler_config.cp_force_single_prefill,
            False,
        )
        restored_fifo_config = pickle.loads(
            pickle.dumps(py_env_configs.runtime_config.fifo_scheduler_config)
        )
        self.assertEqual(restored_fifo_config.max_batch_tokens_without_cache, 2048)
        # Old pickles carry only the two original slots; every field added later
        # must fall back to its default instead of raising.
        fifo_config_type = type(py_env_configs.runtime_config.fifo_scheduler_config)
        legacy_fifo_config = fifo_config_type.__new__(fifo_config_type)
        legacy_fifo_config.__setstate__((32, 8192))
        self.assertEqual(legacy_fifo_config.max_context_batch_size, 32)
        self.assertEqual(legacy_fifo_config.max_batch_tokens_size, 8192)
        self.assertEqual(legacy_fifo_config.max_inited_kv_cache_streams, 0)
        self.assertEqual(legacy_fifo_config.max_batch_tokens_without_cache, 0)

        # Verify frontend and DashSc pre-stop windows are configured independently.
        self.assertEqual(
            py_env_configs.server_config.frontend_pre_stop_drain_seconds, 2.5
        )
        self.assertEqual(
            py_env_configs.server_config.dash_sc_grpc_pre_stop_drain_seconds, 9.0
        )

        # Verify runtime_config (warm_up is now in RuntimeConfig)
        self.assertEqual(py_env_configs.runtime_config.warm_up, True)  # bool in C++
        self.assertEqual(py_env_configs.runtime_config.warm_up_with_loss, False)
        self.assertEqual(py_env_configs.runtime_config.model_warm_up, True)

        # Verify load_config: the flag came from LOADER_RECYCLE_HANDLES=false.
        self.assertFalse(py_env_configs.load_config.loader_recycle_handles)
        # MOE_PURE_TP_PRESHARD=true explicitly enables the opt-in path.
        self.assertTrue(py_env_configs.load_config.moe_pure_tp_preshard)
        # Note: max_seq_len is in ModelConfig, not RuntimeConfig or EngineConfig
        # It will be set when ModelConfig is created from model_args
        self.assertEqual(py_env_configs.vit_config.mm_image_max_file_size_kb, 2048)
        self.assertEqual(py_env_configs.vit_config.mm_video_max_file_size_kb, 4096)
        self.assertEqual(py_env_configs.generate_env_config.think_mode, "adaptive")
        self.assertEqual(
            py_env_configs.jit_config.remote_jit_dir,
            "dfs://bucket/jit/cache",
        )

        # Verify disable_flashinfer_hybrid_prefill
        self.assertTrue(py_env_configs.fmha_config.disable_flashinfer_hybrid_prefill)
        self.assertEqual(
            py_env_configs.py_hw_kernel_config.prefill_cuda_graph_capture_seq_lens,
            [64, 128, 256],
        )
        self.assertTrue(py_env_configs.py_hw_kernel_config.enable_prefill_cuda_graph)
        self.assertEqual(
            py_env_configs.py_hw_kernel_config.prefill_cuda_graph_max_requests, 4
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
            "--disable_flashinfer_hybrid_prefill",
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
            py_env_configs.runtime_config.fifo_scheduler_config.cp_force_single_prefill,
            False,
        )
        self.assertEqual(
            py_env_configs.runtime_config.fifo_scheduler_config.max_inited_kv_cache_streams,
            16,
        )

        # Verify runtime_config (warm_up is now in RuntimeConfig)
        self.assertEqual(py_env_configs.runtime_config.warm_up, False)  # bool in C++
        self.assertEqual(py_env_configs.runtime_config.warm_up_with_loss, False)
        self.assertEqual(py_env_configs.runtime_config.model_warm_up, True)

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
        self.assertTrue(py_env_configs.fmha_config.disable_flashinfer_hybrid_prefill)
        self.assertFalse(py_env_configs.py_hw_kernel_config.enable_prefill_cuda_graph)
        self.assertEqual(
            py_env_configs.py_hw_kernel_config.prefill_cuda_graph_max_requests, 8
        )
        self.assertEqual(
            py_env_configs.py_hw_kernel_config.prefill_cuda_graph_capture_seq_lens,
            HWKernelConfig().prefill_cuda_graph_capture_seq_lens,
        )

    def test_prefill_cuda_graph_cli_binding_and_validation(self):
        from argparse import ArgumentTypeError

        from rtp_llm.server.server_args import hw_kernel_group_args, server_args

        configs = server_args.setup_args(
            [
                "--enable_prefill_cuda_graph",
                "1",
                "--prefill_cuda_graph_max_requests",
                "3",
                "--prefill_cuda_graph_capture_config",
                "7,19,31",
            ]
        )
        self.assertTrue(configs.py_hw_kernel_config.enable_prefill_cuda_graph)
        self.assertEqual(configs.py_hw_kernel_config.prefill_cuda_graph_max_requests, 3)
        self.assertEqual(
            configs.py_hw_kernel_config.prefill_cuda_graph_capture_seq_lens,
            [7, 19, 31],
        )

        self.assertEqual(
            hw_kernel_group_args.PREFILL_CUDA_GRAPH_MAX_REQUESTS_LIMIT,
            HWKernelConfig.prefill_cuda_graph_max_requests_limit,
        )
        self.assertEqual(
            hw_kernel_group_args.PREFILL_CUDA_GRAPH_MAX_CAPTURE_TOKENS,
            HWKernelConfig.prefill_cuda_graph_max_capture_tokens,
        )
        with self.assertRaisesRegex(ArgumentTypeError, "must not exceed 64"):
            hw_kernel_group_args._prefill_cuda_graph_max_requests("65")
        with self.assertRaisesRegex(ArgumentTypeError, "maximum is 64"):
            hw_kernel_group_args._parse_prefill_cuda_graph_capture_config(
                ",".join(str(i) for i in range(1, 66))
            )
        for invalid_config in (
            "0,32",
            "-1,32",
            f"32,{hw_kernel_group_args.PREFILL_CUDA_GRAPH_MAX_CAPTURE_TOKENS + 1}",
        ):
            with self.subTest(invalid_config=invalid_config):
                with self.assertRaises(ArgumentTypeError):
                    hw_kernel_group_args._parse_prefill_cuda_graph_capture_config(
                        invalid_config
                    )

        self.assertEqual(
            hw_kernel_group_args._parse_prefill_cuda_graph_capture_config("64:1"),
            list(range(1, 65)),
        )
        with self.assertRaisesRegex(ArgumentTypeError, "maximum is 64"):
            hw_kernel_group_args._parse_prefill_cuda_graph_capture_config("65:1")

        for bucket_count in (64, 65):
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as config_file:
                config_file.write("\n".join(str(i) for i in range(1, bucket_count + 1)))
                config_path = config_file.name
            try:
                if bucket_count == 64:
                    self.assertEqual(
                        hw_kernel_group_args._parse_prefill_cuda_graph_capture_config(
                            config_path
                        ),
                        list(range(1, 65)),
                    )
                else:
                    with self.assertRaisesRegex(ArgumentTypeError, "maximum is 64"):
                        hw_kernel_group_args._parse_prefill_cuda_graph_capture_config(
                            config_path
                        )
            finally:
                os.unlink(config_path)

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

    def test_cmd_args_override_env_vars(self):
        """Test that command line arguments override environment variables."""
        # Set environment variables
        os.environ["MODEL_TYPE"] = "qwen"
        os.environ["CHECKPOINT_PATH"] = "/path/to/qwen/checkpoint"
        os.environ["ACT_TYPE"] = "BF16"
        os.environ["TP_SIZE"] = "4"
        os.environ["CONCURRENCY_LIMIT"] = "32"
        os.environ["DISABLE_FLASHINFER_HYBRID_PREFILL"] = "1"

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
            "--disable_flashinfer_hybrid_prefill",
            "false",
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
        self.assertFalse(
            py_env_configs.fmha_config.disable_flashinfer_hybrid_prefill
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

        # Not set via env or cmd args: verify the default value
        self.assertTrue(py_env_configs.fmha_config.disable_flashinfer_hybrid_prefill)

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
        init_grammar_group_args(parser, g, cfgs.grammar_admission_config)
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
