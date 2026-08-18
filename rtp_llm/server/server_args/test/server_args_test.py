import importlib
import io
import json
import logging
import os
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
        os.environ["MM_IMAGE_MAX_FILE_SIZE_KB"] = "2048"
        os.environ["MM_VIDEO_MAX_FILE_SIZE_KB"] = "4096"
        os.environ["THINK_MODE"] = "adaptive"
        os.environ["DISABLE_FLASHINFER_HYBRID_PREFILL"] = "1"

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
        self.assertEqual(py_env_configs.vit_config.mm_image_max_file_size_kb, 2048)
        self.assertEqual(py_env_configs.vit_config.mm_video_max_file_size_kb, 4096)
        self.assertEqual(py_env_configs.generate_env_config.think_mode, "adaptive")

        # Verify disable_flashinfer_hybrid_prefill
        self.assertTrue(py_env_configs.fmha_config.disable_flashinfer_hybrid_prefill)

    def test_runtime_tuning_args_are_bound_to_configs(self):
        """CLI values bind to the configs and win over a conflicting environment."""
        from rtp_llm.server.server_args.server_args import setup_args

        with patch.dict(
            os.environ,
            {
                "RUNTIME_MEM_SAFETY_RATIO": "0.5",
                "RUNTIME_MEM_NO_WARMUP_FLOOR_MB": "9999",
            },
            clear=True,
        ):
            py_env_configs = setup_args(
                [
                    "--runtime_mem_safety_ratio",
                    "0.08",
                    "--runtime_mem_no_warmup_floor_mb",
                    "3072",
                ]
            )

        # CLI wins over the environment values above (provided_args precedence).
        self.assertEqual(py_env_configs.kv_cache_config.runtime_mem_safety_ratio, 0.08)
        self.assertEqual(
            py_env_configs.kv_cache_config.runtime_mem_no_warmup_floor_mb, 3072
        )

    def test_empty_string_untyped_env_keeps_legacy_semantics_on_both_paths(self):
        from rtp_llm.server.server_args.server_args import EnvArgumentParser

        for env_only in (True, False):
            with self.subTest(env_only=env_only):
                parser = EnvArgumentParser()
                parser.add_argument("--label", env_name="LABEL", default="default")
                sys.argv = ["prog"]
                with patch.dict(os.environ, {"LABEL": ""}, clear=True):
                    parsed = parser.parse_args() if env_only else parser.parse_args([])

                self.assertEqual(parsed.label, "")

    def test_empty_bounded_runtime_memory_env_uses_default_on_all_paths(self):
        from rtp_llm.ops import KVCacheConfig
        from rtp_llm.server.server_args.generate_args_from_env_clean import (
            generate_args_list,
        )
        from rtp_llm.server.server_args.server_args import setup_args
        from rtp_llm.server.server_args.util import DEFAULT_RESERVER_RUNTIME_MEM_MB

        defaults = KVCacheConfig()
        cases = (
            (
                "RUNTIME_MEM_SAFETY_RATIO",
                "--runtime_mem_safety_ratio",
                "runtime_mem_safety_ratio",
            ),
            (
                "RUNTIME_MEM_NO_WARMUP_FLOOR_MB",
                "--runtime_mem_no_warmup_floor_mb",
                "runtime_mem_no_warmup_floor_mb",
            ),
        )
        for env_name, option, attribute in cases:
            for env_only in (True, False):
                with self.subTest(env_name=env_name, env_only=env_only):
                    sys.argv = ["prog"]
                    with patch.dict(os.environ, {env_name: ""}, clear=True):
                        configs = setup_args() if env_only else setup_args([])

                    self.assertEqual(
                        getattr(configs.kv_cache_config, attribute),
                        getattr(defaults, attribute),
                    )

            with self.subTest(env_name=env_name, generator=True):
                with patch.dict(os.environ, {env_name: ""}, clear=True):
                    generated = generate_args_list(only_env_vars=True)
                self.assertNotIn(option, generated)

        for env_only in (True, False):
            with self.subTest(env_name="RESERVER_RUNTIME_MEM_MB", env_only=env_only):
                sys.argv = ["prog"]
                with patch.dict(
                    os.environ, {"RESERVER_RUNTIME_MEM_MB": ""}, clear=True
                ):
                    configs = setup_args() if env_only else setup_args([])

                self.assertEqual(
                    configs.runtime_config.reserve_runtime_mem_mb,
                    DEFAULT_RESERVER_RUNTIME_MEM_MB,
                )

        with patch.dict(
            os.environ, {"RESERVER_RUNTIME_MEM_MB": ""}, clear=True
        ):
            generated = generate_args_list(only_env_vars=True)
        self.assertNotIn("--reserver_runtime_mem_mb", generated)

    def test_explicit_empty_cli_value_for_bounded_runtime_memory_is_rejected(self):
        from rtp_llm.server.server_args.server_args import setup_args

        for option in (
            "--reserver_runtime_mem_mb",
            "--runtime_mem_safety_ratio",
            "--runtime_mem_no_warmup_floor_mb",
        ):
            with (
                self.subTest(option=option),
                patch.dict(os.environ, {}, clear=True),
                patch("sys.stderr", new_callable=io.StringIO),
            ):
                with self.assertRaises(SystemExit) as caught:
                    setup_args([option, ""])
                self.assertEqual(caught.exception.code, 2)

    def test_env_mappings_are_isolated_between_parser_instances(self):
        from rtp_llm.server.server_args.server_args import EnvArgumentParser

        parser_a = EnvArgumentParser(env_prefix="A")
        parser_b = EnvArgumentParser(env_prefix="B")
        parser_a.add_argument("--value", env_name="VALUE")
        parser_b.add_argument("--value", env_name="VALUE")

        mappings_a = parser_a.get_env_mappings()
        mappings_b = parser_b.get_env_mappings()
        self.assertEqual(mappings_a["value"], "A_VALUE")
        self.assertEqual(mappings_b["value"], "B_VALUE")
        self.assertEqual(mappings_a["help"], "A_HELP")
        self.assertEqual(mappings_b["help"], "B_HELP")
        self.assertNotIn("B_VALUE", mappings_a.values())
        self.assertNotIn("A_VALUE", mappings_b.values())

    def test_forward_warmup_is_on_by_default(self):
        """The Python service entrypoint preserves the legacy warmup default."""
        from rtp_llm.server.server_args.server_args import setup_args

        self.assertTrue(setup_args([]).runtime_config.warm_up)

    def test_runtime_memory_env_boundaries_are_bound_to_config(self):
        from rtp_llm.server.server_args.server_args import setup_args

        for safety_ratio, no_warmup_floor_mb in (
            ("0", "0"),
            ("0.999999", "3072"),
        ):
            with (
                self.subTest(
                    safety_ratio=safety_ratio,
                    no_warmup_floor_mb=no_warmup_floor_mb,
                ),
                patch.dict(
                    os.environ,
                    {
                        "RUNTIME_MEM_SAFETY_RATIO": safety_ratio,
                        "RUNTIME_MEM_NO_WARMUP_FLOOR_MB": no_warmup_floor_mb,
                    },
                    clear=True,
                ),
            ):
                py_env_configs = setup_args([])

                self.assertEqual(
                    py_env_configs.kv_cache_config.runtime_mem_safety_ratio,
                    float(safety_ratio),
                )
                self.assertEqual(
                    py_env_configs.kv_cache_config.runtime_mem_no_warmup_floor_mb,
                    int(no_warmup_floor_mb),
                )

    def test_strict_env_values_abort_with_usage_error_naming_the_env_var(self):
        """Strict converters reject bad env values through argparse, not a bare traceback.

        setup_args([]) takes the CLI-mixed branch (args is not None), where env values are
        converted by hand after argparse has run. The strict runtime-memory knobs use bounded_*
        converters,
        which raise argparse.ArgumentTypeError; that is routed through parser.error() so the exit
        code (2) and the usage-error form match the env-only path. The message names the env var,
        which argparse itself cannot do on the env-only path.
        """
        from rtp_llm.server.server_args.server_args import setup_args

        for env_name, bad_value in (
            ("RUNTIME_MEM_SAFETY_RATIO", "1.0"),  # upper bound is exclusive
            ("RUNTIME_MEM_SAFETY_RATIO", "-0.1"),
            ("RUNTIME_MEM_SAFETY_RATIO", "nan"),
            ("RUNTIME_MEM_SAFETY_RATIO", "abc"),
            ("RUNTIME_MEM_NO_WARMUP_FLOOR_MB", "-1"),
        ):
            with self.subTest(env_name=env_name, bad_value=bad_value):
                with (
                    patch.dict(os.environ, {env_name: bad_value}, clear=True),
                    # argparse prints usage to stderr before exiting; keep test output readable.
                    patch("sys.stderr", new_callable=io.StringIO) as stderr,
                ):
                    with self.assertRaises(SystemExit) as caught:
                        setup_args([])

                self.assertEqual(caught.exception.code, 2)
                self.assertIn(env_name, stderr.getvalue())

    def test_runtime_memory_strict_env_does_not_change_existing_converters(self):
        """An existing converter keeps taking main's generic ArgumentTypeError branch.

        main routes every argparse.ArgumentTypeError through parser.error(), so an invalid
        str2bool env value is a usage error for reasons that predate this branch. What must
        stay true is that it is not routed through the strict/lenient policy added here for
        the runtime-memory knobs; the two branches are told apart by their message shape.
        """
        from rtp_llm.server.server_args.server_args import EnvArgumentParser
        from rtp_llm.server.server_args.util import str2bool

        parser = EnvArgumentParser()
        parser.add_argument("--existing_bool", env_name="EXISTING_BOOL", type=str2bool)

        with (
            patch.dict(os.environ, {"EXISTING_BOOL": "invalid"}, clear=True),
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
        ):
            with self.assertRaises(SystemExit) as caught:
                parser.parse_args([])

        self.assertEqual(caught.exception.code, 2)
        self.assertIn("EXISTING_BOOL (existing_bool):", stderr.getvalue())
        self.assertNotIn("invalid value for", stderr.getvalue())

    def test_invalid_reserver_env_still_falls_back_on_the_cli_mixed_path(self):
        """The one knob that deliberately keeps the opposite contract.

        non_negative_mib_int raises ValueError rather than ArgumentTypeError precisely so an
        invalid RESERVER_RUNTIME_MEM_MB keeps its pre-warmup-feature behavior on this path: warn
        and use the default instead of aborting. Pinned so the divergence stays deliberate --
        making it strict is a behaviour change that needs a release note.
        """
        from rtp_llm.server.server_args.server_args import setup_args
        from rtp_llm.server.server_args.util import DEFAULT_RESERVER_RUNTIME_MEM_MB

        for bad_value in ("-1", "not-an-int"):
            with self.subTest(bad_value=bad_value):
                with patch.dict(
                    os.environ, {"RESERVER_RUNTIME_MEM_MB": bad_value}, clear=True
                ):
                    py_env_configs = setup_args([])

                self.assertEqual(
                    py_env_configs.runtime_config.reserve_runtime_mem_mb,
                    DEFAULT_RESERVER_RUNTIME_MEM_MB,
                )

    def test_env_only_path_binds_new_knobs_and_aborts_on_invalid_values(self):
        """The env-only branch, which setup_args([]) never exercises.

        setup_args([]) passes a non-None args list and therefore takes the CLI-mixed branch. With
        sys.argv holding only the program name and args=None, every env value is instead synthesised
        into argv and converted by argparse itself. RESERVER_RUNTIME_MEM_MB is the interesting one:
        it aborts here even though the same value falls back on the CLI-mixed path, which is the
        deliberate asymmetry documented on non_negative_mib_int.
        """
        from rtp_llm.server.server_args.server_args import setup_args

        sys.argv = ["prog"]

        with patch.dict(
            os.environ,
            {
                "RUNTIME_MEM_SAFETY_RATIO": "0.08",
                "RUNTIME_MEM_NO_WARMUP_FLOOR_MB": "3072",
            },
            clear=True,
        ):
            py_env_configs = setup_args()

        self.assertEqual(py_env_configs.kv_cache_config.runtime_mem_safety_ratio, 0.08)
        self.assertEqual(
            py_env_configs.kv_cache_config.runtime_mem_no_warmup_floor_mb, 3072
        )
        # Out-of-range values abort on this path as well, via argparse's own conversion.
        for env_name, bad_value in (
            ("RUNTIME_MEM_SAFETY_RATIO", "1.0"),
            # Lenient on the CLI-mixed path, but env-only hands it to argparse, which rejects it.
            ("RESERVER_RUNTIME_MEM_MB", "-1"),
        ):
            with self.subTest(env_name=env_name, bad_value=bad_value):
                sys.argv = ["prog"]
                with (
                    patch.dict(os.environ, {env_name: bad_value}, clear=True),
                    patch("sys.stderr", new_callable=io.StringIO),
                ):
                    with self.assertRaises(SystemExit) as caught:
                        setup_args()
                self.assertEqual(caught.exception.code, 2)

    def test_env_generator_emits_only_the_three_bounded_runtime_memory_values(self):
        from rtp_llm.server.server_args.generate_args_from_env_clean import (
            generate_args_list,
        )
        from rtp_llm.server.server_args.server_args import setup_args

        env_values = {
            "RESERVER_RUNTIME_MEM_MB": "1536",
            "RUNTIME_MEM_NO_WARMUP_FLOOR_MB": "3072",
            "RUNTIME_MEM_SAFETY_RATIO": "0.08",
        }
        with patch.dict(os.environ, env_values, clear=True):
            generated = generate_args_list(only_env_vars=True)
            generated_pairs = dict(zip(generated[::2], generated[1::2]))

            self.assertEqual(
                generated_pairs,
                {
                    "--reserver_runtime_mem_mb": "1536",
                    "--runtime_mem_no_warmup_floor_mb": "3072",
                    "--runtime_mem_safety_ratio": "0.08",
                },
            )

            configs = setup_args(generated)
            self.assertEqual(configs.runtime_config.reserve_runtime_mem_mb, 1536)
            self.assertEqual(configs.kv_cache_config.runtime_mem_safety_ratio, 0.08)
            self.assertEqual(
                configs.kv_cache_config.runtime_mem_no_warmup_floor_mb, 3072
            )

    def test_env_generator_keeps_invalid_bounded_value_for_argparse_to_reject(self):
        from rtp_llm.server.server_args.generate_args_from_env_clean import (
            generate_args_list,
        )
        from rtp_llm.server.server_args.server_args import setup_args

        with patch.dict(
            os.environ, {"RUNTIME_MEM_SAFETY_RATIO": "not-a-ratio"}, clear=True
        ):
            generated = generate_args_list(only_env_vars=True)
            self.assertIn(
                ["--runtime_mem_safety_ratio", "not-a-ratio"],
                [generated[index : index + 2] for index in range(0, len(generated), 2)],
            )
            with patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as caught:
                    setup_args(generated)
            self.assertEqual(caught.exception.code, 2)

    def test_env_generator_treats_invalid_reserver_as_fail_fast_argv(self):
        from rtp_llm.server.server_args.generate_args_from_env_clean import (
            generate_args_list,
        )
        from rtp_llm.server.server_args.server_args import setup_args

        for bad_value in ("not-an-int", "-1"):
            with self.subTest(bad_value=bad_value):
                with patch.dict(
                    os.environ,
                    {"RESERVER_RUNTIME_MEM_MB": bad_value},
                    clear=True,
                ):
                    generated = generate_args_list(only_env_vars=True)

                self.assertIn(
                    ["--reserver_runtime_mem_mb", bad_value],
                    [
                        generated[index : index + 2]
                        for index in range(0, len(generated), 2)
                    ],
                )
                with (
                    patch.dict(os.environ, {}, clear=True),
                    patch("sys.stderr", new_callable=io.StringIO),
                ):
                    with self.assertRaises(SystemExit) as caught:
                        setup_args(generated)
                self.assertEqual(caught.exception.code, 2)

    def test_no_warmup_compat_anchor_defaults(self):
        """Pin the literals behind the "no-warmup sizing is unchanged" promise.

        The pre-feature formula hardcoded max(2048 MiB, 5% * total); the C++
        defaults (kDefaultRuntimeMemorySafetyRatio / kDefaultRuntimeNoWarmupFloorMb
        in ConfigModules.h) now carry those numbers. Every other test compares
        against freshly constructed configs, so nothing else fails if the
        defaults drift -- these literal assertions are the backward-compat
        anchor: changing either value resizes the KV cache of every deployment
        that never runs a traced warmup, and needs that impact assessed first.
        """
        from rtp_llm.ops import KVCacheConfig

        config = KVCacheConfig()
        self.assertEqual(
            config.runtime_mem_safety_ratio,
            0.05,
            msg="no-warmup compat anchor: pre-feature formula reserved 5% of total GPU",
        )
        self.assertEqual(
            config.runtime_mem_no_warmup_floor_mb,
            2048,
            msg="no-warmup compat anchor: pre-feature formula floored at 2048 MiB",
        )

    def test_runtime_tuning_summary_logs_defaults_at_info(self):
        """assertLogs renders the record, so a %-placeholder/argument mismatch in
        _log_runtime_tuning_summary fails here instead of only at runtime."""
        from rtp_llm.server.server_args.server_args import setup_args

        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertLogs(level="INFO") as logs,
        ):
            setup_args([])

        records = [
            record
            for record in logs.records
            if record.getMessage().startswith("Runtime memory tuning:")
        ]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].levelno, logging.INFO)
        message = records[0].getMessage()
        # One contiguous string on purpose: this must stay byte-identical to
        # PYTHON_LOG_SENTINEL in rtp_llm/test/smoke/multi_inst_case_runner.py, which
        # greps it as a single substring. Asserting the halves separately would let a
        # field inserted between them pass here and fail smoke.
        self.assertIn("Runtime memory tuning: world_rank=", message)
        self.assertIn("configured_role_type=", message)
        self.assertIn("vit_separation=", message)
        self.assertNotIn(" role_type=", message)
        self.assertIn("tuned=none", message)
        # The reserver default lives at the argparse layer (1024), not in the C++
        # RuntimeConfig (0): rendering 1024 with tuned=none proves the summary
        # compares against the right default, else this default run would list it
        # in the tuned field.
        self.assertIn("reserver_runtime_mem_mb=1024", message)

    def test_runtime_tuning_summary_lists_non_default_values_in_tuned_field(self):
        """Tuned knobs stay at INFO and surface in the same-line tuned=[...] field.

        Collectors filter on the field, not the level: legal tuning is
        configuration, not an incident. WARNING is reserved for genuinely
        actionable conditions.
        """
        from rtp_llm.server.server_args.server_args import setup_args

        for args, expected_fragments in (
            # The only knob whose default is argparse-level (1024) rather than a
            # freshly constructed C++ config: it must still land in tuned=[...].
            (
                ["--reserver_runtime_mem_mb", "8192"],
                ("reserver_runtime_mem_mb=8192", "(default 1024)"),
            ),
        ):
            with self.subTest(args=args):
                with (
                    patch.dict(os.environ, {}, clear=True),
                    self.assertLogs(level="INFO") as logs,
                ):
                    setup_args(args)

                records = [
                    record
                    for record in logs.records
                    if record.getMessage().startswith("Runtime memory tuning:")
                ]
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0].levelno, logging.INFO)
                message = records[0].getMessage()
                self.assertIn("tuned=[", message)
                self.assertNotIn("tuned=none", message)
                for fragment in expected_fragments:
                    self.assertIn(fragment, message)

    def test_runtime_tuning_summary_reports_reserver_env_fallback(self):
        from rtp_llm.server.server_args.server_args import setup_args

        with (
            patch.dict(os.environ, {"RESERVER_RUNTIME_MEM_MB": "-1"}, clear=True),
            self.assertLogs(level="INFO") as logs,
        ):
            setup_args([])

        messages = [
            record.getMessage()
            for record in logs.records
            if record.getMessage().startswith("Runtime memory tuning:")
        ]
        self.assertEqual(len(messages), 1)
        self.assertIn("env_fallback=[reserver_runtime_mem_mb]", messages[0])

    def test_production_parser_help_renders_without_error(self):
        """Tripwire for bare % in any help string.

        argparse expands help text with the % operator, so a literal percent
        sign written as a bare % (instead of %%) makes format_help() raise
        ValueError. That only fires when help is rendered, so servers start
        fine and nothing else catches it -- this test does, for every argument
        the production parser registers.
        """
        from rtp_llm.server.server_args.server_args import (
            EnvArgumentParser,
            PyEnvConfigs,
            init_all_group_args,
        )

        parser = EnvArgumentParser(description="help rendering tripwire")
        py_env_configs = PyEnvConfigs()
        parser.set_root_config(py_env_configs)
        init_all_group_args(parser, py_env_configs)
        parser.add_argument(
            "--percent_help_tripwire",
            help="test-only literal percent: 5%%",
        )

        help_text = parser.format_help()
        # Keep the escape assertion independent of production wording while format_help() above
        # continues to check every argument registered by the production parser.
        self.assertIn("test-only literal percent: 5%", help_text)

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
        self.assertTrue(py_env_configs.fmha_config.disable_flashinfer_hybrid_prefill)

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
