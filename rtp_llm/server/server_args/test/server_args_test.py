import importlib
import json
import logging
import os
import pickle
import sys
from io import StringIO
from unittest import TestCase, main
from unittest.mock import patch

from rtp_llm.utils.pre_import_config import str2bool

# The pickle state tuple is produced in C++ (rtp_llm/cpp/pybind/ConfigInit.cc) and
# consumed there by index, so its length is a cross-layer contract this test file
# can only observe, not enforce.
_ARITY_TRIPWIRE_MSG = (
    "config pickle state arity changed: after adding or removing a field, update "
    "the accepted t.size() list in that config's __setstate__ in "
    "rtp_llm/cpp/pybind/ConfigInit.cc, keep the previous arity accepted so older "
    "states still load, and append new fields at the end so the existing indices "
    "keep pointing at the same fields"
)


class ServerArgsPyEnvConfigsTest(TestCase):
    """Test that environment variables and command line arguments are correctly set to py_env_configs structure."""


class PreImportConfigTest(TestCase):
    def test_server_parser_reuses_pre_import_bool_parser(self):
        from rtp_llm.server.server_args.util import str2bool as server_str2bool

        self.assertIs(server_str2bool, str2bool)


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
        # Note: max_seq_len is in ModelConfig, not RuntimeConfig or EngineConfig
        # It will be set when ModelConfig is created from model_args

    def test_runtime_tuning_args_are_bound_to_configs(self):
        from rtp_llm.server.server_args.server_args import setup_args

        py_env_configs = setup_args(
            [
                "--runtime_mem_safety_ratio",
                "0.08",
                "--runtime_mem_no_warmup_floor_mb",
                "3072",
                "--moe_skew_mult",
                "1.75",
            ]
        )

        self.assertEqual(py_env_configs.kv_cache_config.runtime_mem_safety_ratio, 0.08)
        self.assertEqual(
            py_env_configs.kv_cache_config.runtime_mem_no_warmup_floor_mb, 3072
        )
        self.assertNotIn("RUNTIME_MEM_SAFETY_RATIO", os.environ)
        self.assertNotIn("RUNTIME_MEM_NO_WARMUP_FLOOR_MB", os.environ)
        self.assertEqual(py_env_configs.moe_config.moe_skew_mult, 1.75)
        self.assertNotIn("MOE_SKEW_MULT", os.environ)

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

    def test_kv_cache_config_pickle_preserves_runtime_memory_tuning(self):
        from rtp_llm.server.server_args.server_args import setup_args

        config = setup_args(
            [
                "--runtime_mem_safety_ratio",
                "0.08",
                "--runtime_mem_no_warmup_floor_mb",
                "3072",
            ]
        ).kv_cache_config
        self.assertEqual(len(config.__getstate__()), 56, msg=_ARITY_TRIPWIRE_MSG)
        restored = pickle.loads(pickle.dumps(config))

        self.assertEqual(restored.runtime_mem_safety_ratio, 0.08)
        self.assertEqual(restored.runtime_mem_no_warmup_floor_mb, 3072)

    def test_moe_config_pickle_preserves_warmup_skew(self):
        from rtp_llm.server.server_args.server_args import setup_args

        config = setup_args(
            ["--moe_skew_mult", "1.75", "--fp4_moe_op", "trtllm"]
        ).moe_config
        self.assertEqual(len(config.__getstate__()), 14, msg=_ARITY_TRIPWIRE_MSG)
        restored = pickle.loads(pickle.dumps(config))

        self.assertEqual(restored.moe_skew_mult, 1.75)
        self.assertEqual(restored.fp4_moe_op, "trtllm")

    def test_legacy_state_arities_keep_config_defaults(self):
        """Exercise the shorter tuples __setstate__ still accepts.

        ConfigInit.cc keeps them so a state pickled before these fields existed
        still loads, with the appended fields falling back to their config
        defaults. The round-trip tests above only ever feed the current arity, so
        without this the compatibility branches are never executed.

        Truncating the live __getstate__() also pins the field *order*, which the
        arity assertions do not: the C++ side reads the appended fields at fixed
        indices (t[54]/t[55], t[12]/t[13]), so moving a field earlier in
        __getstate__ makes these unpickle into the wrong slots and fail here.
        """
        from rtp_llm.ops import KVCacheConfig, MoeConfig
        from rtp_llm.server.server_args.server_args import setup_args

        py_env_configs = setup_args(
            [
                "--runtime_mem_safety_ratio",
                "0.08",
                "--runtime_mem_no_warmup_floor_mb",
                "3072",
                "--moe_skew_mult",
                "1.75",
                "--fp4_moe_op",
                "trtllm",
            ]
        )
        # Compare against freshly constructed configs rather than literals: the
        # C++ struct initializers are the single source of truth for these
        # defaults (see tests/test_warmup_bindings.py).
        for config, legacy_arity, defaults, fields in (
            (
                py_env_configs.kv_cache_config,
                54,
                KVCacheConfig(),
                ("runtime_mem_safety_ratio", "runtime_mem_no_warmup_floor_mb"),
            ),
            (
                py_env_configs.moe_config,
                12,
                MoeConfig(),
                ("moe_skew_mult", "fp4_moe_op"),
            ),
        ):
            with self.subTest(config=type(config).__name__):
                state = config.__getstate__()
                for field in fields:
                    self.assertNotEqual(
                        getattr(config, field),
                        getattr(defaults, field),
                        msg=f"{field} must differ from its default for this test to prove anything",
                    )

                legacy = type(config)()
                legacy.__setstate__(tuple(state[:legacy_arity]))

                for field in fields:
                    self.assertEqual(getattr(legacy, field), getattr(defaults, field))

    def test_runtime_tuning_args_reject_invalid_ranges(self):
        """Each rejection names its argument and echoes the offending value.

        A bare assertRaises(SystemExit) passes for any parse failure, including a
        typo in the option name, so every case also matches stderr: the argument
        name proves the right validator fired, and the echoed value proves the
        message is diagnosable without re-running with a debugger.
        """
        from rtp_llm.server.server_args.server_args import setup_args
        from rtp_llm.server.server_args.util import MAX_RUNTIME_MEMORY_MIB

        # Import the shared bound instead of re-deriving it from struct.calcsize:
        # two independent derivations of the same limit can drift apart silently.
        first_unrepresentable_mib = MAX_RUNTIME_MEMORY_MIB + 1
        for argument, value in (
            # Upper bound is exclusive, so 1.0 itself is out of range.
            ("--runtime_mem_safety_ratio", "1.0"),
            ("--runtime_mem_safety_ratio", "-0.01"),
            # math.isfinite branch: rejected before any range comparison.
            ("--runtime_mem_safety_ratio", "inf"),
            ("--runtime_mem_safety_ratio", "nan"),
            ("--runtime_mem_no_warmup_floor_mb", "-1"),
            ("--runtime_mem_no_warmup_floor_mb", str(first_unrepresentable_mib)),
            ("--moe_skew_mult", "-1"),
            ("--moe_skew_mult", "inf"),
            ("--moe_skew_mult", "nan"),
            # The bound is exclusive: exactly 1.0 degenerates to uniform routing.
            ("--moe_skew_mult", "1.0"),
            ("--reserver_runtime_mem_mb", "-1"),
        ):
            with self.subTest(argument=argument, value=value):
                with (
                    patch("sys.stderr", new_callable=StringIO) as stderr,
                    self.assertRaises(SystemExit),
                ):
                    setup_args([argument, value])

                stderr_text = stderr.getvalue()
                self.assertIn(argument, stderr_text)
                self.assertIn(repr(value), stderr_text)

    def test_runtime_tuning_args_accept_in_range_bounds(self):
        """The rejections above are worthless if the bounds also reject valid input."""
        from rtp_llm.server.server_args.server_args import setup_args
        from rtp_llm.server.server_args.util import MAX_RUNTIME_MEMORY_MIB

        py_env_configs = setup_args(
            [
                # 0 and the largest value below the exclusive upper bound.
                "--runtime_mem_safety_ratio",
                "0",
                "--runtime_mem_no_warmup_floor_mb",
                str(MAX_RUNTIME_MEMORY_MIB),
                # Just above the exclusive lower bound of 1.0.
                "--moe_skew_mult",
                "1.000001",
            ]
        )
        self.assertEqual(py_env_configs.kv_cache_config.runtime_mem_safety_ratio, 0.0)
        self.assertEqual(
            py_env_configs.kv_cache_config.runtime_mem_no_warmup_floor_mb,
            MAX_RUNTIME_MEMORY_MIB,
        )
        self.assertEqual(py_env_configs.moe_config.moe_skew_mult, 1.000001)

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
        self.assertIn("world_rank=", records[0].getMessage())

    def test_runtime_tuning_summary_warns_on_non_default_values(self):
        from rtp_llm.server.server_args.server_args import setup_args

        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertLogs(level="INFO") as logs,
        ):
            setup_args(["--moe_skew_mult", "1.75"])

        records = [
            record
            for record in logs.records
            if record.getMessage().startswith("Runtime memory tuning:")
        ]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].levelno, logging.WARNING)
        message = records[0].getMessage()
        self.assertIn("moe_skew_mult=1.75", message)
        self.assertIn("(default 2.0)", message)

    def test_provided_dest_probe_rejects_accumulating_actions(self):
        """Pin the probe constraint documented on _provided_argument_dests.

        The probe pre-seeds every dest with a sentinel object; an accumulating
        action reads that value back and tries to extend it, which cannot work.
        Anyone adding an append/extend/count argument must rework the probe
        first -- this test is the tripwire that says so.
        """
        from rtp_llm.server.server_args.server_args import EnvArgumentParser

        parser = EnvArgumentParser(description="append probe constraint test")
        parser.add_argument("--tag", action="append")

        with self.assertRaises(Exception):
            parser._provided_argument_dests(["--tag", "a"])

    def test_strict_runtime_env_dests_all_exist(self):
        """Guard the fail-fast set against a silent dest rename.

        _STRICT_RUNTIME_ENV_DESTS is matched against action.dest by string. Renaming
        an argument's dest without updating the set drops its env fail-fast with no
        error anywhere -- the set just stops matching. Assert here rather than at
        startup: this is an internal-consistency bug that should fail CI, not refuse
        to boot a production server.
        """
        from rtp_llm.server.server_args.server_args import (
            _STRICT_RUNTIME_ENV_DESTS,
            EnvArgumentParser,
            PyEnvConfigs,
            init_all_group_args,
        )

        parser = EnvArgumentParser(description="strict dest existence check")
        py_env_configs = PyEnvConfigs()
        parser.set_root_config(py_env_configs)
        init_all_group_args(parser, py_env_configs)

        registered = {action.dest for action in parser._actions}
        missing = _STRICT_RUNTIME_ENV_DESTS - registered
        self.assertEqual(
            missing,
            set(),
            msg=f"_STRICT_RUNTIME_ENV_DESTS names dests no argument registers: {sorted(missing)}",
        )

    def test_invalid_runtime_env_reports_name_and_raw_value(self):
        from rtp_llm.server.server_args.server_args import setup_args

        os.environ["MOE_SKEW_MULT"] = "not-a-float"
        with (
            patch("sys.stderr", new_callable=StringIO) as stderr,
            self.assertRaises(SystemExit),
        ):
            setup_args([])

        self.assertIn("MOE_SKEW_MULT='not-a-float'", stderr.getvalue())

    def test_equals_form_cli_overrides_invalid_strict_env(self):
        from rtp_llm.server.server_args.server_args import setup_args

        os.environ["MOE_SKEW_MULT"] = "invalid"
        for args in (["--moe_skew_mult=1.2"], None):
            with self.subTest(args=args):
                if args is None:
                    sys.argv = ["prog", "--moe_skew_mult=1.2"]
                config = setup_args(args).moe_config
                self.assertEqual(config.moe_skew_mult, 1.2)

    def test_abbreviated_cli_option_is_not_overwritten_by_env(self):
        """An abbreviation is a CLI-provided value, so the env must not fill it.

        argparse accepts any unambiguous prefix, so --moe_skew_mul sets
        moe_skew_mult. _provided_argument_dests has to agree with argparse about
        that: when it does not, the mixed-path setattr silently replaces the
        explicit CLI value with the env one (the main symptom, asserted first),
        and for _STRICT_RUNTIME_ENV_DESTS an unconvertible env aborts startup
        even though the CLI supplied a valid value (asserted second).
        """
        from rtp_llm.server.server_args.server_args import setup_args

        os.environ["MOE_SKEW_MULT"] = "3.0"
        for args in (["--moe_skew_mul", "1.75"], None):
            with self.subTest(env="valid", args=args):
                if args is None:
                    sys.argv = ["prog", "--moe_skew_mul", "1.75"]
                config = setup_args(args).moe_config
                self.assertEqual(config.moe_skew_mult, 1.75)

        os.environ["MOE_SKEW_MULT"] = "invalid"
        for args in (["--moe_skew_mul", "1.75"], None):
            with self.subTest(env="invalid", args=args):
                if args is None:
                    sys.argv = ["prog", "--moe_skew_mul", "1.75"]
                config = setup_args(args).moe_config
                self.assertEqual(config.moe_skew_mult, 1.75)

    def test_empty_string_env_value_is_preserved(self):
        from rtp_llm.server.server_args.server_args import EnvArgumentParser

        parser = EnvArgumentParser(description="empty string env test")
        parser.add_argument(
            "--string_value",
            env_name="TEST_STRING_VALUE",
            type=str,
            default="non-empty-default",
        )
        os.environ["TEST_STRING_VALUE"] = ""

        for args in ([], None):
            with self.subTest(args=args):
                sys.argv = ["prog"]
                self.assertEqual(parser.parse_args(args).string_value, "")

    def test_unconvertible_legacy_env_fallback_and_fail_fast(self):
        """Preserve the legacy mixed-path fallback and env-only fail-fast paths."""
        from rtp_llm.server.server_args.server_args import EnvArgumentParser

        parser = EnvArgumentParser(description="invalid env handling test")
        parser.add_argument(
            "--legacy_value",
            env_name="TEST_LEGACY_VALUE",
            type=int,
            default=0,
        )
        os.environ["TEST_LEGACY_VALUE"] = "not-an-int"

        with self.assertLogs(level="WARNING") as logs:
            parsed_args = parser.parse_args([])

        self.assertEqual(parsed_args.legacy_value, 0)
        output = "\n".join(logs.output)
        self.assertIn("TEST_LEGACY_VALUE='not-an-int'", output)
        self.assertIn("using parser default 0", output)

        sys.argv = ["prog"]
        with (
            patch("sys.stderr", new_callable=StringIO) as stderr,
            self.assertRaises(SystemExit),
        ):
            parser.parse_args()

        self.assertIn("TEST_LEGACY_VALUE='not-an-int'", stderr.getvalue())

    def test_argument_type_error_env_remains_fail_fast_on_mixed_path(self):
        from rtp_llm.server.server_args.server_args import EnvArgumentParser

        parser = EnvArgumentParser(description="argument type error env test")
        parser.add_argument(
            "--bool_value",
            env_name="TEST_BOOL_VALUE",
            type=str2bool,
            default=False,
        )
        os.environ["TEST_BOOL_VALUE"] = "invalid"

        with (
            patch("sys.stderr", new_callable=StringIO) as stderr,
            self.assertRaises(SystemExit),
        ):
            parser.parse_args([])

        self.assertIn("TEST_BOOL_VALUE='invalid'", stderr.getvalue())

    def test_invalid_topology_and_port_envs_fail_fast_without_cmd_args(self):
        from rtp_llm.server.server_args.server_args import setup_args

        for env_name in (
            "WORLD_SIZE",
            "TP_SIZE",
            "DP_SIZE",
            "EP_SIZE",
            "LOCAL_WORLD_SIZE",
            "START_PORT",
        ):
            with (
                self.subTest(env_name=env_name),
                patch.dict(os.environ, {env_name: "not-an-int"}, clear=True),
                patch("sys.stderr", new_callable=StringIO) as stderr,
            ):
                sys.argv = ["prog"]
                with self.assertRaises(SystemExit):
                    setup_args()
                self.assertIn(f"{env_name}='not-an-int'", stderr.getvalue())

    def test_invalid_runtime_memory_envs_fail_fast(self):
        """Only the arguments this feature introduces are strict.

        Widening the strict set would change startup behaviour for deployments
        carrying stale env values the parser has always ignored, so the
        lenient CLI-mixed path above still covers everything else.
        """
        from rtp_llm.server.server_args.server_args import setup_args

        for env_name in (
            "RUNTIME_MEM_SAFETY_RATIO",
            "RUNTIME_MEM_NO_WARMUP_FLOOR_MB",
            "MOE_SKEW_MULT",
        ):
            with (
                self.subTest(env_name=env_name),
                patch.dict(os.environ, {env_name: "not-an-int"}, clear=True),
                patch("sys.stderr", new_callable=StringIO) as stderr,
            ):
                with self.assertRaises(SystemExit):
                    setup_args([])
                self.assertIn(f"{env_name}='not-an-int'", stderr.getvalue())

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


if __name__ == "__main__":
    main()
