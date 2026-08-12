import importlib
import io
import json
import logging
import os
import pickle
import sys
from unittest import TestCase, main
from unittest.mock import patch

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
        # MOE_PURE_TP_PRESHARD=false must override the True default.
        self.assertFalse(py_env_configs.load_config.moe_pure_tp_preshard)
        # Note: max_seq_len is in ModelConfig, not RuntimeConfig or EngineConfig
        # It will be set when ModelConfig is created from model_args

    def test_runtime_tuning_args_are_bound_to_configs(self):
        """CLI values bind to the configs and win over a conflicting environment."""
        from rtp_llm.server.server_args.server_args import setup_args

        for cli_args in (
            [
                "--runtime_mem_safety_ratio",
                "0.08",
                "--runtime_mem_no_warmup_floor_mb",
                "3072",
                "--moe_skew_mult",
                "1.75",
            ],
            [
                "--runtime_mem_safety_ratio=0.08",
                "--runtime_mem_no_warmup_floor_mb=3072",
                "--moe_skew_mult=1.75",
            ],
            [
                "--runtime_mem_safety=0.08",
                "--runtime_mem_no_warmup_f=3072",
                "--moe_skew_m=1.75",
            ],
        ):
            with (
                self.subTest(cli_args=cli_args),
                patch.dict(
                    os.environ,
                    {
                        "RUNTIME_MEM_SAFETY_RATIO": "0.5",
                        "RUNTIME_MEM_NO_WARMUP_FLOOR_MB": "9999",
                        "MOE_SKEW_MULT": "3.0",
                    },
                    clear=True,
                ),
            ):
                py_env_configs = setup_args(cli_args)

            # CLI wins over the environment values above (provided_args precedence).
            self.assertEqual(
                py_env_configs.kv_cache_config.runtime_mem_safety_ratio, 0.08
            )
            self.assertEqual(
                py_env_configs.kv_cache_config.runtime_mem_no_warmup_floor_mb, 3072
            )
            self.assertEqual(py_env_configs.moe_config.moe_skew_mult, 1.75)

    def test_equals_cli_in_sys_argv_wins_over_environment(self):
        from rtp_llm.server.server_args.server_args import setup_args

        sys.argv = ["prog", "--moe_skew_mult=1.75"]
        with patch.dict(os.environ, {"MOE_SKEW_MULT": "3.0"}, clear=True):
            py_env_configs = setup_args()

        self.assertEqual(py_env_configs.moe_config.moe_skew_mult, 1.75)

    def test_invalid_sensitive_env_value_is_redacted(self):
        from rtp_llm.server.server_args.server_args import EnvArgumentParser
        from rtp_llm.server.server_args.util import bounded_int

        for dest, env_name in (
            ("api_token", "API_TOKEN"),
            ("openai_api_key", "OPENAI_API_KEY"),
        ):
            with self.subTest(dest=dest):
                parser = EnvArgumentParser()
                parser.add_argument(f"--{dest}", env_name=env_name, type=bounded_int)
                secret = "short-secret-value"
                with (
                    patch.dict(os.environ, {env_name: secret}, clear=True),
                    patch("sys.stderr", new_callable=io.StringIO) as stderr,
                    self.assertRaises(SystemExit),
                ):
                    parser.parse_args([])

                self.assertNotIn(secret, stderr.getvalue())
                self.assertIn("<redacted:18 chars>", stderr.getvalue())

    def test_ordinary_token_arguments_are_not_treated_as_secrets(self):
        from rtp_llm.server.server_args.server_args import EnvArgumentParser

        for dest in ("tokenizer_path", "max_batch_tokens_size"):
            with self.subTest(dest=dest):
                self.assertFalse(EnvArgumentParser._is_sensitive_dest(dest))
                self.assertEqual(
                    EnvArgumentParser._shown_env_value(dest, "visible-value"),
                    "visible-value",
                )

    def test_lone_dash_positional_does_not_crash_cli_precedence_scan(self):
        from rtp_llm.server.server_args.server_args import EnvArgumentParser

        parser = EnvArgumentParser()
        parser.add_argument("input")

        self.assertEqual(parser.parse_args(["-"]).input, "-")

    def test_empty_string_env_value_keeps_legacy_semantics_on_both_paths(self):
        from rtp_llm.server.server_args.server_args import EnvArgumentParser

        for env_only in (True, False):
            with self.subTest(env_only=env_only):
                parser = EnvArgumentParser()
                parser.add_argument("--label", env_name="LABEL", default="default")
                sys.argv = ["prog"]
                with patch.dict(os.environ, {"LABEL": ""}, clear=True):
                    parsed = parser.parse_args() if env_only else parser.parse_args([])

                self.assertEqual(parsed.label, "")

    def test_env_arg_generator_does_not_emit_converter_objects(self):
        from rtp_llm.server.server_args.generate_args_from_env_clean import (
            generate_args_list,
        )

        grpc_json = '{"client_config": {}, "server_config": {}}'
        with patch.dict(
            os.environ,
            {
                "CP_ROTATE_METHOD": "ALL_GATHER",
                "GRPC_CONFIG_JSON": grpc_json,
            },
            clear=True,
        ):
            generated = generate_args_list(only_env_vars=True)

        # These converters return enum/pybind objects. The legacy generator does
        # not emit string-valued options, so it must not stringify those objects
        # into argv values that the same converter cannot parse on the next pass.
        self.assertNotIn("--cp_rotate_method", generated)
        self.assertNotIn("--grpc_config_json", generated)

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
        converted by hand after argparse has run. The three warmup knobs use bounded_* converters,
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
            ("MOE_SKEW_MULT", "0.99"),
            ("MOE_SKEW_MULT", "inf"),
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
                "MOE_SKEW_MULT": "1.75",
            },
            clear=True,
        ):
            py_env_configs = setup_args()

        self.assertEqual(py_env_configs.kv_cache_config.runtime_mem_safety_ratio, 0.08)
        self.assertEqual(
            py_env_configs.kv_cache_config.runtime_mem_no_warmup_floor_mb, 3072
        )
        self.assertEqual(py_env_configs.moe_config.moe_skew_mult, 1.75)

        # Out-of-range values abort on this path as well, via argparse's own conversion.
        for env_name, bad_value in (
            ("RUNTIME_MEM_SAFETY_RATIO", "1.0"),
            ("MOE_SKEW_MULT", "0.99"),
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

    def test_skew_rollback_value_parses_on_both_paths(self):
        """moe_skew_mult=1.0 is the documented "disable the skew" rollback value.

        It is a legal value rather than a rejected one, so it does not fall out of the range tests.
        Pinning it here means a rollback knob that stopped parsing would fail in CI instead of at
        the moment an operator needs it.
        """
        from rtp_llm.server.server_args.server_args import setup_args

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                setup_args(["--moe_skew_mult", "1.0"]).moe_config.moe_skew_mult, 1.0
            )

        sys.argv = ["prog"]
        with patch.dict(os.environ, {"MOE_SKEW_MULT": "1.0"}, clear=True):
            self.assertEqual(setup_args().moe_config.moe_skew_mult, 1.0)

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

        config = setup_args(["--moe_skew_mult", "1.75"]).moe_config
        self.assertEqual(len(config.__getstate__()), 13, msg=_ARITY_TRIPWIRE_MSG)
        restored = pickle.loads(pickle.dumps(config))

        self.assertEqual(restored.moe_skew_mult, 1.75)

    def test_moe_config_pickle_state_covers_every_to_string_field(self):
        """Field-level tripwire: the arity pins above cannot see the failure mode that
        actually happened -- a field added to the struct and to_string() but never to
        the pickle tuple (fp4_moe_op). Cross-checking the to_string() field inventory
        against __getstate__() makes that mode fail loudly: adding a struct field
        without extending the ConfigInit.cc make_tuple/__setstate__ indices breaks
        this equation.

        MoeConfig only: its to_string() lists every struct field, so the inventory is
        authoritative. KVCacheConfig's to_string() covers 36 of its 56 pickled fields,
        so the same cross-check would be vacuous there; it keeps the arity pin only.
        """
        from rtp_llm.ops import MoeConfig

        config = MoeConfig()
        to_string_fields = [
            line.split(":", 1)[0]
            for line in config.to_string().splitlines()
            if ":" in line
        ]
        # Parse sanity: the inventory must be non-trivial and duplicate-free, or the
        # arithmetic below stops meaning anything.
        self.assertGreaterEqual(len(to_string_fields), 14)
        self.assertEqual(len(to_string_fields), len(set(to_string_fields)))

        # Tracked serialization gap, not endorsed: fp4_moe_op predates this tripwire and was never
        # added to the pickle tuple. Current workers reparse configuration and do not consume this
        # pickled field, but a future direct consumer would silently get the struct default. Fixing
        # it touches the ConfigInit.cc tuple/setter and this whitelist in one commit.
        known_unpickled = {"fp4_moe_op"}
        self.assertLessEqual(known_unpickled, set(to_string_fields))

        self.assertEqual(
            len(config.__getstate__()),
            len(to_string_fields) - len(known_unpickled),
            msg=(
                "MoeConfig pickle state no longer matches its to_string() field "
                "inventory. A new struct field must be added to MoeConfig::to_string() "
                "AND to the __getstate__ make_tuple + __setstate__ indices in "
                "rtp_llm/cpp/pybind/ConfigInit.cc in the same commit. A direct pickle "
                "consumer otherwise silently receives the field default. Known "
                "un-pickled field(s): "
                f"{sorted(known_unpickled)}."
            ),
        )

    def test_legacy_state_arities_keep_config_defaults(self):
        """Exercise the shorter tuples __setstate__ still accepts.

        ConfigInit.cc keeps them so a state pickled before these fields existed
        still loads, with the appended fields falling back to their config
        defaults. The round-trip tests above only ever feed the current arity, so
        without this the compatibility branches are never executed.

        Truncating the live __getstate__() also pins the field *order*, which the
        arity assertions do not: the C++ side reads the appended fields at fixed
        indices (t[54]/t[55], t[12]), so moving a field earlier in
        __getstate__ makes these unpickle into the wrong slots and fail here.
        """
        from rtp_llm.ops import KVCacheConfig, MoeConfig

        kv_config = KVCacheConfig()
        kv_config.runtime_mem_safety_ratio = 0.08
        kv_config.runtime_mem_no_warmup_floor_mb = 3072
        moe_config = MoeConfig()
        moe_config.moe_skew_mult = 1.75
        # Compare against freshly constructed configs rather than literals: the
        # C++ struct initializers are the single source of truth for these
        # defaults (see tests/test_warmup_bindings.py).
        for config, legacy_arity, defaults, fields in (
            (
                kv_config,
                54,
                KVCacheConfig(),
                ("runtime_mem_safety_ratio", "runtime_mem_no_warmup_floor_mb"),
            ),
            (
                moe_config,
                12,
                MoeConfig(),
                ("moe_skew_mult",),
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

                # pybind's pickle setter constructs into an uninitialized instance. Calling
                # __setstate__ on type(config)() does not exercise that path and can leave this
                # compatibility test passing without running the C++ arity branch.
                legacy = type(config).__new__(type(config))
                legacy.__setstate__(tuple(state[:legacy_arity]))

                for field in fields:
                    self.assertEqual(getattr(legacy, field), getattr(defaults, field))

                # Self-prove that the C++ setter ran: one shorter than the accepted legacy state
                # must hit ConfigInit.cc's Invalid state guard for both config types.
                invalid = type(config).__new__(type(config))
                with self.assertRaisesRegex(RuntimeError, "Invalid state"):
                    invalid.__setstate__(tuple(state[: legacy_arity - 1]))

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

    def test_moe_skew_mult_default_anchor(self):
        """Pin the declared default behind the PREFILL KV-cache change.

        Not a compat anchor -- the opposite: 2.0 (kDefaultMoeSkewMult in
        ConfigModules.h) means every PD PREFILL deployment running a traced warmup
        reserves for a rank 0 carrying twice the mean load, shrinking the whole
        cluster's KV cache via the min reduction. Every other test compares against
        a freshly constructed MoeConfig, so this literal assertion is the only thing
        that fails when the value drifts, which is the point: changing it changes
        released behaviour and belongs in the release note.
        """
        from rtp_llm.ops import MoeConfig

        self.assertEqual(
            MoeConfig().moe_skew_mult,
            2.0,
            msg=(
                "declared default: PREFILL warmup skews rank 0 to 2x the mean load; "
                "changing it requires updating the release note and rollout guidance"
            ),
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
            (
                ["--moe_skew_mult", "1.75"],
                ("moe_skew_mult=1.75", "(default 2.0)"),
            ),
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
