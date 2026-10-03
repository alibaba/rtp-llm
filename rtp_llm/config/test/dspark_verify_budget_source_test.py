"""Source/parser gates and isolated compiled native config tests; no GPU calls."""

import argparse
import ast
import contextlib
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]


def source(relative):
    return (ROOT / relative).read_text()


def parser_namespace():
    """Execute real parser classes without importing RTP/torch/native extensions."""
    tree = ast.parse(source("rtp_llm/server/server_args/server_args.py"))
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, (ast.Import, ast.ImportFrom))
            and not getattr(node, "module", "").startswith("rtp_llm")
        )
        or isinstance(node, ast.ClassDef)
    ]
    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "parser_source", "exec"), ns)
    tree = ast.parse(
        source("rtp_llm/server/server_args/speculative_decoding_group_args.py")
    )
    nodes = [node for node in tree.body if not isinstance(node, ast.ImportFrom)]
    ns["str2bool"] = lambda value: str(value).lower() in ("1", "true")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "group_source", "exec"), ns)
    return ns


class VerifyBudgetSourceTest(unittest.TestCase):
    def parse(self, env, args):
        ns = parser_namespace()
        config = SimpleNamespace()
        parser = ns["EnvArgumentParser"]()
        parser.set_root_config(config)
        ns["init_speculative_decoding_group_args"](parser, config)
        with patch.dict(os.environ, env, clear=True), patch.object(
            sys, "argv", ["test"]
        ):
            parser.parse_args(args)
        return config

    def test_actual_parser_default_env_cli(self):
        self.assertEqual(self.parse({}, []).sp_dspark_verify_tokens, 0)
        self.assertFalse(self.parse({}, []).sp_dspark_adaptive_verify)
        self.assertEqual(self.parse({}, []).sp_dspark_verify_mode, "")
        for budget in (0, 1, 3, 7):
            config = self.parse({"SP_DSPARK_VERIFY_TOKENS": str(budget)}, None)
            self.assertEqual(config.sp_dspark_verify_tokens, budget)
        config = self.parse(
            {"SP_DSPARK_VERIFY_TOKENS": "3"}, ["--sp_dspark_verify_tokens", "1"]
        )
        self.assertEqual(config.sp_dspark_verify_tokens, 1)
        config = self.parse(
            {"SP_DSPARK_VERIFY_TOKENS": "3"}, ["--gen_num_per_cycle", "7"]
        )
        self.assertEqual(
            (config.sp_dspark_verify_tokens, config.gen_num_per_cycle), (3, 7)
        )
        config = self.parse({"SP_DSPARK_ADAPTIVE_VERIFY": "1"}, [])
        self.assertTrue(config.sp_dspark_adaptive_verify)

    def test_verify_mode_parser(self):
        for mode in ("", "static", "adaptive"):
            self.assertEqual(
                self.parse({"SP_DSPARK_VERIFY_MODE": mode}, []).sp_dspark_verify_mode,
                mode,
            )
            self.assertEqual(
                self.parse({}, ["--sp_dspark_verify_mode", mode]).sp_dspark_verify_mode,
                mode,
            )
        self.assertEqual(
            self.parse(
                {"SP_DSPARK_VERIFY_MODE": "static"},
                ["--sp_dspark_verify_mode", "adaptive"],
            ).sp_dspark_verify_mode,
            "adaptive",
        )
        for mode in ("bad", "STATIC", " adaptive"):
            for args in (None, [], ["--gen_num_per_cycle", "7"]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
                    (SystemExit, argparse.ArgumentTypeError)
                ):
                    self.parse({"SP_DSPARK_VERIFY_MODE": mode}, args)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
                (SystemExit, argparse.ArgumentTypeError)
            ):
                self.parse({}, ["--sp_dspark_verify_mode", mode])

    def test_malformed_env_never_silently_defaults_with_cli(self):
        for invalid in ("bad", "1.5", str(1 << 70)):
            for args in (None, [], ["--gen_num_per_cycle", "7"]):
                with self.subTest(invalid=invalid, args=args):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
                        (SystemExit, argparse.ArgumentTypeError)
                    ):
                        self.parse({"SP_DSPARK_VERIFY_TOKENS": invalid}, args)

    def test_range_values_reach_native_validator_unchanged(self):
        for value in (-1, 8):
            self.assertEqual(
                self.parse(
                    {}, ["--sp_dspark_verify_tokens", str(value)]
                ).sp_dspark_verify_tokens,
                value,
            )

    def test_native_validation_and_pickle_wiring_source(self):
        cpp = source("rtp_llm/cpp/config/ConfigModules.cc")
        method = cpp.split(
            "bool SpeculativeExecutionConfig::isAdaptiveVerify() const {", 1
        )[1].split("\nstd::string", 1)[0]
        for guard in (
            "type != SP_TYPE_DSPARK",
            "sp_dspark_verify_tokens != 0",
            "gen_num_per_cycle <= 0",
            "sp_dspark_verify_tokens < 0 || sp_dspark_verify_tokens > gen_num_per_cycle",
        ):
            self.assertIn(guard, method)
        self.assertIn(
            "return adaptive ? gen_num_per_cycle : verifyBudgetPerRequest();", method
        )
        self.assertIn('sp_dspark_verify_mode == "static"', method)
        self.assertIn(
            'sp_dspark_verify_mode == "adaptive" || sp_dspark_adaptive_verify', method
        )
        self.assertIn(
            "int64_t SpeculativeExecutionConfig::verifyBudgetPerRequest() const", cpp
        )
        bindings = source("rtp_llm/cpp/pybind/ConfigInit.cc")
        self.assertIn(
            '.def("verifySteps", &SpeculativeExecutionConfig::verifySteps)', bindings
        )
        self.assertIn(
            '.def("verifyBudgetPerRequest", &SpeculativeExecutionConfig::verifyBudgetPerRequest)',
            bindings,
        )
        self.assertIn("if (t.size() < 10 || t.size() > 17)", bindings)
        self.assertIn("if (t.size() >= 14)", bindings)
        self.assertIn("if (t.size() >= 15)", bindings)
        self.assertIn("if (t.size() >= 16)", bindings)
        self.assertIn(
            "c.sp_dspark_verify_tokens = t[next_field++].cast<int64_t>();", bindings
        )
        self.assertIn(
            "c.sp_dspark_adaptive_verify = t[next_field++].cast<bool>();", bindings
        )
        self.assertIn(
            "c.sp_dspark_verify_mode = t[next_field].cast<std::string>();", bindings
        )
        self.assertIn("c.verifySteps();", bindings)

    @unittest.skipUnless(shutil.which("g++"), "requires host C++ compiler")
    def test_compiled_config_mode_and_legacy_matrix(self):
        # Compile the actual declarations and methods in isolation. This tests
        # native config semantics, but does not build pybind or the executor.
        header = source("rtp_llm/cpp/config/ConfigModules.h")
        declarations = header.split("enum SpeculativeType", 1)[1].split(
            "struct VitConfig", 1
        )[0]
        cpp = source("rtp_llm/cpp/config/ConfigModules.cc")
        methods = (
            "bool SpeculativeExecutionConfig::isAdaptiveVerify() const {"
            + cpp.split(
                "bool SpeculativeExecutionConfig::isAdaptiveVerify() const {", 1
            )[1].split("std::string SpeculativeExecutionConfig::to_string() const", 1)[
                0
            ]
        )
        program = (
            "#include <cstdint>\n#include <string>\n#include <stdexcept>\n#include <cassert>\n"
            + "enum SpeculativeType"
            + declarations
            + methods
            + r"""
int main() {
    for (int gamma : {4, 7}) {
        SpeculativeExecutionConfig c;
        c.type = SP_TYPE_DSPARK;
        c.gen_num_per_cycle = gamma;
        c.sp_dspark_verify_mode = "adaptive";
        int limit = gamma == 7 ? 146 : 256;
        c.validateVerifyBatchSize(0);
        c.validateVerifyBatchSize(limit);
        for (int batch : {-1, limit + 1, 2147483647}) {
            bool threw = false;
            try { c.validateVerifyBatchSize(batch); }
            catch (const std::invalid_argument&) { threw = true; }
            assert(threw);
        }
        c.sp_dspark_verify_mode = "static";
        c.validateVerifyBatchSize(2147483647);
        c.sp_dspark_verify_mode = "";
        c.type = SP_TYPE_MTP;
        c.validateVerifyBatchSize(2147483647);
    }
    for (int type = 0; type <= 6; ++type) {
        for (int gamma : {-1, 0, 1, 7}) {
            for (int budget : {-1, 0, 1, 3, 7, 8}) {
                for (std::string mode : {"", "static", "adaptive", "invalid"}) {
                    for (bool legacy : {false, true}) {
                        SpeculativeExecutionConfig c;
                        c.type = static_cast<SpeculativeType>(type);
                        c.gen_num_per_cycle = gamma;
                        c.sp_dspark_verify_tokens = budget;
                        c.sp_dspark_adaptive_verify = legacy;
                        c.sp_dspark_verify_mode = mode;
                        bool valid = mode != "invalid";
                        if (type != 6) valid &= budget == 0 && !legacy && mode.empty();
                        else {
                            valid &= gamma > 0 && budget >= 0 && budget <= gamma;
                            if (mode == "static") valid &= !legacy && (budget == 0 || budget == gamma);
                        }
                        bool adaptive = type == 6 && (mode == "adaptive" || (mode.empty() && legacy));
                        int expected_budget = budget == 0 ? gamma : budget;
                        for (int method = 0; method < 3; ++method) {
                            bool threw = false;
                            try {
                                if (method == 0) assert(c.isAdaptiveVerify() == adaptive);
                                if (method == 1) assert(c.verifySteps() == (adaptive ? gamma : expected_budget));
                                if (method == 2) assert(c.verifyBudgetPerRequest() == expected_budget);
                            } catch (const std::invalid_argument&) { threw = true; }
                            assert(threw == !valid);
                        }
                    }
                }
            }
        }
    }
}
"""
        )
        with tempfile.TemporaryDirectory(prefix="dspark_verify_config_") as temp:
            binary = str(Path(temp) / "config_test")
            subprocess.run(
                ["g++", "-std=c++17", "-x", "c++", "-", "-o", binary],
                input=program,
                text=True,
                check=True,
                capture_output=True,
            )
            subprocess.run([binary], check=True, capture_output=True)

    def test_graph_role_geometry_source_contract(self):
        header = re.sub(r"\s+", " ", source("rtp_llm/cpp/models/PyWrappedModel.h"))
        self.assertIn(
            "const int64_t verify_steps = params.sp_config.verifySteps();", header
        )
        self.assertIn(
            "const bool adaptive_dspark = params.sp_config.isAdaptiveVerify();", header
        )
        self.assertIn(
            "adaptive_dspark ? params.sp_config.verifyBudgetPerRequest() : verify_steps",
            header,
        )
        self.assertIn(
            "params.sp_config.gen_num_per_cycle + static_cast<int>(!params.sp_config.sp_dspark_sample_from_anchor)",
            header,
        )
        self.assertIn("DSparkModelRole::COMMIT) {", header)
        self.assertIn(
            "graph_params.num_tokens_per_bs = compact_verify_steps + 1;", header
        )
        self.assertIn(
            "(params.model_id ? params.sp_config.gen_num_per_cycle : compact_verify_steps) + 1",
            header,
        )
        self.assertIn(
            "graph_params.is_ragged_target_verify = adaptive_dspark && is_target_verify_decode;",
            header,
        )
        self.assertIn("graph_params.require_exact_decode_geometry = true;", header)
        self.assertIn(
            "graph_params.sp_steps = (dspark_model_role_ == DSparkModelRole::PROPOSE || (params.model_id && dspark_model_role_ != DSparkModelRole::COMMIT)) ? params.sp_config.gen_num_per_cycle : verify_steps;",
            header,
        )
        runner = source("rtp_llm/cpp/cuda_graph/cuda_graph_runner.cc")
        self.assertGreaterEqual(
            runner.count("is_ragged_target_verify = is_ragged_target_verify_"), 2
        )
        # Explicit formula matrix, tied to the source branches above. This is
        # NOT execution of PyWrappedModel or evidence of actual graph replay.
        for batch in (1, 3, 16):
            for budget, rows in ((0, 8), (1, 2), (3, 4), (7, 8)):
                effective = 7 if budget == 0 else budget
                self.assertEqual(batch * (effective + 1), batch * rows)
                self.assertEqual(batch * (7 + int(False)), batch * 7)
                self.assertEqual(batch * (7 + int(True)), batch * 8)

    def test_confidence_execution_uses_effective_mode(self):
        executor = source("rtp_llm/cpp/normal_engine/speculative/MtpExecutor.cc")
        self.assertIn(
            "dspark_adaptive_verify_        = is_dspark_ && params.sp_config.isAdaptiveVerify();",
            executor,
        )
        self.assertRegex(
            executor,
            r"if \(dspark_adaptive_verify_\) \{\s+const int64_t confidence_features",
        )
        self.assertRegex(
            executor,
            r'if \(dspark_adaptive_verify_\) \{\s+RTP_LLM_PROFILE_SCOPE\("executor.mtp.decode_step\(dspark_confidence_plan\)"\);\s+auto confidence',
        )

    def test_startup_validation_precedes_loading(self):
        setup = source("rtp_llm/config/server_config_setup.py").split(
            "def setup_and_configure_server", 1
        )[1]
        self.assertLess(
            setup.index("sp_config.verifySteps()"),
            setup.index("fetch_model_files_to_local(py_env_configs)"),
        )
        self.assertLess(
            setup.index("sp_config.validateVerifyBatchSize("),
            setup.index("fetch_model_files_to_local(py_env_configs)"),
        )
        executor = source("rtp_llm/cpp/normal_engine/speculative/MtpExecutor.cc")
        self.assertIn(
            "params.sp_config.validateVerifyBatchSize(params.runtime_config.max_generate_batch_size);",
            executor,
        )
        bindings = source("rtp_llm/cpp/pybind/ConfigInit.cc")
        self.assertIn(
            '.def("validateVerifyBatchSize", &SpeculativeExecutionConfig::validateVerifyBatchSize)',
            bindings,
        )
        factory = source("rtp_llm/model_factory.py").split("def from_model_configs", 1)[
            1
        ]
        self.assertLess(
            factory.index("sp_config.verifySteps()"),
            factory.index("model = ModelFactory._create_model("),
        )


if __name__ == "__main__":
    unittest.main()
