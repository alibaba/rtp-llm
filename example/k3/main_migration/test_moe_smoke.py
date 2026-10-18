"""CPU checks for SiTU evidence and the protocol smoke's complete CLI audit."""

import ast
import contextlib
import copy
import io
import json
import logging
import os
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

from example.k3.main_migration import audit_orthogonal_smoke
from example.k3.main_migration.moe_smoke import audit_situ, check_situ


def situ_record():
    return {
        "activation": "situ",
        "backend": "deep_gemm.fp8_fp4_mega_moe",
        "gate_beta": 4.0,
        "up_beta": 25.0,
        "kwargs": {"activation_beta": 4.0, "activation_linear_beta": 25.0},
    }


class MoeSmokeTest(unittest.TestCase):
    def test_production_evidence_is_opt_in_and_preserves_actual_kwargs(self):
        source = (
            pathlib.Path(__file__).resolve().parents[3]
            / "rtp_llm/models_py/modules/factory/fused_moe/impl/cuda/executors/mega_moe.py"
        )
        tree = ast.parse(source.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_record_moe_activation_evidence"
        )
        namespace = {"os": os, "logging": logging, "json": json}
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        record_evidence = namespace[function.name]

        def actual_kernel():
            pass

        for value in (None, "0", "true", " 1", "1"):
            with self.subTest(value=value), patch.dict(os.environ, {}, clear=True):
                if value is not None:
                    os.environ["RTP_LLM_SMOKE_EVIDENCE"] = value
                with patch.object(logging, "info") as logged:
                    for path in ("mega_moe", "k3_fused_shared"):
                        kwargs = {
                            "activation_beta": 4.0,
                            "activation_linear_beta": None,
                        }
                        record_evidence(
                            actual_kernel, "situ", 4.0, None, kwargs, 7, path
                        )
                    record_evidence(
                        actual_kernel, "swiglu", None, None, {}, 7, "mega_moe"
                    )
                self.assertEqual(logged.call_count, 2 if value == "1" else 0)
                for call in logged.call_args_list:
                    self.assertEqual(call.args[0], "[RTP_MOE_ACTIVATION] %s")
                    evidence = json.loads(call.args[1])
                    self.assertEqual(
                        evidence["backend"], f"{actual_kernel.__module__}.actual_kernel"
                    )
                    self.assertEqual(evidence["layer_id"], 7)
                    self.assertIn(
                        evidence["execution_path"], ("mega_moe", "k3_fused_shared")
                    )
                    self.assertEqual(evidence["kwargs"], kwargs)
                    check_situ(evidence)

    def test_actual_keyword_mapping_preserves_independent_scales(self):
        check_situ(situ_record())
        record = situ_record()
        record["up_beta"] = None
        record["kwargs"]["activation_linear_beta"] = None
        check_situ(record)

    def test_swapped_dropped_or_legacy_scales_fail(self):
        for kwargs in (
            {"activation_beta": 25.0, "activation_linear_beta": 4.0},
            {"activation_beta": 4.0},
            {"activation_alpha": 4.0, "activation_beta": 25.0},
            {"situ_beta": 4.0, "situ_linear_beta": 25.0},
        ):
            with self.subTest(kwargs=kwargs):
                record = situ_record()
                record["kwargs"] = kwargs
                with self.assertRaises(AssertionError):
                    check_situ(record)

    def test_invalid_scales_and_backend_fail(self):
        for field, value in (
            ("gate_beta", 0),
            ("gate_beta", float("nan")),
            ("gate_beta", True),
            ("up_beta", -1),
            ("up_beta", float("inf")),
            ("backend", ""),
        ):
            with self.subTest(field=field, value=value):
                record = situ_record()
                record[field] = value
                with self.assertRaises(AssertionError):
                    check_situ(record)

    def _logs(self, root):
        prefill, decode = root / "prefill", root / "decode"
        for directory, ranks in ((prefill, 4), (decode, 8)):
            directory.mkdir()
            for rank in range(ranks):
                (directory / f"main_{rank}.log").write_text(
                    "[INFO] [RTP_MOE_ACTIVATION] " + json.dumps(situ_record()) + "\n"
                )
        return prefill, decode

    def test_asymmetric_ranks_require_real_evidence_for_each_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefill, decode = self._logs(pathlib.Path(tmp))
            result = audit_situ(prefill, decode, prefill_ranks=4)
            self.assertEqual(len(result), 12)
            (decode / "main_7.log").write_text("")
            with self.assertRaisesRegex(AssertionError, "decode rank 7"):
                audit_situ(prefill, decode, prefill_ranks=4)

    def test_malformed_event_reports_its_log_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefill, decode = self._logs(pathlib.Path(tmp))
            (prefill / "main_0.log").write_text("[RTP_MOE_ACTIVATION] broken\n")
            with self.assertRaisesRegex(AssertionError, "main_0.log:1"):
                audit_situ(prefill, decode, prefill_ranks=4)

    def test_protocol_cli_reads_situ_helper_and_fails_closed(self):
        # Isolate the already-tested path/runtime auditors while exercising
        # main()'s real helper import, all-rank log parsing and result writing.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            prefill, decode = self._logs(root)
            result_path, output = root / "result.json", root / "audit.json"
            result_path.write_text(
                json.dumps({"protocol_requests": [{"passed": True}]})
            )
            argv = [
                "audit_orthogonal_smoke.py",
                "--result",
                str(result_path),
                "--prefill-log-dir",
                str(prefill),
                "--decode-log-dir",
                str(decode),
                "--prefill-ranks",
                "4",
                "--decode-dp",
                "1",
                "--output",
                str(output),
            ]
            for mode in ("complete", "missing_event", "missing_log"):
                with self.subTest(mode=mode):
                    if mode == "missing_event":
                        (decode / "main_7.log").write_text("")
                    elif mode == "missing_log":
                        (decode / "main_7.log").unlink()
                    if output.exists():
                        output.unlink()
                    verdict = {"passed": True, "errors": []}
                    with (
                        patch.object(sys, "argv", argv),
                        patch.object(
                            audit_orthogonal_smoke,
                            "audit",
                            return_value=copy.deepcopy(verdict),
                        ),
                        patch.object(
                            audit_orthogonal_smoke, "read_events", return_value={}
                        ),
                        patch(
                            "runtime_smoke.audit_request_runtime",
                            return_value={"passed": True},
                        ),
                        contextlib.redirect_stdout(io.StringIO()),
                    ):
                        status = audit_orthogonal_smoke.main()
                    saved = json.loads(output.read_text())
                    self.assertEqual(status, 0 if mode == "complete" else 1)
                    self.assertEqual(saved["passed"], mode == "complete")
                    if mode == "complete":
                        self.assertEqual(len(saved["situ_configuration"]), 12)
                    else:
                        self.assertTrue(saved["errors"])


if __name__ == "__main__":
    unittest.main()
