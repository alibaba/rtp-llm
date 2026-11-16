"""Regression checks for the shared deployment and black-box acceptance contract."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import framework_smoke as smoke
import result_check
import server_launcher
from text_smoke import Case


class FrameworkSmokeTest(unittest.TestCase):
    def test_cache_answers_require_retrieving_the_prefix(self):
        args = SimpleNamespace(
            base_url="http://127.0.0.1:1",
            decode_health_url="http://127.0.0.1:2/health",
            decode_role_addrs=[],
            namespace="prefix",
            chunk_tokens=65536,
        )
        runner = smoke.FrameworkRunner(args)
        with patch.object(
            runner,
            "fit_prompt",
            side_effect=lambda head, tail, count, **kw: (
                head + " x" * 100 + tail,
                [0] * count,
            ),
        ):
            case = runner.fixture("cache", 1023, value="CEDAR-1234")
        self.assertEqual(
            result_check.answer_from_question(case.prompt), case.expected_json
        )
        self.assertNotIn("CEDAR-1234", case.prompt.splitlines()[-1])
        branch = runner.partial_prefix(case)
        self.assertEqual(
            result_check.answer_from_question(branch.prompt), branch.expected_json
        )
        self.assertEqual(branch.expected_json, {"value": "MAPLE"})
        self.assertEqual(branch.prompt.count("检索 key="), 1)
        self.assertNotIn("MAPLE", branch.prompt.splitlines()[-1])
        prefix = case.prompt[: -len(smoke.VALUE_QUERY.format(key="cedar"))]
        self.assertTrue(branch.prompt.startswith(prefix))

    def test_draft_rounds_alone_do_not_certify_accepted_tokens(self):
        args = SimpleNamespace(
            base_url="http://127.0.0.1:1",
            decode_health_url="http://127.0.0.1:2/health",
            decode_role_addrs=[],
            block_size=1024,
        )
        runner = smoke.FrameworkRunner(args)
        case = Case("copy", "copy", ".", "any", require_mtp=True)
        response = dict(
            choices=[dict(finish_reason="stop", message={"content": "V"})],
            aux_info=dict(
                pd_sep=True,
                input_len=10,
                output_len=12,
                iter_count=4,
                speculative_draft_rounds=4,
                speculative_accepted_tokens_per_pos=[0, 0, 0],
            ),
            debug_info={"output_ids": [[1] * 12]},
        )
        with self.assertRaisesRegex(smoke.SmokeFailure, "no accepted draft token"):
            runner.validate(case, response, 0.1, 128)
        response["aux_info"]["speculative_accepted_tokens_per_pos"] = [3, 2, 1]
        row = runner.validate(case, response, 0.1, 128)
        self.assertEqual(row["mtp_accepted_tokens"], 6)

    def test_bounded_deployments_keep_graph_and_mtp_without_runtime_hooks(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "config.json").write_text('{"num_hidden_layers":93}')
            for model in ("kimi_k3", "kimi_k31"):
                for role, tp, dp in (
                    ("PREFILL", 8, 1),
                    ("DECODE", 8, 1),
                    ("DECODE", 4, 2),
                ):
                    args = SimpleNamespace(
                        checkpoint=checkpoint,
                        draft_checkpoint=checkpoint,
                        role=role,
                        tp_size=tp,
                        dp_size=dp,
                        ep_size=8,
                        start_port=27188,
                        peer_port=28188,
                        peer_ip="127.0.0.1",
                        server=sys.executable,
                        model_type=model,
                        orthogonal_smoke=True,
                        compact_smoke=True,
                        fp8_gemm=True,
                        fp8_kv_cache=True,
                    )
                    env, argv = server_launcher.launch_config(args)
                    options = dict(zip(argv[1::2], argv[2::2]))
                    self.assertEqual(env["MODEL_TYPE"], model)
                    self.assertEqual(env["SP_MODEL_TYPE"], model + "_mtp")
                    self.assertEqual(env["GEN_NUM_PER_CIRCLE"], "3")
                    self.assertEqual(options["--seq_size_per_block"], "1024")
                    self.assertEqual(options["--max_context_batch_size"], "4")
                    if role == "DECODE":
                        self.assertEqual(options["--decode_capture_config"], "1,2,4")
                        self.assertEqual(options["--enable_cuda_graph"], "1")
                    else:
                        self.assertEqual(options["--reuse_cache_tail_blocks"], "5")
                    override = json.loads(options["--json_model_override_args"])
                    self.assertEqual(
                        override.get("kda_nvfp4", False), model == "kimi_k31"
                    )
                    self.assertFalse(
                        any(
                            "SMOKE_EVIDENCE" in key
                            or key.startswith("DSA_DCP_")
                            or key == "PYTHONPATH"
                            for key in env
                        )
                    )

    def fixture(self):
        roles = [{"http_port": 28000}, {"http_port": 28036}]
        rows = [
            dict(
                name=name,
                phase="formal",
                pd_sep=True,
                finish_reason="stop",
                content='{"value":"V"}',
                expected_json={"value": "V"},
                output_len=12,
                input_len=smoke.CHUNK_INPUT_TOKENS if name == "chunked_cold" else 16384,
                effective_reuse_len=0,
                expected_reuse="miss",
                decode_owner_rank=0,
                selected_decode_role_addr=roles[0],
                observed_decode_role_addrs=[roles[0]],
            )
            for name in smoke.CASES
        ]
        for row in rows:
            if row["name"] == "cross_owner_reuse":
                row.update(
                    decode_owner_rank=1,
                    selected_decode_role_addr=roles[1],
                    observed_decode_role_addrs=[roles[1]],
                )
            if row["name"] in ("cache_device", "cache_host"):
                tier = "device" if row["name"] == "cache_device" else "memory"
                row.update(
                    effective_reuse_len=8192,
                    expected_reuse="hit",
                    prefill_cache_tiers=dict(
                        total=8192, local=8192, device=0, memory=0, disk=0
                    ),
                )
                row["prefill_cache_tiers"][tier] = 8192
        members = {name: [name] for name in smoke.STAGES}
        members.update(
            cache_pressure=[f"pressure_{i}" for i in range(3)],
            mixed_cache=["mixed_cached", "mixed_partial", "mixed_recent", "mixed_miss"],
            decode_refill=[f"decode_{i}" for i in range(4)],
        )
        return dict(
            model_type="kimi_k31",
            cases=rows,
            block_size=1024,
            max_inflight=4,
            decode_role_addrs=roles,
            formal_request_retries=0,
            stages=[
                dict(
                    name=name,
                    case_names=names,
                    passed=True,
                    transport="http",
                    concurrent=False,
                )
                for name, names in members.items()
            ],
            cancellation=dict(
                client_disconnected=True,
                request={"stream": True},
                first_content_event={"choices": [{"delta": {"content": "0"}}]},
            ),
        )

    def check_fixture(self, result):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "accuracy.json"
            (path.parent / "cancellation.json").write_text(
                json.dumps(result["cancellation"])
            )
            with patch.object(
                result_check, "audit_raw_answers", return_value=[]
            ), patch.object(
                result_check, "_raw_consistency", return_value=[]
            ), patch.object(
                result_check, "_check_protocol", return_value=[]
            ):
                return result_check._check_compact(result, path, {"errors": []})

    def test_acceptance_rejects_missing_coverage_and_forced_batches(self):
        baseline = self.fixture()
        self.assertTrue(self.check_fixture(baseline)["passed"])
        for mutation in (
            lambda result: result["stages"].pop(),
            lambda result: result["stages"][0].update(transport="pd_group_rpc"),
            lambda result: result.update(max_inflight=64),
            lambda result: next(
                row for row in result["cases"] if row["name"] == "cross_owner_reuse"
            ).update(
                decode_owner_rank=0,
                selected_decode_role_addr=result["decode_role_addrs"][0],
            ),
            lambda result: result["cancellation"].update(first_content_event={}),
            lambda result: next(
                row for row in result["cases"] if row["name"] == "cache_host"
            )["prefill_cache_tiers"].update(memory=0, device=8192),
        ):
            result = copy.deepcopy(baseline)
            mutation(result)
            self.assertFalse(self.check_fixture(result)["passed"])

    def test_dp1_keeps_cache_and_protocol_requirements_without_cross_owner_case(self):
        result = self.fixture()
        result["decode_role_addrs"] = result["decode_role_addrs"][:1]
        result["cases"] = [
            row for row in result["cases"] if row["name"] != "cross_owner_reuse"
        ]
        result["stages"] = [
            stage for stage in result["stages"] if stage["name"] != "cross_owner_reuse"
        ]
        for row in result["cases"]:
            row.update(
                decode_owner_rank=0,
                selected_decode_role_addr=result["decode_role_addrs"][0],
                observed_decode_role_addrs=result["decode_role_addrs"],
            )
        self.assertTrue(self.check_fixture(result)["passed"])
        invalid_owner = copy.deepcopy(result)
        invalid_owner["cases"][0]["decode_owner_rank"] = 1
        self.assertFalse(self.check_fixture(invalid_owner)["passed"])
        no_host_hit = copy.deepcopy(result)
        next(row for row in no_host_hit["cases"] if row["name"] == "cache_host")[
            "prefill_cache_tiers"
        ].update(memory=0, device=8192)
        self.assertFalse(self.check_fixture(no_host_hit)["passed"])


if __name__ == "__main__":
    unittest.main()
