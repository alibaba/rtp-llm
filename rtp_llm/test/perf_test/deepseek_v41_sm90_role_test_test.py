import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rtp_llm.test.perf_test.deepseek_v41_sm90_role_test import (
    LENGTHS,
    configure_jit_cache,
    recipe,
    route_for_worker,
    validate_responses,
    validate_startup,
)


class FixedRoleTest(unittest.TestCase):
    def test_jit_paths_are_shared_across_bazel_targets_without_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            origin = root / "wheel/site-packages/humming/__init__.py"
            origin.parent.mkdir(parents=True)
            origin.touch()
            results = []
            for target in ("smoke", "perf"):
                alias = root / target / "humming/__init__.py"
                alias.parent.mkdir(parents=True)
                alias.symlink_to(origin)
                with mock.patch(
                    "importlib.util.find_spec",
                    return_value=SimpleNamespace(origin=str(alias)),
                ), mock.patch.object(sys, "path", []), mock.patch.dict(
                    os.environ,
                    {"PYTORCH_KERNEL_CACHE_PATH": str(root / "torch")},
                    clear=True,
                ):
                    results.append(configure_jit_cache())
                    self.assertEqual(sys.path[0], str(origin.parent.parent))
                    self.assertEqual(os.environ["PYTHONPATH"], sys.path[0])
                    self.assertTrue((root / "torch").is_dir())
                    self.assertNotIn("HOME", os.environ)
            self.assertEqual(results[0], results[1])

    def test_prefill_is_cp8_not_eight_independent_requests(self):
        for mode in ("smoke", "perf"):
            options, env, batches, output = recipe(
                "prefill", mode, "/model", "/tokenizer"
            )
            self.assertEqual((options["tp_size"], options["ep_size"]), (8, 8))
            self.assertEqual(options["cp_rotate_method"], "ALL_GATHER")
            self.assertEqual(options["use_all_gather"], 0)
            self.assertEqual(env["DSV41_SM90_MOE_COMM"], "v2")
            self.assertEqual(env["DSV41_SM90_DECODE_GRAPH"], "0")
            self.assertEqual((batches, output, LENGTHS), ((1,), 1, (32768, 65536)))

    def test_decode_dp8_capacity_matches_fixed_batches(self):
        for mode in ("smoke", "perf"):
            options, env, batches, output = recipe(
                "decode", mode, "/model", "/tokenizer"
            )
            self.assertEqual((options["tp_size"], options["ep_size"]), (1, 8))
            self.assertNotIn("cp_rotate_method", options)
            self.assertEqual(int(env["DSV41_SM90_DECODE_GRAPH_MAX_ROWS"]), max(batches))
            self.assertEqual(env["DSV41_SM90_DECODE_GRAPH"], "1")
            self.assertEqual(env["ROLE_TYPE"], "DECODE")
            self.assertGreater(output, 1)

    def test_each_response_must_succeed_and_match_actual_lengths(self):
        def response(**changes):
            fields = dict(
                success=True,
                input_len=32768,
                output_len=32,
                prefill_time=1.0,
                total_time=2.0,
                wait_time=0.0,
                decode_time_per_token=0.1,
            )
            fields.update(changes)
            return SimpleNamespace(**fields)

        validate_responses([response()], 1, 32768, 32)
        for bad in (
            {"success": False},
            {"input_len": 32767},
            {"output_len": 1},
            {"total_time": float("nan")},
            {"prefill_time": -1.0},
            {"decode_time_per_token": -0.1},
        ):
            with self.subTest(bad=bad), self.assertRaises(AssertionError):
                validate_responses([response(**bad)], 1, 32768, 32)
        with self.assertRaises(AssertionError):
            validate_responses([], 1, 32768, 32)

    def test_decode_route_uses_direct_generate_endpoint(self):
        worker = {"address": "127.0.0.1:9001", "dp_rank": 3}
        route = route_for_worker(worker, 8000, "decode")
        self.assertEqual(
            route,
            {
                "role": "PDFUSION",
                "ip": "127.0.0.1",
                "http_port": 8000,
                "grpc_port": 9001,
            },
        )
        self.assertEqual(route_for_worker(worker, 8000, "prefill")["role"], "PREFILL")

    def test_startup_rejects_dp8_prefill_and_legacy(self):
        def text(role, dp, cp):
            return "\n".join(
                [
                    f"parallelism: world=8 tp=0/1 ep={rank}/8 dp={rank % dp}/{dp}"
                    for rank in range(8)
                ]
                + [f"cp_enabled={cp > 1} cp_size={cp}", f"SM90 DeepEP V2: role={role}"]
                + (
                    [f"CP={cp}: max_tokens_per_rank 66560 -> 8320"] * 8
                    if cp > 1
                    else []
                )
            )

        validate_startup(text("prefill", 1, 8), "prefill")
        validate_startup(
            text("prefill", 1, 8).replace("cp_enabled=True cp_size=8", ""), "prefill"
        )
        validate_startup(text("decode", 8, 1), "decode")
        for bad in (
            text("prefill", 8, 1),
            text("prefill", 1, 1),
            text("prefill", 1, 8).replace("DeepEP V2", "legacy"),
            text("prefill", 1, 8).replace("cp_enabled=True", "cp_enabled=False"),
        ):
            with self.subTest(bad=bad), self.assertRaises(AssertionError):
                validate_startup(bad, "prefill")


if __name__ == "__main__":
    unittest.main()
