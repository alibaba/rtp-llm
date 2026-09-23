"""CPU contracts for the retained exact-input target, not retired A/B ladders."""

import ast
import os
import typing
import unittest
from pathlib import Path

PERF = Path(__file__).resolve().parent


def _resolve_build_envs(build_text):
    tree = ast.parse(build_text)
    assignments = {}
    target_env_exprs = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            assignments[node.targets[0].id] = node.value
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Name) and call.func.id == "py_test":
                keywords = {
                    kw.arg: kw.value for kw in call.keywords if kw.arg is not None
                }
                name = keywords.get("name")
                env = keywords.get("env")
                if (
                    isinstance(name, ast.Constant)
                    and isinstance(name.value, str)
                    and env is not None
                ):
                    target_env_exprs[name.value] = env

    resolving = set()

    def resolve(expr):
        if isinstance(expr, ast.Name):
            if expr.id in resolving:
                raise AssertionError("cyclic BUILD env reference: " + expr.id)
            resolving.add(expr.id)
            try:
                return resolve(assignments[expr.id])
            finally:
                resolving.remove(expr.id)
        if isinstance(expr, ast.Dict):
            return {resolve(k): resolve(v) for k, v in zip(expr.keys, expr.values)}
        if isinstance(expr, ast.Constant):
            return expr.value
        if (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Name)
            and expr.func.id == "dict"
        ):
            if len(expr.args) > 1:
                raise AssertionError("unexpected dict positional arguments")
            out = {} if not expr.args else dict(resolve(expr.args[0]))
            for kw in expr.keywords:
                if kw.arg is None:
                    out.update(resolve(kw.value))
                else:
                    out[kw.arg] = resolve(kw.value)
            return out
        raise AssertionError("unsupported BUILD env expression: " + ast.dump(expr))

    return {name: resolve(expr) for name, expr in target_env_exprs.items()}


class ReferenceProfileTest(unittest.TestCase):
    def test_exact_recipe_cannot_inherit_hostile_selectors(self):
        targets = _resolve_build_envs((PERF / "BUILD").read_text())
        profile = targets["v4_flash_sm120_cep4pp2_32k_reference"]
        expected = {
            "DSV4_MOE_PREQUANT_INPUT": "1",
            "DSV4_MOE_PREQUANT_INPUT_REQUIRED": "1",
            "DSV4_MOE_EXTENT_COUNT_FUSE": "1",
            "DSV4_MOE_FORWARD_COUNT_PLAN": "1",
            "DSV4_NCCL_EP_MXFP8_DISPATCH_PACK": "1",
            "DSV4_CP_COMPACT_COMPRESSOR": "1",
            "DSV4_CP_COMPACT_TRANSPORT_FUSION": "1",
            "DSV4_MOE_DEVICE_META": "1",
            "DSV4_MOE_LOCAL_REPLAY": "1",
            "DSV4_MOE_LOCAL_REPLAY_REQUIRED": "1",
            "DSV4_MOE_ACTIVE_SCALE_FUSION": "1",
            "DSV4_MOE_ACTIVE_SCALE_FUSION_REQUIRED": "1",
            "DSV4_MOE_FP4_BACKEND": "flashinfer",
            "DSV4_MHC_PRE_GEMM_BACKEND": "tilelang_single",
            "DSV4_M1_HOST_META": "0",
            "DSV4_MOE_LOCAL_REPLAY_EXCHANGE": "0",
            "DSV4_SHARED_EXPERT_REPLAY": "0",
            "DSV4_SM120_SHARED_EXPERT_NATIVE_DG": "0",
            "DSV4_FWD_PROFILE": "0",
            "PERF_MEASURE_RUNS": "8",
            "PERF_GRID_WARMUP_RUNS": "2",
            "RTP_LLM_PERF_UNSET_TPSYNC": "1",
            "RTP_LLM_PP_ROUND_DEVICE_SYNC": "1",
            "RTP_LLM_PP_ROUND_FWD_EVENT_SYNC": "1",
        }
        hostile = {key: "hostile" for key in expected}
        hostile.update(profile)
        self.assertEqual({key: hostile[key] for key in expected}, expected)
        self.assertNotIn("PERF_REQUIRED_TRACE", profile)
        self.assertNotIn("DSV4_FWD_PROFILE_RANK", profile)
        self.assertNotIn("DSV4_DEEPGEMM_SHADOW_PATH", profile)
        self.assertNotIn("RTP_LLM_LOG_TPSYNC", profile)

    def test_clean_grid_is_prefill_only(self):
        source = (PERF / "BUILD").read_text()
        tree = ast.parse(source)
        args = next(
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "_CEP4PP2_PREFILL_ARGS"
                for t in node.targets
            )
        )
        options = dict(zip(args[::2], args[1::2]))
        self.assertEqual(options["--partial"], "2")
        self.assertEqual(options["--batch_size"], "1")
        self.assertEqual(options["--decode_test_length"], "1")
        self.assertEqual(options["--use_batch_decode_scheduler"], "0")
        self.assertFalse(
            {"--dataset", "--dataset_name", "--dataset_path"} & options.keys()
        )
        targets = _resolve_build_envs(source)
        env = targets["v4_flash_sm120_cep4pp2_32k_prefill"]
        self.assertEqual(env, targets["v4_flash_sm120_cep4pp2_32k_reference"])
        self.assertEqual(env["DSV4_FWD_PROFILE"], "0")
        self.assertEqual(env["PERF_PROFILE_RUNS"], "0")
        self.assertNotIn("PERF_REQUIRED_TRACE", env)
        target = next(
            n.value
            for n in tree.body
            if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call)
            and any(
                k.arg == "name"
                and isinstance(k.value, ast.Constant)
                and k.value.value == "v4_flash_sm120_cep4pp2_32k_prefill"
                for k in n.value.keywords
            )
        )
        kw = {k.arg: k.value for k in target.keywords}
        self.assertEqual(ast.literal_eval(kw["main"]), "batch_decode_test.py")
        self.assertEqual(kw["args"].id, "_CEP4PP2_PREFILL_ARGS")

    def test_timing_runner_unsets_presence_sensitive_logger(self):
        tree = ast.parse((PERF / "batch_decode_test.py").read_text())
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_unset_tpsync_for_timing"
        )
        ns = {"os": os, "MutableMapping": typing.MutableMapping}
        exec(
            compile(
                ast.Module(body=[node], type_ignores=[]),
                str(PERF / "batch_decode_test.py"),
                "exec",
            ),
            ns,
        )
        env = {
            "RTP_LLM_PERF_UNSET_TPSYNC": "1",
            "RTP_LLM_LOG_TPSYNC": "0",
            "OTHER": "keep",
        }
        ns[node.name](env)
        self.assertEqual(env, {"OTHER": "keep"})
        env = {"RTP_LLM_PERF_UNSET_TPSYNC": "0", "RTP_LLM_LOG_TPSYNC": "anything"}
        ns[node.name](env)
        self.assertEqual(env["RTP_LLM_LOG_TPSYNC"], "anything")


if __name__ == "__main__":
    unittest.main()
