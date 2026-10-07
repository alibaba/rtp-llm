"""Recorder contracts without loading RTP native extensions or model weights."""

import ast
import importlib.util
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "utils/prefill_input_log.py"
spec = importlib.util.spec_from_file_location("prefill_input_log_test_subject", SOURCE)
log = importlib.util.module_from_spec(spec)
spec.loader.exec_module(log)


class PrefillInputLogTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env = mock.patch.dict(
            os.environ, MEGA_MOE_LOG_INPUTS="1", MEGA_MOE_SNAPSHOT_DIR=self.temp.name
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def rows(self):
        files = list(Path(self.temp.name).glob("prefill_ops_*.jsonl"))
        self.assertEqual(len(files), 1)
        return [json.loads(line) for line in files[0].read_text().splitlines()]

    def test_tensor_metadata_and_scalars(self):
        base = torch.arange(30).reshape(5, 6)
        view = base[1:, ::2]
        with mock.patch.object(
            torch.Tensor, "item", side_effect=AssertionError("item")
        ), mock.patch.object(
            torch.Tensor, "cpu", side_effect=AssertionError("cpu")
        ), mock.patch.object(
            torch.cuda, "synchronize", side_effect=AssertionError("sync")
        ):
            meta = log.describe_inputs(
                {"x": view, "empty": torch.empty(0), "nested": (None, True, 7, 0.5)}
            )
        self.assertEqual(meta["x"]["shape"], (4, 3))
        self.assertEqual(meta["x"]["stride"], (6, 2))
        self.assertEqual(meta["x"]["storage_offset"], 6)
        self.assertFalse(meta["x"]["contiguous"])
        self.assertEqual(meta["x"]["data_ptr"], view.data_ptr())
        self.assertEqual(meta["empty"]["numel"], 0)
        self.assertEqual(meta["nested"][2], {"type": "int", "value": 7})
        self.assertEqual(log.describe_inputs(3, typed_scalars=False), 3)

    def test_unknown_object_cycle_and_meta(self):
        class Opaque:
            def __repr__(self):
                raise AssertionError("repr must not run")

        value = [Opaque()]
        value.append(value)
        result = log.describe_inputs(value)
        self.assertTrue(result[1]["cycle"])
        self.assertIn("Opaque", result[0]["type"])
        self.assertNotIn("data_ptr", log.describe_inputs(torch.empty(2, device="meta")))

    def test_explicit_object_adapters(self):
        cache = type("LayerKVCache", (), {})()
        cache.kv_cache_base = torch.empty(2, 3)
        cache.seq_size_per_block = 64
        result = log.describe_inputs(cache)["fields"]
        self.assertEqual(result["kv_cache_base"]["shape"], (2, 3))
        self.assertEqual(result["seq_size_per_block"]["value"], 64)
        inputs = type("PyModelInputs", (), {})()
        inputs.input_ids = torch.ones(2, dtype=torch.long)
        self.assertIn("input_ids", log.describe_inputs(inputs)["fields"])

    def test_dispatch_parent_scope_and_result(self):
        x = torch.arange(6).reshape(2, 3)

        def native(value, scale):
            # Inputs must already be visible while the operator is executing.
            self.assertTrue(any(r.get("op") == "native" for r in self.rows()))
            return value * scale

        with log.prefill_input_snapshot(True):
            with log.prefill_stage("decoder", 4):
                result = log.trace_call("native", native, x, scale=2)
        self.assertTrue(torch.equal(result, x * 2))
        rows = self.rows()
        call = next(r for r in rows if r["event"] == "call" and r["op"] == "native")
        aten = next(
            r for r in rows if r["event"] == "call" and r["backend"] == "torch_dispatch"
        )
        self.assertEqual(aten["parent_call_id"], call["call_id"])
        self.assertEqual(aten["layer_id"], 4)
        self.assertEqual(rows[-1]["state"], "python_forward_returned")
        self.assertIsNone(log._CURRENT.get())

    def test_scalar_defaults_and_non_string_dictionary_keys(self):
        metadata = log.describe_inputs({(1, 2): "first", (3, 4): "second"})
        self.assertEqual(len(metadata["items"]), 2)
        with log.prefill_input_snapshot(True):
            log.trace_call("defaults", lambda value=7: value)
            torch.add(torch.ones(1), 2)
        calls = [row for row in self.rows() if row["event"] == "call"]
        default = next(row for row in calls if row["op"] == "defaults")
        self.assertEqual(default["inputs"]["value"]["value"], 7)
        add = next(row for row in calls if row["op"] == "aten.add.Tensor")
        self.assertEqual(add["inputs"]["alpha"]["value"], 1)

    def test_disabled_and_non_prefill(self):
        for flag, phase in [("0", True), ("1", False), ("true", True)]:
            with mock.patch.dict(os.environ, MEGA_MOE_LOG_INPUTS=flag):
                with log.prefill_input_snapshot(phase):
                    self.assertEqual(log.trace_call("add", lambda x: x + 1, 1), 2)
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_exception_preserved_and_inputs_flushed(self):
        error = ValueError("bad shape")

        def bad(x):
            self.assertEqual(self.rows()[-1]["event"], "call")
            raise error

        with self.assertRaises(ValueError) as caught:
            with log.prefill_input_snapshot(True):
                log.trace_call("bad", bad, torch.empty(3))
        self.assertIs(caught.exception, error)
        self.assertEqual(self.rows()[-1]["state"], "python_forward_failed")
        self.assertIsNone(log._CURRENT.get())
        self.assertIsNone(log._PARENT.get())

    def test_replaces_previous_prefill_and_keeps_all_calls(self):
        for count in (3, 5):
            with log.prefill_input_snapshot(True):
                for i in range(count):
                    log.trace_call("chunk", lambda x: x, i)
            calls = [r for r in self.rows() if r["event"] == "call"]
            self.assertEqual(len(calls), count)
        self.assertEqual(len({r["prefill_sequence"] for r in self.rows()}), 1)

    def test_more_than_eight_mib_not_dropped(self):
        with log.prefill_input_snapshot(True):
            for i in range(9):
                log.trace_call("large", lambda x: x, "a" * (1024 * 1024))
        path = next(Path(self.temp.name).glob("*.jsonl"))
        self.assertGreater(path.stat().st_size, 8 * 1024 * 1024)
        self.assertEqual(len([r for r in self.rows() if r["event"] == "call"]), 9)

    def test_write_failure_does_not_change_inference(self):
        with mock.patch.object(
            log._Recorder, "disable", autospec=True, wraps=None
        ) as disabled:
            # A directory creation failure disables only diagnostic output.
            disabled.side_effect = lambda recorder: setattr(recorder, "enabled", False)
            with mock.patch.object(Path, "mkdir", side_effect=OSError("read only")):
                with log.prefill_input_snapshot(True):
                    result = log.trace_call("ok", lambda: 42)
        self.assertEqual(result, 42)
        self.assertEqual(disabled.call_count, 1)

    def test_write_failure_once_and_exception_preserved(self):
        with log.prefill_input_snapshot(True):
            recorder = log._CURRENT.get()
            with mock.patch.object(
                recorder.output, "write", side_effect=OSError("full")
            ), mock.patch.object(log.logging, "exception") as report:
                self.assertEqual(log.trace_call("ok", lambda: 5), 5)
                self.assertEqual(log.trace_call("ok", lambda: 6), 6)
                self.assertEqual(report.call_count, 1)

    def test_rank_and_pid_isolation(self):
        with mock.patch.object(
            log.dist, "is_initialized", return_value=True
        ), mock.patch.object(log.dist, "is_available", return_value=True):
            for rank, pid in [(0, 101), (1, 101), (1, 102)]:
                with mock.patch.object(
                    log.dist, "get_rank", return_value=rank
                ), mock.patch.object(log.os, "getpid", return_value=pid):
                    with log.prefill_input_snapshot(True):
                        pass
        self.assertEqual(len(list(Path(self.temp.name).glob("*.jsonl"))), 3)

    def test_triton_grid_not_evaluated_extra_times(self):
        grids = []
        launches = []

        class Kernel:
            arg_names = ["x", "BLOCK"]

            def __getitem__(self, grid):
                def launch(x, BLOCK=8):
                    resolved = grid({"BLOCK": BLOCK}) if callable(grid) else grid
                    launches.append((resolved, x, BLOCK))
                    return "compiled_kernel"

                return launch

        def grid(meta):
            grids.append(meta)
            return (meta["BLOCK"],)

        kernel = Kernel()
        with log.prefill_input_snapshot(True):
            result = log.trace_triton("kernel", kernel, grid, torch.empty(2), BLOCK=16)
        self.assertEqual(result, "compiled_kernel")
        self.assertEqual(len(grids), 1)
        self.assertEqual(launches[0][0], (16,))
        self.assertTrue(any(r["event"] == "triton_grid" for r in self.rows()))

    def test_qwen35_real_forward_scope_including_embedding(self):
        # Execute the real model's forward methods with tiny CPU layer fixtures.
        # No model weights/native extensions are needed to test the scope boundary.
        source = (ROOT / "model_desc/qwen3_next.py").read_text()
        tree = ast.parse(source)
        classes = []
        for name, bases in [
            ("Qwen3NextModel", []),
            ("Qwen35Model", [ast.Name(id="Qwen3NextModel", ctx=ast.Load())]),
        ]:
            cls = next(
                n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name
            )
            methods = [
                n
                for n in cls.body
                if isinstance(n, ast.FunctionDef)
                and n.name in ("forward", "word_embedding")
            ]
            classes.append(
                ast.ClassDef(
                    name=name, bases=bases, keywords=[], body=methods, decorator_list=[]
                )
            )
        ns = dict(
            torch=torch,
            trace_call=log.trace_call,
            prefill_stage=log.prefill_stage,
            prefill_input_snapshot=log.prefill_input_snapshot,
            get_primary_attention_inputs=lambda inputs, cache: inputs.attention_inputs,
            select_attention_inputs_for_layer=lambda inputs, cache, i: inputs.attention_inputs,
            select_fmha_impl_for_layer=lambda *args: None,
            Qwen3NextMetadata=lambda **kw: types.SimpleNamespace(**kw),
            prepare_causal_conv1d_metadata=lambda **kw: None,
            HybridAttentionType=types.SimpleNamespace(LINEAR="linear"),
            PyModelOutputs=lambda x: x,
            mega_moe_prefill_snapshot=lambda flag: __import__(
                "contextlib"
            ).nullcontext(),
        )
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                )
            ]
            + classes,
            type_ignores=[],
        )
        exec(
            compile(
                ast.fix_missing_locations(module),
                str(ROOT / "model_desc/qwen3_next.py"),
                "exec",
            ),
            ns,
        )
        model = ns["Qwen35Model"]()
        model.kv_cache = None
        model.parallelism_config = types.SimpleNamespace(
            prefill_cp_config=types.SimpleNamespace(is_enabled=lambda: False)
        )
        model.embed_tokens = lambda ids, *args: torch.nn.functional.embedding(
            ids, torch.arange(32.0).reshape(8, 4)
        )
        model.multimodal_embedding_injector = lambda emb, features, locs: emb + features

        class Layer:
            layer_type = "linear"

            def __call__(self, hidden, residual, *args, **kwargs):
                return hidden * 2, residual + 1

        model.layers = [Layer(), Layer()]
        model.norm = lambda h, r: (h + r, r)
        model.prepare_fmha_impl = lambda inputs: None
        inputs = types.SimpleNamespace(
            input_ids=torch.tensor([1, 2]),
            combo_position_ids=None,
            embedding_inputs=types.SimpleNamespace(
                combo_tokens_type_ids=None, text_tokens_mask=None
            ),
            multimodal_inputs=types.SimpleNamespace(
                multimodal_features=torch.ones(2, 4), mm_features_locs=None
            ),
            attention_inputs=types.SimpleNamespace(
                is_prefill=True, is_target_verify=False, cu_seqlens_device=None
            ),
        )
        fusion = types.ModuleType("prefill_fusion")
        fusion.METADATA = "unused"
        fusion.enabled = lambda _: False
        fusion.gdn_prefill_backend = lambda _: "native"
        with mock.patch.dict(
            "sys.modules",
            {"rtp_llm.models_py.triton_kernels.common.prefill_fusion": fusion},
        ):
            with mock.patch.dict(os.environ, MEGA_MOE_LOG_INPUTS="0"):
                reference = model.forward(inputs)
            actual = model.forward(inputs)
            self.assertTrue(torch.equal(reference, actual))
            rows = self.rows()
            stages = {r["stage"] for r in rows}
            self.assertTrue({"embedding", "prepare", "decoder", "final_norm"} <= stages)
            self.assertEqual(
                {r["layer_id"] for r in rows if r["stage"] == "decoder"}, {0, 1}
            )
            previous = next(Path(self.temp.name).glob("*.jsonl")).read_bytes()
            for prefill, verify in [(False, False), (True, True)]:
                inputs.attention_inputs.is_prefill = prefill
                inputs.attention_inputs.is_target_verify = verify
                model.forward(inputs)
                self.assertEqual(
                    next(Path(self.temp.name).glob("*.jsonl")).read_bytes(), previous
                )


if __name__ == "__main__":
    unittest.main()
