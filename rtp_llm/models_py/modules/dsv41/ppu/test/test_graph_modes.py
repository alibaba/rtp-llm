"""Pure source checks for the isolated N9 launcher and receipt contract."""

import ast
import copy
import importlib.util
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace


PUBLIC = Path(__file__).resolve().parents[6]
DRIVER = Path(__file__).resolve().parents[1] / "tools" / "driver_graph_modes.py"
SPEC = importlib.util.spec_from_file_location("n9_receipt_driver", DRIVER)
DRIVER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER_MODULE)


def simulated_receipts():
    # Synthetic records exercise only the offline receiver. Device receipt
    # generation is exclusively in the native runner path.
    records = []
    for rank in range(8):
        for step in (1, 2):
            live = rank == 0
            records.append({
                "rank": rank, "replay": step, "schema": "n9-native-grouped-decode-replay-v1",
                "valid": [live], "positions": [256 + step if live else -1],
                "swa_slots": [[step + 1 if live else -1]],
                "token_ids": [23], "engram_windows": [[23, step, -1, -1]],
                "domains": {str(i): {"nonzero_bytes": int(live)} for i in range(51)},
                "reserved_unchanged": [True] * 51,
                "layer_calls": [step] * 40,
                "addresses": {"ids": 41 + rank},
                "received": [[6 if rank == 2 else 0]] * 40,
            })
    return records


def receive(records):
    return DRIVER_MODULE.validate_native_receipts([
        SimpleNamespace(read_text=lambda r=record: json.dumps(r)) for record in records
    ])


class NativeReceiptContract(unittest.TestCase):
    def test_actual_expert_owner_is_independent_of_request_owner(self):
        result = receive(simulated_receipts())
        self.assertEqual(result["owner"], 0)
        self.assertEqual(result["received_per_layer_per_step"], 6)

    def test_stale_slot_window_or_position_rejected(self):
        for key in ("positions", "swa_slots", "engram_windows"):
            with self.subTest(key=key):
                records = simulated_receipts()
                records[1][key] = records[0][key]
                with self.assertRaises(ValueError):
                    receive(records)

    def test_idle_write_and_dummy_page_mutation_rejected(self):
        for key, value in (("swa_slots", [[2]]), ("reserved_unchanged", [False] * 51)):
            with self.subTest(key=key):
                records = simulated_receipts()
                records[2][key] = value
                with self.assertRaises(ValueError):
                    receive(records)

    def test_missing_compute_or_pointer_change_rejected(self):
        for key in ("layer_calls", "addresses", "received"):
            with self.subTest(key=key):
                records = simulated_receipts()
                records[1][key] = copy.deepcopy(records[0][key])
                if key == "addresses":
                    records[1][key]["ids"] += 1
                if key == "received":
                    records[1][key][0][0] += 1
                with self.assertRaises(ValueError):
                    receive(records)

    def test_replay_identity_set_rejected_if_incomplete_or_duplicated(self):
        records = simulated_receipts()
        for incomplete in (records[:-1], records + records[:1]):
            with self.assertRaises(ValueError):
                receive(incomplete)


class SourceBoundaries(unittest.TestCase):
    def test_shared_commit_layout_and_serving_gate(self):
        commit = (PUBLIC / "rtp_llm/cpp/normal_engine/speculative/MtpBatchStreamProcessor.cc").read_text()
        runner = (PUBLIC / "rtp_llm/cpp/cuda_graph/cuda_graph_runner.cc").read_text()
        tail = commit.split("void MtpBatchStreamProcessor::updateDecodePostDSparkCommitInput(", 1)[1]
        tail = tail.split("void MtpBatchStreamProcessor::updateDecodePostDraftModelInput(", 1)[0]
        self.assertNotIn("USE_PPU", tail)
        self.assertNotIn("accepted_rows", tail)
        self.assertIn("model_input.last_hidden_states = target_features;", tail)
        self.assertIn("state.seq_len_sum != expected_tokens", runner)

    def test_new_switches_are_opt_in(self):
        launch = ast.parse((PUBLIC / "rtp_llm/tools/v41/launch.py").read_text())
        preset = next(node for node in launch.body if isinstance(node, ast.FunctionDef) and node.name == "preset_env")
        args = {a.arg: d for a, d in zip(preset.args.kwonlyargs, preset.args.kw_defaults)}
        for name in ("shared_grouped_graph", "shared_graph_async"):
            self.assertIs(args[name].value, False)

    def test_real_parser_keeps_graph_opt_in_and_dspark_isolation(self):
        predecessor = Path(__file__).with_name("test_dspark_modes.py")
        spec = importlib.util.spec_from_file_location("n9_predecessor_helpers", predecessor)
        helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helpers)
        helpers.load_light(
            "rtp_llm.models_py.modules.dsv41.ppu.moe_precision",
            helpers.PPU / "moe_precision.py",
        )
        runtime = helpers.load_light("n9_runtime_config", PUBLIC / "rtp_llm/config/v41_runtime.py")
        tree = ast.parse((PUBLIC / "rtp_llm/tools/v41/launch.py").read_text())
        selected = ast.Module(body=[
            node for node in tree.body if isinstance(node, ast.FunctionDef)
            and node.name in {"preset_env", "bounded_prefill_wait_timeout_ms"}
        ], type_ignores=[])
        namespace = dict(os=os, json=json, V41RuntimeConfig=runtime.V41RuntimeConfig,
                         PPU_STRATEGY=runtime.PPU_STRATEGY, env_bool=runtime.env_bool,
                         MEGAMOE_DSPARK_ASYNC_FLAGS=runtime.MEGAMOE_DSPARK_ASYNC_FLAGS,
                         DEFAULT_PREFILL_WAIT_TIMEOUT_MS=1800000, MAX_PREFILL_WAIT_TIMEOUT_MS=86400000)
        exec(compile(selected, "launch.py", "exec"), namespace)
        preset = namespace["preset_env"]
        ordinary = preset("decode", "/weights", parent={"RTP_V41_PPU_SHARED_GROUPED_GRAPH": "1"})
        self.assertFalse(runtime.V41RuntimeConfig.from_env(ordinary).cuda_graph)
        for asynchronous in (False, True):
            env = preset("decode", "/weights", parent={"GENERATION_PREFILL_CAPTURE_CONFIG": "512"},
                         shared_grouped_graph=True, shared_graph_async=asynchronous)
            self.assertTrue(runtime.V41RuntimeConfig.from_env(env).shared_grouped_graph)
            self.assertEqual(env["RTP_LLM_STREAM_ASYNC"], str(int(asynchronous)))
            self.assertEqual(env["RTP_LLM_DEVICE_INPUT"], str(int(asynchronous)))
            self.assertNotIn("GENERATION_PREFILL_CAPTURE_CONFIG", env)
        for role, options in (
            ("prefill", {"shared_grouped_graph": True}),
            ("decode", {"shared_graph_async": True}),
        ):
            with self.subTest(role=role, options=options), self.assertRaises(ValueError):
                preset(role, "/weights", parent={}, **options)
        env = preset("decode", "/weights", parent={},
                     shared_grouped_graph=True, dspark_gamma=3)
        self.assertEqual(runtime.V41RuntimeConfig.from_env(env).speculative, "DSPARK")
        env = preset("decode", "/weights", parent={"RTP_V41_PPU_MOE_BACKEND": "megamoe"},
                     shared_decode_graph=True)
        self.assertEqual(runtime.V41RuntimeConfig.from_env(env).moe_precision, "w4a4")
        self.assertEqual(env["RTP_LLM_STREAM_ASYNC"], "0")
        env = preset(
            "decode", "/weights", shared_decode_graph=True, dspark_gamma=3,
            parent={"RTP_V41_PPU_MOE_BACKEND": "megamoe", "RTP_LLM_MTP_ASYNC_PREPARE": "1"},
        )
        self.assertEqual(runtime.V41RuntimeConfig.from_env(env).speculative, "DSPARK")
        self.assertEqual(env["RTP_LLM_MTP_ASYNC_PREPARE"], "0")

    def test_fixed_ep_forward_has_no_dynamic_host_sizes(self):
        src = DRIVER.parents[1] / "graph_moe.py"
        tree = ast.parse(src.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        forward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
        attrs = {node.attr for node in ast.walk(forward) if isinstance(node, ast.Attribute)}
        self.assertFalse(attrs & {"item", "tolist", "cpu", "synchronize", "nonzero", "dispatch", "_forward_deepep"})
        self.assertTrue({"all_gather_into_tensor", "all_reduce", "gemm"} <= attrs)


if __name__ == "__main__":
    unittest.main()
