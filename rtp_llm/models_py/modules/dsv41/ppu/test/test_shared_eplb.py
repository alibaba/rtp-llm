"""Stdlib contracts; real migration/continuation is the native EPLB target.

These checks can run without importing Torch or any native binding. They do
not count as device smoke evidence.
"""

import ast
import dataclasses
import importlib.util
import os
import struct
import sys
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch


def _root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "rtp_llm/config/v41_runtime.py").is_file():
            return parent
    raise RuntimeError("source runfiles are required")


ROOT = _root()
PPU = ROOT / "rtp_llm/models_py/modules/dsv41/ppu"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ADAPTER = _load("_n8_storage_contract", PPU / "shared_eplb.py")
RUNTIME = _load("_n8_config_contract", ROOT / "rtp_llm/config/v41_runtime.py")


class SharedEplbContracts(unittest.TestCase):
    def test_logical_target_and_draft_are_independent_of_physical_slots(self):
        for logical in (384, 128):
            self.assertEqual(ADAPTER.physical_expert_geometry(logical, logical, 8), (logical, logical // 8))
            self.assertEqual(ADAPTER.physical_expert_geometry(logical, logical + 8, 8), (logical + 8, logical // 8 + 1))
            self.assertEqual(ADAPTER.physical_expert_geometry(logical, logical + 32, 8), (logical + 32, logical // 8 + 4))
        for values in ((384, 383, 8), (384, 385, 8), (384, 776, 8), (128, 136, 0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ADAPTER.physical_expert_geometry(*values)

    def test_transposed_and_padded_expert_views_cover_exact_storage_bytes(self):
        # Three experts. Six visible uint16 elements occupy a padded16-byte
        # expert row. Transposing inner dimensions must keep that same row.
        expected = (3, 16, 4)
        self.assertEqual(ADAPTER.expert_storage_geometry((3, 2, 3), (8, 3, 1), 2, 2, 52, 3), expected)
        self.assertEqual(ADAPTER.expert_storage_geometry((3, 3, 2), (8, 1, 3), 2, 2, 52, 3), expected)
        # Independent byte labels verify offsets and preserved stride padding.
        storage = bytes(range(52))
        rows = [storage[4 + expert * 16:4 + (expert + 1) * 16] for expert in range(3)]
        self.assertEqual(rows[2], bytes(range(36, 52)))
        self.assertEqual(b"".join(rows), storage[4:])

    def test_overlapping_or_incomplete_storage_is_rejected(self):
        for shape, stride, size in [((3, 2, 3), (5, 3, 1), 52),
                                    ((3, 2, 3), (8, 1, 1), 52),
                                    ((3, 2, 3), (8, 3, 1), 51),
                                    ((2, 2, 3), (8, 3, 1), 52),
                                    ((3, 2, 3), (8, 0, 1), 52)]:
            with self.subTest(shape=shape, stride=stride, size=size), self.assertRaises(ValueError):
                ADAPTER.expert_storage_geometry(shape, stride, 2, 2, size, 3)

    def test_inverse_scale_identity_does_not_round_or_normalize(self):
        for scalar in (1, 0.0009765625, 4096.0):
            encoded = ADAPTER.scalar_identity(scalar)
            self.assertEqual(struct.unpack("!d", bytes.fromhex(encoded))[0], scalar)
        for scalar in (True, 0, -1, float("inf"), float("nan"), "1.0"):
            with self.subTest(scalar=scalar), self.assertRaises(ValueError):
                ADAPTER.scalar_identity(scalar)

    def test_defaults_and_isolated_opt_in(self):
        base = RUNTIME.V41RuntimeConfig(
            role="DECODE", world_size=8, physical_tp=1, attention_dp=8,
            ep=8, enp=8, cp_method="PREFILL_CP", source_cp=8, max_seq_len=512,
        )
        self.assertIs(base.validate(), base)
        self.assertEqual(base.summary()["local_experts"], 48)
        for mode in ("NONE", "STATS", "EPLB", "ALL"):
            enabled = dataclasses.replace(base, shared_eplb=True, eplb=mode, redundant_experts=8)
            self.assertEqual(enabled.validate().summary()["local_experts"], 49)
            self.assertEqual(enabled.summary()["logical_experts"], 384)
        for overrides in ({"eplb": "EPLB"}, {"redundant_experts": 8},
                          {"shared_eplb": True, "redundant_experts": 1}, {"ep": 0}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                dataclasses.replace(base, **overrides).validate()

    def test_launcher_clears_previous_eplb_startup_identity(self):
        # Compile only the actual pure launcher function; no framework import
        # stubs, patched factories or accelerator packages are involved.
        tree = ast.parse((ROOT / "rtp_llm/tools/v41/launch.py").read_text())
        names = {"bounded_prefill_wait_timeout_ms", "preset_env"}
        selected = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
        class ConfigCapture:
            @classmethod
            def from_env(cls, env):
                return cls()

        scope = dict(os=os, PPU_STRATEGY=RUNTIME.PPU_STRATEGY, env_bool=RUNTIME.env_bool,
                     V41RuntimeConfig=ConfigCapture, DEFAULT_PREFILL_WAIT_TIMEOUT_MS=1800000,
                     MAX_PREFILL_WAIT_TIMEOUT_MS=86400000)
        exec(compile(selected, str(ROOT / "rtp_llm/tools/v41/launch.py"), "exec"), scope)
        parent = {"EPLB_MODE": "ALL", "REDUNDANT_EXPERT": "32", "EPLB_UPDATE_TIME": "1",
                  "EPLB_STATS_WINDOW_SIZE": "1", "RTP_V41_PPU_EPLB": "1"}
        clean = scope["preset_env"]("decode", "/checkpoint", parent=parent)
        self.assertEqual({key: clean[key] for key in parent}, {
            "EPLB_MODE": "NONE", "REDUNDANT_EXPERT": "0", "EPLB_UPDATE_TIME": "0",
            "EPLB_STATS_WINDOW_SIZE": "100", "RTP_V41_PPU_EPLB": "0"})
        enabled = scope["preset_env"](
            "decode", "/checkpoint", parent={}, ppu_model_path="shared", shared_eplb=True,
            eplb_mode="EPLB", redundant_experts=8, eplb_update_time=16, eplb_stats_window_size=1,
        )
        self.assertEqual(enabled["REDUNDANT_EXPERT"], "8")
        self.assertEqual(enabled["EPLB_MODE"], "EPLB")
        self.assertEqual(enabled["RTP_V41_PPU_EPLB"], "1")
        with self.assertRaises(ValueError):
            scope["preset_env"]("decode", "/checkpoint", parent={}, eplb_mode="EPLB")

    def test_native_observer_cannot_drive_or_fabricate_migration(self):
        tree = ast.parse((PPU / "tools/driver_shared_eplb.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, ("step", "rearrange", "_plan", "zero_", "fill_", "copy_"))
        text = (PPU / "shared_eplb.py").read_text()
        self.assertIn("from rtp_llm.eplb.runtime import EplbLayer", text)
        self.assertNotIn("RuntimeExpertBalancer(", text)

    def test_commit_only_draft_has_no_routed_ffn_binding(self):
        for filename in ("shared_model.py", "shared_dspark_model.py"):
            tree = ast.parse((PPU / filename).read_text())
            value = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Attribute) and target.attr == "_shared_eplb_layers"
                                 for target in node.targets))
            collect = compile(ast.Expression(value), filename, "eval")
            commit_only = SimpleNamespace(v4=SimpleNamespace(layers=[SimpleNamespace(attn=object()) for _ in range(3)]))
            self.assertEqual(eval(collect, {"self": commit_only}), ())
            bindings = (object(), object(), object())
            routed = SimpleNamespace(v4=SimpleNamespace(layers=[SimpleNamespace(ffn=SimpleNamespace(eplb_layer=b)) for b in bindings]))
            self.assertEqual(eval(collect, {"self": routed}), bindings)


def _observation_contracts():
    path = PPU / "tools/driver_shared_eplb.py"
    tree = ast.parse(path.read_text())
    names = {"NativeObservationLedger", "migration_window_basis", "epoch_moved_slots"}
    body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    scope = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), scope)
    return scope


class NativeObservationContracts(unittest.TestCase):
    """In-memory protocol inputs only; these are not native report artifacts."""

    def setUp(self):
        self.contracts = _observation_contracts()
        self.ledger = self.contracts["NativeObservationLedger"]("decode")

    def rows(self, prefix=None, *, after=False, role="decode"):
        pending = self.ledger.pending
        round_id = self.ledger.round if after else self.ledger.round + 1
        business = prefix is not None
        rows = []
        for rank in range(8):
            idle = not business or role == "decode" and rank != 3
            # The receiver can be an idle source when peers dispatch work.
            receives = 6 if after and business and rank == 5 else 0
            rows.append({"rank": rank, "round": round_id, "phase": "after" if after else "before",
                         "idle": idle, "warmup": False, "bindings_stable": True, "map_signature": "layout",
                         "business_call": self.ledger.business_calls + int(after and business),
                         "request_ids": [23] if role == "prefill" else [],
                         "sequence_prefixes": [] if role == "prefill" else [1 if idle else prefix],
                         "input_token_ids": [0 if idle else 5], "prefill": role == "prefill",
                         "input_lengths": [prefix] if role == "prefill" and business else [1],
                         "context_prefixes": [0] if role == "prefill" else [],
                         "count_nonzero": int(bool(receives)), "received_assignments": receives,
                         "layer0_received": receives, "map_unchanged": True, "output_finite": True,
                         "counters_unchanged": not business, "raw_report": after and business,
                         "raw_writes": after and not idle, "raw_pools": after and not idle,
                         "idle_routes_zero": idle, "output_zero": idle})
        if after:
            self.assertEqual(pending["round"], round_id)
        return rows

    def observe(self, prefix=None, *, role="decode"):
        self.ledger.begin(self.rows(prefix, role=role))
        return self.ledger.finish(self.rows(prefix, after=True, role=role))

    def test_b02_empty_startup_now_preserves_business_identity(self):
        for _ in range(3):
            frame = self.observe()
            self.assertEqual((frame["kind"], frame["business_forward"], frame["global_received_assignments"]),
                             ("all_empty", 0, 0))
        self.assertEqual(self.ledger.empty_total, 3)
        self.assertEqual(self.observe(257)["business_forward"], 1)

    def test_mixed_real_idle_receivers_and_inter_request_gaps(self):
        for index, prefix in enumerate((257, 258, 259, 260, 261, 17), 1):
            self.assertEqual(self.observe()["business_forward"], index - 1)
            frame = self.observe(prefix)
            self.assertEqual(frame["business_forward"], index)
            self.assertEqual(frame["global_layer0_received"], 6)
            self.assertEqual(frame["stage"], "fresh" if index == 6 else "trigger")
        self.assertEqual((self.ledger.round, self.ledger.business_calls), (12, 6))

    def test_prefill_empty_and_two_business_reports(self):
        self.ledger = self.contracts["NativeObservationLedger"]("prefill")
        self.assertEqual(self.observe(role="prefill")["business_forward"], 0)
        self.assertEqual(self.observe(257, role="prefill")["stage"], "trigger")
        self.assertEqual(self.observe(role="prefill")["kind"], "all_empty")
        self.assertEqual(self.observe(17, role="prefill")["stage"], "fresh")

    def test_all_empty_rejects_business_metadata_and_invalid_ranks(self):
        for field, value in (("request_ids", [12]), ("sequence_prefixes", [17]),
                             ("input_lengths", [257]), ("rank", 7), ("rank", False),
                             ("bindings_stable", False), ("map_signature", "diverged")):
            with self.subTest(field=field, value=value):
                rows = self.rows()
                rows[0][field] = value
                with self.assertRaises(AssertionError):
                    self.ledger.begin(rows)

    def test_all_empty_rejects_counts_writes_or_execution(self):
        for field, value in (("count_nonzero", 1), ("received_assignments", 1), ("layer0_received", 1),
                             ("raw_report", True), ("raw_writes", True), ("raw_pools", True),
                             ("counters_unchanged", False), ("map_unchanged", False), ("output_zero", False),
                             ("business_call", 1)):
            with self.subTest(field=field):
                self.ledger = self.contracts["NativeObservationLedger"]("decode")
                self.ledger.begin(self.rows())
                rows = self.rows(after=True)
                rows[2][field] = value
                with self.assertRaises(AssertionError):
                    self.ledger.finish(rows)

    def test_mixed_idle_rejects_source_writes_not_received_assignments(self):
        for field, value in (("raw_writes", True), ("raw_pools", True), ("idle_routes_zero", False),
                             ("raw_report", False)):
            with self.subTest(field=field):
                self.ledger = self.contracts["NativeObservationLedger"]("decode")
                self.ledger.begin(self.rows(257))
                rows = self.rows(257, after=True)
                rows[5][field] = value
                with self.assertRaises(AssertionError):
                    self.ledger.finish(rows)

    def test_wrong_phase_and_fresh_before_trigger_rejected(self):
        with self.assertRaises(AssertionError):
            self.ledger.finish([])
        with self.assertRaises(AssertionError):
            self.ledger.begin(self.rows(17))
        self.ledger.begin(self.rows())
        with self.assertRaises(AssertionError):
            self.ledger.begin(self.rows())
        rows = self.rows(after=True)
        rows[1]["round"] += 1
        with self.assertRaises(AssertionError):
            self.ledger.finish(rows)

    def test_zero_windows_do_not_credit_real_migration(self):
        classify = self.contracts["migration_window_basis"]
        empty1 = self.observe()
        business = self.observe(257)
        empty3 = self.observe()
        empty4 = self.observe()
        self.assertEqual(classify([empty1]), "zero_count_window")
        self.assertEqual(classify([empty1, business]), "actual_business_window")
        self.assertEqual(classify([business, empty3]), "actual_business_window")
        self.assertEqual(classify([empty3, empty4]), "zero_count_window")
        with self.assertRaises(AssertionError):
            classify([empty1, empty3])
        with self.assertRaises(AssertionError):
            classify([business])
        with self.assertRaises(AssertionError):
            classify([{**empty1, "global_layer0_received": 1}])

    def test_epoch_slot_proof_recomputes_exact_movement(self):
        recompute = self.contracts["epoch_moved_slots"]
        old = list(range(384)) + list(range(8))
        new = old.copy()
        new[1], new[3] = new[3], new[1]
        epoch = {"old_physical_to_logical": old, "physical_to_logical": new,
                 "global_moved_slots": [1, 3], "global_moved_slot_count": 2}
        self.assertEqual(recompute(epoch), {1, 3})
        with self.assertRaises(AssertionError):
            recompute({**epoch, "global_moved_slots": [1, 4]})

    def test_completed_fixture_delegates_without_observer_work(self):
        path = PPU / "test/native_shared_eplb_role.py"
        tree = ast.parse(path.read_text())
        fixture = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "NativeSharedEplbModel")

        class Parent:
            def forward(self, inputs, fmha_impl=None):
                return inputs, fmha_impl

        class TerminalEvidence:
            done = True

            def before(self, inputs):
                raise AssertionError("completed fixture must not observe again")

            after = before

        scope = {"DeepSeekV41SharedPPUModel": Parent, "os": os}
        exec(compile(ast.Module(body=[fixture], type_ignores=[]), str(path), "exec"), scope)
        model = scope["NativeSharedEplbModel"]()
        model._n8_evidence = TerminalEvidence()
        inputs, fmha = object(), object()
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(model.forward(inputs, fmha), (inputs, fmha))


if __name__ == "__main__":
    unittest.main()
