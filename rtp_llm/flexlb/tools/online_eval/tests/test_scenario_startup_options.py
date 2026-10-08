"""Birth settings preserve legacy performance and Master logging ownership."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from runtime.harness import default_perf
from runtime.perf_presets import ROOT, capture_defaults, load_performance_file, load_preset, preset_names
from scenario import compile_scenarios
from scenario.backend import make_env_spec
from scenario.catalog import handlers
from scenario.compiler import environment
from test_scenario_runtime import source


def spec(raw):
    return make_env_spec(
        environment(raw, "test", "single-nonbatch"),
        "single-nonbatch",
        {"master_base": 28000},
    )


class StartupOptionsTest(unittest.TestCase):
    def test_capture_record_supplies_topology_and_rejects_untracked_edits(self):
        for name, expected in (
            ("glm_5_3_l20d", (125, 536, 31218, 46157)),
            ("deepseek_v4_flash_l20c", (48, 192, 21553, 221484)),
        ):
            with self.subTest(name=name):
                actual = spec({"perf_preset": name})
                self.assertEqual((actual.n_prefill, actual.n_decode,
                    actual.prefill_cache_blocks, actual.decode_cache_blocks), expected)
                capture = capture_defaults(name)
                self.assertEqual(tuple(capture.values()), expected)
        source = ROOT / "data/performance/glm_5_3_l20d.json"
        document = json.loads(source.read_text())
        document["prefill"]["memory_cache"]["capacity_blocks"] += 1
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "performance.json"
            changed.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_performance_file(changed)

    def test_capture_deviations_require_explicit_baseline_and_reason(self):
        with self.assertRaisesRegex(Exception, "model_override"):
            environment({"perf_preset": "glm_5_3_l20d", "n_prefill": 64},
                "test", "single-nonbatch")
        declared = environment({"perf_preset": "glm_5_3_l20d", "n_prefill": 64,
            "model_override": {"baseline": "glm_5_3_l20d", "reason": "故障注入"}},
            "test", "single-nonbatch")
        self.assertEqual(declared["n_prefill"], 64)

    def test_paired_master_baseline_is_rendered_and_explicit_deviations_are_audited(self):
        from flexlb_cfg import render_env, render_process_config, ConfigOverride
        from runtime import stress
        expected_caps = {
            "single-nonbatch": 1024,
            "batch-window": 2,
            "single-batch": 2,
            "window-nonbatch": 64,
        }
        for profile, cap in expected_caps.items():
            plan = environment({"perf_preset": "glm_5_3_l20d"}, "test", profile)
            override = ConfigOverride(**plan["config_overrides"])
            config = json.loads(render_env(profile, override))
            self.assertEqual(config["dispatcher"]["maxInflightPerPrefillWorker"], cap)
            self.assertEqual(plan["resolved_config"], config)
            self.assertEqual(plan["config_overrides"].get("max_inflight_per_prefill_worker"),
                             1024 if profile == "single-nonbatch" else None)
            affinity = config["router"]["roles"]["prefill"]["cacheAffinity"]
            self.assertEqual(affinity, {"maxExtraTtftMs": 1000000000, "minPrefixHitPercent": 5})
            envelope = json.loads(render_process_config(profile, override))
            self.assertEqual(json.loads(dict(envelope["zone_process_setting"]["process_info"]["envs"])["FLEXLB_CONFIG"]), config)
        raw = {"perf_preset": "glm_5_3_l20d", "config_overrides": {"cache_affinity_max_extra_ttft_ms": 20}}
        with self.assertRaisesRegex(Exception, "model_override"):
            environment(raw, "test", "single-nonbatch")
        raw["model_override"] = {"baseline": "glm_5_3_l20d", "reason": "亲和策略对照"}
        self.assertEqual(environment(raw, "test", "single-nonbatch")["config_overrides"]["cache_affinity_max_extra_ttft_ms"], 20)
        args = stress.parse_args(["--performance", str(ROOT / "data/performance/glm_5_3_l20d.json"), "--dry-run"])
        self.assertEqual(json.loads(stress._config(args))["router"]["roles"]["prefill"]["cacheAffinity"], affinity)
        for mode, cap in (("sn", 1024), ("wb", 2), ("sb", 2), ("wn", 64)):
            args = stress.parse_args(["--performance", str(ROOT / "data/performance/glm_5_3_l20d.json"),
                                      "--master-mode", mode, "--dry-run"])
            self.assertEqual(json.loads(stress._config(args))["dispatcher"]["maxInflightPerPrefillWorker"], cap)
        batch_override = environment({"perf_preset": "glm_5_3_l20d",
                                      "config_overrides": {"max_inflight_per_prefill_worker": 3}},
                                     "test", "batch-window")
        self.assertEqual(batch_override["resolved_config"]["dispatcher"]["maxInflightPerPrefillWorker"], 3)
        with self.assertRaisesRegex(Exception, "model_override"):
            environment({"perf_preset": "glm_5_3_l20d",
                         "config_overrides": {"max_inflight_per_prefill_worker": 3}},
                        "test", "single-nonbatch")
        self.assertNotIn("master", load_preset("glm_5_3_l20d")[0])
        for bad in (-1, True, 2**63):
            with self.assertRaises(ValueError):
                ConfigOverride(cache_affinity_max_extra_ttft_ms=bad)
        for bad in (-1, 101, float("nan"), True):
            with self.assertRaises(ValueError):
                ConfigOverride(cache_affinity_min_prefix_hit_percent=bad)
        document = json.loads((ROOT / "data/performance/glm_5_3_l20d.json").read_text())
        document["master"]["config_overrides"]["cache_affinity_min_prefix_hit_percent"] = 20
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "changed.json"
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_performance_file(path)

    def test_paired_master_cap_requires_valid_profile_scope(self):
        from runtime.perf_presets import PairedMasterSettings

        document = json.loads((ROOT / "data/performance/glm_5_3_l20d.json").read_text())
        master = document["master"]
        self.assertEqual(master["config_overrides"]["max_inflight_per_prefill_worker"], 1024)
        for scope in ({}, {"max_inflight_per_prefill_worker": ["unknown"]},
                      {"max_inflight_per_prefill_worker": []},
                      {"other": ["single-nonbatch"]},
                      {"prefill_expression": ["single-nonbatch"]}):
            with self.subTest(scope=scope), self.assertRaisesRegex(ValueError, "profile_scope"):
                PairedMasterSettings.from_record({**master, "profile_scope": scope})

    def test_preset_registry_is_total_and_rejects_typos_before_launch(self):
        self.assertEqual(("default", "glm_5_3_l20d", "deepseek_v4_flash_l20c"), preset_names())
        for name in preset_names():
            self.assertIsInstance(load_preset(name)[0], dict)
            self.assertEqual(spec({"perf_preset": name}).perf, load_preset(name)[0])
        for typo in ("fault_env", "production_scale_2026092", "Default", "", None):
            with self.subTest(typo=typo), self.assertRaisesRegex(ValueError, "perf_preset"):
                load_preset(typo)
            with self.subTest(typo=typo), self.assertRaises(Exception):
                spec({"perf_preset": typo})

    def test_waiting_cap_merges_birth_perf_without_adding_batch_limits(self):
        for cap in (0, 16):
            with self.subTest(cap=cap):
                expected = copy.deepcopy(default_perf())
                expected.setdefault("prefill", {})["max_waiting_batches"] = cap
                actual = spec(
                    {"perf_preset": "default", "prefill_max_waiting_batches": cap}
                )
                self.assertEqual(actual.perf, expected)
                self.assertNotIn("max_batch_tokens", actual.perf["prefill"])
                self.assertNotIn("max_batch_requests", actual.perf["prefill"])
        self.assertEqual(spec({"perf_preset": "default"}).perf, default_perf())

    def test_explicit_perf_replacement_and_cap_compose_without_mutating_input(self):
        perf = dict(fixed_ms=3000, scale=1, max_batch_tokens=1024, max_batch_requests=0)
        raw = dict(prefill_perf=perf, prefill_max_waiting_batches=16)
        before = copy.deepcopy(raw)
        self.assertEqual(spec(raw).perf["prefill"], dict(perf, max_waiting_batches=16))
        self.assertEqual(raw, before)

    def test_master_logging_and_diagnostic_api_are_independent(self):
        baseline = environment({}, "test", "single-nonbatch")["resolved_config"]
        for logging in (False, True):
            for diagnostic in (False, True):
                raw = dict(master_debug_log=logging, debug_enabled=diagnostic)
                actual = spec(raw)
                self.assertIs(actual.master_debug_log, logging)
                self.assertEqual(
                    actual.master_env,
                    {"FLEXLB_DEBUG_ENABLED": "true"} if diagnostic else {},
                )
                self.assertEqual(
                    environment(raw, "test", "single-nonbatch")["resolved_config"],
                    baseline,
                )
        self.assertFalse(spec({}).master_debug_log)

    def test_variant_options_survive_environment_epochs_without_leaking(self):
        doc = source()
        doc["profiles"] = ["single-nonbatch"]
        doc["variants"] = [
            dict(id="ordinary"),
            dict(
                id="configured",
                environment_overrides=dict(
                    master_debug_log=True,
                    prefill_max_waiting_batches=16,
                    prefill_perf=dict(fixed_ms=100.0, scale=1.0,
                                      max_batch_tokens=1024, max_batch_requests=0),
                ),
            ),
        ]
        doc["stages"].insert(
            1,
            dict(
                id="replace",
                action="environment_reconfigure",
                params=dict(config_overrides=dict(ordering="fifo")),
            ),
        )
        plans = compile_scenarios([("test", doc)], handlers=handlers())
        for index, plan in enumerate(plans):
            initial = plan["environment"]
            later = plan["stages"][1]["params"]["environments"]["single-nonbatch"]
            self.assertEqual(
                initial.get("master_debug_log"), later.get("master_debug_log")
            )
            self.assertEqual(
                initial.get("prefill_max_waiting_batches"),
                later.get("prefill_max_waiting_batches"),
            )
            actual = make_env_spec(later, "single-nonbatch", {"master_base": 28000})
            if index:
                self.assertTrue(actual.master_debug_log)
                self.assertEqual(
                    actual.perf["prefill"],
                    dict(fixed_ms=100.0, scale=1.0, max_batch_tokens=1024,
                         max_batch_requests=0, max_waiting_batches=16),
                )
            else:
                self.assertNotIn("prefill_max_waiting_batches", initial)
                self.assertNotIn("master_debug_log", initial)
                self.assertEqual(actual.perf, default_perf())

    def test_strict_types_reject_before_process_construction(self):
        for field, values in (
            ("master_debug_log", (None, 0, 1, "true", [], {})),
            (
                "prefill_max_waiting_batches",
                (None, True, False, -1, 1.5, 16.0, "16", [], {}),
            ),
        ):
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(
                    ValueError
                ):
                    environment({field: value}, "test", "single-nonbatch")


if __name__ == "__main__":
    unittest.main()
