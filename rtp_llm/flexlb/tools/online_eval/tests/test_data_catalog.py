"""Input layout and cross-language artifact pinning contracts."""
import json
import hashlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from traffic import catalog as data_catalog

ROOT = Path(__file__).resolve().parents[1]


class DataCatalogTest(unittest.TestCase):
    def test_default_model_companions_and_provenance(self):
        entry, paths = data_catalog.verify_default_model()
        self.assertEqual("frontend", entry["source"])
        self.assertEqual("frontend_20260921", entry["profile_symbol"])
        self.assertEqual(paths["model"].parent, paths["manifest"].parent)
        self.assertEqual(paths["model"].parent, paths["java_fixture"].parent)

    def test_missing_companion_fails_at_catalog_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            source = data_catalog.catalog()["models"][data_catalog.catalog()["default_trace"]]
            for relative in (source["model"], source["java_fixture"], source["calibration_profile"]):
                path = destination / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(data_catalog.DATA / relative, path)
            shutil.copyfile(data_catalog.CATALOG, destination / "catalog.json")
            with mock.patch.object(data_catalog, "DATA", destination), \
                    mock.patch.object(data_catalog, "CATALOG", destination / "catalog.json"):
                with self.assertRaisesRegex(ValueError, "missing manifest"):
                    data_catalog.verify_default_model()

    def test_data_and_config_classification_and_binary_admission(self):
        from scenario.loader import load_document
        config_roots = {path.name for path in (ROOT / "config").iterdir()}
        self.assertEqual({"README.md", "load_client_env.txt", "mode_profiles.yaml",
                          "performance_presets.json", "report_views", "scenarios", "suites.yaml"},
                         config_roots, "config contains an unclassified input")
        data_roots = {path.name for path in (ROOT / "data").iterdir()}
        self.assertEqual({"README.md", "catalog.json", "calibration", "performance",
                          "traffic_models"}, data_roots, "data contains an unclassified artifact")
        self.assertFalse((ROOT / "config/traffic_profiles").exists())
        self.assertFalse((ROOT / "config/perf_presets").exists())
        self.assertTrue(all(p.is_file() for p in (ROOT / "config/scenarios").iterdir()))
        self.assertEqual([], list((ROOT / "data").rglob("*.yaml")))
        listed = {str((ROOT / "data" / entry["model"]).relative_to(ROOT))
                  for entry in data_catalog.catalog()["models"].values()}
        captures = []
        for name, entry in data_catalog.catalog()["models"].items():
            data_catalog.validate_model_name(entry)
            _, _, manifest, _ = data_catalog.verify_model(name)
            provenance = manifest.get("provenance") or {}
            start, end = provenance.get("source_start"), provenance.get("source_end")
            self.assertIsInstance(start, int)
            self.assertIsInstance(end, int)
            captures.append((name, entry, start, end))
        for index, (name_a, entry_a, start_a, end_a) in enumerate(captures):
            for name_b, entry_b, start_b, end_b in captures[index + 1:]:
                if (entry_a["source"], entry_a["codec"], entry_a["codec_version"]) == \
                        (entry_b["source"], entry_b["codec"], entry_b["codec_version"]):
                    self.assertLessEqual(max(0, min(end_a, end_b) - max(start_a, start_b)),
                                         5 * 60 * 1000, f"overlapping captures: {name_a}, {name_b}")
        with self.assertRaisesRegex(ValueError, "unconventional snapshot name"):
            data_catalog.validate_model_name(dict(codec="prefix_lineage", codec_version=2,
                                                  model="traffic_models/frontend_20260922.xz"))
        tracked = subprocess.check_output(
            ["git", "ls-files", "--cached", "--", "data/traffic_models/*.xz"],
            cwd=ROOT, text=True).splitlines()
        self.assertEqual(listed, set(tracked), "tracked captures require catalog registration")

        def sources(node):
            if isinstance(node, dict):
                if node.get("kind") == "trace" and isinstance(node.get("parameters"), dict):
                    yield node
                for value in node.values():
                    yield from sources(value)
            elif isinstance(node, list):
                for value in node:
                    yield from sources(value)

        for case in (ROOT / "config/scenarios").rglob("*.yaml"):
            for source in sources(load_document(case)):
                params = source["parameters"]
                model_path = (case.parent / params["path"]).resolve()
                relative = str(model_path.relative_to(ROOT)) if model_path.is_relative_to(ROOT) else None
                self.assertIn(relative, listed, f"{case}: trace model is not registered")
                manifest = json.loads(model_path.with_suffix(".manifest.json").read_text())
                self.assertEqual(manifest["sha256"], params["sha256"],
                                 f"{case}: trace SHA differs from its manifest")
                self.assertEqual(manifest["count"], params["count"],
                                 f"{case}: trace count differs from its manifest")
        # Every catalog entry is selectable by run_stress; workload scenarios may also
        # pin a model, but admitting a capture does not require a new case.

    def test_unregistered_capture_is_ignored_at_stage_boundary(self):
        candidate = ROOT / "data/traffic_models/prefix_lineage_v2_unregistered.xz"
        self.assertFalse(candidate.exists())
        candidate.write_bytes(b"not a registered capture")
        try:
            result = subprocess.run(["git", "check-ignore", "--quiet", str(candidate)], cwd=ROOT)
            self.assertEqual(0, result.returncode, "unregistered binary can enter a normal git add")
        finally:
            candidate.unlink()

    def test_profile_symbol_keeps_fixed_seed_trace_and_provenance(self):
        from traffic.realistic import write_trace
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixed.jsonl"
            semantics = write_trace(path, {"seed": 12345, "count": 10000,
                                           "output_tokens": 420}, "fixed")
            self.assertEqual("3ab3bb294b96acdcf188ff08897a1b8e41537903c4a56d6570a594c42c415d69",
                             hashlib.sha256(path.read_bytes()).hexdigest())
            manifest = json.loads(data_catalog.model_entry()[1]["manifest"].read_text())
            self.assertEqual(manifest["provenance"], semantics["calibration"]["provenance"])

    def test_catalog_models_are_selectable_and_materialize(self):
        from runtime import stress
        from traffic.traffic_source import validate_plan

        with tempfile.TemporaryDirectory() as directory:
            for name in data_catalog.catalog()["models"]:
                with self.subTest(name=name):
                    args = stress.parse_args(["--dry-run", "--traffic-model", name,
                                              "--limit", "8"])
                    self.assertEqual(name, args.traffic_model)
                    output = Path(directory) / f"{name}.jsonl"
                    stress._traffic(args, output)
                    self.assertEqual(8, validate_plan(output))
                    evidence = json.loads(output.with_suffix(".manifest.json").read_text())
                    self.assertEqual(8, evidence["request_count"])


if __name__ == "__main__":
    unittest.main()
