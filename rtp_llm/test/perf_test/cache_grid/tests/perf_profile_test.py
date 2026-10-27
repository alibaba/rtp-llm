import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.cache_grid.config.perf_profile import (
    ProfileError,
    engine_args_section,
    engine_section,
    extract_embedded_profile,
    fingerprint,
    load_profile,
    merge_engine_args,
    resolve_int,
    resolve_label,
    resolve_str,
    resolve_title,
    set_engine_arg,
    validate_profile,
)


def _minimal_profile(**overrides):
    base = {"schema_version": 1}
    base.update(overrides)
    return base


class ValidateProfileTest(unittest.TestCase):
    def test_minimal_valid_profile(self):
        validate_profile({"schema_version": 1})

    def test_full_profile(self):
        validate_profile(
            {
                "schema_version": 1,
                "label": "test",
                "engine": {"tp_size": 8, "dp_size": 1},
                "engine_args": {"--use_deepep_moe": "1", "--fp8_kv_cache": 1},
                "cache_grid": {"cache_alignment": 512},
                "chart": {"title": "Test", "model_label": "T"},
            }
        )

    def test_rejects_non_dict(self):
        with self.assertRaisesRegex(ProfileError, "must be a JSON object"):
            validate_profile([1, 2, 3])

    def test_rejects_missing_schema_version(self):
        with self.assertRaisesRegex(ProfileError, "missing schema_version"):
            validate_profile({})

    def test_rejects_wrong_schema_version(self):
        with self.assertRaisesRegex(ProfileError, "unsupported schema_version 99"):
            validate_profile({"schema_version": 99})

    def test_rejects_bool_in_engine_args(self):
        with self.assertRaisesRegex(ProfileError, "bool is not a valid CLI value"):
            validate_profile({"schema_version": 1, "engine_args": {"--flag": False}})

    def test_rejects_invalid_engine_section(self):
        with self.assertRaisesRegex(ProfileError, "engine must be a JSON object"):
            validate_profile({"schema_version": 1, "engine": "bad"})

    def test_rejects_invalid_engine_args_section(self):
        with self.assertRaisesRegex(ProfileError, "engine_args must be a JSON object"):
            validate_profile({"schema_version": 1, "engine_args": [1, 2]})

    def test_rejects_explicit_null_object_sections(self):
        for section in ("engine", "engine_args", "cache_grid", "chart"):
            with self.subTest(section=section), self.assertRaisesRegex(
                ProfileError, rf"{section} must be a JSON object"
            ):
                validate_profile({"schema_version": 1, section: None})


class LoadProfileTest(unittest.TestCase):
    def test_load_from_disk(self):
        profile = _minimal_profile(label="test")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(profile, tmp)
            tmp.flush()
            loaded = load_profile(tmp.name)
        self.assertEqual(loaded, profile)

    def test_load_rejects_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp.write("{bad json")
            tmp.flush()
            with self.assertRaisesRegex(ProfileError, "invalid JSON"):
                load_profile(tmp.name)


class FingerprintTest(unittest.TestCase):
    def test_same_profile_same_fingerprint(self):
        p = _minimal_profile(engine={"tp_size": 8})
        self.assertEqual(fingerprint(p), fingerprint(p))

    def test_key_order_does_not_matter(self):
        p1 = {"schema_version": 1, "a": 1, "b": 2}
        p2 = {"schema_version": 1, "b": 2, "a": 1}
        self.assertEqual(fingerprint(p1), fingerprint(p2))

    def test_different_profiles_different_fingerprints(self):
        p1 = _minimal_profile(engine={"tp_size": 8})
        p2 = _minimal_profile(engine={"tp_size": 4})
        self.assertNotEqual(fingerprint(p1), fingerprint(p2))

    def test_fingerprint_is_hex_string(self):
        fp = fingerprint(_minimal_profile())
        self.assertEqual(len(fp), 64)
        int(fp, 16)


class ResolveTest(unittest.TestCase):
    def test_cli_wins_over_profile(self):
        self.assertEqual(
            resolve_int(
                {"schema_version": 1, "engine": {"tp_size": 8}},
                "engine",
                "tp_size",
                4,
                1,
            ),
            4,
        )

    def test_removed_runtime_alignment_profile_field_is_rejected(self):
        with self.assertRaisesRegex(ProfileError, "generator.cache_alignment"):
            validate_profile(
                {"schema_version": 1, "cache_grid": {"expected_block_size": 4096}}
            )

    def test_profile_wins_over_default(self):
        self.assertEqual(
            resolve_int(
                {"schema_version": 1, "engine": {"tp_size": 8}},
                "engine",
                "tp_size",
                None,
                1,
            ),
            8,
        )

    def test_default_when_all_absent(self):
        self.assertEqual(
            resolve_int({"schema_version": 1}, "engine", "tp_size", None, 1),
            1,
        )

    def test_explicit_zero_wins(self):
        self.assertEqual(
            resolve_int(
                {"schema_version": 1, "cache_grid": {"cache_alignment": 512}},
                "cache_grid",
                "cache_alignment",
                0,
                256,
            ),
            0,
        )

    def test_invalid_profile_value_falls_back_to_default(self):
        self.assertEqual(
            resolve_int(
                {"schema_version": 1, "engine": {"tp_size": "bad"}},
                "engine",
                "tp_size",
                None,
                1,
            ),
            1,
        )

    def test_resolve_str_cli_wins(self):
        self.assertEqual(
            resolve_str(
                {"schema_version": 1, "engine": {"model_type": "profile"}},
                "engine",
                "model_type",
                "cli",
                "default",
            ),
            "cli",
        )

    def test_resolve_str_empty_cli_falls_through(self):
        self.assertEqual(
            resolve_str(
                {"schema_version": 1, "engine": {"model_type": "profile"}},
                "engine",
                "model_type",
                "",
                "default",
            ),
            "profile",
        )

    def test_resolve_label_cli_wins(self):
        profile = {"schema_version": 1, "chart": {"model_label": "P"}}
        self.assertEqual(resolve_label(profile, "CLI", "D"), "CLI")

    def test_resolve_label_profile_wins(self):
        profile = {"schema_version": 1, "chart": {"model_label": "P"}}
        self.assertEqual(resolve_label(profile, None, "D"), "P")

    def test_resolve_label_uses_model_metadata(self):
        profile = {"schema_version": 1, "engine": {"model_type": "qwen_2"}}
        self.assertEqual(resolve_label(profile, None, "Model"), "qwen_2")
        profile["model_label"] = "My Model"
        self.assertEqual(resolve_label(profile, None, "Model"), "My Model")
        profile["chart"] = {"model_label": "Chart Model"}
        self.assertEqual(resolve_label(profile, None, "Model"), "Chart Model")
        self.assertEqual(resolve_label(profile, "CLI", "Model"), "CLI")

    def test_resolve_label_default_when_none(self):
        self.assertEqual(resolve_label(None, None, "D"), "D")
        self.assertEqual(resolve_label({"engine": None}, None, "D"), "D")

    def test_resolve_title(self):
        profile = {"schema_version": 1, "chart": {"title": "PT"}}
        self.assertEqual(resolve_title(profile, None, "DT"), "PT")
        self.assertEqual(resolve_title(profile, "CT", "DT"), "CT")
        self.assertEqual(resolve_title(None, None, "DT"), "DT")


class SetEngineArgTest(unittest.TestCase):
    def test_append_when_absent(self):
        result = set_engine_arg(["--other", "1"], "tp_size", "8")
        self.assertEqual(result, ["--other", "1", "--tp_size", "8"])

    def test_update_space_separated(self):
        result = set_engine_arg(["--tp_size", "4", "--other", "1"], "tp_size", "8")
        self.assertEqual(result, ["--tp_size", "8", "--other", "1"])

    def test_update_equals_form(self):
        result = set_engine_arg(["--tp_size=4", "--other", "1"], "tp_size", "8")
        self.assertEqual(result, ["--tp_size=8", "--other", "1"])

    def test_bare_flag_gets_value_inserted(self):
        result = set_engine_arg(["--reuse_cache", "--other"], "reuse_cache", "1")
        self.assertEqual(result, ["--reuse_cache", "1", "--other"])


class MergeEngineArgsTest(unittest.TestCase):
    def test_injects_missing_args(self):
        profile = {
            "schema_version": 1,
            "engine": {"tp_size": 8, "dp_size": 1},
            "engine_args": {"--use_deepep_moe": "1"},
        }
        result = merge_engine_args(profile, ["--other", "1"])
        self.assertIn("--tp_size", result)
        self.assertIn("8", result)
        self.assertIn("--use_deepep_moe", result)
        self.assertNotIn("--dp_size", result)

    def test_cli_wins_over_profile(self):
        profile = {
            "schema_version": 1,
            "engine": {"tp_size": 8},
        }
        result = merge_engine_args(profile, ["--tp_size", "4"])
        idx = result.index("--tp_size")
        self.assertEqual(result[idx + 1], "4")

    def test_cli_equals_form_wins(self):
        profile = {
            "schema_version": 1,
            "engine": {"tp_size": 8},
        }
        result = merge_engine_args(profile, ["--tp_size=4"])
        self.assertIn("--tp_size=4", result)
        self.assertNotIn("8", result)

    def test_skip_namespace_keys(self):
        profile = {
            "schema_version": 1,
            "engine": {"dp_size": 4, "max_seq_len": 65536, "concurrency_limit": 32},
        }
        result = merge_engine_args(profile, [])
        self.assertEqual(result, [])

    def test_include_namespace_keys_when_requested(self):
        profile = {
            "schema_version": 1,
            "engine": {"dp_size": 4},
        }
        result = merge_engine_args(profile, [], skip_namespace_keys=False)
        self.assertIn("--dp_size", result)


class ExtractEmbeddedProfileTest(unittest.TestCase):
    def test_extracts_profile(self):
        data = {
            "schema_version": 2,
            "mode": "prefix_cache_grid",
            "metrics": [],
            "profile": {"schema_version": 1, "label": "test"},
        }
        result = extract_embedded_profile(data)
        self.assertEqual(result, {"schema_version": 1, "label": "test"})

    def test_returns_none_when_absent(self):
        self.assertIsNone(
            extract_embedded_profile(
                {"schema_version": 2, "mode": "prefix_cache_grid", "metrics": []}
            )
        )

    def test_returns_none_when_not_dict(self):
        self.assertIsNone(extract_embedded_profile({"profile": "not a dict"}))

    def test_returns_none_for_non_dict_input(self):
        self.assertIsNone(extract_embedded_profile([1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
