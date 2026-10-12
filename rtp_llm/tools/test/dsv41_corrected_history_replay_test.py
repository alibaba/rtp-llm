"""Admission controls for the current ten-layer corrected history dataset."""

import copy
import unittest
from types import SimpleNamespace

import torch

from rtp_llm.tools.dsv41_corrected_history_replay import (
    FLAGS,
    HASHES,
    KERNEL_SHA256,
    NUM_LAYERS,
    PROFILE,
    PROMPTS,
    PROVIDER_SHA256,
    admit_traces,
    validate_recipe,
    validate_schedule,
)


def fixture(prompt=0):
    emitted = ([11111, 16, 1], [223, 22, 201])[prompt]
    identities = [{"input_ids": list(PROMPTS[prompt]), "start_pos": 0}]
    identities.extend(
        {"input_ids": [token], "start_pos": 5 + step}
        for step, token in enumerate(emitted)
    )
    reference, native, comparisons, consumed = [], [], [], []
    for step, identity in enumerate(identities):
        consumed.extend(identity["input_ids"])
        tensors = {
            f"layer{layer:02d}.ffn.{kind}": torch.ones(
                len(identity["input_ids"]), 2, dtype=torch.bfloat16
            )
            for layer in range(NUM_LAYERS)
            for kind in ("input", "output")
        }
        reference.append({"metadata": identity, "tensors": tensors})
        metadata = {
            "input_ids": identity["input_ids"],
            "cache_spans": [
                [1, 1, 1, 1, identity["start_pos"], len(identity["input_ids"])]
            ],
            "prompt_index": prompt,
            "prompt_token_ids": list(PROMPTS[prompt]),
            "consumed_token_ids": consumed.copy(),
            "precision_profile": PROFILE,
            "sparse_prefill_reference_scope": "short-single-initial",
            "mhc_collapse_reference": True,
            "decode_tilelang_reference": True,
            "decode_tilelang_reference_scope": "single-token-planar",
            "decode_tilelang_provider_sha256": PROVIDER_SHA256,
            **dict.fromkeys(FLAGS, True),
            **dict.fromkeys(HASHES, KERNEL_SHA256),
        }
        native.append({"metadata": metadata, "tensors": tensors})
        comparisons.append(
            {
                "profile": PROFILE,
                "prompt_index": prompt,
                "step": step,
                "matching_prefix": True,
                "candidate_identity": identity,
                "reference_identity": identity,
            }
        )
    return reference, native, comparisons, identities


def schedule():
    return SimpleNamespace(
        n_layers=40,
        compress_ratios=[0, 0, *([2] * 18), *([1] * 20)],
        kv_source_layers=[2, 8, 14, 20],
        index_source_layers=[2, 8, 14, 20, 24, 28, 32, 36],
    ), SimpleNamespace(layer_ids=[1, 14])


class CorrectedHistoryAdmissionTest(unittest.TestCase):
    def test_both_complete_four_step_histories(self):
        for prompt in (0, 1):
            reference, native, comparisons, expected = fixture(prompt)
            identities, recipe = admit_traces(reference, native, comparisons, prompt)
            self.assertEqual(identities, expected)
            self.assertEqual(recipe["precision_profile"], PROFILE)
            self.assertIs(recipe["mhc_collapse_reference"], True)
            self.assertIs(recipe["decode_tilelang_reference"], True)
            self.assertEqual(recipe["decode_tilelang_provider_sha256"], PROVIDER_SHA256)

    def test_exact_boolean_recipe_flags_are_required(self):
        for name in (*FLAGS, "mhc_collapse_reference", "decode_tilelang_reference"):
            for value in (False, None, 1, "true", "missing"):
                _, native, _, _ = fixture()
                if value == "missing":
                    native[-1]["metadata"].pop(name)
                else:
                    native[-1]["metadata"][name] = value
                with self.subTest(name=name, value=value), self.assertRaisesRegex(
                    ValueError, name
                ):
                    validate_recipe(
                        native,
                        kernel_sha256=KERNEL_SHA256,
                        provider_sha256=PROVIDER_SHA256,
                    )

    def test_provider_identities_and_scopes_are_pinned(self):
        for name in (
            *HASHES,
            "precision_profile",
            "sparse_prefill_reference_scope",
            "decode_tilelang_reference_scope",
            "decode_tilelang_provider_sha256",
        ):
            _, native, _, _ = fixture()
            native[1]["metadata"][name] = "changed"
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                validate_recipe(
                    native, kernel_sha256=KERNEL_SHA256, provider_sha256=PROVIDER_SHA256
                )
        for kernel, provider in ((None, PROVIDER_SHA256), (KERNEL_SHA256, "a" * 64)):
            with self.assertRaisesRegex(ValueError, "pinned providers"):
                validate_recipe(
                    fixture()[1], kernel_sha256=kernel, provider_sha256=provider
                )

    def test_divergent_or_unaudited_prefixes_are_rejected(self):
        for name, value in (
            ("matching_prefix", False),
            ("matching_prefix", 1),
            ("step", True),
            ("prompt_index", False),
            ("profile", "baseline"),
            ("candidate_identity", {"input_ids": [455], "start_pos": 7}),
        ):
            reference, native, comparisons, _ = fixture()
            comparisons[-1][name] = value
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "audited matching"
            ):
                admit_traces(reference, native, comparisons, 0)
        reference, native, comparisons, _ = copy.deepcopy(fixture())
        native[-1]["metadata"]["input_ids"] = [455]
        with self.assertRaisesRegex(ValueError, "divergent"):
            admit_traces(reference, native, comparisons, 0)

    def test_complete_layer_and_step_coverage_is_required(self):
        for layers in (7, 9, 11, True, 10.0):
            with self.assertRaisesRegex(ValueError, "ten-layer"):
                admit_traces(*fixture()[:3], 0, layers=layers)
        for index in range(3):
            values = list(fixture()[:3])
            values[index] = values[index][:-1]
            with self.assertRaisesRegex(ValueError, "every qualified"):
                admit_traces(*values, 0)
        for stream in (0, 1):
            values = copy.deepcopy(fixture()[:3])
            values[stream][-1]["tensors"].pop("layer09.ffn.output")
            with self.assertRaisesRegex(ValueError, "complete FFN"):
                admit_traces(*values, 0)

    def test_prompt_and_consumed_history_are_required(self):
        for name in ("prompt_index", "prompt_token_ids", "consumed_token_ids"):
            reference, native, comparisons, _ = fixture()
            native[-1]["metadata"][name] = None
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "recorded token"
            ):
                admit_traces(reference, native, comparisons, 0)
        with self.assertRaisesRegex(ValueError, "prompt differs"):
            admit_traces(*fixture(1)[:3], 0)
        for prompt in (True, False, 2, "0"):
            with self.assertRaisesRegex(ValueError, "ten-layer"):
                admit_traces(*fixture()[:3], prompt)

    def test_both_retained_sources_and_engram_schedule_are_required(self):
        args, layout = schedule()
        validate_schedule(args, layout)
        for name, value in (
            ("n_layers", 40.0),
            ("kv_source_layers", [2, 14, 20]),
            ("index_source_layers", [2, 8, 14, 20]),
            ("compress_ratios", [0, 0, 2]),
        ):
            args, layout = schedule()
            setattr(args, name, value)
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "schedule"
            ):
                validate_schedule(args, layout)
        args, layout = schedule()
        layout.layer_ids = [1]
        with self.assertRaisesRegex(ValueError, "schedule"):
            validate_schedule(args, layout)


if __name__ == "__main__":
    unittest.main()
