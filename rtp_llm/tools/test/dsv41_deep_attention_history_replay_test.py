"""Admission and byte controls for the qualified deep matching-prefix dataset."""

import copy
import unittest

import torch

from rtp_llm.tools.dsv41_deep_attention_history_replay import (
    FLAGS,
    HASHES,
    KERNEL_SHA256,
    NUM_LAYERS,
    PROFILE,
    admit_traces,
    require_exact,
    validate_recipe,
)


def fixture(prompt=0):
    count = 3 if prompt == 0 else 4
    identities = [{"input_ids": [2, 3], "start_pos": 0}]
    identities += [
        {"input_ids": [10 + step], "start_pos": step + 1} for step in range(1, count)
    ]
    reference, native, comparisons = [], [], []
    for step, identity in enumerate(identities):
        tensors = {
            f"layer{layer:02d}.ffn.{kind}": torch.ones(1, 2, dtype=torch.bfloat16)
            for layer in range(NUM_LAYERS)
            for kind in ("input", "output")
        }
        reference.append(
            {
                "metadata": {
                    "input_ids": [identity["input_ids"]],
                    "start_pos": identity["start_pos"],
                },
                "tensors": tensors,
            }
        )
        metadata = dict(
            precision_profile=PROFILE,
            sparse_prefill_reference_scope="short-single-initial",
            prompt_index=prompt,
            prompt_token_ids=identities[0]["input_ids"],
            consumed_token_ids=sum(
                (row["input_ids"] for row in identities[: step + 1]), []
            ),
            input_ids=identity["input_ids"],
            cache_spans=[
                [1, 1, 1, 1, identity["start_pos"], len(identity["input_ids"])]
            ],
        )
        metadata.update({name: True for name in FLAGS})
        metadata.update({name: KERNEL_SHA256 for name in HASHES})
        native.append({"metadata": metadata, "tensors": tensors})
        comparisons.append(
            dict(
                profile=PROFILE,
                prompt_index=prompt,
                step=step,
                matching_prefix=True,
                candidate_identity=identity,
                reference_identity=identity,
            )
        )
    return reference, native, comparisons, identities


class DeepAttentionHistoryAdmissionTest(unittest.TestCase):
    def test_complete_current_recipe_for_both_matching_histories(self):
        for prompt in (0, 1):
            reference, native, comparisons, wanted = fixture(prompt)
            identities, recipe = admit_traces(reference, native, comparisons, prompt)
            self.assertEqual(identities, wanted)
            self.assertEqual(recipe["precision_profile"], PROFILE)
            self.assertEqual(
                set(recipe),
                {
                    "precision_profile",
                    "sparse_prefill_reference_scope",
                    *FLAGS,
                    *HASHES,
                },
            )

    def test_recipe_requires_exact_boolean_flags(self):
        for name in FLAGS:
            for bad in (False, None, 1, "true", "missing"):
                reference, native, comparisons, _ = fixture()
                if bad == "missing":
                    native[1]["metadata"].pop(name)
                else:
                    native[1]["metadata"][name] = bad
                with self.subTest(name=name, bad=bad), self.assertRaisesRegex(
                    ValueError, name
                ):
                    admit_traces(reference, native, comparisons, 0)

    def test_every_kernel_identity_and_scope_is_pinned(self):
        for name in (*HASHES, "precision_profile", "sparse_prefill_reference_scope"):
            _, native, _, _ = fixture()
            native[-1]["metadata"][name] = "changed"
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                validate_recipe(native, kernel_sha256=KERNEL_SHA256)
        for digest in (None, "a" * 64):
            with self.assertRaises(ValueError):
                validate_recipe(fixture()[1], kernel_sha256=digest)

    def test_divergent_or_unaudited_prefixes_are_rejected(self):
        for name, value in (
            ("matching_prefix", False),
            ("matching_prefix", 1),
            ("candidate_identity", {"input_ids": [455], "start_pos": 3}),
            ("reference_identity", {"input_ids": [1], "start_pos": 3}),
            ("step", True),
            ("prompt_index", False),
        ):
            reference, native, comparisons, _ = fixture()
            comparisons[1][name] = value
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "audited matching"
            ):
                admit_traces(reference, native, comparisons, 0)
        reference, native, comparisons, _ = fixture()
        native[-1]["metadata"]["input_ids"] = [455]
        with self.assertRaisesRegex(ValueError, "divergent"):
            admit_traces(reference, native, comparisons, 0)

    def test_layer_step_and_boundary_coverage_is_required(self):
        for layers in (3, 6, 8, True, 7.0):
            with self.assertRaisesRegex(ValueError, "seven-layer"):
                admit_traces(*fixture()[:3], 0, layers=layers)
        for prompt in (False, True, 2, "0"):
            with self.assertRaises(ValueError):
                admit_traces(*fixture()[:3], prompt)
        for index in range(3):
            values = list(fixture()[:3])
            values[index] = values[index][:-1]
            with self.assertRaisesRegex(ValueError, "every qualified"):
                admit_traces(*values, 0)
        reference, native, comparisons, _ = copy.deepcopy(fixture())
        native[0]["tensors"].pop("layer06.ffn.output")
        with self.assertRaisesRegex(ValueError, "complete FFN"):
            admit_traces(reference, native, comparisons, 0)

    def test_recorded_consumed_token_history_is_required(self):
        for name in ("prompt_index", "prompt_token_ids", "consumed_token_ids"):
            reference, native, comparisons, _ = fixture()
            native[-1]["metadata"][name] = None
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "recorded token"
            ):
                admit_traces(reference, native, comparisons, 0)

    def test_normalized_byte_controls_retain_zero_sign_and_dtype(self):
        expected = torch.tensor([[0.0, 1.0]], dtype=torch.bfloat16)
        exact = require_exact(expected, expected[None].clone(), "normalized")
        self.assertTrue(exact["normalized_bytes_equal"])
        changed = expected.clone()
        changed[0, 0] = -0.0
        self.assertTrue(torch.equal(expected, changed))
        for value in (changed, expected.float()):
            with self.assertRaises(AssertionError):
                require_exact(expected, value, "changed")


if __name__ == "__main__":
    unittest.main()
