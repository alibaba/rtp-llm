"""Check logical sparse positions across the two attention storage layouts."""

import unittest

import torch

from rtp_llm.tools.dsv41_attention_history_replay import (
    official_history_control,
    validate_native_recipe,
)
from rtp_llm.tools.dsv41_attention_replay import (
    aligned_sparse_inputs,
    history_identities,
)


class AttentionReplayAlignmentTest(unittest.TestCase):
    def setUp(self):
        kv = torch.arange(3 * 512).reshape(1, 3, 512).bfloat16()
        self.reference = {
            "kv": kv,
            "indices": torch.tensor([[[0, -1, -1], [0, 1, 2]]]),
        }
        self.captured = {
            "kv": kv[0, [2, 0, 1]].unsqueeze(1),
            "indices": torch.tensor([[[1, -1, -1]], [[0, 1, 2]]]),
            "lengths": torch.tensor([1, 3]),
        }

    def test_layout_alignment_preserves_set_and_reports_order(self):
        kv, selections = aligned_sparse_inputs(self.reference, self.captured, 2)
        self.assertTrue(kv["bitwise_equal"])
        self.assertTrue(all(row["set_equal"] for row in selections))
        self.assertTrue(selections[0]["order_equal"])
        self.assertFalse(selections[1]["order_equal"])
        self.assertEqual(selections[1]["native_aligned"], [2, 0, 1])

    def test_selected_position_mismatch_is_rejected(self):
        self.captured["indices"][0, 0, 0] = 2
        with self.assertRaisesRegex(AssertionError, "position set differs"):
            aligned_sparse_inputs(self.reference, self.captured, 2)

    def test_invalid_indices_and_lengths_are_rejected(self):
        for invalid in (-2, 3):
            with self.subTest(index=invalid):
                self.captured["indices"][0, 0, 0] = invalid
                with self.assertRaisesRegex(AssertionError, "position set differs"):
                    aligned_sparse_inputs(self.reference, self.captured, 2)
        self.captured["indices"][0, 0, 0] = 1
        self.captured["lengths"][0] = 4
        with self.assertRaisesRegex(ValueError, "row length"):
            aligned_sparse_inputs(self.reference, self.captured, 2)

    def test_equal_duplicate_sets_are_rejected(self):
        self.reference["indices"][0, 1] = torch.tensor([0, 0, 2])
        self.captured["indices"][1, 0] = torch.tensor([1, 1, 0])
        with self.assertRaisesRegex(AssertionError, "position set differs"):
            aligned_sparse_inputs(self.reference, self.captured, 2)

    def test_decode_query_is_independent_of_full_swa_ring(self):
        kv = torch.arange(131 * 512).reshape(1, 131, 512).bfloat16()
        reference = {
            "kv": kv,
            "indices": torch.tensor([[[0, 1, 2, 3, 4, 5, 128, 129, 130, -1]]]),
        }
        captured = {
            "kv": torch.cat((kv[0, 128:], kv[0, :128])).unsqueeze(1),
            "indices": torch.tensor([[[3, 4, 5, 6, 7, 8, 0, 1, 2, -1]]]),
            "lengths": torch.tensor([10]),
        }
        values, selections = aligned_sparse_inputs(reference, captured, 128)
        self.assertTrue(values["bitwise_equal"])
        self.assertEqual(len(selections), 1)
        self.assertTrue(selections[0]["set_equal"])
        self.assertTrue(selections[0]["order_equal"])

    def test_query_count_and_swa_segment_admission(self):
        for rows in (0, -1, 1.5):
            with self.subTest(window_rows=rows):
                with self.assertRaisesRegex(ValueError, "SWA segment"):
                    aligned_sparse_inputs(self.reference, self.captured, rows)
        captured = {**self.captured, "lengths": torch.tensor([1])}
        with self.assertRaisesRegex(ValueError, "query row count"):
            aligned_sparse_inputs(self.reference, captured, 2)


class AttentionReplayHistoryTest(unittest.TestCase):
    def traces(self, steps=((0, (10, 20, 30, 40, 50)), (5, (60,)), (6, (70,)))):
        return [
            {"metadata": {"start_pos": start, "input_ids": [list(ids)]}}
            for start, ids in steps
        ]

    def test_contiguous_history_and_matching_native_cache_spans(self):
        reference = self.traces()
        native = [
            {
                "metadata": {
                    "input_ids": trace["metadata"]["input_ids"][0],
                    "cache_spans": [[1, 1, 1, 1, start, span]],
                }
            }
            for trace, start, span in zip(reference, (0, 5, 6), (5, 1, 1))
        ]
        identities = history_identities(reference, native)
        self.assertEqual([entry["start_pos"] for entry in identities], [0, 5, 6])
        self.assertEqual(identities[-1]["input_ids"], [70])

    def test_gaps_overlap_and_resets_are_rejected(self):
        for start in (0, 4, 6):
            with self.subTest(start=start):
                with self.assertRaisesRegex(ValueError, "gap, overlap or reset"):
                    history_identities(
                        self.traces(((0, (10, 20, 30, 40, 50)), (start, (60,))))
                    )
        with self.assertRaisesRegex(ValueError, "gap, overlap or reset"):
            history_identities(self.traces(((5, (60,)),)))

    def test_prefill_and_decode_shapes_are_bounded(self):
        with self.assertRaisesRegex(ValueError, "initial prefill"):
            history_identities([])
        for steps in (((0, (1,)),), ((0, (1, 2)), (2, (3, 4)))):
            with self.subTest(steps=steps):
                with self.assertRaisesRegex(ValueError, "single-token decode"):
                    history_identities(self.traces(steps))
        with self.assertRaisesRegex(ValueError, "first SWA ring"):
            history_identities(self.traces(((0, tuple(range(128))), (128, (1,)))))

    def test_divergent_current_or_earlier_prefix_is_rejected(self):
        reference = self.traces()
        for step in range(3):
            native = self.traces()
            native[step]["metadata"]["input_ids"][0][-1] += 1
            with self.subTest(step=step):
                with self.assertRaisesRegex(ValueError, "divergent consumed prefixes"):
                    history_identities(reference, native)
        with self.assertRaisesRegex(ValueError, "different step counts"):
            history_identities(reference, reference[:2])

    def test_native_history_cannot_silently_use_a_different_recipe(self):
        traces = self.traces()
        recipe = {
            "precision_profile": "tp8_projection_mhc_fp32_compressor_recipe",
            "compressor_fused": True,
            "hca_bf16_projection": True,
        }
        for trace in traces:
            trace["metadata"].update(recipe)
        validate_native_recipe(traces)
        for field, invalid in (
            ("precision_profile", "baseline"),
            ("compressor_fused", False),
            ("hca_bf16_projection", False),
        ):
            with self.subTest(field=field):
                traces[1]["metadata"][field] = invalid
                with self.assertRaisesRegex(
                    ValueError, "composed TP8/mHC/compressor recipe"
                ):
                    validate_native_recipe(traces)
                traces[1]["metadata"][field] = recipe[field]

    def down_recipe_traces(
        self, *, sparse=False, routed=False, post=False, engram=False
    ):
        profile = "tp8_projection_mhc_fp32_compressor_recipe"
        if sparse:
            profile += "_sparse_sink_div_rn_prefill"
        profile += "_expert_down_reference" if routed else "_shared_down_reference"
        if post:
            profile += "_mhc_post_reference"
        if engram:
            profile += "_engram_reference"
        metadata = {
            "precision_profile": profile,
            "compressor_fused": True,
            "hca_bf16_projection": True,
            "sparse_sink_div_rn_prefill": sparse,
            "sparse_sink_div_rn_decode": False,
            "shared_down_reference": True,
            "routed_down_reference": routed,
            "down_reference_kernel_sha256": "a" * 64,
        }
        if post:
            metadata["mhc_post_reference"] = True
        if engram:
            metadata.update(
                engram_projection_reference=True,
                engram_gate_reference=True,
                engram_reference_kernel_sha256="a" * 64,
            )
        traces = self.traces()
        for trace in traces:
            trace["metadata"].update(metadata)
        return traces

    def test_output_gemm_recipes_select_the_matching_attention_control(self):
        for sparse in (False, True):
            for routed in (False, True):
                with self.subTest(sparse=sparse, routed=routed):
                    traces = self.down_recipe_traces(sparse=sparse, routed=routed)
                    recipe = validate_native_recipe(
                        traces, reference_kernel_sha256="a" * 64
                    )
                    for field in (
                        "precision_profile",
                        "sparse_sink_div_rn_prefill",
                        "sparse_sink_div_rn_decode",
                        "shared_down_reference",
                        "routed_down_reference",
                        "down_reference_kernel_sha256",
                    ):
                        self.assertEqual(recipe[field], traces[0]["metadata"][field])
                    control = "tp8_projection_compressor_recipe"
                    if sparse:
                        control += "_sparse_sink_div_rn_prefill"
                    self.assertEqual(official_history_control(recipe), control)

    def test_output_gemm_recipe_requires_explicit_and_exact_flags(self):
        for field in (
            "compressor_fused",
            "hca_bf16_projection",
            "sparse_sink_div_rn_prefill",
            "sparse_sink_div_rn_decode",
            "shared_down_reference",
            "routed_down_reference",
        ):
            for invalid in (None, "true", 1, "flip", "missing"):
                with self.subTest(field=field, invalid=invalid):
                    traces = self.down_recipe_traces(sparse=True, routed=True)
                    expected = traces[1]["metadata"][field]
                    if invalid == "missing":
                        del traces[1]["metadata"][field]
                    else:
                        traces[1]["metadata"][field] = (
                            not expected if invalid == "flip" else invalid
                        )
                    with self.assertRaises(ValueError):
                        validate_native_recipe(traces, reference_kernel_sha256="a" * 64)

    def test_output_gemm_recipe_checks_released_kernel_identity(self):
        for digest in (None, "a" * 63, "g" * 64, "b" * 64):
            with self.subTest(reference_digest=digest):
                with self.assertRaisesRegex(ValueError, "kernel hash differs"):
                    validate_native_recipe(
                        self.down_recipe_traces(), reference_kernel_sha256=digest
                    )
        for digest in (None, "b" * 64):
            traces = self.down_recipe_traces()
            traces[1]["metadata"]["down_reference_kernel_sha256"] = digest
            with self.subTest(trace_digest=digest):
                with self.assertRaisesRegex(ValueError, "kernel hash differs"):
                    validate_native_recipe(traces, reference_kernel_sha256="a" * 64)

    def test_post_recipes_retain_independent_attention_controls(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                traces = self.down_recipe_traces(sparse=sparse, routed=True, post=True)
                recipe = validate_native_recipe(
                    traces, reference_kernel_sha256="a" * 64
                )
                prior = validate_native_recipe(
                    self.down_recipe_traces(sparse=sparse, routed=True),
                    reference_kernel_sha256="a" * 64,
                )
                self.assertEqual(
                    recipe,
                    {
                        **prior,
                        "precision_profile": traces[0]["metadata"]["precision_profile"],
                        "mhc_post_reference": True,
                    },
                )
                self.assertEqual(
                    official_history_control(recipe), official_history_control(prior)
                )
                self.assertNotIn("mhc_post_reference", prior)

    def test_post_recipes_require_explicit_boolean_flag_and_kernel_identity(self):
        for sparse in (False, True):
            for invalid in (None, False, 1, "true", "missing"):
                with self.subTest(sparse=sparse, post_flag=invalid):
                    traces = self.down_recipe_traces(
                        sparse=sparse, routed=True, post=True
                    )
                    if invalid == "missing":
                        del traces[1]["metadata"]["mhc_post_reference"]
                    else:
                        traces[1]["metadata"]["mhc_post_reference"] = invalid
                    with self.assertRaisesRegex(ValueError, "mhc_post_reference"):
                        validate_native_recipe(traces, reference_kernel_sha256="a" * 64)
            for digest in (None, "a" * 63, "b" * 64):
                with self.subTest(sparse=sparse, kernel_digest=digest):
                    with self.assertRaisesRegex(ValueError, "kernel hash differs"):
                        validate_native_recipe(
                            self.down_recipe_traces(
                                sparse=sparse, routed=True, post=True
                            ),
                            reference_kernel_sha256=digest,
                        )

    def test_history_cannot_mix_post_recipes_or_apply_unrecorded_post(self):
        for sparse in (False, True):
            traces = self.down_recipe_traces(sparse=sparse, routed=True, post=True)
            traces[-1] = self.down_recipe_traces(sparse=sparse, routed=True)[-1]
            with self.subTest(sparse=sparse):
                with self.assertRaisesRegex(ValueError, "mix recipes across steps"):
                    validate_native_recipe(traces, reference_kernel_sha256="a" * 64)
        for invalid in (True, None, 0, "false"):
            traces = self.down_recipe_traces(routed=True)
            traces[1]["metadata"]["mhc_post_reference"] = invalid
            with self.subTest(unrecorded_post=invalid):
                with self.assertRaisesRegex(ValueError, "mhc_post_reference"):
                    validate_native_recipe(traces, reference_kernel_sha256="a" * 64)

    def test_engram_recipes_preserve_the_independent_attention_control(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                traces = self.down_recipe_traces(
                    sparse=sparse, routed=True, post=True, engram=True
                )
                recipe = validate_native_recipe(
                    traces, reference_kernel_sha256="a" * 64
                )
                prior = validate_native_recipe(
                    self.down_recipe_traces(sparse=sparse, routed=True, post=True),
                    reference_kernel_sha256="a" * 64,
                )
                self.assertEqual(
                    recipe,
                    {
                        **prior,
                        "precision_profile": traces[0]["metadata"]["precision_profile"],
                        "engram_projection_reference": True,
                        "engram_gate_reference": True,
                        "engram_reference_kernel_sha256": "a" * 64,
                    },
                )
                self.assertEqual(
                    official_history_control(recipe), official_history_control(prior)
                )
                self.assertNotIn("engram_projection_reference", prior)
                self.assertNotIn("engram_gate_reference", prior)

    def test_engram_recipes_require_exact_flags_and_matching_kernel_identity(self):
        for sparse in (False, True):
            for field in ("engram_projection_reference", "engram_gate_reference"):
                for invalid in (None, False, 1, "true", "missing"):
                    with self.subTest(sparse=sparse, field=field, invalid=invalid):
                        traces = self.down_recipe_traces(
                            sparse=sparse, routed=True, post=True, engram=True
                        )
                        if invalid == "missing":
                            del traces[1]["metadata"][field]
                        else:
                            traces[1]["metadata"][field] = invalid
                        with self.assertRaisesRegex(ValueError, field):
                            validate_native_recipe(
                                traces, reference_kernel_sha256="a" * 64
                            )
            for invalid in (None, "", "a" * 63, "b" * 64):
                traces = self.down_recipe_traces(
                    sparse=sparse, routed=True, post=True, engram=True
                )
                traces[1]["metadata"]["engram_reference_kernel_sha256"] = invalid
                with self.subTest(sparse=sparse, digest=invalid):
                    with self.assertRaisesRegex(ValueError, "Engram kernel hash"):
                        validate_native_recipe(traces, reference_kernel_sha256="a" * 64)

    def test_history_rejects_mixed_or_unrecorded_engram_substitutions(self):
        for sparse in (False, True):
            traces = self.down_recipe_traces(
                sparse=sparse, routed=True, post=True, engram=True
            )
            traces[-1] = self.down_recipe_traces(sparse=sparse, routed=True, post=True)[
                -1
            ]
            with self.subTest(sparse=sparse):
                with self.assertRaisesRegex(ValueError, "mix recipes across steps"):
                    validate_native_recipe(traces, reference_kernel_sha256="a" * 64)
            for field in ("engram_projection_reference", "engram_gate_reference"):
                for invalid in (True, None, 0, "false"):
                    traces = self.down_recipe_traces(
                        sparse=sparse, routed=True, post=True
                    )
                    traces[1]["metadata"][field] = invalid
                    with self.subTest(field=field, invalid=invalid):
                        with self.assertRaisesRegex(ValueError, field):
                            validate_native_recipe(
                                traces, reference_kernel_sha256="a" * 64
                            )
            traces = self.down_recipe_traces(sparse=sparse, routed=True, post=True)
            traces[1]["metadata"]["engram_reference_kernel_sha256"] = "a" * 64
            with self.assertRaisesRegex(ValueError, "unexpected Engram kernel hash"):
                validate_native_recipe(traces, reference_kernel_sha256="a" * 64)
        for routed, post in ((False, True), (True, False)):
            with self.subTest(routed=routed, post=post):
                with self.assertRaisesRegex(ValueError, "composed TP8/mHC"):
                    validate_native_recipe(
                        self.down_recipe_traces(routed=routed, post=post, engram=True),
                        reference_kernel_sha256="a" * 64,
                    )

    def test_four_step_history_checks_the_first_post_eos_consumed_token(self):
        traces = self.traces()
        end = sum(len(trace["metadata"]["input_ids"][0]) for trace in traces)
        traces.append({"metadata": {"input_ids": [[1]], "start_pos": end}})
        identities = history_identities(traces)
        self.assertEqual(len(identities), 4)
        self.assertEqual(identities[-1], {"input_ids": [1], "start_pos": end})
        native = [
            {
                "metadata": {
                    "input_ids": identity["input_ids"],
                    "cache_spans": [
                        [1, 1, 1, 1, identity["start_pos"], len(identity["input_ids"])]
                    ],
                }
            }
            for identity in identities
        ]
        self.assertEqual(history_identities(traces, native), identities)
        native[-1]["metadata"]["input_ids"] = [455]
        with self.assertRaisesRegex(ValueError, "divergent consumed prefixes"):
            history_identities(traces, native)

    def test_history_cannot_combine_individually_valid_recipes(self):
        traces = self.down_recipe_traces()
        traces[-1] = self.down_recipe_traces(sparse=True, routed=True)[-1]
        with self.assertRaisesRegex(ValueError, "mix recipes across steps"):
            validate_native_recipe(traces, reference_kernel_sha256="a" * 64)
        with self.assertRaisesRegex(ValueError, "nonempty recipe"):
            validate_native_recipe([])

    def test_legacy_recipe_rejects_unrecorded_sparse_or_down_substitutions(self):
        traces = self.traces()
        for trace in traces:
            trace["metadata"].update(
                precision_profile="tp8_projection_mhc_fp32_compressor_recipe",
                compressor_fused=True,
                hca_bf16_projection=True,
            )
        recipe = validate_native_recipe(traces)
        self.assertEqual(
            official_history_control(recipe), "tp8_projection_compressor_recipe"
        )
        for field in (
            "sparse_sink_div_rn_prefill",
            "sparse_sink_div_rn_decode",
            "shared_down_reference",
            "routed_down_reference",
            "mhc_post_reference",
        ):
            with self.subTest(field=field):
                traces[1]["metadata"][field] = True
                with self.assertRaisesRegex(ValueError, "recipe flag differs"):
                    validate_native_recipe(traces)
                del traces[1]["metadata"][field]
        traces[1]["metadata"]["down_reference_kernel_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "unexpected down-reference hash"):
            validate_native_recipe(traces)


if __name__ == "__main__":
    unittest.main()
