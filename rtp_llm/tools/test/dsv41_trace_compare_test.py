import copy
import unittest

import torch

from rtp_llm.tools.dsv41_trace_compare import compare_traces, tensor_difference


def traces():
    reference = {
        "metadata": {"input_ids": [[3, 4]], "start_pos": 5},
        "tensors": {
            "layer00.ffn.input": torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
            "layer00.ffn.output": torch.tensor([[[2.0, 3.0], [4.0, 5.0]]]),
            "final_norm": torch.tensor([[[0.5, 1.0], [1.5, 2.0]]]),
            "logits_fp32": torch.tensor([[1.0, 4.0, 3.0, 0.0]]),
            "local_logits_bf16_head": torch.tensor([[1.0, 4.0]]),
        },
    }
    candidate = copy.deepcopy(reference)
    candidate["metadata"] = {"input_ids": [3, 4], "cache_spans": [[11, 0, 0, 0, 5, 2]]}
    candidate["tensors"] = {
        name: value[0] if value.ndim == 3 else value
        for name, value in candidate["tensors"].items()
        if name != "local_logits_bf16_head"
    }
    return reference, candidate


class TraceCompareTest(unittest.TestCase):
    def test_matching_inputs_and_singleton_batch_alignment(self):
        result = compare_traces(*traces())
        self.assertIsNone(result["first_bitwise_difference"])
        self.assertTrue(result["greedy_equal"])
        self.assertEqual(result["logits_fp32"]["max_abs"], 0)
        head = result["head_precision"]["reference"]
        self.assertEqual(head["vocabulary_range"], [0, 2])
        self.assertEqual(head["watched_ids"], [0, 1])

    def test_first_boundary_and_changed_greedy(self):
        reference, candidate = traces()
        candidate["tensors"]["layer00.ffn.output"][1, 0] += 0.5
        candidate["tensors"]["logits_fp32"][0, 2] = 5
        result = compare_traces(reference, candidate)
        self.assertEqual(result["first_bitwise_difference"], "layer00.ffn.output")
        error = result["boundaries"]["layer00.ffn.output"]
        self.assertEqual(error["max_abs"], 0.5)
        self.assertEqual(error["mean_abs"], 0.125)
        self.assertEqual(error["different_values"], 1)
        self.assertFalse(result["greedy_equal"])

    def test_mismatched_inputs_positions_or_spans_are_rejected(self):
        for field in ("token", "position", "count", "batch"):
            reference, candidate = traces()
            with self.subTest(field=field):
                if field == "token":
                    candidate["metadata"]["input_ids"][0] = 8
                elif field == "position":
                    candidate["metadata"]["cache_spans"][0][4] = 6
                elif field == "count":
                    candidate["metadata"]["cache_spans"][0][5] = 3
                else:
                    candidate["metadata"]["cache_spans"] *= 2
                with self.assertRaises(ValueError):
                    compare_traces(reference, candidate)

    def test_missing_boundaries_shapes_and_nonfinite_values_are_rejected(self):
        for defect in ("missing", "gap", "shape", "rows", "nonfinite"):
            reference, candidate = traces()
            with self.subTest(defect=defect):
                tensors = candidate["tensors"]
                if defect == "missing":
                    del tensors["layer00.ffn.input"]
                elif defect == "gap":
                    for direction in ("input", "output"):
                        tensors[f"layer02.ffn.{direction}"] = tensors.pop(
                            f"layer00.ffn.{direction}"
                        )
                elif defect == "shape":
                    tensors["final_norm"] = tensors["final_norm"][:, :1]
                elif defect == "rows":
                    for trace in (reference, candidate):
                        trace["tensors"]["final_norm"] = torch.ones(3, 2)
                else:
                    tensors["final_norm"][0, 0] = float("nan")
                with self.assertRaises(ValueError):
                    compare_traces(reference, candidate)

    def test_bitwise_signed_zero_and_zero_norm(self):
        result = tensor_difference(torch.tensor([[0.0]]), torch.tensor([[-0.0]]))
        self.assertFalse(result["bitwise_equal"])
        self.assertEqual(result["different_values"], 0)
        self.assertIsNone(result["relative_l2"])
        self.assertIsNone(result["cosine"])


if __name__ == "__main__":
    unittest.main()
