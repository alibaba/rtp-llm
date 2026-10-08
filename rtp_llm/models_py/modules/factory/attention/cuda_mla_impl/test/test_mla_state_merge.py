import runpy
import unittest
from pathlib import Path

import torch


merge_mla_states_in_place = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "mla_state_merge.py")
)["merge_mla_states_in_place"]


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class MlaStateMergeTest(unittest.TestCase):
    def test_sequential_slices_match_softmax_state_reference(self):
        torch.manual_seed(31)
        device = "cuda:0"
        output = torch.randn(23, 12, 128, dtype=torch.bfloat16, device=device)
        lse = torch.randn(23, 12, dtype=torch.float32, device=device) + 2
        pieces = [
            (torch.randn_like(output), torch.randn_like(lse) + 1),
            (torch.randn_like(output), torch.randn_like(lse) + 3),
        ]
        reference_lse = torch.logsumexp(
            torch.stack([lse] + [part_lse for _, part_lse in pieces]), dim=0
        )
        weights = [torch.exp(item - reference_lse) for item in
                   [lse] + [part_lse for _, part_lse in pieces]]
        reference = sum(
            state.float() * weight.unsqueeze(-1)
            for state, weight in zip([output] + [p[0] for p in pieces], weights)
        ).to(torch.bfloat16)
        for partial, partial_lse in pieces:
            merge_mla_states_in_place(output, lse, partial, partial_lse)
        torch.testing.assert_close(output, reference, rtol=0.01, atol=0.015)
        torch.testing.assert_close(lse, reference_lse, rtol=1e-5, atol=1e-5)

    def test_empty_state_takes_partial(self):
        output = torch.zeros(3, 12, 128, dtype=torch.bfloat16, device="cuda:0")
        lse = torch.full((3, 12), -float("inf"), device="cuda:0")
        partial = torch.randn_like(output)
        partial_lse = torch.randn_like(lse)
        merge_mla_states_in_place(output, lse, partial, partial_lse)
        torch.testing.assert_close(output, partial, rtol=0, atol=0)
        torch.testing.assert_close(lse, partial_lse, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
