from unittest import TestCase, main, skipUnless

import torch

from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_selection import (
    select_qsa_paged_tokens,
)
from rtp_llm.models_py.model_desc.test.qwen4_exp.qsa_fixtures import QSARuntimeFixture


class Qwen4ExpQSAGraphTest(QSARuntimeFixture, TestCase):
    @skipUnless(torch.cuda.is_available(), "CUDA is required for draft graph replay")
    def test_draft_graph_replays_ragged_lengths_and_new_pages(self):
        for lanes in (1, 2, 3, 4):
            with self.subTest(lanes=lanes):
                self._check_draft_graph_replay(lanes)

    @skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_paged_topk_prevalidated_path_replays_new_lengths(self):
        device = torch.device("cuda")
        logits = torch.tensor([[0.5, 2.0, 1.0]], device=device)
        lengths = torch.tensor([9], dtype=torch.int32, device=device)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            select_qsa_paged_tokens(
                logits,
                lengths,
                compress_ratio=4,
                token_budget=8,
                validate_lengths=False,
            )
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = select_qsa_paged_tokens(
                logits,
                lengths,
                compress_ratio=4,
                token_budget=8,
                validate_lengths=False,
            )
        logits.copy_(torch.tensor([[3.0, 0.5, 2.0]], device=device))
        lengths.copy_(torch.tensor([12], dtype=torch.int32, device=device))
        graph.replay()
        expected = select_qsa_paged_tokens(
            logits,
            lengths,
            compress_ratio=4,
            token_budget=8,
        )
        torch.testing.assert_close(replayed, expected)


if __name__ == "__main__":
    main()
