"""Weight-free single-request DSpARK round simulator.

This module deliberately models the protocol boundary rather than MiniMax-M3
attention math: fixed-width query construction, sequential Markov proposals,
target verification, commit/correction, and stale speculative-slot overwrite.
It is a local correctness oracle until the shared C++ executor and a real
MiniMax-M3 DSpARK checkpoint are wired into this branch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, List, Optional, Sequence

import torch

from rtp_llm.models_py.speculative.dspark_reference import sample_markov_drafts

TargetLogits = Callable[[Sequence[int]], torch.Tensor]


@dataclass(frozen=True)
class DSparkRoundTrace:
    anchor: int
    query_ids: List[int]
    proposals: List[int]
    accepted_draft_tokens: int
    committed_tokens: List[int]
    committed_sequence: List[int]
    proposal_overwrites: List[int]
    correction_position: Optional[int]
    draft_cache: List[int]

    def to_dict(self):
        return asdict(self)


class SingleRequestDSparkSimulator:
    """Execute greedy DSpARK rounds for one logical request on CPU.

    The target callback receives the committed/teacher-forced token prefix and
    returns next-token logits. Draft base logits are supplied per round so a
    test can force full acceptance, first/middle rejection, and overwrite.
    """

    def __init__(
        self,
        *,
        prompt_tokens: Sequence[int],
        proposal_width: int,
        noise_token_id: int,
        markov_w1: torch.Tensor,
        markov_w2: torch.Tensor,
        target_logits: TargetLogits,
        sample_from_anchor: bool = True,
        draft_to_target: Optional[torch.Tensor] = None,
    ) -> None:
        if not prompt_tokens:
            raise ValueError("DSpARK simulation requires a non-empty prompt")
        if proposal_width <= 0:
            raise ValueError("proposal_width must be positive")
        if noise_token_id < 0:
            raise ValueError("noise_token_id must be non-negative")
        self.committed = [int(token) for token in prompt_tokens]
        self.proposal_width = int(proposal_width)
        self.noise_token_id = int(noise_token_id)
        self.markov_w1 = markov_w1
        self.markov_w2 = markov_w2
        self.target_logits = target_logits
        self.sample_from_anchor = bool(sample_from_anchor)
        self.draft_to_target = draft_to_target
        # This token list stands in for committed and speculative KV slots.
        # Entries beyond len(committed) are unreachable stale rows.
        self.draft_cache = list(self.committed)

    def _write_cache(self, position: int, token: int) -> bool:
        overwritten = position < len(self.draft_cache)
        if overwritten:
            self.draft_cache[position] = token
        else:
            if position != len(self.draft_cache):
                raise RuntimeError("simulator cache writes must stay contiguous")
            self.draft_cache.append(token)
        return overwritten

    def _target_argmax(self, prefix: Sequence[int]) -> int:
        logits = self.target_logits(prefix)
        if logits.dim() != 1 or logits.numel() != self.markov_w1.shape[0]:
            raise ValueError("target_logits must return [target_vocab]")
        return int(logits.argmax().item())

    def run_greedy_round(self, base_logits: torch.Tensor) -> DSparkRoundTrace:
        """Run propose -> Markov -> target verify -> commit for one request."""
        if base_logits.dim() != 2 or base_logits.shape[0] != self.proposal_width:
            raise ValueError("base_logits must be [proposal_width, padded_draft_vocab]")

        anchor = self.committed[-1]
        query_width = self.proposal_width + int(not self.sample_from_anchor)
        query_ids = [anchor] + [self.noise_token_id] * (query_width - 1)
        sampled, _ = sample_markov_drafts(
            base_logits.unsqueeze(0),
            torch.tensor([anchor], dtype=torch.long),
            self.markov_w1,
            self.markov_w2,
            temperature=0.0,
            draft_to_target=self.draft_to_target,
        )
        proposals = [int(token) for token in sampled[0].tolist()]

        speculative_start = len(self.committed)
        proposal_overwrites = []
        for offset, token in enumerate(proposals):
            position = speculative_start + offset
            if self._write_cache(position, token):
                proposal_overwrites.append(position)

        # Target verify is teacher-forced over every proposal row. Logical
        # acceptance is a prefix decision made only after all rows exist.
        verify_prefix = list(self.committed)
        target_tokens = []
        for proposal in proposals:
            target_tokens.append(self._target_argmax(verify_prefix))
            verify_prefix.append(proposal)
        bonus_token = self._target_argmax(verify_prefix)

        accepted = 0
        while (
            accepted < self.proposal_width
            and proposals[accepted] == target_tokens[accepted]
        ):
            accepted += 1

        correction_position = None
        if accepted == self.proposal_width:
            committed_tokens = proposals + [bonus_token]
        else:
            committed_tokens = proposals[:accepted] + [target_tokens[accepted]]
            correction_position = speculative_start + accepted

        for offset, token in enumerate(committed_tokens):
            self._write_cache(speculative_start + offset, token)
        self.committed.extend(committed_tokens)

        return DSparkRoundTrace(
            anchor=anchor,
            query_ids=query_ids,
            proposals=proposals,
            accepted_draft_tokens=accepted,
            committed_tokens=committed_tokens,
            committed_sequence=list(self.committed),
            proposal_overwrites=proposal_overwrites,
            correction_position=correction_position,
            draft_cache=list(self.draft_cache),
        )


def run_demo() -> List[DSparkRoundTrace]:
    """Run full-accept, middle-reject, then stale-slot-overwrite rounds."""
    vocab = 6
    width = 3
    w1 = torch.eye(vocab)
    w2 = torch.zeros(vocab, vocab)
    for previous in range(vocab):
        w2[(previous + 1) % vocab, previous] = 20.0

    def target_logits(prefix: Sequence[int]) -> torch.Tensor:
        logits = torch.full((vocab,), -20.0)
        logits[(int(prefix[-1]) + 1) % vocab] = 20.0
        return logits

    simulator = SingleRequestDSparkSimulator(
        prompt_tokens=[0],
        proposal_width=width,
        noise_token_id=vocab - 1,
        markov_w1=w1,
        markov_w2=w2,
        target_logits=target_logits,
    )
    traces = [simulator.run_greedy_round(torch.zeros(width, vocab))]

    middle_reject = torch.zeros(width, vocab)
    middle_reject[1, 2] = 50.0
    traces.append(simulator.run_greedy_round(middle_reject))
    traces.append(simulator.run_greedy_round(torch.zeros(width, vocab)))
    return traces


if __name__ == "__main__":
    import json

    print(json.dumps([trace.to_dict() for trace in run_demo()], indent=2))
