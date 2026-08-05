"""Model-independent DSpark proposer base.

DSpark evaluates one runtime-fixed query block ``[anchor, noise, ...]`` per
round and corrects the resulting base logits left-to-right with a low-rank
Markov bias.  :class:`DSparkProposerMixin` owns everything identical across model
families — the engine input contract, query-block geometry, committed-row
mapping and the greedy Markov sampling tail — as an add-on base class::

    class DeepSeekV4DSparkModel(DSparkProposerMixin, DeepSeekV4Model): ...

following the same shape as ``QWen_VL(QWen, MultiModalMixin)``.  A model
family implements only the hooks that genuinely differ per model: feature
projection, KV injection + non-causal query-block attention, and the logits
head.

Engine input contract (all token rows are request-major):

* ``input_ids``: ``[B * width]`` query block.  Column zero is the anchor;
  the remaining columns are forced to the configured noise token here.
* ``input_hiddens``: target auxiliary features, flattenable to
  ``[rows, aux_feature_dim]``.
* ``dspark_ctx_lengths`` / ``dspark_ctx_starts``: number and source-row
  start of newly committed feature rows for each request.
* ``attention_inputs.prefix_lengths``: committed sequence length *after*
  those context rows and immediately before the query block.

The greedy sampling tail reproduces the reference proposer numerics exactly:
``softmax(base + bias)`` then ``argmax`` per step, the sampled token feeding
the next step's Markov bias.  Stochastic (Gumbel) draft sampling is
intentionally not implemented yet; it extends the tail without changing the
hook surface.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs


class DSparkMarkovHead(nn.Module):
    """Low-rank Markov transition head: ``bias(prev) = w2 @ w1[prev]``.

    Weights stay replicated in checkpoint dtype — the head runs a serial
    per-step loop, so TP-sharding it would add an all-reduce per step.
    """

    def __init__(
        self,
        markov_w1: torch.Tensor,
        markov_w2: torch.Tensor,
        *,
        vocab_size: int,
        rank: int,
    ):
        super().__init__()
        if tuple(markov_w1.shape) != (int(vocab_size), int(rank)):
            raise ValueError(
                f"unexpected DSpark markov_w1 shape: {tuple(markov_w1.shape)}, "
                f"expected [{vocab_size},{rank}]"
            )
        if tuple(markov_w2.shape) != tuple(markov_w1.shape):
            raise ValueError(
                "DSpark markov_w2 shape must match markov_w1, got "
                f"{tuple(markov_w2.shape)} vs {tuple(markov_w1.shape)}"
            )
        self.markov_w1 = markov_w1
        self.markov_w2 = markov_w2

    def embed(self, previous_tokens: torch.Tensor) -> torch.Tensor:
        return F.embedding(previous_tokens, self.markov_w1)

    def bias(self, previous_tokens: torch.Tensor) -> torch.Tensor:
        return F.linear(self.embed(previous_tokens), self.markov_w2).float()


def optional_tensor(value: Any) -> Optional[torch.Tensor]:
    if value is None or not isinstance(value, torch.Tensor):
        return None
    return value if value.numel() > 0 else None


def map_context_rows(
    starts: torch.Tensor,
    lengths: torch.Tensor,
    prefix_lengths: torch.Tensor,
    row_count: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map source feature rows to request ids and absolute positions.

    Rows outside every ``[start, start + length)`` interval are padding from a
    dense target-verify output and receive ``(-1, -1)``.  Keeping this
    transform independent from the feature projection also makes its
    packed/dense layout semantics directly unit-testable on CPU.
    """
    device = starts.device
    if row_count == 0:
        return (
            torch.empty(0, dtype=torch.int32, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
        )
    batch_size = int(starts.numel())
    if batch_size == 0:
        raise ValueError("cannot map non-empty DSpark context without requests")

    rows = torch.arange(row_count, device=device, dtype=torch.long)
    req = torch.searchsorted(starts.contiguous(), rows, right=True) - 1
    safe_req = req.clamp(0, batch_size - 1)
    valid = (req >= 0) & (rows >= starts[safe_req])
    valid = valid & (rows < starts[safe_req] + lengths[safe_req])

    local_offset = rows - starts[safe_req]
    positions = prefix_lengths[safe_req] - lengths[safe_req] + local_offset
    req = torch.where(valid, req, torch.full_like(req, -1))
    positions = torch.where(valid, positions, torch.full_like(positions, -1))
    return req.to(torch.int32), positions.to(torch.int32)


class DSparkProposerMixin:
    """Add-on base class granting a model the DSpark proposal capability.

    Subclasses call :meth:`init_dspark_proposer` once during construction,
    assign :attr:`markov_head` when weights load, and implement the three
    model-specific hooks.  Everything else — per-round orchestration, the
    query-block/committed-row geometry and the sampling tail — is inherited.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def init_dspark_proposer(
        self,
        *,
        width: int,
        noise_token_id: int,
        aux_feature_dim: int,
        hidden_dim: int,
        vocab_size: int,
    ) -> None:
        if int(width) <= 0:
            raise ValueError(f"DSpark width must be positive, got {width}")
        if int(noise_token_id) < 0:
            raise ValueError(
                f"DSpark noise token id must be non-negative, got {noise_token_id}"
            )
        for name, value in (
            ("aux_feature_dim", aux_feature_dim),
            ("hidden_dim", hidden_dim),
            ("vocab_size", vocab_size),
        ):
            if int(value) <= 0:
                raise ValueError(f"DSpark {name} must be positive, got {value}")

        self._dspark_width = int(width)
        self._dspark_noise_token_id = int(noise_token_id)
        self._dspark_aux_feature_dim = int(aux_feature_dim)
        self._dspark_hidden_dim = int(hidden_dim)
        self._dspark_vocab_size = int(vocab_size)
        self.markov_head: Optional[DSparkMarkovHead] = None

    # ------------------------------------------------------------------
    # Model-specific hooks
    # ------------------------------------------------------------------

    def combine_hidden_states(self, features: torch.Tensor) -> torch.Tensor:
        """Project packed target feature rows ``[rows, aux_feature_dim]`` to
        draft-input features ``[rows, hidden_dim]`` (bf16)."""
        raise NotImplementedError

    def forward_query_block(
        self,
        query_ids: torch.Tensor,
        query_positions: torch.Tensor,
        main_x: torch.Tensor,
        context_req_ids: torch.Tensor,
        context_positions: torch.Tensor,
        prefix_lengths: torch.Tensor,
        active_requests: torch.Tensor,
        inputs: PyModelInputs,
        fmha_impl: Any,
    ) -> torch.Tensor:
        """Commit ``main_x`` rows into the draft KV cache and evaluate the
        non-causal query block.  Returns backbone hidden states; called for
        empty batches too so collective layers stay balanced."""
        raise NotImplementedError

    def compute_draft_logits(
        self, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reduce backbone hidden states to ``(normalized [B*width, dim],
        base_logits [B, width, vocab])``."""
        raise NotImplementedError

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        """Translate draft-vocab token ids to target-vocab ids.  Identity by
        default; reduced-vocabulary checkpoints override."""
        return draft_ids

    # ------------------------------------------------------------------
    # Shared per-round flow
    # ------------------------------------------------------------------

    def dspark_empty_outputs(
        self, batch_size: int, device: torch.device
    ) -> PyModelOutputs:
        """Zero-filled outputs with the exact serving shapes and dtypes."""
        width = self._dspark_width
        outputs = PyModelOutputs(
            torch.zeros(
                (batch_size * width, self._dspark_hidden_dim),
                dtype=torch.bfloat16,
                device=device,
            )
        )
        outputs.draft_tokens = torch.zeros(
            (batch_size, width), dtype=torch.int32, device=device
        )
        outputs.draft_probs = torch.zeros(
            (batch_size, width, self._dspark_vocab_size),
            dtype=torch.float32,
            device=device,
        )
        return outputs

    def run_draft_step(
        self,
        inputs: PyModelInputs,
        fmha_impl: Any,
        device: torch.device,
    ) -> PyModelOutputs:
        """Execute one DSpark draft round on an initialized model.

        The subclass owns readiness checks (weights loaded, KV cache
        present); this method owns the input contract and the
        model-independent round shape."""
        width = self._dspark_width

        attention_inputs = inputs.attention_inputs
        input_lengths = optional_tensor(
            getattr(attention_inputs, "input_lengths", None)
        )
        batch_size = int(input_lengths.numel()) if input_lengths is not None else 0
        expected_tokens = batch_size * width
        if int(inputs.input_ids.numel()) != expected_tokens:
            raise RuntimeError(
                "DSpark input_ids must contain exactly B*gamma tokens: "
                f"numel={inputs.input_ids.numel()}, batch={batch_size}, "
                f"gamma={width}"
            )

        prefix = optional_tensor(getattr(attention_inputs, "prefix_lengths", None))
        if batch_size > 0 and (prefix is None or int(prefix.numel()) < batch_size):
            raise RuntimeError(
                "DSpark requires prefix_lengths with one value per request"
            )
        prefix_lengths = (
            prefix[:batch_size].to(device=device, dtype=torch.int32)
            if prefix is not None
            else torch.empty(0, dtype=torch.int32, device=device)
        )

        raw_ids = inputs.input_ids.to(device=device, dtype=torch.int32).view(
            batch_size, width
        )
        anchors = raw_ids[:, 0].clone()
        query_ids = torch.full_like(raw_ids, self._dspark_noise_token_id)
        if batch_size > 0:
            query_ids[:, 0].copy_(anchors)
        query_positions = prefix_lengths.to(torch.long).view(batch_size, 1)
        query_positions = query_positions + torch.arange(
            width, device=device, dtype=torch.long
        ).view(1, width)

        main_x, context_req_ids, context_positions = self._parse_committed_context(
            inputs, prefix_lengths.to(torch.long), batch_size, device
        )
        ctx_lengths = optional_tensor(getattr(inputs, "dspark_ctx_lengths", None))
        active_requests = (
            ctx_lengths[:batch_size].to(device=device) > 0
            if ctx_lengths is not None
            else torch.zeros(batch_size, dtype=torch.bool, device=device)
        )

        hidden = self.forward_query_block(
            query_ids,
            query_positions,
            main_x,
            context_req_ids,
            context_positions,
            prefix_lengths,
            active_requests,
            inputs,
            fmha_impl,
        )

        # Empty DP ranks must still execute every collective layer above so EP
        # stays balanced; only the non-collective head is skipped.
        if batch_size == 0:
            return self.dspark_empty_outputs(0, device)

        normalized, base_logits = self.compute_draft_logits(hidden)
        draft_tokens, draft_probs = self._sample_sequential_markov(
            base_logits, anchors
        )
        outputs = PyModelOutputs(normalized)
        outputs.draft_tokens = draft_tokens
        outputs.draft_probs = draft_probs
        return outputs

    def _parse_committed_context(
        self,
        inputs: PyModelInputs,
        prefix_lengths: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``main_x, req_ids, positions`` for received target rows.

        Explicit ``starts`` may describe a dense target-verify buffer with
        holes after each request's accepted prefix.  ``searchsorted`` maps all
        source rows to requests without a device-to-host length read; hole
        rows retain a ``-1`` request/position and are skipped by the cache
        writer.
        """
        aux_dim = self._dspark_aux_feature_dim
        hidden = optional_tensor(getattr(inputs, "input_hiddens", None))
        lengths = optional_tensor(getattr(inputs, "dspark_ctx_lengths", None))

        def _empty() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return (
                torch.empty(
                    (0, self._dspark_hidden_dim),
                    dtype=torch.bfloat16,
                    device=device,
                ),
                torch.empty(0, dtype=torch.int32, device=device),
                torch.empty(0, dtype=torch.int32, device=device),
            )

        if batch_size == 0:
            return _empty()
        if hidden is None:
            raise RuntimeError("DSpark requires target features in input_hiddens")
        if lengths is None or int(lengths.numel()) < batch_size:
            raise RuntimeError(
                "DSpark requires dspark_ctx_lengths with one value per request"
            )

        lengths = lengths[:batch_size].to(device=device, dtype=torch.long)
        starts_in = optional_tensor(getattr(inputs, "dspark_ctx_starts", None))
        if starts_in is None:
            starts = lengths.cumsum(0) - lengths
        else:
            if int(starts_in.numel()) < batch_size:
                raise RuntimeError(
                    "DSpark dspark_ctx_starts has fewer values than the batch"
                )
            starts = starts_in[:batch_size].to(device=device, dtype=torch.long)

        if hidden.numel() % aux_dim != 0:
            raise RuntimeError(
                "DSpark target feature tensor cannot be reshaped to the "
                f"configured width {aux_dim}: shape={tuple(hidden.shape)}"
            )
        features = hidden.reshape(-1, aux_dim).to(device=device)
        row_count = int(features.shape[0])
        if row_count == 0:
            return _empty()

        main_x = self.combine_hidden_states(features)
        req, positions = map_context_rows(
            starts, lengths, prefix_lengths, row_count
        )
        return main_x, req, positions

    def _sample_sequential_markov(
        self,
        base_logits: torch.Tensor,
        anchor_ids: torch.Tensor,
        need_probabilities: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the left-to-right greedy Markov correction chain.

        Numerics follow the reference proposer exactly: per step,
        ``probs = softmax(base + bias)`` in fp32 and the argmax over ``probs``
        becomes both the proposal token and the next step's Markov input.
        No host synchronization occurs.
        """
        if self.markov_head is None:
            raise RuntimeError("DSpark markov head is not loaded")
        if base_logits.dim() != 3:
            raise ValueError(
                "DSpark base logits must be [B,width,V], got "
                f"{tuple(base_logits.shape)}"
            )
        batch, width, _vocab = (int(v) for v in base_logits.shape)
        if width != self._dspark_width:
            raise ValueError(
                f"DSpark base logits width {width} does not match the "
                f"configured width {self._dspark_width}"
            )
        if tuple(anchor_ids.shape) != (batch,):
            raise ValueError(
                f"DSpark anchors must be [{batch}], got {tuple(anchor_ids.shape)}"
            )

        previous = anchor_ids
        tokens = []
        probabilities = [] if need_probabilities else None
        for step in range(width):
            logits = base_logits[:, step] + self.markov_head.bias(previous)
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
            next_token = torch.argmax(probs, dim=-1)
            if probabilities is not None:
                probabilities.append(probs)
            tokens.append(next_token.to(torch.int32))
            previous = self.map_draft_to_target(next_token)

        draft_tokens = torch.stack(tokens, dim=1).contiguous()
        draft_probs = (
            torch.stack(probabilities, dim=1).contiguous()
            if probabilities is not None
            else None
        )
        return draft_tokens, draft_probs
