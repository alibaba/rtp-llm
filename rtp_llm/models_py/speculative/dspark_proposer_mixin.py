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

Engine input contract — two standard-slot calls per round (all token rows
are request-major):

* **Commit** (``dspark_call_phase=COMMIT``, incremental-prefill shape): ``input_ids`` = the committed
  tokens, ``attention_inputs.input_lengths`` = newly committed rows per
  request, ``attention_inputs.prefix_lengths`` = where they start,
  ``input_hiddens`` = the matching target feature rows, flattenable to
  ``[rows, aux_feature_dim]`` (a zero-copy view of the shared MTP hidden
  buffer).
* **Propose** (``dspark_call_phase=PROPOSE``, fixed-width block): ``input_ids`` = ``[B * width]`` query
  block (column zero is the anchor; the remaining columns are forced to
  the configured noise token here), ``attention_inputs.prefix_lengths`` =
  committed sequence length immediately before the query block. No
  feature input — the block reads the committed feature KV.

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

from rtp_llm.models_py.modules.dsv4.kv_cache_utils import primary_attention_inputs
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
    committed_ends: torch.Tensor,
    row_count: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map source feature rows to request ids and absolute positions.

    ``committed_ends`` is each request's sequence length after this commit
    (``prefix + newly committed``); row positions count back from it.  Rows
    outside every ``[start, start + length)`` interval are padding from a
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
    positions = committed_ends[safe_req] - lengths[safe_req] + local_offset
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

    def commit_feature_rows(
        self,
        main_x: torch.Tensor,
        context_req_ids: torch.Tensor,
        context_positions: torch.Tensor,
        committed_ends: torch.Tensor,
        inputs: PyModelInputs,
        commit_ctx: Any = None,
    ) -> None:
        """Write projected feature rows ``main_x`` into every draft layer's
        KV cache (per-layer KV projection of the same rows — features are not
        propagated through the layers).  ``committed_ends`` is each request's
        sequence length after this commit.  ``commit_ctx`` is whatever the
        model's ``map_commit_rows`` returned for this call (e.g. a CPContext
        describing the row layout); it never outlives the call."""
        raise NotImplementedError

    def map_commit_rows(
        self,
        starts: torch.Tensor,
        lengths: torch.Tensor,
        committed_ends: torch.Tensor,
        row_count: int,
        inputs: PyModelInputs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """Map each committed feature row to ``(request id, absolute
        position, commit_ctx)``.  The default assumes the rows are the
        request-major packed layout described by ``starts``/``lengths``;
        models whose engine supplies a different row layout (e.g. rank-local
        rows under prefill CP) override this hook, derive the layout from the
        CP metadata on ``inputs.attention_inputs``, and return the derived
        context as the opaque third value for ``commit_feature_rows``."""
        req, positions = map_context_rows(starts, lengths, committed_ends, row_count)
        return req, positions, None

    def forward_query_block(
        self,
        query_ids: torch.Tensor,
        query_positions: torch.Tensor,
        prefix_lengths: torch.Tensor,
        active_requests: torch.Tensor,
        inputs: PyModelInputs,
        fmha_impl: Any,
    ) -> torch.Tensor:
        """Evaluate the non-causal query block against the previously
        committed feature KV.  Returns backbone hidden states; called for
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
        return outputs

    def run_commit_step(
        self,
        inputs: PyModelInputs,
        device: torch.device,
    ) -> PyModelOutputs:
        """Commit target feature rows into the draft KV cache.

        A standard incremental-prefill call: ``input_lengths`` is the number
        of newly committed rows per request (prompt suffix at seeding, the
        dense gamma+1 verify rows at the decode tail — rejected rows are
        overwritten in place next round, the same self-healing MTP relies
        on), ``prefix_lengths`` is where they start, and ``input_hiddens``
        carries the feature rows packed in the same request-major order.
        Produces no logits."""
        attention_inputs = primary_attention_inputs(
            inputs.attention_inputs, getattr(self, "kv_cache", None)
        )
        input_lengths = optional_tensor(
            getattr(attention_inputs, "input_lengths", None)
        )
        batch_size = int(input_lengths.numel()) if input_lengths is not None else 0
        hidden = optional_tensor(getattr(inputs, "input_hiddens", None))

        def _empty_outputs() -> PyModelOutputs:
            return PyModelOutputs(
                torch.zeros(
                    (0, self._dspark_hidden_dim),
                    dtype=torch.bfloat16,
                    device=device,
                )
            )

        if batch_size == 0 or hidden is None or hidden.numel() == 0:
            return _empty_outputs()

        aux_dim = self._dspark_aux_feature_dim
        if hidden.numel() % aux_dim != 0:
            raise RuntimeError(
                "DSpark target feature tensor cannot be reshaped to the "
                f"configured width {aux_dim}: shape={tuple(hidden.shape)}"
            )
        features = hidden.reshape(-1, aux_dim).to(device=device)
        row_count = int(features.shape[0])

        prefix = optional_tensor(getattr(attention_inputs, "prefix_lengths", None))
        if prefix is None or int(prefix.numel()) < batch_size:
            raise RuntimeError(
                "DSpark commit requires prefix_lengths with one value per request"
            )
        prefix_lengths = prefix[:batch_size].to(device=device, dtype=torch.long)
        lengths = input_lengths[:batch_size].to(device=device, dtype=torch.long)
        starts = lengths.cumsum(0) - lengths

        # Under prefill CP the framework rewrites input_lengths to rank-local
        # chunk lengths; committed ends are global sequence positions, so read
        # the pre-split lengths back off the CP metadata when present.  (For
        # dense forwards the metadata is absent or equals input_lengths.)
        committed_lengths = lengths
        cp_info = getattr(attention_inputs, "context_parallel_info", None)
        global_lengths = (
            optional_tensor(getattr(cp_info, "prefill_actual_input_lengths_cpu", None))
            if cp_info is not None
            else None
        )
        if global_lengths is not None and int(global_lengths.numel()) >= batch_size:
            committed_lengths = global_lengths[:batch_size].to(
                device=device, dtype=torch.long
            )
        committed_ends = prefix_lengths + committed_lengths

        # Row windows are the plain prefix sum of the standard input_lengths;
        # positions continue each request's committed prefix. All rows are
        # payload — the commit call's geometry is exactly its row layout
        # (models with a different engine-supplied layout override
        # map_commit_rows).
        req, positions, commit_ctx = self.map_commit_rows(
            starts, lengths, committed_ends, row_count, inputs
        )

        main_x = self.combine_hidden_states(features)
        self.commit_feature_rows(
            main_x,
            req,
            positions,
            committed_ends.to(torch.int32),
            inputs,
            commit_ctx=commit_ctx,
        )
        # The generic prefill CUDA graph owns a row-aligned output buffer even
        # though the executor only needs this call's KV-cache side effect.
        return PyModelOutputs(main_x)

    def run_propose_step(
        self,
        inputs: PyModelInputs,
        fmha_impl: Any,
        device: torch.device,
    ) -> PyModelOutputs:
        """Evaluate one fixed-width proposal block on an initialized model.

        The query block reads the committed feature KV written by
        :meth:`run_commit_step`; the call carries no feature input."""
        width = self._dspark_width

        attention_inputs = primary_attention_inputs(
            inputs.attention_inputs, getattr(self, "kv_cache", None)
        )
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

        # Live requests always carry a non-empty committed prefix (at least
        # the anchor's context); zero-prefix slots are CUDA-graph padding.
        active_requests = prefix_lengths > 0

        hidden = self.forward_query_block(
            query_ids,
            query_positions,
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
        draft_tokens = self._sample_sequential_markov(base_logits, anchors)
        outputs = PyModelOutputs(normalized)
        outputs.draft_tokens = draft_tokens
        return outputs

    def _sample_sequential_markov(
        self,
        base_logits: torch.Tensor,
        anchor_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run the left-to-right greedy Markov correction chain.

        Numerics follow the reference proposer exactly: per step the argmax
        over ``base + bias`` (softmax is monotone, so it never changes the
        argmax) becomes both the proposal token and the next step's Markov
        input.  No host synchronization occurs.

        The proposal is deterministic, so its q distribution is the point
        mass on the emitted token; rejection sampling receives that one-hot
        from the engine (built C++-side) instead of a full-vocabulary
        softmax materialized here.
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
        for step in range(width):
            logits = base_logits[:, step] + self.markov_head.bias(previous)
            next_token = torch.argmax(logits.float(), dim=-1)
            tokens.append(next_token.to(torch.int32))
            previous = self.map_draft_to_target(next_token)

        return torch.stack(tokens, dim=1).contiguous()
