"""Model-independent DSpark proposer base.

DSpark evaluates one runtime-fixed query block ``[anchor, noise, ...]`` per
round and corrects the resulting base logits left-to-right with a low-rank
Markov bias.  :class:`DSparkProposerMixin` owns everything identical across model
families — the engine input contract, query-block geometry, committed-row
mapping and normalized proposal-hidden production — as an add-on base class::

    class DeepSeekV4DSparkModel(DSparkProposerMixin, DeepSeekV4Model): ...

following the same shape as ``QWen_VL(QWen, MultiModalMixin)``.  A model
family implements only the hooks that genuinely differ per model: feature
projection, KV injection + non-causal query-block attention, and proposal
hidden-state reduction/normalization.

Engine input contract — two standard-slot calls per round (all token rows
are request-major):

* **Commit** (``forward_commit``, incremental-prefill shape): ``input_ids`` = the committed
  tokens, ``attention_inputs.input_lengths`` = newly committed rows per
  request, ``attention_inputs.prefix_lengths`` = where they start,
  ``input_hiddens`` = the matching target feature rows, flattenable to
  ``[rows, aux_feature_dim]`` (a zero-copy view of the shared MTP hidden
  buffer).
* **Propose** (``forward_propose``, fixed-width block): ``input_ids`` = ``[B * width]`` query
  block (column zero is the anchor; the remaining columns are forced to
  the configured noise token here), ``attention_inputs.prefix_lengths`` =
  committed sequence length immediately before the query block. No
  feature input — the block reads the committed feature KV.

The Python model stops at normalized hidden-state production. The C++ model
wrapper applies the regular lm_head, then the speculative executor applies the
sequential Markov correction and samples with the framework's high-performance
sampler using each request's real sampling parameters.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from rtp_llm.models_py.modules.dsv4.kv_cache_utils import primary_attention_inputs
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs


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

    Subclasses call :meth:`init_dspark_proposer` once during construction and
    implement the three model-specific hooks. Everything else — per-round
    orchestration and query-block/committed-row geometry — is inherited.
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
        ):
            if int(value) <= 0:
                raise ValueError(f"DSpark {name} must be positive, got {value}")

        self._dspark_width = int(width)
        self._dspark_noise_token_id = int(noise_token_id)
        self._dspark_aux_feature_dim = int(aux_feature_dim)
        self._dspark_hidden_dim = int(hidden_dim)

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

    def compute_draft_hidden_states(self, hidden: torch.Tensor) -> torch.Tensor:
        """Reduce and normalize backbone output to ``[B*width, dim]``."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared per-round flow
    # ------------------------------------------------------------------

    def dspark_empty_outputs(
        self, batch_size: int, device: torch.device
    ) -> PyModelOutputs:
        """Zero-filled normalized hidden states with serving geometry."""
        width = self._dspark_width
        return PyModelOutputs(
            torch.zeros(
                (batch_size * width, self._dspark_hidden_dim),
                dtype=torch.bfloat16,
                device=device,
            )
        )

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
        # The device mirror feeds the tensor math below; the host field stays
        # the batch-size probe. Copying the pinned host buffer to the device is
        # a blocking H2D transfer, which CUDA rejects mid graph capture.
        lengths_source = optional_tensor(
            getattr(attention_inputs, "input_lengths_device", None)
        )
        if lengths_source is None:
            lengths_source = input_lengths
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

        prefix = optional_tensor(
            getattr(attention_inputs, "prefix_lengths_device", None)
        )
        if prefix is None:
            prefix = optional_tensor(getattr(attention_inputs, "prefix_lengths", None))
        if prefix is None or int(prefix.numel()) < batch_size:
            raise RuntimeError(
                "DSpark commit requires prefix_lengths with one value per request"
            )
        prefix_lengths = prefix[:batch_size].to(device=device, dtype=torch.long)
        lengths = lengths_source[:batch_size].to(device=device, dtype=torch.long)
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
        # The fixed-width commit CUDA graph owns a row-aligned output buffer even
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

        # Read the CUDA-resident mirror: copying the pinned host field to the
        # device is a blocking H2D transfer, which CUDA rejects while the decode
        # graph is capturing.
        prefix = optional_tensor(
            getattr(attention_inputs, "prefix_lengths_device", None)
        )
        if prefix is None:
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
        query_ids = torch.full_like(raw_ids, self._dspark_noise_token_id)
        if batch_size > 0:
            query_ids[:, 0].copy_(raw_ids[:, 0])
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

        return PyModelOutputs(self.compute_draft_hidden_states(hidden))
