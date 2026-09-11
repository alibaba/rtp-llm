"""Target-only continuation of a completed local CED prefill.

The scheduler owns sampling and EP coordination. This component preserves the
canonical materialized boundary; it does not implement PD or DSpark acceptance.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from rtp_llm.models_py.modules.dsv41.engram import committed_history
from rtp_llm.models_py.modules.dsv41.inputs import V41ModelRows


class V41TargetContinuation:
    @classmethod
    @torch.inference_mode()
    def from_prefill(cls, executor, result):
        cache = executor.cache
        end = executor.plan.extends[-1].encoder_rows.end
        if (
            executor.canonical is None
            or executor.next_extend != len(executor.plan.extends)
            or result.output is None
            or result.context.cache is not cache
            or result.context.epoch != cache.active_epoch
            or result.context.end != end
            or cache.poisoned
        ):
            raise ValueError("target continuation requires the completed final prefill")
        executor.progress.require_handoff(end, executor.plan.protected_checkpoint_end)
        if any(cache.swa_ends.get(layer) != end for layer in cache.swa) or any(
            owner.materialized_end != end for owner in cache.owners.values()
        ):
            raise ValueError("prefill handoff has incomplete target/draft/global state")
        tail = executor.canonical.rows(
            max(0, end - 3), end, device=executor.target.embedding.device
        )
        return cls(executor.target, cache, end, tail, result.output)

    def __init__(self, target, cache, materialized_end, tail, output):
        tail.validate()
        if (
            type(materialized_end) is not int
            or not 0 < materialized_end <= cache.max_tokens
            or tail.token_ids.numel() != min(3, materialized_end)
            or output.hidden_states.shape[0] == 0
        ):
            raise ValueError("target continuation requires a complete canonical tail")
        self.target, self.cache, self.output = target, cache, output
        self.materialized_end = materialized_end
        self.identity = cache.identity
        self.request_id = cache.request_id
        self.epoch = cache.active_epoch
        self.history_ids = torch.zeros(
            (1, 3), dtype=torch.int32, device=target.embedding.device
        )
        self.history_valid = torch.zeros_like(self.history_ids, dtype=torch.bool)
        self.history_ids[:, -tail.token_ids.numel() :].copy_(tail.token_ids)
        self.history_valid[:, -tail.token_ids.numel() :].copy_(tail.text_mask)

    def _validate_boundary(self):
        if (
            self.cache.poisoned
            or self.cache.identity != self.identity
            or self.cache.request_id != self.request_id
            or self.cache.active_epoch != self.epoch
            or any(
                self.cache.swa_ends.get(layer) != self.materialized_end
                for layer in range(40)
            )
            or any(
                owner.materialized_end != self.materialized_end
                for owner in self.cache.owners.values()
            )
        ):
            raise RuntimeError("target continuation boundary is stale or incomplete")

    def logits(self):
        """Sampling does not materialize the selected output token."""
        self._validate_boundary()
        return self.target.logits(self.output.hidden_states[-1:])

    @torch.inference_mode()
    def advance(self, token_id: int | None):
        """Materialize one selected token, or participate with zero local rows.

        EOS/stop and output limits belong to the caller: a terminal sampled
        token need not be materialized. Empty ranks still execute all EP layers.
        Only the three canonical predecessors survive each successful step.
        """
        self._validate_boundary()
        if token_id is not None and (
            type(token_id) is not int or not 0 <= token_id < 129280
        ):
            raise ValueError("target decode requires a canonical integer token ID")
        count = int(token_id is not None)
        end = self.materialized_end + count
        if end > min(self.cache.max_tokens, 1048576):
            raise ValueError("target decode exceeds the admitted context capacity")
        ids = torch.tensor(
            [] if token_id is None else [token_id],
            dtype=torch.int32,
            device=self.history_ids.device,
        )
        rows = V41ModelRows(
            ids,
            torch.full_like(ids, -1),
            torch.ones(count, dtype=torch.bool, device=ids.device),
            self.history_ids[:count].clone(),
            self.history_valid[:count].clone(),
        )
        context = self.cache.begin_forward(
            epoch=self.epoch + 1, start=self.materialized_end, end=end
        )
        try:
            output = self.target(rows, context, execution_mode="full")
            if (
                context.completed_layers != set(range(40))
                or any(self.cache.swa_ends.get(layer) != end for layer in range(40))
                or any(
                    owner.materialized_end != end
                    for owner in self.cache.owners.values()
                )
            ):
                raise RuntimeError(
                    "a target decode row did not complete all forty layers"
                )
            if count:
                self.history_ids, self.history_valid = committed_history(
                    self.history_ids,
                    self.history_valid,
                    ids[None, :],
                    rows.text_mask[None, :],
                )
                self.output = output
            self.materialized_end, self.epoch = end, context.epoch
            return output, context
        except Exception:
            self.cache.poisoned = True
            raise


@dataclass(frozen=True)
class V41GenerationStep:
    token_id: int | None
    finish_reason: str | None
    active_ranks: int
    context: object | None


class V41TargetGeneration:
    """Greedy target generation for one independent request per EP rank.

    All ranks call step in the same order until globally_complete. A locally
    finished or cancelled request keeps participating with zero target rows.
    PD admission, DSpark and batched request scheduling are separate owners.
    """

    def __init__(self, continuation, *, max_output_tokens: int, process_group=None):
        continuation._validate_boundary()
        if (
            type(max_output_tokens) is not int
            or max_output_tokens <= 0
            or continuation.materialized_end + max_output_tokens
            > min(continuation.cache.max_tokens, 1048576)
        ):
            raise ValueError("generation requires an admitted positive output budget")
        if not dist.is_initialized():
            raise RuntimeError("target generation requires its initialized EP group")
        if process_group is not None and process_group is not dist.group.WORLD:
            raise ValueError("target generation must use the model's WORLD EP group")
        self.continuation = continuation
        self.max_output_tokens = max_output_tokens
        self.process_group = process_group
        self.generated_token_ids: list[int] = []
        self.finish_reason: str | None = None
        self.globally_complete = False
        self.failed = False
        self._active = torch.zeros(
            (), dtype=torch.int32, device=continuation.history_ids.device
        )

    def cancel(self):
        """Stop local sampling; the caller must continue collective steps."""
        if self.failed:
            raise RuntimeError("target generation has failed")
        if self.finish_reason is None:
            self.finish_reason = "cancelled"

    @torch.inference_mode()
    def step(self) -> V41GenerationStep:
        if self.failed or self.globally_complete:
            raise RuntimeError("target generation is no longer runnable")
        token = None
        try:
            self.continuation._validate_boundary()
            if self.finish_reason is None:
                logits = self.continuation.logits()
                if logits.shape != (1, 129280) or not bool(
                    torch.isfinite(logits).all()
                ):
                    raise RuntimeError(
                        "target sampling requires finite full-vocab logits"
                    )
                token = int(logits.argmax(-1).item())
                self.generated_token_ids.append(token)
                if token == self.continuation.target.config.eos_token_id:
                    self.finish_reason = "stop"
                elif len(self.generated_token_ids) == self.max_output_tokens:
                    self.finish_reason = "length"
            self._active.fill_(int(self.finish_reason is None))
            dist.all_reduce(
                self._active, op=dist.ReduceOp.SUM, group=self.process_group
            )
            active = int(self._active.item())
            self.globally_complete = active == 0
            context = None
            if active:
                _, context = self.continuation.advance(
                    token if self.finish_reason is None else None
                )
            return V41GenerationStep(token, self.finish_reason, active, context)
        except Exception:
            self.failed = True
            self.continuation.cache.poisoned = True
            raise
