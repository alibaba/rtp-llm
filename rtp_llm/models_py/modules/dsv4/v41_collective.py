"""Explicit, fixed-capacity V4.1 EP collectives behind the serving gate.

Prefill uses stage-local BF16 gathering; decode gathers group-32 FP8 payloads.
Both return local expert contributions in FP32 and sum peers in rank order.
The scheduler must give all members the same invocation/chunk/graph schedule.
"""

import hashlib
import json
from dataclasses import asdict, dataclass

import torch
from torch import nn

from rtp_llm.models_py.modules.dsv4.v41_dispatch import V41DispatchLayout, pack_dispatch
from rtp_llm.models_py.modules.dsv4.v41_expert import _check_limit
from rtp_llm.models_py.modules.dsv4.v41_moe import V41LocalMoE


@dataclass(frozen=True)
class V41EPPlan:
    dispatch: V41DispatchLayout
    role: str
    pp_rank: int
    layer_id: int
    world_ranks: tuple
    generation: str
    rows_per_rank: int
    swiglu_limit: float = 10.0

    def __post_init__(self):
        _check_limit(self.swiglu_limit)
        if type(self.dispatch) is not V41DispatchLayout:
            raise ValueError("V4.1 EP requires its immutable dispatch layout")
        if self.role not in ("prefill", "decode"):
            raise ValueError("V4.1 EP role must be prefill or decode")
        world = 4 if self.role == "prefill" else 8
        if (
            type(self.world_ranks) is not tuple
            or len(self.world_ranks) != world
            or any(type(rank) is not int or rank < 0 for rank in self.world_ranks)
            or len(set(self.world_ranks)) != world
            or self.dispatch.experts % world
            or type(self.rows_per_rank) is not int
            or self.rows_per_rank <= 0
            or type(self.generation) is not str
            or not self.generation
        ):
            raise ValueError(
                "V4.1 EP requires exact membership, generation and row bounds"
            )
        if type(self.pp_rank) is not int or type(self.layer_id) is not int:
            raise ValueError("V4.1 EP stage/layer IDs must be integers")
        begin, end = (0, 40)
        if self.role == "prefill":
            if self.pp_rank not in (0, 1):
                raise ValueError("V4.1 prefill requires two pipeline stages")
            begin, end = (0, 21) if self.pp_rank == 0 else (21, 40)
        elif self.pp_rank != 0:
            raise ValueError("V4.1 decode has one pipeline stage")
        if not begin <= self.layer_id < end:
            raise ValueError("V4.1 EP global layer is outside its pipeline stage")

    @property
    def fingerprint(self):
        metadata = {
            "protocol": "dsv41-fixed-ep-v1",
            "input": "bf16" if self.role == "prefill" else "group32_fp8",
            "return": "fp32_rank_order_sum",
            "plan": asdict(self),
        }
        return hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).digest()


class V41CollectiveMoE(nn.Module):
    """A bounded baseline with explicit group admission and invocation storage.

    Call ``admit`` collectively outside graph capture before use. The supplied
    group must be the exact EP group, including its stage-local membership.
    No WORLD inference, global workspace cache or transport fallback is used.
    Failures require worker/group teardown by the caller. This component is
    deliberately not registered as a serving strategy yet.
    """

    def __init__(self, local, plan, process_group):
        super().__init__()
        if not isinstance(local, V41LocalMoE) or type(plan) is not V41EPPlan:
            raise ValueError("V4.1 EP requires local experts and an immutable plan")
        if process_group is None:
            raise ValueError("V4.1 EP requires an explicit process group")
        self.local, self.plan, self.process_group = local, plan, process_group
        self._admitted = False
        self._identity = None

    def _runtime_identity(self):
        local, experts = self.local, self.local.experts
        return (
            self.plan,
            id(self.process_group),
            experts.device,
            id(local),
            id(experts),
            local.expert_start_id,
            experts.experts,
            experts.dim,
            experts.backend,
            experts.swiglu_limit,
            local.max_tokens,
            local.masked_token_limit,
            local.max_topk,
        )

    def _local_matches(self):
        dist, group, plan = torch.distributed, self.process_group, self.plan
        world = len(plan.world_ranks)
        local = self.local
        return (
            dist.get_backend(group) == "nccl"
            and dist.get_world_size(group) == world
            and tuple(dist.get_process_group_ranks(group)) == plan.world_ranks
            and local.experts.backend == "deepgemm"
            and local.experts.dim == plan.dispatch.hidden
            and local.experts.experts == plan.dispatch.experts // world
            and local.expert_start_id == dist.get_rank(group) * local.experts.experts
            and local.max_topk >= plan.dispatch.topk
            and local.max_tokens >= world * plan.rows_per_rank
            and local.experts.swiglu_limit == plan.swiglu_limit
            and (
                plan.role == "prefill"
                or local.masked_token_limit >= world * plan.rows_per_rank
            )
        )

    def admit(self):
        dist, group = torch.distributed, self.process_group
        device = self.local.experts.device
        if not dist.is_initialized() or device.type != "cuda":
            raise RuntimeError("V4.1 EP admission requires CUDA and initialized NCCL")
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("V4.1 EP admission must precede graph capture")
            self._admitted = False
            world = dist.get_world_size(group)
            signature = torch.tensor(
                [*self.plan.fingerprint, int(self._local_matches())],
                dtype=torch.uint8,
                device=device,
            ).view(1, 33)
            signatures = torch.empty((world, 33), dtype=torch.uint8, device=device)
            dist.all_gather_into_tensor(signatures, signature, group=group)
            gathered = signatures.cpu()
            if not bool((gathered[:, :32] == gathered[0, :32]).all()) or not bool(
                (gathered[:, 32] == 1).all()
            ):
                raise RuntimeError(
                    "V4.1 EP peer plan or local expert admission mismatch"
                )
        self._identity = self._runtime_identity()
        self._admitted = True

    def forward(self, x, weights, indices, *, valid_rows=None):
        plan, group, device = self.plan, self.process_group, self.local.experts.device
        if not self._admitted or self._identity != self._runtime_identity():
            raise RuntimeError("V4.1 EP requires admission for this group/plan/device")
        if (
            x.ndim != 2
            or x.shape[1] != plan.dispatch.hidden
            or x.dtype != torch.bfloat16
            or x.device != device
            or not x.is_contiguous()
            or x.requires_grad
            or x.shape[0] > plan.rows_per_rank
            or weights.shape != (x.shape[0], plan.dispatch.topk)
            or weights.dtype != torch.float32
            or weights.device != device
            or not weights.is_contiguous()
            or weights.requires_grad
            or indices.shape != weights.shape
            or indices.dtype not in (torch.int32, torch.int64)
            or indices.device != device
            or not indices.is_contiguous()
        ):
            raise ValueError(
                "V4.1 EP inputs disagree with the admitted row/route contract"
            )
        if valid_rows is None:
            valid_rows = torch.ones(x.shape[0], device=device, dtype=torch.bool)
        if (
            valid_rows.shape != (x.shape[0],)
            or valid_rows.dtype != torch.bool
            or valid_rows.device != device
            or not valid_rows.is_contiguous()
        ):
            raise ValueError("V4.1 EP requires a contiguous device row mask")
        with torch.cuda.device(device):
            if plan.role == "prefill" and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "V4.1 prefill EP uses eager contiguous expert counts"
                )
            return self._forward(x, weights, indices, valid_rows)

    def _forward(self, x, weights, indices, valid_rows):
        from rtp_llm.models_py.modules.dsv4.v41_dispatch import _validate_routes

        plan, group, dist = self.plan, self.process_group, torch.distributed
        world, capacity = len(plan.world_ranks), plan.rows_per_rank
        rows, dim = x.shape

        def gather(tensor):
            result = torch.empty(
                (world * capacity, *tensor.shape[1:]),
                dtype=tensor.dtype,
                device=x.device,
            )
            dist.all_gather_into_tensor(result, tensor, group=group)
            return result

        if plan.role == "decode":
            payload = plan.dispatch.allocate(capacity, device=x.device)
            pack_dispatch(x, weights, indices, valid_rows, plan.dispatch, out=payload)
            partial = self.local.forward_received(
                gather(payload), plan.dispatch, layout="masked"
            )
        else:
            active = _validate_routes(
                weights, indices, valid_rows, plan.dispatch.experts
            )
            padded_x = torch.zeros(
                (capacity, dim), device=x.device, dtype=torch.bfloat16
            )
            padded_w = torch.zeros(
                (capacity, plan.dispatch.topk), device=x.device, dtype=torch.float32
            )
            padded_i = torch.full(
                (capacity, plan.dispatch.topk), -1, device=x.device, dtype=torch.int32
            )
            padded_x[:rows].copy_(torch.where(active.any(-1)[:, None], x, 0))
            padded_w[:rows].copy_(torch.where(active, weights, 0))
            padded_i[:rows].copy_(torch.where(active, indices, -1).int())
            partial = self.local(
                gather(padded_x),
                gather(padded_w),
                gather(padded_i),
                layout="contiguous",
            )
        received = torch.empty_like(partial)
        dist.all_to_all_single(received, partial, group=group)
        output = torch.zeros((capacity, dim), device=x.device, dtype=torch.float32)
        for peer in range(world):
            output.add_(received[peer * capacity : (peer + 1) * capacity])
        return output[:rows]
