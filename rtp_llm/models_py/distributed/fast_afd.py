"""Point-to-point routed MoE execution for the first FastAFD topology.

The attention ranks own attention, routing and shared experts. One expert rank
owns routed experts. Each attention rank sends one request at a time and waits
for its result. The expert rank visits active attention ranks in rank order,
which permits different microbatch counts and layer positions on each rank.

Every attention rank must call ``finish`` once after its forward_micro_batch,
including when it receives only dummy inputs. No zero-element NCCL transfer is
issued: the fixed-size request header carries empty requests and end markers.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, recv, send

_PROTOCOL_VERSION = 1
_HEADER_LENGTH = 9
_STATUS_LENGTH = 1
_MAX_TOKENS = 131072


class _Kind(IntEnum):
    REQUEST = 1
    FINISH = 2
    ABORT = 3
    STOP = 4


class _Status(IntEnum):
    OK = 0
    BAD_VERSION = 1
    BAD_KIND = 2
    BAD_LAYER = 3
    BAD_SHAPE = 4
    BAD_DTYPE = 5
    EXPERT_FAILED = 6
    BAD_OUTPUT = 7
    RESOURCE_FAILED = 8
    BAD_STEP = 9
    STEP_FAILED = 10
    ALL_IDLE = 11


_DTYPE_CODE = {torch.float16: 1, torch.bfloat16: 2}
_IDS_CODE = {torch.int32: 1, torch.int64: 2}
_CODE_IDS = {value: key for key, value in _IDS_CODE.items()}


@dataclass
class _PendingRequest:
    rank: int
    layer_idx: int
    token_count: int
    hidden: torch.Tensor
    ids: torch.Tensor
    weights: torch.Tensor


def _check_common(
    hidden_size: int,
    top_k: int,
    expert_count: int,
    device: torch.device | str,
    activation_dtype: torch.dtype,
) -> torch.device:
    if hidden_size <= 0 or top_k <= 0 or expert_count <= 0:
        raise ValueError("hidden_size, top_k and expert_count must be positive")
    if top_k > expert_count:
        raise ValueError("top_k cannot exceed expert_count")
    if activation_dtype not in _DTYPE_CODE:
        raise ValueError("FastAFD supports only float16 and bfloat16 activations")
    result = torch.device(device)
    if result.type not in ("cpu", "cuda"):
        raise ValueError("FastAFD requires a CPU test device or a CUDA device")
    if result.type == "cuda" and result.index is None:
        result = torch.device("cuda", torch.cuda.current_device())
    return result


def _header(
    device: torch.device,
    kind: _Kind,
    step_id: int,
    layer_idx: int = -1,
    token_count: int = 0,
    hidden_size: int = 0,
    top_k: int = 0,
    dtype_code: int = 0,
    ids_code: int = 0,
) -> torch.Tensor:
    return torch.tensor(
        [
            _PROTOCOL_VERSION,
            int(kind),
            step_id,
            layer_idx,
            token_count,
            hidden_size,
            top_k,
            dtype_code,
            ids_code,
        ],
        device=device,
        dtype=torch.int64,
    )


def _send_status(device: torch.device, rank: int, status: _Status) -> None:
    send(
        torch.tensor([int(status)], device=device, dtype=torch.int64),
        rank,
        Group.DP_AND_TP,
    )


class FastAFDClient:
    """Synchronous routed-expert proxy for one attention rank.

    The caller computes gate/top-k and shared experts locally. ``forward``
    returns only the routed-expert contribution. The caller adds its shared
    expert contribution using the normal GenericMoeLayer merge path.
    """

    def __init__(
        self,
        service_rank: int,
        hidden_size: int,
        top_k: int,
        device: torch.device | str,
        activation_dtype: torch.dtype,
        expert_count: int,
        topk_ids_dtype: torch.dtype = torch.int32,
    ) -> None:
        self.device = _check_common(
            hidden_size, top_k, expert_count, device, activation_dtype
        )
        if service_rank < 1:
            raise ValueError("service_rank must be the rank after attention ranks")
        if topk_ids_dtype not in _IDS_CODE:
            raise ValueError("FastAFD top-k ids must be int32 or int64")
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            if service_rank != world_size - 1 or rank >= service_rank:
                raise ValueError(
                    "FastAFD requires N attention ranks and one final expert rank"
                )
        self.service_rank = service_rank
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.activation_dtype = activation_dtype
        self.expert_count = expert_count
        self.topk_ids_dtype = topk_ids_dtype
        self._finished = True
        self._step_id = -1
        self._stopped = False
        self._failed = False
        self.global_idle = False

    def begin_step(self) -> None:
        """Reopen the client for the next forward_micro_batch invocation."""
        if self._stopped:
            raise RuntimeError("FastAFD client was stopped")
        if self._failed:
            raise RuntimeError("FastAFD client failed in a previous step")
        if not self._finished:
            raise RuntimeError("FastAFD previous step was not finished")
        self._step_id += 1
        self._finished = False
        self.global_idle = False

    def _status(self) -> _Status:
        status = torch.empty(_STATUS_LENGTH, device=self.device, dtype=torch.int64)
        recv(status, self.service_rank, Group.DP_AND_TP)
        try:
            return _Status(int(status.item()))
        except ValueError as exc:
            raise RuntimeError(
                "FastAFD expert service returned an unknown status"
            ) from exc

    def _validate(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        if not isinstance(layer_idx, int) or layer_idx < 0:
            raise ValueError("layer_idx must be a nonnegative integer")
        if not all(
            isinstance(tensor, torch.Tensor)
            for tensor in (hidden_states, topk_ids, topk_weights)
        ):
            raise TypeError("FastAFD request fields must be tensors")
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_size:
            raise ValueError("hidden_states must have shape [tokens, hidden_size]")
        token_count = hidden_states.shape[0]
        if token_count > _MAX_TOKENS:
            raise ValueError("FastAFD request exceeds maximum token count")
        if topk_ids.shape != (token_count, self.top_k):
            raise ValueError("topk_ids must have shape [tokens, top_k]")
        if topk_weights.shape != (token_count, self.top_k):
            raise ValueError("topk_weights must have shape [tokens, top_k]")
        if (
            hidden_states.device != self.device
            or topk_ids.device != self.device
            or topk_weights.device != self.device
        ):
            raise ValueError("FastAFD tensors must be on the configured device")
        if (
            hidden_states.dtype != self.activation_dtype
            or topk_ids.dtype != self.topk_ids_dtype
            or topk_weights.dtype != torch.float32
        ):
            raise ValueError("FastAFD tensor dtypes do not match the protocol")

    def forward(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self._finished:
            raise RuntimeError("FastAFD client was already finished")
        try:
            self._validate(layer_idx, hidden_states, topk_ids, topk_weights)
        except (TypeError, ValueError) as exc:
            # The expert service may already be waiting for this rank. Mark it
            # done before propagating the local validation error.
            try:
                self.abort()
            except Exception as abort_exc:
                raise exc from abort_exc
            raise

        token_count = hidden_states.shape[0]
        send(
            _header(
                self.device,
                _Kind.REQUEST,
                self._step_id,
                layer_idx,
                token_count,
                self.hidden_size,
                self.top_k,
                _DTYPE_CODE[self.activation_dtype],
                _IDS_CODE[self.topk_ids_dtype],
            ),
            self.service_rank,
            Group.DP_AND_TP,
        )
        status = self._status()
        if status != _Status.OK:
            self._finished = True  # The service removes a rejected rank.
            self._failed = True
            raise RuntimeError(
                f"FastAFD request rejected by expert service: {status.name}"
            )

        if token_count == 0:
            # Both peers still exchange a completion status for this step.
            status = self._status()
            if status != _Status.OK:
                error = RuntimeError(f"FastAFD empty request failed: {status.name}")
                try:
                    self.abort()
                except Exception as abort_exc:
                    raise error from abort_exc
                raise error
            return torch.empty_like(hidden_states)

        send(hidden_states.contiguous(), self.service_rank, Group.DP_AND_TP)
        send(topk_ids.contiguous(), self.service_rank, Group.DP_AND_TP)
        send(topk_weights.contiguous(), self.service_rank, Group.DP_AND_TP)
        status = self._status()
        if status != _Status.OK:
            error = RuntimeError(f"FastAFD routed expert failed: {status.name}")
            try:
                self.abort()
            except Exception as abort_exc:
                raise error from abort_exc
            raise error
        result = torch.empty(
            (token_count, self.hidden_size),
            device=self.device,
            dtype=self.activation_dtype,
        )
        recv(result, self.service_rank, Group.DP_AND_TP)
        return result

    def finish(self) -> None:
        """End this step after every attention rank reaches the service barrier."""
        if self._finished:
            return
        send(
            _header(self.device, _Kind.FINISH, self._step_id),
            self.service_rank,
            Group.DP_AND_TP,
        )
        self._finished = True
        status = self._status()
        self.global_idle = status == _Status.ALL_IDLE
        if status not in (_Status.OK, _Status.ALL_IDLE):
            self._failed = True
            raise RuntimeError(f"FastAFD finish rejected: {status.name}")

    def abort(self) -> None:
        """Signal a local failure without leaving the expert service waiting."""
        if self._finished:
            return
        self._failed = True
        send(
            _header(self.device, _Kind.ABORT, self._step_id),
            self.service_rank,
            Group.DP_AND_TP,
        )
        self._finished = True
        status = self._status()
        if status != _Status.OK:
            raise RuntimeError(f"FastAFD abort rejected: {status.name}")

    def stop(self) -> None:
        """Permanently leave the service after the final completed engine step."""
        if self._stopped or self._failed:
            # ABORT and non-OK status end the expert service with an error;
            # there is no receiver left for a subsequent STOP.
            return
        if not self._finished:
            raise RuntimeError("FastAFD cannot stop during an active step")
        # The previous FINISH ACK is a barrier, so the service is now waiting
        # for this rank's header in the following step. STOP is ACKed as soon
        # as it is received; other attention ranks need not stop together.
        send(
            _header(self.device, _Kind.STOP, self._step_id + 1),
            self.service_rank,
            Group.DP_AND_TP,
        )
        self._stopped = True
        status = self._status()
        if status != _Status.OK:
            self._failed = True
            raise RuntimeError(f"FastAFD stop rejected: {status.name}")


class FastAFDExpertService:
    """Run prebuilt routed FusedMoe modules on a single expert rank."""

    def __init__(
        self,
        attention_ranks: Sequence[int],
        fused_moe_by_layer: Mapping[int, torch.nn.Module],
        hidden_size: int,
        top_k: int,
        device: torch.device | str,
        activation_dtype: torch.dtype,
        expert_count: int,
    ) -> None:
        self.device = _check_common(
            hidden_size, top_k, expert_count, device, activation_dtype
        )
        ranks = tuple(attention_ranks)
        if not ranks or ranks != tuple(range(len(ranks))):
            raise ValueError(
                "FastAFD requires contiguous attention ranks starting at zero"
            )
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            if world_size != len(ranks) + 1 or rank != len(ranks):
                raise ValueError("FastAFD expert service must be the final world rank")
        if not fused_moe_by_layer:
            raise ValueError("FastAFD requires at least one routed MoE layer")
        if any(not isinstance(layer, int) or layer < 0 for layer in fused_moe_by_layer):
            raise ValueError("FastAFD layer indices must be nonnegative integers")
        for fused_moe in fused_moe_by_layer.values():
            if not callable(fused_moe):
                raise TypeError("each FastAFD expert layer must be callable")
            if getattr(fused_moe, "includes_shared_expert", False):
                raise ValueError(
                    "FastAFD shared experts must remain on attention ranks"
                )
            actual_experts = getattr(fused_moe, "expert_num", expert_count)
            if actual_experts != expert_count:
                raise ValueError("FastAFD expert count disagrees with FusedMoe")
            if getattr(fused_moe, "topk_ids_dtype", torch.int64) not in _IDS_CODE:
                raise ValueError("FastAFD expert layer needs int32 or int64 top-k ids")
        self.attention_ranks = ranks
        self.fused_moe_by_layer = dict(fused_moe_by_layer)
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.activation_dtype = activation_dtype
        self.expert_count = expert_count
        self._step_id = 0
        self._live_attention_ranks = set(ranks)
        self.global_idle = False

    def _validate_header(self, values: list[int]) -> _Status:
        (
            version,
            kind,
            step_id,
            layer_idx,
            tokens,
            hidden,
            top_k,
            dtype_code,
            ids_code,
        ) = values
        if version != _PROTOCOL_VERSION:
            return _Status.BAD_VERSION
        if kind != _Kind.REQUEST:
            return _Status.BAD_KIND
        if step_id != self._step_id:
            return _Status.BAD_STEP
        if layer_idx not in self.fused_moe_by_layer:
            return _Status.BAD_LAYER
        if (
            tokens < 0
            or tokens > _MAX_TOKENS
            or hidden != self.hidden_size
            or top_k != self.top_k
        ):
            return _Status.BAD_SHAPE
        if (
            dtype_code != _DTYPE_CODE[self.activation_dtype]
            or ids_code not in _CODE_IDS
        ):
            return _Status.BAD_DTYPE
        return _Status.OK

    def _prepare_request(
        self, rank: int, values: list[int], failures: list[str]
    ) -> _PendingRequest | None:
        status = self._validate_header(values)
        if status != _Status.OK:
            _send_status(self.device, rank, status)
            failures.append(f"attention rank {rank} sent {status.name}")
            return None

        _, _, _, layer_idx, token_count, _, _, _, ids_code = values
        try:
            hidden = torch.empty(
                (token_count, self.hidden_size),
                device=self.device,
                dtype=self.activation_dtype,
            )
            ids = torch.empty(
                (token_count, self.top_k),
                device=self.device,
                dtype=_CODE_IDS[ids_code],
            )
            weights = torch.empty(
                (token_count, self.top_k), device=self.device, dtype=torch.float32
            )
        except Exception as exc:
            _send_status(self.device, rank, _Status.RESOURCE_FAILED)
            failures.append(f"attention rank {rank} buffer allocation failed: {exc}")
            return None
        _send_status(self.device, rank, _Status.OK)
        return _PendingRequest(rank, layer_idx, token_count, hidden, ids, weights)

    def _execute_batch(
        self,
        layer_idx: int,
        batch: list[_PendingRequest],
        responses: dict[int, tuple[_Status, torch.Tensor | None]],
        failures: list[str],
    ) -> None:
        fused_moe = self.fused_moe_by_layer[layer_idx]
        total_tokens = sum(request.token_count for request in batch)
        try:
            if len(batch) == 1:
                hidden = batch[0].hidden
                ids = batch[0].ids
                weights = batch[0].weights
            else:
                hidden = torch.cat([request.hidden for request in batch], dim=0)
                ids = torch.cat([request.ids for request in batch], dim=0)
                weights = torch.cat([request.weights for request in batch], dim=0)
            ids = ids.to(dtype=getattr(fused_moe, "topk_ids_dtype"))
            result = fused_moe(
                hidden_states=hidden,
                topk_weights=weights,
                topk_ids=ids,
                activation="SiGLU",
            )
        except Exception as exc:
            failures.append(
                f"layer {layer_idx} ranks {[request.rank for request in batch]} "
                f"expert failed: {exc}"
            )
            for request in batch:
                responses[request.rank] = (_Status.EXPERT_FAILED, None)
            return
        if (
            not isinstance(result, torch.Tensor)
            or result.shape != (total_tokens, self.hidden_size)
            or result.dtype != self.activation_dtype
            or result.device != self.device
        ):
            failures.append(
                f"layer {layer_idx} ranks {[request.rank for request in batch]} "
                "returned bad output"
            )
            for request in batch:
                responses[request.rank] = (_Status.BAD_OUTPUT, None)
            return

        offset = 0
        for request in batch:
            responses[request.rank] = (
                _Status.OK,
                result[offset : offset + request.token_count],
            )
            offset += request.token_count

    def _execute_requests(
        self, requests: list[_PendingRequest], failures: list[str]
    ) -> dict[int, tuple[_Status, torch.Tensor | None]]:
        responses: dict[int, tuple[_Status, torch.Tensor | None]] = {}
        by_layer: dict[int, list[_PendingRequest]] = {}
        for request in requests:
            if request.token_count == 0:
                responses[request.rank] = (_Status.OK, None)
            else:
                by_layer.setdefault(request.layer_idx, []).append(request)

        # Bound temporary concat buffers. Large prefills still work, but only
        # requests fitting this cap are combined into a single expert launch.
        for layer_idx, layer_requests in by_layer.items():
            batch: list[_PendingRequest] = []
            batch_tokens = 0
            for request in layer_requests:
                if batch and batch_tokens + request.token_count > _MAX_TOKENS:
                    self._execute_batch(layer_idx, batch, responses, failures)
                    batch = []
                    batch_tokens = 0
                batch.append(request)
                batch_tokens += request.token_count
            if batch:
                self._execute_batch(layer_idx, batch, responses, failures)
        return responses

    def serve_until_done(self) -> bool:
        """Serve one step; return true after every attention rank has stopped."""
        if not self._live_attention_ranks:
            return True
        active = set(self._live_attention_ranks)
        failures: list[str] = []
        sentinels: dict[int, _Status] = {}
        aborted_ranks: set[int] = set()
        had_request = False
        while active:
            requests: list[_PendingRequest] = []
            # One header from each rank in this round. Do not wait for the
            # ranks to name the same layer: microbatch counts may differ.
            for rank in self.attention_ranks:
                if rank not in active:
                    continue
                header = torch.empty(
                    _HEADER_LENGTH, device=self.device, dtype=torch.int64
                )
                recv(header, rank, Group.DP_AND_TP)
                values = [int(value) for value in header.tolist()]
                version, kind, step_id = values[:3]
                if kind in (_Kind.FINISH, _Kind.ABORT, _Kind.STOP):
                    active.remove(rank)
                    if kind == _Kind.STOP:
                        # A departing AG must not wait for the other AGs' step
                        # barrier: engines can be stopped one rank at a time.
                        self._live_attention_ranks.remove(rank)
                        if version != _PROTOCOL_VERSION:
                            status = _Status.BAD_VERSION
                        elif step_id != self._step_id:
                            status = _Status.BAD_STEP
                        else:
                            status = _Status.OK
                        _send_status(self.device, rank, status)
                        if status != _Status.OK:
                            failures.append(f"attention rank {rank} sent {status.name}")
                    elif version != _PROTOCOL_VERSION:
                        failures.append(f"attention rank {rank} sent BAD_VERSION")
                        sentinels[rank] = _Status.BAD_VERSION
                    elif step_id != self._step_id:
                        failures.append(f"attention rank {rank} sent BAD_STEP")
                        sentinels[rank] = _Status.BAD_STEP
                    elif kind == _Kind.ABORT:
                        failures.append(f"attention rank {rank} aborted")
                        sentinels[rank] = _Status.OK
                        aborted_ranks.add(rank)
                    else:
                        sentinels[rank] = _Status.OK
                    continue
                had_request = True
                request = self._prepare_request(rank, values, failures)
                if request is None:
                    active.remove(rank)
                else:
                    requests.append(request)

            # Clients that received header ACK may be blocked in payload send.
            # Receive every payload before the combined expert computation.
            for request in requests:
                if request.token_count == 0:
                    continue
                recv(request.hidden, request.rank, Group.DP_AND_TP)
                recv(request.ids, request.rank, Group.DP_AND_TP)
                recv(request.weights, request.rank, Group.DP_AND_TP)

            responses = self._execute_requests(requests, failures)
            for request in requests:
                status, output = responses[request.rank]
                _send_status(self.device, request.rank, status)
                if output is not None:
                    send(output.contiguous(), request.rank, Group.DP_AND_TP)

        self.global_idle = not had_request and not failures
        # Delaying sentinel ACK until all ranks finish makes this a true step
        # barrier and prevents idle ranks from queuing future-step heartbeats.
        for rank in self.attention_ranks:
            if rank in sentinels:
                status = sentinels[rank]
                if failures and status == _Status.OK and rank not in aborted_ranks:
                    status = _Status.STEP_FAILED
                elif self.global_idle and status == _Status.OK:
                    status = _Status.ALL_IDLE
                _send_status(self.device, rank, status)
        self._step_id += 1
        if failures:
            raise RuntimeError("FastAFD service failures: " + "; ".join(failures))
        return not self._live_attention_ranks
