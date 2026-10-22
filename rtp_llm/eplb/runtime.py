"""Step-boundary EPLB for full experts distributed over the WORLD group."""

import logging
import random
from collections.abc import Sequence

import torch
import torch.distributed as dist

from rtp_llm.eplb.policy import rebalance_experts_global


class EplbLayer:
    """Borrow fixed routing tensors and actual executor weights.

    Every weight/scale has local physical experts on dimension zero. Other
    dimensions may be non-contiguous. Neither storage nor tensor addresses are
    replaced after CUDA graph capture. Gate IDs are logical IDs or -1.
    """

    def __init__(
        self,
        layer_id: int,
        logical_to_physical: torch.Tensor,
        logical_count: torch.Tensor,
        weights: Sequence[torch.Tensor],
        ep_rank: int = 0,
        replica_offset: int = 0,
    ) -> None:
        self.layer_id = layer_id
        self.logical_to_physical = logical_to_physical
        self.logical_count = logical_count
        self.weights = tuple(weights)
        self.ep_rank = ep_rank
        # Replicated TP inputs must select the same copy on every rank. Storage
        # ownership is separate from rotation for independent DP/CP inputs.
        self.replica_offset = replica_offset
        if (
            layer_id < 0
            or logical_count.ndim != 1
            or logical_count.numel() == 0
            or logical_to_physical.ndim != 2
            or logical_to_physical.size(0) != logical_count.numel()
            or logical_to_physical.dtype not in (torch.int32, torch.int64)
            or logical_count.dtype not in (torch.int32, torch.int64)
            or logical_count.device != logical_to_physical.device
        ):
            raise ValueError(
                "EPLB requires a nonnegative layer ID and integer routing maps"
            )
        counts = logical_count.cpu().long()
        mapping = logical_to_physical.cpu().long()
        if bool(((counts <= 0) | (counts > mapping.size(1))).any()):
            raise ValueError("Every logical expert must have valid physical replicas")
        valid = torch.arange(mapping.size(1), device="cpu").unsqueeze(
            0
        ) < counts.unsqueeze(1)
        physical_slots = mapping[valid]
        self.num_logical_experts = counts.numel()
        self.num_physical_experts = physical_slots.numel()
        if (
            mapping.size(1) != self.num_physical_experts - self.num_logical_experts + 1
            or not torch.equal(
                physical_slots.sort().values,
                torch.arange(self.num_physical_experts, device="cpu"),
            )
            or bool((mapping[~valid] != -1).any())
        ):
            raise ValueError("EPLB maps must cover every physical slot exactly once")
        if not self.weights or self.weights[0].ndim == 0:
            raise ValueError(
                "EPLB requires actual executor weights with an expert dimension"
            )
        self.num_local_experts = self.weights[0].size(0)
        if (
            self.num_local_experts <= 0
            or self.num_physical_experts % self.num_local_experts != 0
            or not 0 <= ep_rank < self.num_physical_experts // self.num_local_experts
            or any(
                w.ndim == 0
                or w.size(0) != self.num_local_experts
                or w.device != logical_to_physical.device
                for w in self.weights
            )
        ):
            raise ValueError(
                "EPLB weights must have matching local experts and routing device"
            )
        self.physical_to_logical = torch.empty(
            self.num_physical_experts, dtype=torch.int64, device="cpu"
        )
        logical_ids = (
            torch.arange(self.num_logical_experts, device="cpu")
            .unsqueeze(1)
            .expand_as(mapping)
        )
        self.physical_to_logical[physical_slots] = logical_ids[valid]

    def map_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """Map [..., top_k] IDs without host synchronization; preserve -1."""
        logical = ids.clamp_min(0).long()
        position = torch.arange(ids.numel(), device=ids.device).reshape(ids.shape)
        if ids.ndim > 1:
            position = position // ids.size(-1) + position % ids.size(-1)
        replica = (position + self.replica_offset) % self.logical_count[logical]
        physical = self.logical_to_physical[logical, replica]
        return torch.where(ids >= 0, physical, -1).to(ids.dtype)


def _shuffle_weights(
    layer: EplbLayer, next_physical: torch.Tensor, ep_size: int
) -> None:
    """Stage all destinations before writing any source; communicate raw bytes."""
    local = layer.num_local_experts
    rank = layer.ep_rank
    old = layer.physical_to_logical.tolist()
    new = next_physical.tolist()
    new_local = new[rank * local : (rank + 1) * local]
    staging = [
        torch.empty(w.shape, dtype=w.dtype, device=w.device) for w in layer.weights
    ]
    ops = []
    send_buffers = []
    # Each destination rank receives a logical expert only once. Prefer its
    # existing local copy; otherwise share receivers across existing senders.
    for expert in range(layer.num_logical_experts):
        sources = [
            (peer, old[peer * local : (peer + 1) * local].index(expert))
            for peer in range(ep_size)
            if expert in old[peer * local : (peer + 1) * local]
        ]
        source_by_rank = dict(sources)
        remote_index = 0
        for peer in range(ep_size):
            peer_new = new[peer * local : (peer + 1) * local]
            if expert not in peer_new:
                continue
            destination = peer_new.index(expert)
            if peer in source_by_rank:
                source_rank, source_slot = peer, source_by_rank[peer]
            else:
                source_rank, source_slot = sources[remote_index % len(sources)]
                remote_index += 1
            if rank == peer == source_rank:
                for out, weight in zip(staging, layer.weights):
                    out[destination].copy_(weight[source_slot])
            elif rank == source_rank:
                for weight in layer.weights:
                    payload = (
                        weight[source_slot].contiguous().reshape(-1).view(torch.uint8)
                    )
                    send_buffers.append(payload)
                    ops.append(dist.P2POp(dist.isend, payload, peer))
            elif rank == peer:
                for out in staging:
                    payload = out[destination].reshape(-1).view(torch.uint8)
                    ops.append(dist.P2POp(dist.irecv, payload, source_rank))
    if ops:
        for request in dist.batch_isend_irecv(ops):
            request.wait()
    for destination, expert in enumerate(new_local):
        first = new_local.index(expert)
        if destination != first:
            for out in staging:
                out[destination].copy_(out[first])
    for weight, out in zip(layer.weights, staging):
        weight.copy_(out)


class RuntimeExpertBalancer:
    """Own the sampled window; borrow graph-stable per-layer executor tensors.

    Call record_step once after a valid model step, before its counts are cleared,
    then balance. Counts include actual physical assignments (including padding).
    Warmup and speculative commit must call neither method. Multi-rank execution
    requires full experts and EP=WORLD with NCCL; the model boundary checks TP/PP.
    """

    def __init__(
        self,
        layers: Sequence[EplbLayer],
        config,
        ep_rank: int,
        ep_size: int,
        model_id: int | str,
    ) -> None:
        self.layers = tuple(sorted(layers, key=lambda layer: layer.layer_id))
        if not self.layers:
            raise ValueError("EPLB requires at least one routed expert layer")
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.model_id = model_id
        self._window_size = config.eplb_stats_window_size
        self._layers_per_update = config.eplb_balance_layer_per_step
        self._method = config.balance_method
        if self._window_size <= 0 or self._layers_per_update <= 0:
            raise ValueError("EPLB stats window and layers per update must be positive")
        if self._method not in ("round", "random", "most_unbalanced_layer", "mix"):
            raise ValueError(f"Unsupported EPLB balance_method={self._method}")
        first = self.layers[0]
        self._device = first.logical_count.device
        self._num_logical = first.num_logical_experts
        self._num_physical = first.num_physical_experts
        self._num_local = first.num_local_experts
        if (
            ep_size <= 0
            or not 0 <= ep_rank < ep_size
            or self._num_physical != self._num_local * ep_size
            or len({layer.layer_id for layer in self.layers}) != len(self.layers)
            or any(
                layer.num_logical_experts != self._num_logical
                or layer.num_physical_experts != self._num_physical
                or layer.num_local_experts != self._num_local
                or layer.logical_count.device != self._device
                or layer.ep_rank != ep_rank
                for layer in self.layers
            )
        ):
            raise ValueError("EPLB layers must share an expert layout and EP rank")
        if ep_size > 1 and (
            self._device.type != "cuda"
            or not dist.is_initialized()
            or dist.get_world_size() != ep_size
            or dist.get_rank() != ep_rank
            or dist.get_backend() != "nccl"
        ):
            raise ValueError("EPLB requires EP=WORLD using CUDA tensors and NCCL")
        self._layer_ids = torch.tensor(
            [layer.layer_id for layer in self.layers],
            dtype=torch.int64,
            device=self._device,
        )
        self._window = torch.empty(
            self._window_size,
            len(self.layers),
            self._num_local,
            dtype=torch.int64,
            device=self._device,
        )
        self._samples = 0
        self._window_index = 0
        self._steps = 0
        self._updates = 0
        self._round_index = 0
        self._mode = -1
        self._interval = -1
        self.update_config(config.eplb_mode, config.eplb_update_time)

    def update_config(self, mode, update_time: int) -> None:
        mode = int(mode)
        if mode not in (0, 1, 2, 3) or update_time <= 0:
            raise ValueError("EPLB mode must be 0..3 and update time must be positive")
        if (mode, update_time) != (self._mode, self._interval):
            self._mode, self._interval = mode, update_time
            self._samples = self._window_index = self._steps = 0

    def record_step(self, counts: torch.Tensor) -> None:
        if self._mode == 0:
            return
        if (
            counts.ndim != 2
            or counts.dtype not in (torch.int32, torch.int64)
            or counts.size(0) <= self.layers[-1].layer_id
            or counts.size(1) != self._num_local
            or counts.device != self._device
        ):
            raise ValueError(
                "EPLB counts must be [model layers, local physical experts]"
            )
        self._window[self._window_index].copy_(counts.index_select(0, self._layer_ids))
        self._window_index = (self._window_index + 1) % self._window_size
        self._samples = min(self._samples + 1, self._window_size)
        self._steps += 1

    def _select_layers(self, rank_loads: torch.Tensor) -> list[int]:
        count = min(self._layers_per_update, len(self.layers))
        method = self._method
        if method == "mix":
            method = "round" if self._updates % 2 == 0 else "most_unbalanced_layer"
        if method == "round":
            selected = [
                (self._round_index + i) % len(self.layers) for i in range(count)
            ]
            self._round_index = (self._round_index + count) % len(self.layers)
            return selected
        if method == "random":
            return random.Random(self._updates).sample(range(len(self.layers)), count)
        peak = rank_loads.amax(-1).sum(0).double()
        total = rank_loads.sum((0, 2)).double()
        imbalance = peak * self.ep_size / total.clamp_min(1)
        return imbalance.argsort(descending=True, stable=True)[:count].tolist()

    @torch.inference_mode()
    def balance(self) -> bool:
        if self._mode < 2 or self._steps < self._interval or self._samples == 0:
            return False
        self._steps = 0
        self._updates += 1
        history = self._window[: self._samples]
        # One window-boundary collective carries logical history and actual
        # per-rank loads. The CPU transfer/CPU policy synchronize only here.
        stats = torch.zeros(
            self._samples,
            len(self.layers),
            self._num_logical + self.ep_size,
            dtype=torch.int64,
            device=self._device,
        )
        physical = torch.stack([layer.physical_to_logical for layer in self.layers])
        local_map = physical[
            :, self.ep_rank * self._num_local : (self.ep_rank + 1) * self._num_local
        ]
        stats[..., : self._num_logical].scatter_add_(
            -1, local_map.to(self._device).expand_as(history), history
        )
        stats[..., self._num_logical + self.ep_rank] = history.sum(-1)
        if self.ep_size > 1:
            dist.all_reduce(stats)
        stats = stats.cpu()
        selected = self._select_layers(stats[..., self._num_logical :])
        next_physical, next_inverse, next_count = rebalance_experts_global(
            stats[:, selected, : self._num_logical], self._num_physical, self._num_local
        )
        changed = False
        for index, layer_index in enumerate(selected):
            layer = self.layers[layer_index]
            if not bool(stats[:, layer_index, : self._num_logical].any()):
                continue
            changed_slots = int(
                (layer.physical_to_logical != next_physical[index]).sum()
            )
            if changed_slots == 0:
                continue
            _shuffle_weights(layer, next_physical[index], self.ep_size)
            layer.logical_to_physical.copy_(next_inverse[index])
            layer.logical_count.copy_(next_count[index])
            layer.physical_to_logical.copy_(next_physical[index])
            if self._device.type == "cuda":
                torch.cuda.current_stream(self._device).synchronize()
            changed = True
            logging.info(
                "[EPLB] applied model_id=%s layer_id=%d update=%d changed_slots=%d ep_rank=%d",
                self.model_id,
                layer.layer_id,
                self._updates,
                changed_slots,
                self.ep_rank,
            )
        if changed:
            # Old physical slots now denote different experts. Discard the
            # previous window even when the update interval is shorter than it.
            self._samples = self._window_index = 0
        return changed
