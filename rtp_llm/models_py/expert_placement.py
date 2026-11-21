"""Static logical-to-physical expert placement for NewLoader MoE layers."""

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional, Tuple


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


@lru_cache(maxsize=16)
def _physical_to_logical_layers(
    layer_count: int,
    logical_expert_count: int,
    physical_expert_count: int,
    ep_size: int,
    num_nodes: int,
    phy2log_path: Optional[str],
    phy2log_mtime_ns: Optional[int],
    phy2log_size: Optional[int],
) -> Tuple[Tuple[int, ...], ...]:
    """Mirror the legacy loader's initial redundant-expert placement."""

    del phy2log_mtime_ns, phy2log_size  # File identity is part of the cache key.
    layer_count = _integer(layer_count, "layer_count", minimum=1)
    logical_expert_count = _integer(
        logical_expert_count, "logical_expert_count", minimum=1
    )
    physical_expert_count = _integer(
        physical_expert_count, "physical_expert_count", minimum=1
    )
    ep_size = _integer(ep_size, "ep_size", minimum=1)
    num_nodes = _integer(num_nodes, "num_nodes", minimum=1)
    redundant_experts = physical_expert_count - logical_expert_count
    if redundant_experts < 0 or redundant_experts > logical_expert_count:
        raise ValueError(
            f"redundant_expert={redundant_experts} must be in "
            f"[0, logical_expert_count={logical_expert_count}]"
        )
    if logical_expert_count % ep_size:
        raise ValueError(
            f"logical_expert_count={logical_expert_count} must be divisible by "
            f"ep_size={ep_size}"
        )
    if physical_expert_count % ep_size:
        raise ValueError(
            f"physical_expert_count={physical_expert_count} must be divisible by "
            f"ep_size={ep_size}"
        )

    if phy2log_path:
        with open(phy2log_path, "r", encoding="utf-8") as stream:
            raw_layers = json.load(stream)
        if not isinstance(raw_layers, list) or len(raw_layers) != layer_count:
            actual = (
                len(raw_layers)
                if isinstance(raw_layers, list)
                else type(raw_layers).__name__
            )
            raise ValueError(f"phy2log must contain {layer_count} layers, got {actual}")
        layers = tuple(tuple(layer) for layer in raw_layers)
    elif redundant_experts == 0:
        identity = tuple(range(logical_expert_count))
        layers = tuple(identity for _ in range(layer_count))
    elif redundant_experts % ep_size == 0:
        if ep_size % num_nodes:
            raise ValueError(
                f"ep_size={ep_size} must be divisible by num_nodes={num_nodes}"
            )
        experts_per_ep = logical_expert_count // ep_size
        redundant_per_ep = redundant_experts // ep_size
        ranks_per_node = ep_size // num_nodes
        layers_list = []
        for _ in range(layer_count):
            layer = []
            for ep_rank in range(ep_size):
                node_id = ep_rank // ranks_per_node
                layer.extend(
                    range(ep_rank * experts_per_ep, (ep_rank + 1) * experts_per_ep)
                )
                for replica_index in range(redundant_per_ep):
                    replica_ep_rank = (
                        ep_rank + 1
                    ) % ranks_per_node + node_id * ranks_per_node
                    layer.append(replica_ep_rank * experts_per_ep + replica_index)
            layers_list.append(tuple(layer))
        layers = tuple(layers_list)
    else:
        layer = tuple(range(logical_expert_count)) + tuple(range(redundant_experts))
        layers = tuple(layer for _ in range(layer_count))

    for layer_index, layer in enumerate(layers):
        if len(layer) != physical_expert_count:
            raise ValueError(
                f"phy2log layer {layer_index} contains {len(layer)} physical "
                f"experts; expected {physical_expert_count}"
            )
        invalid = [
            expert_id
            for expert_id in layer
            if isinstance(expert_id, bool)
            or not isinstance(expert_id, int)
            or not 0 <= expert_id < logical_expert_count
        ]
        if invalid:
            raise ValueError(
                f"phy2log layer {layer_index} contains invalid logical expert "
                f"ids {invalid[:10]}"
            )
        missing = sorted(set(range(logical_expert_count)) - set(layer))
        if missing:
            raise ValueError(
                f"phy2log layer {layer_index} omits logical experts {missing[:10]}"
            )
    return layers


@dataclass(frozen=True)
class ExpertPlacement:
    """One layer's physical expert slots and the current EP-rank view."""

    logical_expert_count: int
    ep_size: int
    ep_rank: int
    physical_to_logical: Tuple[int, ...]

    def __post_init__(self) -> None:
        _integer(self.logical_expert_count, "logical_expert_count", minimum=1)
        _integer(self.ep_size, "ep_size", minimum=1)
        _integer(self.ep_rank, "ep_rank")
        if self.ep_rank >= self.ep_size:
            raise ValueError(
                f"ep_rank={self.ep_rank} must be less than ep_size={self.ep_size}"
            )
        if len(self.physical_to_logical) % self.ep_size:
            raise ValueError(
                f"physical expert count {len(self.physical_to_logical)} must be "
                f"divisible by ep_size={self.ep_size}"
            )
        if len(self.physical_to_logical) < self.logical_expert_count:
            raise ValueError(
                f"physical expert count {len(self.physical_to_logical)} must be at "
                f"least logical_expert_count={self.logical_expert_count}"
            )
        invalid = [
            expert_id
            for expert_id in self.physical_to_logical
            if isinstance(expert_id, bool)
            or not isinstance(expert_id, int)
            or not 0 <= expert_id < self.logical_expert_count
        ]
        if invalid:
            raise ValueError(
                "physical_to_logical contains invalid logical expert ids "
                f"{invalid[:10]}"
            )
        missing = sorted(
            set(range(self.logical_expert_count)) - set(self.physical_to_logical)
        )
        if missing:
            raise ValueError(
                f"physical_to_logical omits logical experts {missing[:10]}"
            )

    @property
    def physical_expert_count(self) -> int:
        return len(self.physical_to_logical)

    @property
    def has_redundancy(self) -> bool:
        return self.physical_expert_count != self.logical_expert_count

    @property
    def is_identity(self) -> bool:
        return self.physical_to_logical == tuple(range(self.logical_expert_count))

    @property
    def physical_experts_per_rank(self) -> int:
        return self.physical_expert_count // self.ep_size

    @property
    def local_physical_start(self) -> int:
        return self.ep_rank * self.physical_experts_per_rank

    @property
    def local_physical_to_logical(self) -> Tuple[int, ...]:
        start = self.local_physical_start
        return self.physical_to_logical[start : start + self.physical_experts_per_rank]

    @property
    def logical_to_physical(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(
            self.global_indices(logical_id)
            for logical_id in range(self.logical_expert_count)
        )

    def local_indices(self, logical_expert_id: int) -> Tuple[int, ...]:
        self._validate_logical_id(logical_expert_id)
        return tuple(
            local_index
            for local_index, candidate in enumerate(self.local_physical_to_logical)
            if candidate == logical_expert_id
        )

    def global_indices(self, logical_expert_id: int) -> Tuple[int, ...]:
        self._validate_logical_id(logical_expert_id)
        return tuple(
            physical_index
            for physical_index, candidate in enumerate(self.physical_to_logical)
            if candidate == logical_expert_id
        )

    def target_ep_ranks(self, logical_expert_id: int) -> Tuple[int, ...]:
        per_rank = self.physical_experts_per_rank
        return tuple(
            sorted(
                {
                    physical_index // per_rank
                    for physical_index in self.global_indices(logical_expert_id)
                }
            )
        )

    def _validate_logical_id(self, logical_expert_id: int) -> None:
        if (
            isinstance(logical_expert_id, bool)
            or not isinstance(logical_expert_id, int)
            or not 0 <= logical_expert_id < self.logical_expert_count
        ):
            raise ValueError(
                f"logical expert id {logical_expert_id!r} is outside "
                f"[0, {self.logical_expert_count})"
            )


def expert_stage_node_count(parallelism_config: Any, ep_size: int) -> int:
    """Count nodes in the expert group, excluding unrelated PP stages.

    Retain legacy PP=1 semantics. Rank placement follows the runtime's
    contiguous local-world ranks; a PP stage may occupy only part of a node.
    """
    world_size = getattr(parallelism_config, "world_size", None)
    world_size = _integer(
        ep_size if world_size is None else world_size,
        "parallelism_config.world_size",
        minimum=1,
    )
    local_world_size = getattr(parallelism_config, "local_world_size", None)
    local_world_size = _integer(
        world_size if local_world_size is None else local_world_size,
        "parallelism_config.local_world_size",
        minimum=1,
    )
    if world_size % local_world_size:
        raise ValueError(
            f"world_size={world_size} must be divisible by local_world_size={local_world_size}"
        )
    if getattr(parallelism_config, "pp_size", 1) <= 1:
        return world_size // local_world_size
    from rtp_llm.models_py.distributed.rank_layout import Group, RankLayout

    layout = RankLayout.from_parallelism_config(parallelism_config)
    ep_size = _integer(ep_size, "ep_size", minimum=1)
    if layout.lane_stride() % ep_size:
        raise ValueError("ep_size must divide the owning PP stage")
    rank = parallelism_config.world_rank
    members = layout.group_of(Group.STAGE, rank)
    first = layout.rank_in_group(Group.STAGE, rank) // ep_size * ep_size
    return len({peer // local_world_size for peer in members[first : first + ep_size]})


def expert_placement_for_layer(
    model_config: Any,
    *,
    layer_index: int,
    logical_expert_count: int,
    ep_size: int,
    ep_rank: int,
    parallelism_config: Any = None,
) -> ExpertPlacement:
    """Resolve the legacy-compatible static placement for one NewLoader layer."""

    layer_index = _integer(layer_index, "layer_index")
    layer_count = getattr(model_config, "num_layers", layer_index + 1)
    layer_count = _integer(layer_count, "model_config.num_layers", minimum=1)
    if layer_index >= layer_count:
        raise ValueError(
            f"layer_index={layer_index} is outside model layer count {layer_count}"
        )

    eplb_config = getattr(model_config, "eplb_config", None)
    physical_expert_count = logical_expert_count
    if eplb_config is not None:
        resolver = getattr(eplb_config, "phy_exp_num", None)
        if callable(resolver):
            physical_expert_count = resolver(logical_expert_count)
        else:
            redundant = getattr(eplb_config, "redundant_expert", 0)
            redundant = _integer(redundant, "eplb_config.redundant_expert")
            physical_expert_count = logical_expert_count + redundant
    physical_expert_count = _integer(
        physical_expert_count, "physical_expert_count", minimum=1
    )

    num_nodes = expert_stage_node_count(parallelism_config, ep_size)

    phy2log_path = getattr(model_config, "phy2log_path", None) or None
    phy2log_mtime_ns = None
    phy2log_size = None
    if phy2log_path is not None:
        phy2log_path = os.fspath(phy2log_path)
        stat = os.stat(phy2log_path)
        phy2log_mtime_ns = stat.st_mtime_ns
        phy2log_size = stat.st_size
    layers = _physical_to_logical_layers(
        layer_count,
        logical_expert_count,
        physical_expert_count,
        ep_size,
        num_nodes,
        phy2log_path,
        phy2log_mtime_ns,
        phy2log_size,
    )
    return ExpertPlacement(
        logical_expert_count=logical_expert_count,
        ep_size=ep_size,
        ep_rank=ep_rank,
        physical_to_logical=layers[layer_index],
    )
