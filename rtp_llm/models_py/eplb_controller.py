"""Dynamic EPLB controller backed by NewModelLoader checkpoint streaming."""

import logging
import random
import threading
from collections import deque
from enum import Enum
from functools import partial
from queue import Queue
from typing import Any, Deque, Dict, Sequence

import torch

from rtp_llm.eplb.eplb import rebalance_experts
from rtp_llm.models_py.expert_placement import ExpertPlacement
from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts

logger = logging.getLogger(__name__)


class _HistoryStats:
    def __init__(self, window_size: int, shape: Sequence[int]):
        if isinstance(window_size, bool) or not isinstance(window_size, int):
            raise TypeError("eplb_stats_window_size must be an integer")
        if window_size <= 0:
            raise ValueError("eplb_stats_window_size must be positive")
        self._window_size = window_size
        self._values: Deque[torch.Tensor] = deque()
        self._total = torch.zeros(tuple(shape), dtype=torch.int64)

    def add(self, value: torch.Tensor) -> torch.Tensor:
        value = value.to(device="cpu", dtype=torch.int64)
        if value.shape != self._total.shape:
            raise ValueError(
                f"EPLB logical stats shape changed from {tuple(self._total.shape)} "
                f"to {tuple(value.shape)}"
            )
        if len(self._values) == self._window_size:
            self._total.sub_(self._values.popleft())
        snapshot = value.clone()
        self._values.append(snapshot)
        self._total.add_(snapshot)
        return self._total


class _SelectLayerMethod(Enum):
    ROUND = "round"
    RANDOM = "random"
    MOST_UNBALANCED_LAYER = "most_unbalanced_layer"
    MIX = "mix"


class _StatsOnlyBalancer:
    """The native executor reports counters, but never owns these packed weights."""

    eplb_stats_only = True


def stage_supports_weight_install(model, experts):
    """The native controller may promote this stage from STATS without reloading."""
    if getattr(model, "eplb_attention_only", False) or getattr(model, "_ft_eplb_stats_only", False):
        return False
    return all(
        isinstance(module, BaseMoEExperts)
        and module.quant_method.supports_eplb_weight_install
        for module in experts
    )


def create_stage_expert_balancer(loader, model, model_config):
    """Only actual routed-expert owners participate on this PP stage."""
    if not model_config.eplb_config.enable_eplb():
        return None

    # The native executors schedule EPLB from the configured global layer IDs,
    # not by discovering Python modules. Compare the two views *before* publishing
    # the model: a missing callback otherwise falls through to legacy weights.
    declared = set(model_config.moe_layer_index)
    invalid = declared.difference(range(model_config.num_layers))
    if invalid:
        raise ValueError(f"EPLB declares out-of-range MoE layers: {sorted(invalid)}")
    if not declared and getattr(model_config, "moe_style", 0):
        raise ValueError("EPLB enabled for a routed model with no declared MoE layers")
    from rtp_llm.config.layer_partition import local_layer_range

    owned = local_layer_range(model_config, loader.load_config.parallelism_config)
    # DSpARK PREFILL owns attention/cache commits only, although the checkpoint
    # declares FFNs for its DECODE role. No routed counters are written there.
    attention_only = getattr(model, "eplb_attention_only", False)
    expected = set() if attention_only else declared.intersection(owned)
    experts = {}
    for module in model.modules():
        if not callable(getattr(module, "bind_eplb_stats", None)):
            continue
        if module.layer_idx in experts:
            raise RuntimeError(
                f"Multiple routed-expert modules found for layer {module.layer_idx}"
            )
        experts[module.layer_idx] = module
    actual = set(experts)
    if actual != expected:
        raise ValueError(
            "EPLB routed-layer ownership mismatch: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    if not expected and not attention_only:
        return None  # Valid dense-only PP stage.
    if not stage_supports_weight_install(model, experts.values()):
        mode = model_config.eplb_config.eplb_mode
        if getattr(mode, "name", mode) != "STATS":
            raise ValueError(
                "This routed expert layout supports EPLB statistics but not weight rebalancing"
            )
        model._newloader_eplb_num_layers = model_config.num_layers
        model._newloader_eplb_bind_stats = partial(_bind_stats, experts, stats_only=True)
        return _StatsOnlyBalancer()
    return NewLoaderExpertBalancer(loader, model, model_config)


class NewLoaderExpertBalancer:
    """Implements the Python callback protocol consumed by C++ ExpertBalancer."""

    def supports_python_weight_install(self) -> bool:
        """Tell C++ that NewLoader installs CPU-staged tensors into Python weights."""
        return True

    def __init__(self, loader: Any, model: torch.nn.Module, model_config: Any):
        self._loader = loader
        self._model = model
        self._model_config = model_config
        self._lock = threading.Lock()

        experts: Dict[int, BaseMoEExperts] = {}
        for module in model.modules():
            if not isinstance(module, BaseMoEExperts):
                continue
            if module.layer_idx in experts:
                raise RuntimeError(
                    f"Multiple routed-expert modules found for layer {module.layer_idx}"
                )
            experts[module.layer_idx] = module
        if not experts:
            raise ValueError("Dynamic EPLB requires at least one NewLoader MoE layer")
        self._experts = experts
        self._staged = None
        self.moe_layer_index = tuple(sorted(experts))

        first = experts[self.moe_layer_index[0]]
        self.num_layers = int(getattr(model_config, "num_layers", len(experts)))
        self.num_experts = first.num_experts
        self.num_replicas = first.physical_num_experts
        self.num_gpu = first.ep_size
        # Zero denotes an ungrouped router (one balancing group).
        self.num_groups = int(getattr(model_config, "moe_n_group", 1)) or 1
        mode = getattr(model_config.eplb_config, "eplb_mode", None)
        parallelism = loader.load_config.parallelism_config
        from rtp_llm.models_py.expert_placement import expert_stage_node_count

        self.num_nodes = expert_stage_node_count(parallelism, self.num_gpu)
        if getattr(parallelism, "pp_size", 1) > 1:
            from rtp_llm.config.layer_partition import local_layer_range
            from rtp_llm.models_py.distributed.rank_layout import RankLayout

            layout = RankLayout.from_parallelism_config(parallelism)
            coord = layout.coord_of(parallelism.world_rank)
            if (
                # Pure TP owns every logical expert on each TP rank; its
                # expert count/stats axis is 1, not the stage collective size.
                (self.num_gpu != layout.lane_stride()
                 and not (self.num_gpu == 1 and parallelism.dp_size == 1))
                or parallelism.pp_rank != coord.pp
                or parallelism.dp_rank != coord.dp
                or parallelism.tp_rank != coord.tp
                or first.ep_rank
                != layout.ep_rank_of(parallelism.world_rank, self.num_gpu)
            ):
                raise ValueError(
                    "Dynamic EPLB requires consistent stage-local EP coordinates"
                )
            owned = local_layer_range(model_config, parallelism)
            if any(layer not in owned for layer in self.moe_layer_index):
                raise ValueError(
                    "Dynamic EPLB model contains MoE layers outside its PP stage"
                )
        # rebalance_experts uses a global (one-node) solve on its non-hierarchical
        # path. Match main's explicit topology error on the hierarchical path.
        if getattr(mode, "name", mode) != "STATS":
            self._validate_plan_topology()

        eplb_config = model_config.eplb_config
        self.select_layer_method = _SelectLayerMethod(eplb_config.balance_method)
        self.force_repack = int(eplb_config.eplb_force_repack) == 1
        self._round_robin: Queue[int] = Queue()
        self._update_count = 0
        self._history = _HistoryStats(
            int(eplb_config.eplb_stats_window_size),
            (self.num_layers, self.num_experts),
        )
        self.phy2log = [list(range(self.num_replicas)) for _ in range(self.num_layers)]
        for layer_id, module in experts.items():
            placement = module._expert_placement
            if (
                placement.logical_expert_count != self.num_experts
                or placement.physical_expert_count != self.num_replicas
                or placement.ep_size != self.num_gpu
            ):
                raise ValueError("All EPLB MoE layers must share one expert topology")
            self.phy2log[layer_id] = list(placement.physical_to_logical)

        # PyWrappedModel binds these before creating CUDA graphs. Keep the
        # callback independent of the controller to avoid a model/controller cycle.
        model._newloader_eplb_num_layers = self.num_layers
        model._newloader_eplb_bind_stats = partial(_bind_stats, experts)

    def _validate_plan_topology(self):
        if self.num_groups <= 0:
            raise ValueError("model_config.moe_n_group must be positive")
        if self.num_groups % self.num_nodes == 0 and self.num_gpu % self.num_nodes:
            raise ValueError(
                f"Hierarchical EPLB requires ep_size={self.num_gpu} divisible by num_nodes={self.num_nodes}"
            )

    def _select_layer(self, gpu_loads: torch.Tensor) -> int:
        method = self.select_layer_method
        if method == _SelectLayerMethod.ROUND:
            return self._next_round_robin()
        if method == _SelectLayerMethod.RANDOM:
            return random.choice(self.moe_layer_index)
        if method == _SelectLayerMethod.MOST_UNBALANCED_LAYER:
            return self._most_unbalanced(gpu_loads)
        if method == _SelectLayerMethod.MIX:
            return (
                self._next_round_robin()
                if self._update_count % 2 == 0
                else self._most_unbalanced(gpu_loads)
            )
        raise RuntimeError(f"Unsupported EPLB balance method {method}")

    def _next_round_robin(self) -> int:
        if self._round_robin.empty():
            for layer_id in self.moe_layer_index:
                self._round_robin.put(layer_id)
        layer_id = self._round_robin.get()
        self._round_robin.put(layer_id)
        return layer_id

    def _most_unbalanced(self, gpu_loads: torch.Tensor) -> int:
        if gpu_loads.dim() != 2 or gpu_loads.shape[0] != self.num_layers:
            raise ValueError(
                f"EPLB gpu_loads must have shape [{self.num_layers}, ep_size]"
            )
        maxima = gpu_loads.max(dim=1).values
        return max(self.moe_layer_index, key=lambda layer_id: int(maxima[layer_id]))

    @torch.inference_mode()
    def create_balance_plan(self, log_stats: torch.Tensor, gpu_loads: torch.Tensor):
        # The C++ controller gates this callback on its *current* mode. It can
        # switch from STATS to EPLB without rebuilding the Python owner.
        self._validate_plan_topology()
        with self._lock:
            self._update_count += 1
            history = self._history.add(log_stats)
            layer_id = self._select_layer(gpu_loads.to(device="cpu"))
            phy2log, _, _ = rebalance_experts(
                history[layer_id : layer_id + 1],
                self.num_replicas,
                self.num_groups,
                self.num_nodes,
                self.num_gpu,
                self.force_repack,
            )
            layer_phy2log = phy2log[0].to(dtype=torch.int32, device="cpu")
            module = self._experts[layer_id]
            placement = ExpertPlacement(
                logical_expert_count=self.num_experts,
                ep_size=module.ep_size,
                ep_rank=module.ep_rank,
                physical_to_logical=tuple(layer_phy2log.tolist()),
            )
            # Treat phy2log as the one placement source of truth.  Rebuild the
            # inverse tensors from it so the loader and runtime cannot observe
            # mutually inconsistent routing metadata after repeated updates.
            logic_expert_cnt, log2phy = module._routing_tensors(placement)
        logger.debug(
            "[NewLoader EPLB] plan layer=%d phy2log=%s",
            layer_id,
            placement.physical_to_logical,
        )
        return (
            torch.tensor([layer_id], dtype=torch.int32).contiguous(),
            logic_expert_cnt.contiguous(),
            log2phy.contiguous(),
            layer_phy2log.contiguous(),
        )

    @torch.inference_mode()
    def load_moe_weight(
        self,
        layer_id_tensor: torch.Tensor,
        ep_rank: int,
        ep_size: int,
        phy2log: torch.Tensor,
    ):
        layer_id = int(layer_id_tensor.item())
        if layer_id not in self._experts:
            raise ValueError(f"Layer {layer_id} is not a NewLoader MoE layer")
        planned_phy2log = [int(value) for value in phy2log.to(device="cpu").tolist()]
        weights = self._loader.load_moe_layer_weights(
            self._model,
            layer_id=layer_id,
            physical_to_logical=planned_phy2log,
            ep_rank=ep_rank,
            ep_size=ep_size,
        )
        # This callback runs inside the C++ loading worker.  A mismatch must
        # therefore become a load failure and participate in the all-rank
        # rollback before any live tensor is touched.
        self._experts[layer_id].validate_eplb_reload_weights(weights)
        with self._lock:
            if self._staged is not None:
                raise RuntimeError("EPLB plan staged before prior acknowledgement")
            # The module owns this entire schema, including calibration tensors.
            # C++ only coordinates ranks and never translates parameter names.
            self._staged = (layer_id, weights)
        return layer_id

    @torch.inference_mode()
    def install_balance_plan(self, layer_id, phy2log, counts, log2phy):
        with self._lock:
            if self._staged is None or self._staged[0] != layer_id:
                raise RuntimeError("EPLB install without the matching staged layer")
            self._experts[layer_id].apply_eplb_plan(
                phy2log, counts, log2phy, self._staged[1], defer_commit=True
            )

    def rollback_balance_plan(self, layer_id: int) -> None:
        with self._lock:
            self._experts[layer_id].finalize_eplb_install(commit=False)
            self._staged = None

    def commit_balance_plan(self, layer_id: int) -> None:
        with self._lock:
            if self._staged is None or self._staged[0] != layer_id:
                raise RuntimeError("EPLB commit without matching staged weights")
            module = self._experts[layer_id]
            module.finalize_eplb_install(commit=True)
            self.phy2log[layer_id] = list(module._expert_placement.physical_to_logical)
            self._staged = None


def _bind_stats(experts, logical, gpu_loads, live_rows=None, *, stats_only=False):
    for layer_id, module in experts.items():
        if (
            stats_only
            and isinstance(module, BaseMoEExperts)
            and module.ep_size > 1
            and module._parallelism_config.get_attn_tp_rank() != 0
        ):
            # The StatsOnly C++ collector sums source counters across EP.
            # Replicated attention-TP inputs must contribute once per DP group;
            # CP origins have attention TP rank zero and contribute separately.
            continue
        module.bind_eplb_stats(logical[layer_id], gpu_loads[layer_id], live_rows)
