"""Fixed logical-to-physical expert layout for one global BF16 MoE pool.

This helper does not enable RL EP collectives or alter the control protocol.
Padding slots carry existing expert weights but are never selected by routing.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import os


@dataclass(frozen=True)
class FixedExpertLayout:
    logical_experts: int
    ep_size: int
    slots_per_rank: int
    physical_to_logical: tuple
    logical_to_physical: tuple

    @property
    def physical_experts(self):
        return self.ep_size * self.slots_per_rank

    @property
    def redundant_slots(self):
        return self.physical_experts - self.logical_experts

    def rank_sources(self, rank):
        if not 0 <= rank < self.ep_size:
            raise ValueError("Invalid expert rank")
        begin = rank * self.slots_per_rank
        return self.physical_to_logical[begin:begin + self.slots_per_rank]

    def primary_experts_per_rank(self):
        result = [0] * self.ep_size
        for slot in self.logical_to_physical:
            result[slot // self.slots_per_rank] += 1
        return tuple(result)

    def layer_source_map(self, layers):
        if layers < 1:
            raise ValueError("At least one layer is required")
        return [list(self.physical_to_logical) for _ in range(layers)]


def balanced_expert_layout(logical_experts, ep_size):
    if logical_experts < ep_size or ep_size < 1:
        raise ValueError("Every EP rank must own at least one logical expert")
    base, extra = divmod(logical_experts, ep_size)
    slots = base + int(extra > 0)
    physical = []
    mapping = []
    logical = 0
    for rank in range(ep_size):
        count = base + int(rank < extra)
        for _ in range(count):
            mapping.append(len(physical));physical.append(logical);logical += 1
        # These never-routed padding slots need no additional checkpoint key.
        while len(physical) < (rank + 1) * slots:
            physical.append(logical % logical_experts)
    result = FixedExpertLayout(logical_experts, ep_size, slots, tuple(physical), tuple(mapping))
    if set(result.physical_to_logical) != set(range(logical_experts)):
        raise ValueError("Expert placement omitted a logical checkpoint expert")
    if any(result.physical_to_logical[p] != e for e, p in enumerate(result.logical_to_physical)):
        raise ValueError("Routing and checkpoint placement disagree")
    return result


def validate_native_fixed_profile(model, parallel, moe, *, quantized):
    if model.model_type != "qwen38_moe_text":
        raise ValueError("The fixed EP layout is restricted to the Qwen3.8 profile")
    actual = (parallel.world_size, parallel.tp_size, parallel.dp_size,
              parallel.ep_size, parallel.ffn_sp_size, parallel.get_ffn_tp_size())
    world32 = os.environ.get("RTP_LLM_RL_WORLD32") == "1"
    if world32 and os.environ.get("RTP_LLM_RL_WORLD72") == "1":
        raise ValueError("Qwen3.8 world32 and world72 are mutually exclusive")
    world = 32 if world32 else 72
    physical = 512 if world32 else 576
    if actual != (world, 8, world // 8, world, 8, 1):
        raise ValueError(f"Expected the explicit Qwen3.8 world{world} logical model")
    if quantized or moe.fake_balance_expert or model.eplb_config.enable_eplb():
        raise ValueError("Fixed BF16 routing forbids quantization, fake routing and dynamic EPLB")
    layout = balanced_expert_layout(model.expert_num, parallel.ep_size)
    if (model.expert_num != 512 or layout.physical_experts != physical
            or model.eplb_config.phy_exp_num(model.expert_num) != physical):
        raise ValueError(f"Expected 512 logical experts and {physical} fixed physical slots")
    source = Path(model.phy2log_path)
    actual_map = json.loads(source.read_text())
    if actual_map != layout.layer_source_map(model.num_layers):
        raise ValueError("The loader's physical checkpoint map differs from the routing LUT")
    return layout
