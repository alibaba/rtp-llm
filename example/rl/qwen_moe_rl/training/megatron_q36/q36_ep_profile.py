"""Candidate guards and exact expert coordinates for Q36 EP8 -> EP4.

This validates supplied model/live-provider metadata. It does not patch native
code, construct a distributed model, or claim the supplied metadata is genuine.
"""
from dataclasses import dataclass
from math import prod


def validate_model(config):
    actual = tuple(config.get(key) for key in ('num_hidden_layers', 'hidden_size',
                   'num_experts', 'num_experts_per_tok', 'num_attention_heads', 'num_key_value_heads'))
    if actual != (40, 2048, 256, 8, 16, 2):
        raise ValueError('EP4 profile is restricted to the exact 35B text configuration')


def validate_native_profile(*, world_size, tp_size, dp_size, ep_size, pp_size,
                            ffn_sp_size, ffn_tp_size, rank, model_config,
                            quantized, eplb, eager, moe_strategy):
    validate_model(model_config)
    if (world_size, tp_size, dp_size, ep_size, pp_size, ffn_sp_size, ffn_tp_size) != (4, 4, 1, 4, 1, 4, 1):
        raise ValueError('EP4 native profile requires exact world4/TP4/DP1/EP4 topology')
    if type(rank) is not int or rank not in range(4):
        raise ValueError('native rank must be in [0, 4)')
    if quantized or eplb or not eager or moe_strategy != 'nccl_bf16_reference':
        raise ValueError('EP4 candidate requires unquantized eager BF16 reference EP')


@dataclass(frozen=True)
class ExpertCoordinate:
    logical_name: str
    shape: tuple[int, ...]
    layer: int
    global_expert: int
    ep_rank: int
    local_expert: int
    tensor_kind: str

    @property
    def canonical_name(self):
        return f'{self.logical_name}/layer-{self.layer}/global-expert-{self.global_expert}/{self.tensor_kind}'

    def to_sr_coordinates(self):
        # Deferred because this CPU contract is usable without Torch.
        from bf16_sr_coordinates_candidate import SRCoordinates
        return SRCoordinates(self.canonical_name, self.shape, self.shape).validate()


def expert_coordinate(*, logical_name, shape, layer, ep_rank, local_expert,
                      source_global_expert, tensor_kind):
    if not isinstance(logical_name, str) or not logical_name:
        raise ValueError('reviewed logical name is required')
    if not isinstance(shape, tuple) or not shape or any(type(x) is not int or x <= 0 for x in shape) or prod(shape) <= 0:
        raise ValueError('positive local tensor shape is required')
    if type(layer) is not int or not 0 <= layer < 40:
        raise ValueError('invalid 35B layer')
    if type(ep_rank) is not int or not 0 <= ep_rank < 8:
        raise ValueError('expected training EP8 rank')
    if type(local_expert) is not int or not 0 <= local_expert < 32:
        raise ValueError('expected 32 local experts on each training rank')
    expected_global = ep_rank * 32 + local_expert
    if type(source_global_expert) is not int or source_global_expert != expected_global:
        raise ValueError('live provider/source expert ownership disagrees with EP8 mapping')
    if tensor_kind not in ('gate_proj', 'up_proj', 'down_proj', 'gate_up_proj'):
        raise ValueError('unsupported or ambiguous expert tensor kind')
    return ExpertCoordinate(logical_name, shape, layer, expected_global, ep_rank, local_expert, tensor_kind)


def destination(ep_rank, local_expert):
    if type(ep_rank) is not int or not 0 <= ep_rank < 8 or type(local_expert) is not int or not 0 <= local_expert < 32:
        raise ValueError('invalid source EP8 expert coordinate')
    global_expert = ep_rank * 32 + local_expert
    return global_expert // 64, global_expert % 64
