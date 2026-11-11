"""Coordinates and IPC metadata for Qwen3.8 EP64-to-EP32 refit."""

import hashlib
import struct
from dataclasses import dataclass


NUM_LAYERS = 92
NUM_EXPERTS = 512
TRAIN_EP = 64
ROLLOUT_EP = 32
TRAIN_EXPERTS = NUM_EXPERTS // TRAIN_EP
ROLLOUT_EXPERTS = NUM_EXPERTS // ROLLOUT_EP
_PACKED_TAILS = {"gate_up_proj": (4096, 8192), "down_proj": (8192, 2048)}
PACKED_SHAPES = {
    f"model.layers.{layer}.mlp.experts.{field}": (NUM_EXPERTS, *tail)
    for layer in range(NUM_LAYERS)
    for field, tail in _PACKED_TAILS.items()
}

# A distinct wire prefix keeps the ordinary IPC descriptor opaque. The native
# RPC still validates the active version and transaction on every write.
_MAGIC = b"Q38EP32\x01"
_HEADER = struct.Struct("!8sHHH32s")


@dataclass(frozen=True)
class ExpertShardLayout:
    rollout_rank: int

    def __post_init__(self):
        if type(self.rollout_rank) is not int or not 0 <= self.rollout_rank < ROLLOUT_EP:
            raise ValueError("Invalid Qwen3.8 EP32 rollout rank")

    @property
    def experts(self):
        begin = self.rollout_rank * ROLLOUT_EXPERTS
        return range(begin, begin + ROLLOUT_EXPERTS)

    @property
    def train_ranks(self):
        return (2 * self.rollout_rank, 2 * self.rollout_rank + 1)

    @classmethod
    def from_load_config(cls, config):
        topology = (
            config.num_layers,
            config.expert_num,
            config.phy_exp_num,
            config.tp_size,
            config.dp_size,
            config.ep_size,
            config.ffn_tp_size,
            config.ffn_tp_rank,
        )
        if topology != (NUM_LAYERS, NUM_EXPERTS, NUM_EXPERTS, 8, 4, ROLLOUT_EP, 1, 0) or (
            config.enable_eplb or config.merge_lora or config.is_ft_style_weight
        ):
            raise ValueError("Local expert refit requires Qwen3.8 TP8/DP4/EP32 with ETP1")
        layout = cls(config.ep_rank)
        for layer in range(NUM_LAYERS):
            selected = config.get_selected_experts(layer, NUM_EXPERTS)
            if tuple(selected) != tuple(layout.experts):
                raise ValueError(f"Qwen3.8 expert ownership changed at layer {layer}")
        return layout

    def source_shapes(self, global_shapes):
        """Keep the full HF key set and only replace the packed expert axis."""
        for name, shape in PACKED_SHAPES.items():
            if name not in global_shapes or tuple(global_shapes[name]) != shape:
                raise ValueError(f"Unexpected Qwen3.8 packed source shape for {name}")
        local_shapes = dict(global_shapes)
        for name, shape in PACKED_SHAPES.items():
            local_shapes[name] = (ROLLOUT_EXPERTS, *shape[1:])
        return local_shapes

    def pack_ipc(self, name, train_ranks, metadata):
        if name not in PACKED_SHAPES or tuple(train_ranks) != self.train_ranks:
            raise ValueError("Unexpected Qwen3.8 expert source name or sender order")
        if not isinstance(metadata, bytes) or not metadata:
            raise ValueError("An opaque CUDA IPC descriptor is required")
        name_digest = hashlib.sha256(name.encode()).digest()
        return _HEADER.pack(_MAGIC, self.rollout_rank, *self.train_ranks, name_digest) + metadata

    def unpack_ipc(self, name, metadata):
        if name not in PACKED_SHAPES:
            if isinstance(metadata, bytes) and metadata.startswith(_MAGIC):
                raise ValueError("Expert IPC metadata cannot be used for a nonexpert source")
            return metadata
        if not isinstance(metadata, bytes) or len(metadata) <= _HEADER.size:
            raise ValueError("Missing Qwen3.8 local expert IPC metadata")
        expected = (_MAGIC, self.rollout_rank, *self.train_ranks, hashlib.sha256(name.encode()).digest())
        if _HEADER.unpack_from(metadata) != expected:
            raise ValueError("Qwen3.8 expert IPC rank, sender order or name mismatch")
        return metadata[_HEADER.size:]
