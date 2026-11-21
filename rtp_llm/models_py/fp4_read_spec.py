"""Read planning for unfused ModelOpt NVFP4 checkpoint projections.

Shapes describe the checkpoint, before backend-specific scale swizzling.
This primitive does not register a quant method or enable model routing.
Separate Q/K/V projections also support grouped-query KV replication. Fused
checkpoint keys, merged projections and expert placement need their own plans.
"""

import math
from typing import Tuple

from rtp_llm.models_py.read_spec import ReadSpec, make_sharded_read_spec


def modelopt_fp4_qkv_read_spec(
    name: str,
    shape: Tuple[int, ...],
    *,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    tp_size: int,
    tp_rank: int,
) -> ReadSpec:
    """Read one separate Q/K/V projection in ModelOpt's packed layout.

    KV heads are shared by consecutive ranks when TP exceeds the number of KV
    heads, matching QKVParallelLinear and the legacy attention partition. The
    returned shard metadata describes unique KV owners, not replica count.
    Scalar scales remain replicated; no scale reduction or swizzle is done here.
    """
    for label, value in (
        ("hidden_size", hidden_size),
        ("num_heads", num_heads),
        ("num_kv_heads", num_kv_heads),
        ("head_dim", head_dim),
        ("tp_size", tp_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
    if (
        isinstance(tp_rank, bool)
        or not isinstance(tp_rank, int)
        or not 0 <= tp_rank < tp_size
    ):
        raise ValueError("tp_rank must be an integer within the TP group")
    if num_heads % num_kv_heads or num_heads % tp_size:
        raise ValueError("Query heads must be divisible by KV heads and TP size")
    if num_kv_heads % tp_size and tp_size % num_kv_heads:
        raise ValueError("KV heads and TP size must be mutually divisible")
    if not isinstance(name, str):
        raise TypeError("Checkpoint tensor name must be a string")
    parts = name.rsplit(".", 2)
    projection = parts[-2] if len(parts) >= 2 else None
    if projection not in ("q_proj", "k_proj", "v_proj"):
        raise ValueError("Expected a separate q_proj/k_proj/v_proj checkpoint tensor")
    is_query = projection == "q_proj"
    heads = num_heads if is_query else num_kv_heads
    shards = tp_size if is_query else math.gcd(num_kv_heads, tp_size)
    rank = tp_rank if is_query else tp_rank // (tp_size // shards)
    return modelopt_fp4_read_spec(
        name,
        shape,
        input_size=hidden_size,
        output_size=heads * head_dim,
        tp_size=shards,
        tp_rank=rank,
        partition_dim=0,
    )


def modelopt_fp4_read_spec(
    name: str,
    shape: Tuple[int, ...],
    *,
    input_size: int,
    output_size: int,
    tp_size: int,
    tp_rank: int,
    partition_dim: int,
) -> ReadSpec:
    """Plan a dense projection's four tensors in their physical layouts.

    ``partition_dim=0`` is column parallel (split output features), and
    ``partition_dim=1`` is row parallel (split input features). ModelOpt
    NVFP4 packs two E2M1 values per byte and uses 16-value scale blocks.
    Scalar scales are replicated, matching the legacy loader's sp_id rule.
    """
    for label, value in (
        ("input_size", input_size),
        ("output_size", output_size),
        ("tp_size", tp_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
    if (
        isinstance(tp_rank, bool)
        or not isinstance(tp_rank, int)
        or not 0 <= tp_rank < tp_size
    ):
        raise ValueError("tp_rank must be an integer within the TP group")
    if (
        isinstance(partition_dim, bool)
        or not isinstance(partition_dim, int)
        or partition_dim not in (0, 1)
    ):
        raise ValueError("partition_dim must be 0 (output) or 1 (input)")
    if input_size % 16:
        raise ValueError("NVFP4 input_size must align to 16-value scale blocks")
    partition_size = (output_size, input_size)[partition_dim]
    if partition_size % tp_size:
        raise ValueError("NVFP4 partition size must be divisible by tp_size")
    if partition_dim == 1 and (input_size // tp_size) % 16:
        raise ValueError(
            "NVFP4 row-parallel shards must align to 16-value scale blocks"
        )

    # Validate names and shape metadata even for scalar/replicated tensors.
    full = ReadSpec.full(name, shape)
    parameter = name.rsplit(".", 1)[-1]
    if parameter in ("weight_scale_2", "input_scale"):
        if math.prod(full.source_shape) != 1:
            raise ValueError(f"{name} must be scalar for an unfused NVFP4 projection")
        return full
    expected_shapes = {
        "weight": (output_size, input_size // 2),
        "weight_scale": (output_size, input_size // 16),
    }
    if parameter not in expected_shapes:
        raise ValueError(f"Unknown ModelOpt NVFP4 checkpoint parameter {name!r}")
    expected = expected_shapes[parameter]
    if full.source_shape != expected:
        raise ValueError(
            f"{name} expected checkpoint shape {expected}, got {full.source_shape}"
        )
    if tp_size == 1:
        return full
    return make_sharded_read_spec(
        name, full.source_shape, partition_dim, tp_size, tp_rank
    )
