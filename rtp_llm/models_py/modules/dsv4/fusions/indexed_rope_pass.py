"""GraphFX replacements for DSV4 indexed RoPE consumers.

Scenario
========
DSV4 FP8 decode and prefill materialize per-token RoPE rows from the full
``freqs_cis`` table and then feed those rows to RoPE consumers.  The pass
removes that materialized gather when the fixed hidden/head dimensions match
the indexed CUDA wrappers; dynamic batch/sequence dimensions are intentionally
left to runtime validation.

Indexed RMSNorm+RoPE same-graph form, seen in decode Q/KV:

    freqs_cis --+
                +--> position_ids.to(int64) --> index_select(dim=0)
                                                     |
                                                     v
                                               contiguous/view
                                                     |
    q_or_kv -----------------------------------------+
                                                     v
                         fused_rmsnorm_rope(q_or_kv, weight_or_None, freqs)

becomes:

    freqs_cis --------+
    position_ids -----+--> dsv4_indexed_rmsnorm_rope(q_or_kv, weight_or_None,
    q_or_kv ----------+                                 freqs_cis, position_ids)

Indexed inv-RoPE+FP8 quant, seen in decode output projection, has the same
materialized gather before the FP8 DeepGEMM consumer:

    freqs_cis --+
                +--> position_ids.to(int64) --> index_select(dim=0)
                                                     |
                                                     v
                                               contiguous/view
                                                     |
    o -----------------------------------------------+
                                                     v
                      fused_inv_rope_fp8_quant(o, materialized_freqs)
                                                     |
                                                     v
                                      (o_fp8, o_scale) -> DeepGEMM

The equivalent replacement keeps the dynamic request dimensions out of FX
matching and lets the CUDA wrapper validate runtime shape/stride/dtype:

    freqs_cis --------+
    position_ids -----+--> dsv4_indexed_inv_rope_fp8_quant(o, freqs_cis, position_ids)
    o ----------------+                                      |
                                                              v
                                               (o_fp8, o_scale) -> DeepGEMM

For cross-graph splits, the producer graph emits a poison/provenance token
instead of a real gather, and the consumer graph resolves it:

    Graph A before: freqs_cis.index_select(0, position_ids).contiguous() -> output
    Graph A after : dsv4_indexed_rope_freqs_token(freqs_cis, position_ids) -> output

    Graph B before: fused_rmsnorm_rope(x, weight, freqs_per_token, rd)
                    or fused_inv_rope_fp8_quant(o, freqs_per_token)
    Graph B after : dsv4_indexed_rmsnorm_rope_from_freqs(x, weight, freqs_token, rd)
                    or dsv4_indexed_inv_rope_fp8_quant_from_freqs(o, freqs_token)

Code key paths
==============
GraphFX install/registry:
``rtp_llm/models_py/modules/dsv4/fusions/graphfx_injector.py`` and
``rtp_llm/models_py/modules/dsv4/fusions/fusion_registry.py``.

Runtime/kernel replacements:
``rtp_llm/models_py/modules/dsv4/fusions/indexed_rope_runtime.py`` and
``rtp_llm/models_py/kernels/cuda/dsv4_indexed_rope/dsv4_indexed_rope.py``.

Concrete DSV4 origins as of 2026-05-18
======================================
Decode same-graph RMSNorm+RoPE:
``rtp_llm/models_py/modules/dsv4/fp8/decode/compute_qkv.py:51-54``
materializes ``freqs_cis`` with ``attn.freqs_cis.index_select``.
``compute_qkv.py:60-64`` consumes it in the Q ``fused_rmsnorm_rope`` path and
``compute_qkv.py:66-72`` consumes it in the KV ``fused_rmsnorm_rope`` path.
The call site is ``rtp_llm/models_py/modules/dsv4/fp8/attention.py:1842``.

Decode cross-graph inv-RoPE+FP8 quant:
``compute_qkv.py:74`` returns the materialized ``DecodeQKV.freqs_cis``.
``rtp_llm/models_py/modules/dsv4/fp8/attention.py:1858`` threads it into
``decode_output_proj`` where
``rtp_llm/models_py/modules/dsv4/fp8/decode/output_proj.py:58-66`` calls
``fused_inv_rope_fp8_quant`` before ``wo_a`` DeepGEMM.

Prefill RMSNorm+RoPE cross-graph:
``rtp_llm/models_py/modules/dsv4/fp8/attention.py:2742-2749`` builds
``common.freqs_cis`` from ``position_ids``.  ``attention.py:3888-3897``
consumes it in the Q and KV ``fused_rmsnorm_rope`` paths.  The metadata is
installed from ``rtp_llm/models_py/modules/dsv4/fp8/prefill_meta.py:32`` and
called by ``rtp_llm/models_py/modules/dsv4/prefill/forward.py:301``.

Prefill output projection also calls ``fused_inv_rope_fp8_quant`` at
``rtp_llm/models_py/modules/dsv4/fp8/attention.py:4211`` and
``attention.py:4243``, but it receives already per-token ``freqs`` from
``common.freqs_cis`` and is not the indexed ``position_ids`` lookup pattern.
"""

from __future__ import annotations

import logging
import operator
import os

import torch

from rtp_llm.models_py.kernels.cuda.dsv4_indexed_rope import (
    dsv4_indexed_inv_rope_fp8_quant,
    dsv4_indexed_rmsnorm_rope,
)
from rtp_llm.models_py.modules.dsv4.fusions.fusion_registry import (
    eliminate_dead_code_preserving_dsv4_side_effects,
    record_dsv4_fusion_hit,
    record_dsv4_fusion_miss,
    register_dsv4_fusion_pass,
)
from rtp_llm.models_py.modules.dsv4.fusions.indexed_rope_runtime import (
    dsv4_indexed_inv_rope_fp8_quant_from_freqs,
    dsv4_indexed_rmsnorm_rope_from_freqs,
    dsv4_indexed_rope_freqs_token,
)

logger = logging.getLogger(__name__)


def _target_name(target) -> str:
    return getattr(target, "__name__", str(target))


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _is_call_function(node: object, name: str) -> bool:
    return isinstance(node, torch.fx.Node) and node.op == "call_function" and _target_name(node.target) == name


def _is_call_method(node: object, name: str) -> bool:
    return isinstance(node, torch.fx.Node) and node.op == "call_method" and _target_name(node.target) == name


def _unwrap_contiguous(node: object) -> object:
    while _is_call_method(node, "contiguous") and node.args:
        node = node.args[0]
    return node


def _unwrap_position_cast(node: object) -> object:
    if not _is_call_method(node, "to") or not node.args:
        return node
    # Keep the original dynamic position tensor when the graph only inserted
    # the production path's dtype cast to int64 for materialized gather.
    for arg in node.args[1:]:
        if arg is torch.long or arg is torch.int64:
            return node.args[0]
    dtype = node.kwargs.get("dtype") if isinstance(node.kwargs, dict) else None
    if dtype in (torch.long, torch.int64):
        return node.args[0]
    return node


def _is_layout_only_node(node: object) -> bool:
    if not isinstance(node, torch.fx.Node):
        return False
    if node.op == "call_method" and _target_name(node.target) in (
        "contiguous",
        "reshape",
        "view",
    ):
        return True
    return node.op == "call_function" and _target_name(node.target) in (
        "reshape",
        "view",
    )


def _has_only_output_or_layout_users(
    node: torch.fx.Node,
    seen: set[torch.fx.Node] | None = None,
) -> bool:
    if seen is None:
        seen = set()
    if node in seen:
        return True
    seen.add(node)
    if not node.users:
        return False
    for user in list(node.users):
        if user.op == "output":
            continue
        if not _is_layout_only_node(user):
            return False
        if not _has_only_output_or_layout_users(user, seen):
            return False
    return True


def _match_index_select_freqs(node: object):
    node = _unwrap_contiguous(node)
    if _is_call_method(node, "index_select") and len(node.args) >= 3:
        freqs_cis, dim, position_ids = node.args[:3]
    elif _is_call_function(node, "index_select") and len(node.args) >= 3:
        freqs_cis, dim, position_ids = node.args[:3]
    elif isinstance(node, torch.fx.Node) and node.op == "call_function" and node.target == operator.getitem:
        if len(node.args) < 2:
            return None
        freqs_cis, position_ids = node.args[:2]
        position_ids = _single_tensor_index(position_ids)
        if position_ids is None:
            return None
        dim = 0
    elif _is_call_function(node, "index.Tensor") or _is_call_function(node, "aten.index.Tensor"):
        if len(node.args) < 2:
            return None
        freqs_cis = node.args[0]
        position_ids = _single_tensor_index(node.args[1])
        if position_ids is None:
            return None
        dim = 0
    else:
        return None
    if dim != 0:
        return None
    return freqs_cis, _unwrap_position_cast(position_ids)


def _single_tensor_index(index: object):
    if isinstance(index, (tuple, list)):
        if len(index) != 1:
            return None
        index = index[0]
    if isinstance(index, slice):
        return None
    return index


def _tensor_meta(node: object):
    if not isinstance(node, torch.fx.Node):
        return None
    return (node.meta or {}).get("tensor_meta")


def _meta_dtype(node: object):
    meta = _tensor_meta(node)
    return getattr(meta, "dtype", None)


def _looks_like_freqs_index_select(node: object) -> bool:
    match = _match_index_select_freqs(node)
    if match is None:
        return False
    freqs_cis, _ = match
    if _meta_dtype(node) == torch.complex64 or _meta_dtype(freqs_cis) == torch.complex64:
        return True
    return "freqs_cis" in str(freqs_cis).lower()


def _kw_or_arg(node: torch.fx.Node, name: str, index: int, default=None):
    if name in node.kwargs:
        return node.kwargs[name]
    return node.args[index] if len(node.args) > index else default


def _mul_two_expr(value: object):
    if not isinstance(value, torch.fx.Node):
        return None
    if value.op != "call_function":
        return None
    if value.target not in (operator.mul, torch.mul):
        return None
    if len(value.args) < 2:
        return None
    lhs, rhs = value.args[:2]
    if lhs == 2:
        return rhs
    if rhs == 2:
        return lhs
    return None


def _fixed_positive_int(value: object) -> bool:
    return isinstance(value, int) and value > 0


def _is_torch_empty_sentinel(node: object) -> bool:
    return (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is torch.empty
        and len(node.args) >= 1
        and node.args[0] == 1
    )


def _match_view_as_real_freqs(node: object):
    if not _is_call_function(node, "view_as_real") or not node.args:
        return None
    freqs_flat = node.args[0]
    if not (_is_call_method(freqs_flat, "view") or _is_call_function(freqs_flat, "view")):
        return None
    if len(freqs_flat.args) < 3:
        return None
    freqs_cis = freqs_flat.args[0]
    rope_half_dim = freqs_flat.args[2]
    return freqs_cis, rope_half_dim


def _match_flat_view(node: object):
    if not (_is_call_method(node, "view") or _is_call_function(node, "view")):
        return None
    if len(node.args) < 3:
        return None
    if node.args[1] != -1:
        return None
    return node.args[0], node.args[2]


def _find_view_of(node: torch.fx.Node):
    views = []
    for user in list(node.users):
        if not (_is_call_method(user, "view") or _is_call_function(user, "view")):
            continue
        if user.args and user.args[0] is node:
            views.append(user)
    return views[0] if len(views) == 1 else None


def _erase_dead_node(gm: torch.fx.GraphModule, node: torch.fx.Node, miss_reason: str) -> bool:
    """Erase a replaced side-effect node only when FX proves it has no users."""
    if node.users:
        record_dsv4_fusion_miss("indexed_rope_fx", miss_reason)
        return False
    gm.graph.erase_node(node)
    return True


def _rewrite_lowered_rmsnorm_rope(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Rewrite the real torch.compile-lowered Triton RMSNorm+RoPE graph.

    Some DSV4 paths do not preserve the Python ``fused_rmsnorm_rope`` call in
    the FX graph.  Dynamo inlines it into a Triton mutation wrapper with
    ``x_ptr/w_ptr/freqs_ri_ptr/out_ptr`` kwargs.  This matcher keeps the pass
    non-invasive by recognizing only that lowered operator contract.
    """
    if not _env_flag("DSV4_INDEXED_ROPE_REWRITE_LOWERED_RMSNORM_ROPE", True):
        return False
    if not _is_call_function(node, "triton_kernel_wrapper_mutation"):
        return False
    triton_kwargs = node.kwargs.get("kwargs") if isinstance(node.kwargs, dict) else None
    if not isinstance(triton_kwargs, dict):
        return False
    required_keys = {"x_ptr", "w_ptr", "freqs_ri_ptr", "out_ptr"}
    if not required_keys.issubset(triton_kwargs):
        return False
    x_flat = triton_kwargs["x_ptr"]
    weight = triton_kwargs["w_ptr"]
    freqs_ri = triton_kwargs["freqs_ri_ptr"]
    out_flat = triton_kwargs["out_ptr"]
    x_match = _match_flat_view(x_flat)
    freqs_match = _match_view_as_real_freqs(freqs_ri)
    if x_match is None or freqs_match is None or not isinstance(out_flat, torch.fx.Node):
        return False
    x, _ = x_match
    freqs_cis, rope_half_dim = freqs_match
    rope_head_dim = triton_kwargs.get("freqs_stride_b")
    if rope_head_dim is None:
        rope_head_dim = _mul_two_expr(rope_half_dim)
    if rope_head_dim is None:
        record_dsv4_fusion_miss("indexed_rope_fx", "lowered_rmsnorm_rope_missing_rope_dim")
        return False
    output_view = _find_view_of(out_flat)
    if output_view is None:
        record_dsv4_fusion_miss("indexed_rope_fx", "lowered_rmsnorm_rope_missing_output_view")
        return False
    if node.users:
        record_dsv4_fusion_miss("indexed_rope_fx", "lowered_rmsnorm_rope_old_mutation_still_used")
        return False
    weight_arg = None if _is_torch_empty_sentinel(weight) else weight
    with gm.graph.inserting_before(output_view):
        fused = gm.graph.call_function(
            dsv4_indexed_rmsnorm_rope_from_freqs,
            args=(x, weight_arg, freqs_cis, rope_head_dim),
            kwargs={
                "eps": 1e-6,
                "inverse": False,
                "out": None,
                "inplace": False,
                "group_heads": None,
            },
        )
    output_view.replace_all_uses_with(fused)
    # The original Triton wrapper mutates ``out_flat`` and returns no value.
    # After all output-view users are replaced by the indexed CUDA op, keeping
    # the wrapper would execute both the old and new kernels because FX DCE
    # treats Triton mutation wrappers as impure.  Remove it only when no FX
    # users remain, so unexpected producer/consumer layouts fail closed.
    _erase_dead_node(gm, node, "lowered_rmsnorm_rope_old_mutation_still_used")
    return True


def _rewrite_rmsnorm_rope(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    if not _env_flag("DSV4_INDEXED_ROPE_REWRITE_RMSNORM_ROPE", True):
        return False
    if not _is_call_function(node, "fused_rmsnorm_rope") or len(node.args) < 4:
        return False
    freqs_match = _match_index_select_freqs(node.args[2])
    if freqs_match is None:
        rope_head_dim = node.args[3]
        with gm.graph.inserting_before(node):
            fused = gm.graph.call_function(
                dsv4_indexed_rmsnorm_rope_from_freqs,
                args=(node.args[0], node.args[1], node.args[2], rope_head_dim),
                kwargs={
                    "eps": node.kwargs.get("eps", 1e-6),
                    "inverse": node.kwargs.get("inverse", False),
                    "out": node.kwargs.get("out", None),
                    "inplace": node.kwargs.get("inplace", False),
                    "group_heads": node.kwargs.get("group_heads", None),
                },
            )
        node.replace_all_uses_with(fused)
        return True
    if (
        bool(node.kwargs.get("inverse", False))
        or node.kwargs.get("out", None) is not None
        or bool(node.kwargs.get("inplace", False))
        or node.kwargs.get("group_heads", None) is not None
    ):
        record_dsv4_fusion_miss("indexed_rope_fx", "rmsnorm_rope_unsupported_kwargs")
        return False
    rope_head_dim = node.args[3]
    if isinstance(rope_head_dim, int) and not _fixed_positive_int(rope_head_dim):
        record_dsv4_fusion_miss("indexed_rope_fx", "rmsnorm_rope_invalid_rope_dim")
        return False
    with gm.graph.inserting_before(node):
        fused = gm.graph.call_function(
            dsv4_indexed_rmsnorm_rope,
            args=(node.args[0], node.args[1], freqs_match[0], freqs_match[1], rope_head_dim),
            kwargs={"eps": node.kwargs.get("eps", 1e-6)},
        )
    node.replace_all_uses_with(fused)
    return True


def _rewrite_inv_rope_quant(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    if not _env_flag("DSV4_INDEXED_ROPE_REWRITE_INV_ROPE_QUANT", True):
        return False
    if not _is_call_function(node, "fused_inv_rope_fp8_quant") or len(node.args) < 2:
        return False
    freqs_match = _match_index_select_freqs(node.args[1])
    n_groups = _kw_or_arg(node, "n_groups", 2)
    heads_per_group = _kw_or_arg(node, "heads_per_group", 3)
    nope_dim = _kw_or_arg(node, "nope_dim", 4)
    rope_head_dim = _kw_or_arg(node, "rope_head_dim", 5)
    quant_group_size = node.kwargs.get("quant_group_size", 128)
    fp8_buf = node.kwargs.get("fp8_buf", None)
    scale_buf = node.kwargs.get("scale_buf", None)
    if any(isinstance(v, int) and not _fixed_positive_int(v) for v in (n_groups, heads_per_group, rope_head_dim)):
        record_dsv4_fusion_miss("indexed_rope_fx", "inv_rope_quant_invalid_group_or_rope_dim")
        return False
    if isinstance(nope_dim, int) and nope_dim < 0:
        record_dsv4_fusion_miss("indexed_rope_fx", "inv_rope_quant_invalid_nope_dim")
        return False
    if quant_group_size != 128:
        record_dsv4_fusion_miss("indexed_rope_fx", "inv_rope_quant_unsupported_quant_group")
        return False
    if freqs_match is None:
        with gm.graph.inserting_before(node):
            fused = gm.graph.call_function(
                dsv4_indexed_inv_rope_fp8_quant_from_freqs,
                args=(
                    node.args[0],
                    node.args[1],
                    n_groups,
                    heads_per_group,
                    nope_dim,
                    rope_head_dim,
                ),
                kwargs={
                    "quant_group_size": quant_group_size,
                    "eps": node.kwargs.get("eps", 1e-10),
                    "fp8_buf": fp8_buf,
                    "scale_buf": scale_buf,
                    "impl": node.kwargs.get("impl", None),
                    "heads_per_cta": node.kwargs.get("heads_per_cta", None),
                },
            )
        node.replace_all_uses_with(fused)
        return True
    if fp8_buf is not None or scale_buf is not None:
        record_dsv4_fusion_miss("indexed_rope_fx", "inv_rope_quant_explicit_output_buffer")
        return False
    with gm.graph.inserting_before(node):
        fused = gm.graph.call_function(
            dsv4_indexed_inv_rope_fp8_quant,
            args=(node.args[0], freqs_match[0], freqs_match[1]),
            kwargs={
                "n_groups": n_groups,
                "heads_per_group": heads_per_group,
                "nope_dim": nope_dim,
                "rope_head_dim": rope_head_dim,
                "quant_group_size": quant_group_size,
                "eps": node.kwargs.get("eps", 1e-10),
            },
        )
    node.replace_all_uses_with(fused)
    return True


def _rewrite_freqs_index_select_token(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    if not _env_flag("DSV4_INDEXED_ROPE_REWRITE_FREQ_TOKEN", True):
        return False
    if not _looks_like_freqs_index_select(node):
        return False
    freqs_match = _match_index_select_freqs(node)
    if freqs_match is None:
        return False
    freqs_cis, position_ids = freqs_match
    replace_node = _unwrap_contiguous(node)
    if isinstance(replace_node, torch.fx.Node) and not _has_only_output_or_layout_users(replace_node):
        record_dsv4_fusion_miss("indexed_rope_fx", "freq_token_has_unsupported_same_graph_consumer")
        return False
    with gm.graph.inserting_before(replace_node):
        token = gm.graph.call_function(
            dsv4_indexed_rope_freqs_token,
            args=(freqs_cis, position_ids),
        )
    node.replace_all_uses_with(token)
    replace_node.replace_all_uses_with(token)
    return True


def apply_indexed_rope_fx_pass(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Replace materialized RoPE-table gather patterns with indexed CUDA ops.

    The pass intentionally ignores dynamic batch/sequence length. It rewrites
    only when the fixed operator contract is visible in the graph:
    ``freqs_cis.index_select(0, position_ids).contiguous()`` feeding either
    ``fused_rmsnorm_rope`` or ``fused_inv_rope_fp8_quant``. The inserted CUDA
    wrapper performs the runtime dtype/shape/stride checks and fails fast.
    """
    replaced = 0
    consumer_replaced = 0
    rewrite_consumers = _env_flag("DSV4_INDEXED_ROPE_REWRITE_CONSUMERS", True)
    rewrite_freq_token = _env_flag("DSV4_INDEXED_ROPE_REWRITE_FREQ_TOKEN", True)
    if os.environ.get("DSV4_FUSION_REGISTRY_DEBUG"):
        logger.info(
            "DSV4 indexed RoPE pass gates: consumers=%s rmsnorm_rope=%s "
            "inv_rope_quant=%s freq_token=%s",
            rewrite_consumers,
            _env_flag("DSV4_INDEXED_ROPE_REWRITE_RMSNORM_ROPE", True),
            _env_flag("DSV4_INDEXED_ROPE_REWRITE_INV_ROPE_QUANT", True),
            rewrite_freq_token,
        )
    if rewrite_consumers:
        for node in list(gm.graph.nodes):
            if not isinstance(node, torch.fx.Node):
                continue
            if _rewrite_rmsnorm_rope(gm, node):
                replaced += 1
                consumer_replaced += 1
                continue
            if _rewrite_lowered_rmsnorm_rope(gm, node):
                replaced += 1
                consumer_replaced += 1
                continue
            if _rewrite_inv_rope_quant(gm, node):
                replaced += 1
                consumer_replaced += 1
        if consumer_replaced and rewrite_freq_token:
            # Consumer rewrites replace old RoPE users but those dead nodes still
            # appear in the gather's users set until DCE runs.  Clean them before
            # deciding whether a remaining gather is producer/output-only.
            eliminate_dead_code_preserving_dsv4_side_effects(gm)
    if rewrite_freq_token:
        for node in list(gm.graph.nodes):
            if not isinstance(node, torch.fx.Node):
                continue
            if _rewrite_freqs_index_select_token(gm, node):
                replaced += 1
    if replaced:
        eliminate_dead_code_preserving_dsv4_side_effects(gm)
        record_dsv4_fusion_hit("indexed_rope_fx", replaced)
        logger.info("DSV4 FX fused %d indexed RoPE patterns", replaced)
    return gm


def register_indexed_rope_pass() -> None:
    register_dsv4_fusion_pass(
        "indexed_rope_fx",
        apply_indexed_rope_fx_pass,
        priority=5,
        env_gate="DSV4_INDEXED_ROPE_CUDA",
    )


register_indexed_rope_pass()
