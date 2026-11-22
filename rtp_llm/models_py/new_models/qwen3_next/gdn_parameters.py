"""Rank-local Qwen3-Next/Qwen3.5 gated-delta checkpoint parameters."""

from __future__ import annotations

import math
from numbers import Real
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from rtp_llm.models_py.layers.linear import ColumnParallelLinear
from rtp_llm.models_py.module_base import RtpModule, copy_weight_
from rtp_llm.models_py.quant_methods.base import QuantizationConfig

if TYPE_CHECKING:
    from rtp_llm.ops import LinearAttentionConfig


def reorder_ba(ts: List[torch.Tensor], la_cfg: LinearAttentionConfig):
    """Reorder ba weight: [head_num_k, group_v*2, hidden] -> [b_all, a_all, hidden].T"""
    if len(ts) != 1 or ts[0].dim() != 2:
        raise ValueError("in_proj_ba transform expects one two-dimensional tensor")
    t = ts[0]
    hidden_size = t.shape[-1]
    head_num_k = la_cfg.linear_num_key_heads
    head_num_v = la_cfg.linear_num_value_heads
    if head_num_v % head_num_k != 0:
        raise ValueError(
            f"linear value heads {head_num_v} must divide into key heads {head_num_k}"
        )
    group_v = head_num_v // head_num_k
    if t.shape[0] != head_num_v * 2:
        raise ValueError(f"in_proj_ba rows must be {head_num_v * 2}, got {t.shape[0]}")
    t = t.reshape(head_num_k, group_v * 2, t.shape[-1])
    b, a = t.split([group_v, group_v], dim=1)
    return torch.cat([b.reshape(-1, hidden_size), a.reshape(-1, hidden_size)], dim=0)


def reorder_qkvz(ts: List[torch.Tensor], la_cfg: LinearAttentionConfig):
    """Reorder qkvz weight: split q,k,v,z per-head then concatenate."""
    if len(ts) != 1 or ts[0].dim() not in (1, 2):
        raise ValueError("in_proj_qkvz transform expects one row-oriented tensor")
    t = ts[0]
    if t.dim() == 1:
        t = t.unsqueeze(1)
    head_num_k = la_cfg.linear_num_key_heads
    head_num_v = la_cfg.linear_num_value_heads
    head_k_dim = la_cfg.linear_key_head_dim
    head_v_dim = la_cfg.linear_value_head_dim
    if head_num_v % head_num_k != 0:
        raise ValueError(
            f"linear value heads {head_num_v} must divide into key heads {head_num_k}"
        )
    group_v = head_num_v // head_num_k
    dim0, dim1 = t.shape
    qkvz_size = head_num_k * head_k_dim * 2 + head_num_v * head_v_dim * 2
    if dim0 == qkvz_size:
        t = t.reshape(head_num_k, head_k_dim * 2 + head_v_dim * group_v * 2, dim1)
        q, k, v, z = torch.split(
            t,
            [head_k_dim, head_k_dim, head_v_dim * group_v, head_v_dim * group_v],
            dim=1,
        )
        return torch.cat(
            [
                q.reshape(-1, dim1),
                k.reshape(-1, dim1),
                v.reshape(-1, dim1),
                z.reshape(-1, dim1),
            ],
            dim=0,
        )
    raise ValueError(f"in_proj_qkvz rows must be {qkvz_size}, got {dim0}")


def reorder_qkvz_scale(
    tensor: torch.Tensor,
    la_cfg: LinearAttentionConfig,
    block_n: int,
) -> torch.Tensor:
    """Reorder a fused per-block QKVZ scale grid without guessing layout."""
    head_k_dim = la_cfg.linear_key_head_dim
    head_v_dim = la_cfg.linear_value_head_dim
    head_num_k = la_cfg.linear_num_key_heads
    head_num_v = la_cfg.linear_num_value_heads
    if head_num_v % head_num_k != 0:
        raise ValueError("linear value heads must be divisible by key heads")
    if head_k_dim % block_n != 0 or head_v_dim % block_n != 0:
        raise ValueError(
            "Qwen3-Next FP8 QKVZ head dimensions must align to the output "
            f"block size {block_n}: key={head_k_dim}, value={head_v_dim}"
        )
    group_v = head_num_v // head_num_k
    per_group = (head_k_dim * 2 + head_v_dim * group_v * 2) // block_n
    expected_rows = head_num_k * per_group
    if tensor.dim() != 2 or tensor.shape[0] != expected_rows:
        raise ValueError(
            f"QKVZ scale grid must have {expected_rows} rows, got {tuple(tensor.shape)}"
        )
    reshaped = tensor.reshape(head_num_k, per_group, tensor.shape[1])
    q_rows = head_k_dim // block_n
    value_rows = head_v_dim * group_v // block_n
    q, k, v, z = torch.split(reshaped, [q_rows, q_rows, value_rows, value_rows], dim=1)
    return torch.cat(
        [
            q.reshape(-1, tensor.shape[1]),
            k.reshape(-1, tensor.shape[1]),
            v.reshape(-1, tensor.shape[1]),
            z.reshape(-1, tensor.shape[1]),
        ],
        dim=0,
    ).contiguous()


def _finite_positive_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite positive real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return result


class Qwen3NextGdnParameters(RtpModule):
    """Own checkpoint parameters independent of runtime graph and device views."""

    def __init__(
        self,
        linear_attn_config: LinearAttentionConfig,
        hidden_size: int,
        rms_norm_eps: float,
        attn_tp_size: int,
        attn_tp_rank: int,
        params_dtype: torch.dtype,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()
        self.linear_attn_config = linear_attn_config
        self.hidden_size = hidden_size
        self.rms_norm_eps = _finite_positive_real(rms_norm_eps, "layernorm_eps")
        self.attn_tp_size = attn_tp_size
        self.attn_tp_rank = attn_tp_rank
        self.tp_size = attn_tp_size
        self.head_k_dim = linear_attn_config.linear_key_head_dim
        self.head_v_dim = linear_attn_config.linear_value_head_dim
        if self.head_k_dim != self.head_v_dim:
            raise ValueError(
                "Qwen3-Next linear attention requires equal key/value head "
                f"dimensions, got {self.head_k_dim}/{self.head_v_dim}"
            )
        num_k_heads = linear_attn_config.linear_num_key_heads
        num_v_heads = linear_attn_config.linear_num_value_heads
        if not (0 <= attn_tp_rank < attn_tp_size):
            raise ValueError(f"Invalid attention TP rank {attn_tp_rank}/{attn_tp_size}")
        if min(num_k_heads, num_v_heads, self.head_k_dim, self.head_v_dim) <= 0:
            raise ValueError(
                "Linear attention head counts and dimensions must be positive"
            )
        if num_k_heads % attn_tp_size or num_v_heads % attn_tp_size:
            raise ValueError(
                "Linear attention head counts must be divisible by attention TP"
            )
        self.local_num_k_heads = num_k_heads // attn_tp_size
        self.local_num_v_heads = num_v_heads // attn_tp_size
        if self.local_num_v_heads % self.local_num_k_heads:
            raise ValueError("Local value heads must be divisible by local key heads")
        self.num_key_value_heads = self.local_num_v_heads // self.local_num_k_heads
        self.linear_conv_kernel_dim = linear_attn_config.linear_conv_kernel_dim
        self.qkvz_size = (
            self.head_k_dim * self.local_num_k_heads * 2
            + self.head_v_dim * self.local_num_v_heads * 2
        )
        self.qkv_size = (
            self.head_k_dim * self.local_num_k_heads * 2
            + self.head_v_dim * self.local_num_v_heads
        )
        self.ba_size = self.local_num_v_heads * 2

        # These projections are already rank-local after the model-specific
        # Q/K/V/Z transform. The common Linear layer still owns quantization,
        # completeness validation, backend preflight and the post-load hook.
        self.in_proj_qkvz = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=self.qkvz_size,
            tp_size=1,
            tp_rank=0,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj_qkvz",
            bias=False,
            params_dtype=params_dtype,
        )
        self.in_proj_ba_w = nn.Parameter(
            torch.empty(hidden_size, self.ba_size, dtype=params_dtype),
            requires_grad=False,
        )
        self.conv1d_w = nn.Parameter(
            torch.empty(self.qkv_size, self.linear_conv_kernel_dim, dtype=params_dtype),
            requires_grad=False,
        )
        self.dt_bias = nn.Parameter(
            torch.empty(self.local_num_v_heads, dtype=params_dtype),
            requires_grad=False,
        )
        self.a_log = nn.Parameter(
            torch.empty(self.local_num_v_heads, dtype=params_dtype),
            requires_grad=False,
        )
        self.norm_w = nn.Parameter(
            torch.empty(self.head_v_dim, dtype=params_dtype), requires_grad=False
        )
        self.out_proj = ColumnParallelLinear(
            input_size=self.local_num_v_heads * self.head_v_dim,
            output_size=hidden_size,
            tp_size=1,
            tp_rank=0,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
            bias=False,
            params_dtype=params_dtype,
        )
        self._qkvz_layouts: Dict[str, str] = {}
        self._qkvz_split_parts: Dict[str, Set[str]] = {}
        self._ba_layout: Optional[str] = None
        self._ba_split_parts: Set[str] = set()
        self._input_projection_packed = False

    def _packed_input_weight(self):
        """Return the live [QKVZ+BA,K] view, never a cached copy of a parameter."""
        if not self._input_projection_packed:
            return None
        q, ba = self.in_proj_qkvz.weight, self.in_proj_ba_w
        width = self.qkvz_size + self.ba_size
        # Moving/replacing individual parameters can break a storage alias.
        # Never read a stale BA suffix after such a move.
        if not (
            q.device == ba.device
            and q.dtype == ba.dtype
            and q.stride() == (1, width)
            and ba.stride() == (width, 1)
            and q.storage_offset() == 0
            and ba.storage_offset() == self.qkvz_size
            and q.untyped_storage().data_ptr() == ba.untyped_storage().data_ptr()
        ):
            return None
        return q.as_strided((width, self.hidden_size), (1, width))

    @torch.no_grad()
    def _pack_dense_input_projection(self):
        proj = self.in_proj_qkvz
        if self._packed_input_weight() is not None:
            return
        # Finalize the child before handing it a view into its parent's GEMM.
        # The normal model post-load walk may finalize the child again.
        proj.process_weights_after_loading()
        full_kn = torch.cat((proj.weight.T, self.in_proj_ba_w), dim=1)
        # Keep the registered objects (and all writes through them) authoritative.
        proj.weight.set_(full_kn[:, : self.qkvz_size].T)
        self.in_proj_ba_w.set_(full_kn[:, self.qkvz_size :])
        self._input_projection_packed = True

    def _prepare_input_projection(self):
        from rtp_llm.models_py.quant_methods.unquantized import UnquantizedLinearMethod

        proj = self.in_proj_qkvz
        if (
            torch.version.cuda is not None
            and torch.version.hip is None
            and self.in_proj_ba_w.is_cuda
            and self.in_proj_ba_w.dtype == torch.bfloat16
            and isinstance(proj.quant_method, UnquantizedLinearMethod)
            and proj.quant_config.quant_type == "modelopt_fp4"
            and getattr(proj.quant_config.source_config, "mixed_attention", False)
        ):
            self._pack_dense_input_projection()

    def _input_project(self, hidden_states):
        proj = self.in_proj_qkvz
        weight = self._packed_input_weight()
        if (
            weight is not None
            and not proj._forward_hooks
            and not proj._forward_pre_hooks
        ):
            fused = torch.nn.functional.linear(hidden_states, weight)
            return fused.split((self.qkvz_size, self.ba_size), dim=-1)
        return proj(hidden_states), torch.mm(hidden_states, self.in_proj_ba_w)

    def validate_runtime_device(self, device: torch.device) -> None:
        if device.type != "cuda":
            raise RuntimeError(
                "Qwen3-Next linear attention requires a CUDA or ROCm accelerator"
            )

    @staticmethod
    def _block_size(layer: ColumnParallelLinear) -> Tuple[int, int]:
        value = getattr(layer.quant_config, "weight_block_size", [128, 128])
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"Invalid FP8 weight_block_size {value!r}")
        if any(type(size) is not int or size <= 0 for size in value):
            raise ValueError(
                f"FP8 block dimensions must be positive integers: {value!r}"
            )
        return value[0], value[1]

    def _split_qkvz_rows(self, tensor: torch.Tensor, unit: int) -> torch.Tensor:
        global_k = self.head_k_dim * self.linear_attn_config.linear_num_key_heads
        global_v = self.head_v_dim * self.linear_attn_config.linear_num_value_heads
        if global_k % unit or global_v % unit:
            raise ValueError(
                f"QKVZ group sizes {global_k}/{global_v} must align to unit {unit}"
            )
        q_rows = global_k // unit
        v_rows = global_v // unit
        expected_rows = q_rows * 2 + v_rows * 2
        if tensor.dim() < 1 or tensor.shape[0] != expected_rows:
            raise ValueError(
                f"QKVZ tensor must have {expected_rows} rows, got {tuple(tensor.shape)}"
            )
        q, k, v, z = torch.split(tensor, [q_rows, q_rows, v_rows, v_rows], dim=0)
        if q_rows % self.attn_tp_size or v_rows % self.attn_tp_size:
            raise ValueError("QKVZ groups must be divisible by attention TP")
        local_q = q_rows // self.attn_tp_size
        local_v = v_rows // self.attn_tp_size
        q_start = self.attn_tp_rank * local_q
        v_start = self.attn_tp_rank * local_v
        return torch.cat(
            [
                q.narrow(0, q_start, local_q),
                k.narrow(0, q_start, local_q),
                v.narrow(0, v_start, local_v),
                z.narrow(0, v_start, local_v),
            ],
            dim=0,
        ).contiguous()

    def _split_out_weight(self, tensor: torch.Tensor) -> torch.Tensor:
        global_input = self.head_v_dim * self.linear_attn_config.linear_num_value_heads
        expected = (self.hidden_size, global_input)
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"out_proj.weight must have shape {expected}, got {tuple(tensor.shape)}"
            )
        local_input = global_input // self.attn_tp_size
        start = self.attn_tp_rank * local_input
        return tensor.narrow(1, start, local_input).contiguous()

    def _split_out_block_scale(self, tensor: torch.Tensor) -> torch.Tensor:
        block_n, block_k = self._block_size(self.out_proj)
        global_input = self.head_v_dim * self.linear_attn_config.linear_num_value_heads
        local_input = global_input // self.attn_tp_size
        start = self.attn_tp_rank * local_input
        expected = (
            math.ceil(self.hidden_size / block_n),
            math.ceil(global_input / block_k),
        )
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"out_proj.weight_scale_inv must have shape {expected}, "
                f"got {tuple(tensor.shape)}"
            )
        if self.attn_tp_size == 1:
            return tensor.contiguous()
        if start % block_k or local_input % block_k:
            raise ValueError(
                "out_proj TP input shard must align to FP8 input blocks: "
                f"start={start}, width={local_input}, block={block_k}"
            )
        return tensor.narrow(1, start // block_k, local_input // block_k).contiguous()

    def _split_ba(self, tensor: torch.Tensor) -> torch.Tensor:
        reordered = reorder_ba([tensor], self.linear_attn_config)
        global_heads = self.linear_attn_config.linear_num_value_heads
        b, a = torch.split(reordered, [global_heads, global_heads], dim=0)
        local = self.local_num_v_heads
        start = self.attn_tp_rank * local
        return (
            torch.cat([b.narrow(0, start, local), a.narrow(0, start, local)], dim=0)
            .t()
            .contiguous()
        )

    @staticmethod
    def _claim_layout(layouts: Dict[str, str], parameter: str, layout: str) -> None:
        current = layouts.get(parameter)
        if current is not None and current != layout:
            raise RuntimeError(
                f"Cannot mix {current} and {layout} checkpoint layouts for {parameter}"
            )
        layouts[parameter] = layout

    def _split_qkv_or_z_rows(
        self, tensor: torch.Tensor, component: str, unit: int
    ) -> Tuple[torch.Tensor, int]:
        global_k = self.head_k_dim * self.linear_attn_config.linear_num_key_heads
        global_v = self.head_v_dim * self.linear_attn_config.linear_num_value_heads
        if global_k % unit or global_v % unit:
            raise ValueError(
                f"QKV/Z group sizes {global_k}/{global_v} must align to unit {unit}"
            )
        q_rows = global_k // unit
        v_rows = global_v // unit
        if component == "qkv":
            expected_rows = q_rows * 2 + v_rows
        elif component == "z":
            expected_rows = v_rows
        else:
            raise ValueError(f"Unknown QKV/Z component {component!r}")
        if tensor.dim() < 1 or tensor.shape[0] != expected_rows:
            raise ValueError(
                f"in_proj_{component} tensor must have {expected_rows} rows, "
                f"got {tuple(tensor.shape)}"
            )
        if q_rows % self.attn_tp_size or v_rows % self.attn_tp_size:
            raise ValueError("QKV/Z groups must be divisible by attention TP")
        local_q = q_rows // self.attn_tp_size
        local_v = v_rows // self.attn_tp_size
        q_start = self.attn_tp_rank * local_q
        v_start = self.attn_tp_rank * local_v
        if component == "qkv":
            q, k, v = torch.split(tensor, [q_rows, q_rows, v_rows], dim=0)
            local_tensor = torch.cat(
                [
                    q.narrow(0, q_start, local_q),
                    k.narrow(0, q_start, local_q),
                    v.narrow(0, v_start, local_v),
                ],
                dim=0,
            )
            offset = 0
        else:
            local_tensor = tensor.narrow(0, v_start, local_v)
            if self.qkv_size % unit:
                raise ValueError(
                    f"Local QKV size {self.qkv_size} must align to unit {unit}"
                )
            offset = self.qkv_size // unit
        return local_tensor.contiguous(), offset

    def _load_split_qkvz(
        self, component: str, parameter: str, tensor: torch.Tensor
    ) -> None:
        if parameter == "input_scale":
            raise ValueError(
                "Split QKV/Z input_scale cannot be represented by one fused runtime "
                "scale; use a fused checkpoint or dynamic activation quantization"
            )
        if parameter == "weight_scale" and tensor.numel() == 1:
            raise ValueError(
                "Split QKV/Z scalar weight scales cannot be fused without "
                "dequantization and requantization"
            )
        # Split QKV/Z checkpoints are copied into disjoint slices of one child
        # parameter.  Tell the FST coordinator to materialize that staged child
        # before taking the target reference; the second part can then mark the
        # logical parameter complete and trigger online quantization.
        self._prepare_child_weight_loading(self.in_proj_qkvz)
        target = getattr(self.in_proj_qkvz, parameter, None)
        if not isinstance(target, nn.Parameter):
            raise RuntimeError(
                f"Unsupported split QKV/Z tensor {self.in_proj_qkvz.prefix}.{parameter}"
            )
        self._claim_layout(self._qkvz_layouts, parameter, "split")
        loaded_parts = self._qkvz_split_parts.setdefault(parameter, set())
        if component in loaded_parts:
            raise RuntimeError(f"Duplicate split QKV/Z tensor {component}.{parameter}")

        unit = 1
        if parameter == "weight":
            if tensor.dim() != 2 or tensor.shape[1] != self.hidden_size:
                raise ValueError(
                    f"in_proj_{component}.weight must have input width "
                    f"{self.hidden_size}, got {tuple(tensor.shape)}"
                )
        elif parameter == "weight_scale_inv":
            block_n, block_k = self._block_size(self.in_proj_qkvz)
            unit = block_n
            expected_columns = math.ceil(self.hidden_size / block_k)
            if tensor.dim() != 2 or tensor.shape[1] != expected_columns:
                raise ValueError(
                    f"in_proj_{component}.weight_scale_inv must have "
                    f"{expected_columns} input blocks, got {tuple(tensor.shape)}"
                )
        elif parameter == "weight_scale":
            if tensor.dim() not in (1, 2):
                raise ValueError(
                    f"in_proj_{component}.weight_scale must be row-oriented, "
                    f"got {tuple(tensor.shape)}"
                )
        else:
            raise RuntimeError(f"Unsupported split QKV/Z parameter {parameter!r}")

        local, offset = self._split_qkv_or_z_rows(tensor, component, unit)
        target_slice = target.data.narrow(0, offset, local.shape[0])
        if local.shape != target_slice.shape and local.numel() == target_slice.numel():
            local = local.reshape(target_slice.shape)
        copy_weight_(
            target_slice,
            local,
            f"{self.in_proj_qkvz.prefix}.{component}.{parameter}",
        )
        loaded_parts.add(component)
        if loaded_parts == {"qkv", "z"}:
            self.in_proj_qkvz._record_parameter_loaded(parameter)
        self._finalize_child_weight_loading(self.in_proj_qkvz)

    def _load_split_ba(self, component: str, tensor: torch.Tensor) -> None:
        if self._ba_layout is not None and self._ba_layout != "split":
            raise RuntimeError("Cannot mix fused and split checkpoint layouts for B/A")
        self._ba_layout = "split"
        if component in self._ba_split_parts:
            raise RuntimeError(f"Duplicate split B/A tensor in_proj_{component}.weight")
        global_heads = self.linear_attn_config.linear_num_value_heads
        expected = (global_heads, self.hidden_size)
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"in_proj_{component}.weight must have shape {expected}, "
                f"got {tuple(tensor.shape)}"
            )
        start = self.attn_tp_rank * self.local_num_v_heads
        local = tensor.narrow(0, start, self.local_num_v_heads).t().contiguous()
        offset = 0 if component == "b" else self.local_num_v_heads
        copy_weight_(
            self.in_proj_ba_w.data.narrow(1, offset, self.local_num_v_heads),
            local,
            f"in_proj_{component}.weight",
        )
        self._ba_split_parts.add(component)
        if self._ba_split_parts == {"a", "b"}:
            self._mark_weight_loaded("in_proj_ba_w")

    def _split_head_tensor(self, tensor: torch.Tensor, label: str) -> torch.Tensor:
        expected = self.linear_attn_config.linear_num_value_heads
        if tensor.dim() != 1 or tensor.shape[0] != expected:
            raise ValueError(
                f"{label} must have shape {(expected,)}, got {tensor.shape}"
            )
        start = self.attn_tp_rank * self.local_num_v_heads
        return tensor.narrow(0, start, self.local_num_v_heads).contiguous()

    def _split_conv1d(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 3:
            if tensor.shape[1] != 1:
                raise ValueError(
                    f"conv1d.weight middle dimension must be 1, got {tensor.shape}"
                )
            tensor = tensor.squeeze(1)
        global_k = self.head_k_dim * self.linear_attn_config.linear_num_key_heads
        global_v = self.head_v_dim * self.linear_attn_config.linear_num_value_heads
        expected = (global_k * 2 + global_v, self.linear_conv_kernel_dim)
        if tuple(tensor.shape) == expected[::-1]:
            tensor = tensor.t().contiguous()
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"conv1d.weight must have shape {expected}, got {tuple(tensor.shape)}"
            )
        q, k, v = torch.split(tensor, [global_k, global_k, global_v], dim=0)
        local_k = global_k // self.attn_tp_size
        local_v = global_v // self.attn_tp_size
        k_start = self.attn_tp_rank * local_k
        v_start = self.attn_tp_rank * local_v
        return torch.cat(
            [
                q.narrow(0, k_start, local_k),
                k.narrow(0, k_start, local_k),
                v.narrow(0, v_start, local_v),
            ],
            dim=0,
        ).contiguous()

    def _projection_tensor(self, module, parameter, tensor):
        if module is self.in_proj_qkvz:
            if parameter == "weight_scale_inv":
                unit, _ = self._block_size(module)
                reordered = reorder_qkvz_scale(tensor, self.linear_attn_config, unit)
                return self._split_qkvz_rows(reordered, unit)
            if parameter == "weight":
                if module.quant_config.quant_type == "none" and (
                    tensor.dim() != 2 or tensor.shape[1] != self.hidden_size
                ):
                    raise ValueError(
                        f"in_proj_qkvz.weight expects input width {self.hidden_size}"
                    )
            if parameter == "weight" or (
                parameter == "weight_scale" and tensor.numel() != 1
            ):
                reordered = reorder_qkvz([tensor], self.linear_attn_config)
                return self._split_qkvz_rows(reordered, 1)
        elif module is self.out_proj:
            if parameter == "weight":
                return self._split_out_weight(tensor)
            if parameter == "weight_scale_inv":
                return self._split_out_block_scale(tensor)
        else:
            raise ValueError("Not a GDN projection")
        if parameter in ("weight_scale", "input_scale"):
            return tensor
        raise ValueError(f"Unsupported GDN projection parameter {parameter!r}")

    def prepare_child_weight_updates(self, module, incoming):
        local = {
            name: self._projection_tensor(module, name, tensor)
            for name, tensor in incoming.items()
        }
        return module.prepare_weight_updates(local)

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        for name, tensor in weights.items():
            if name.startswith(("in_proj_qkvz.", "out_proj.")):
                projection, parameter = name.split(".", 1)
                module = getattr(self, projection)
                local = self._projection_tensor(module, parameter, tensor)
                if module is self.in_proj_qkvz:
                    self._claim_layout(self._qkvz_layouts, parameter, "fused")
                self._load_child_weights(module, {parameter: local})
            elif name.startswith("in_proj_qkv."):
                self._load_split_qkvz("qkv", name.removeprefix("in_proj_qkv."), tensor)
            elif name.startswith("in_proj_z."):
                self._load_split_qkvz("z", name.removeprefix("in_proj_z."), tensor)
            elif name == "in_proj_ba.weight":
                if self._ba_layout is not None and self._ba_layout != "fused":
                    raise RuntimeError(
                        "Cannot mix split and fused checkpoint layouts for B/A"
                    )
                self._ba_layout = "fused"
                if not self._assign_weight(
                    self, "in_proj_ba_w", self._split_ba(tensor)
                ):
                    raise RuntimeError("Failed to load in_proj_ba.weight")
            elif name == "in_proj_b.weight":
                self._load_split_ba("b", tensor)
            elif name == "in_proj_a.weight":
                self._load_split_ba("a", tensor)
            elif name == "conv1d.weight":
                if not self._assign_weight(
                    self, "conv1d_w", self._split_conv1d(tensor)
                ):
                    raise RuntimeError("Failed to load conv1d.weight")
            elif name == "dt_bias":
                if not self._assign_weight(
                    self, "dt_bias", self._split_head_tensor(tensor, name)
                ):
                    raise RuntimeError("Failed to load dt_bias")
            elif name == "A_log":
                if not self._assign_weight(
                    self, "a_log", self._split_head_tensor(tensor, name)
                ):
                    raise RuntimeError("Failed to load A_log")
            elif name == "norm.weight":
                if tuple(tensor.shape) != tuple(self.norm_w.shape):
                    raise ValueError(
                        f"linear_attn.norm.weight must have shape "
                        f"{tuple(self.norm_w.shape)}, got {tuple(tensor.shape)}"
                    )
                if not self._assign_weight(self, "norm_w", tensor.contiguous()):
                    raise RuntimeError("Failed to load linear attention norm")
            else:
                raise RuntimeError(f"Unsupported Qwen3-Next linear tensor {name!r}")
