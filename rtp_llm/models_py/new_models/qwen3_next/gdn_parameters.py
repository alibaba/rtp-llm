"""Rank-local Qwen3-Next/Qwen3.5 gated-delta checkpoint parameters."""

from __future__ import annotations

import math
from numbers import Real
from typing import TYPE_CHECKING, Any, Dict, Optional, Set, Tuple

import torch
import torch.nn as nn

from rtp_llm.models_py.layers.linear import ColumnParallelLinear
from rtp_llm.models_py.module_base import CheckpointLoadTarget, RtpModule, copy_weight_
from rtp_llm.models_py.new_models.gdn_checkpoint_layout import (
    reorder_ba,
    reorder_qkvz,
    reorder_qkvz_scale,
)
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.runtime_weight_update import TensorUpdate

if TYPE_CHECKING:
    from rtp_llm.ops import LinearAttentionConfig


def _finite_positive_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite positive real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return result


class Qwen3NextGdnParameters(RtpModule):
    """Own checkpoint parameters independent of runtime graph and device views."""

    supports_independent_weight_updates = True
    _state_parameters = {
        "in_proj_ba.weight": "in_proj_ba_w",
        "conv1d.weight": "conv1d_w",
        "dt_bias": "dt_bias",
        "A_log": "a_log",
        "norm.weight": "norm_w",
    }

    def online_checkpoint_target(self, target):
        name = target.name
        if name in self._state_parameters or name in ("in_proj_b.weight", "in_proj_a.weight"):
            return target
        projection = name.partition(".")[0]
        if projection in ("in_proj_qkv", "in_proj_z"):
            return CheckpointLoadTarget(self.in_proj_qkvz, name, True)
        return None  # Other projections resolve to the exact child parameter.

    def _local_state_tensor(self, name, tensor):
        if name == "in_proj_ba.weight":
            return self._split_ba(tensor)
        if name == "conv1d.weight":
            return self._split_conv1d(tensor)
        if name in ("dt_bias", "A_log"):
            return self._split_head_tensor(tensor, name)
        if name == "norm.weight":
            if tensor.shape != self.norm_w.shape:
                raise ValueError(f"linear_attn.norm.weight must have shape {tuple(self.norm_w.shape)}")
            return tensor.contiguous()
        raise NotImplementedError(f"No GDN state update for {name!r}")

    def prepare_weight_updates(self, weights):
        updates = []
        split_ba = None
        for name, tensor in weights.items():
            if name in ("in_proj_b.weight", "in_proj_a.weight"):
                if "in_proj_ba.weight" in weights:
                    raise ValueError("Cannot update fused and split B/A in one request")
                if tensor.dtype != self.in_proj_ba_w.dtype:
                    raise TypeError(f"{name} requires {self.in_proj_ba_w.dtype}, got {tensor.dtype}")
                if split_ba is None:
                    split_ba = self.in_proj_ba_w.detach().to(device="cpu", copy=True)
                component = name.removeprefix("in_proj_").removesuffix(".weight")
                local, offset = self._split_ba_component(component, tensor)
                split_ba.narrow(1, offset, self.local_num_v_heads).copy_(local)
                continue
            local = self._local_state_tensor(name, tensor)
            parameter = self._state_parameters[name]
            target = self._parameters[parameter]
            if tensor.dtype != target.dtype:
                raise TypeError(f"{name} requires {target.dtype}, got {tensor.dtype}")
            updates.append(TensorUpdate(parameter, target, local))
        if split_ba is not None:
            updates.append(TensorUpdate("in_proj_ba_w", self.in_proj_ba_w, split_ba))
        return updates

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
        grouped_control_source: bool = False,
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
        self._grouped_controls = None
        if grouped_control_source and getattr(quant_config, "quant_type", None) in ("awq", "gptq"):
            from rtp_llm.models_py.new_models.qwen3_next.grouped_controls import GroupedControlSource

            self._grouped_controls = GroupedControlSource(quant_config, prefix, hidden_size, num_v_heads, params_dtype)

        # These projections are already rank-local after the model-specific
        # Q/K/V/Z transform. The common Linear layer still owns quantization,
        # completeness validation, backend preflight and the post-load hook.
        self.in_proj_qkvz = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=self.qkvz_size,
            tp_size=1,
            tp_rank=0,
            quant_config=quant_config,
            quantization_aliases=("in_proj_qkv", "in_proj_z"),
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
        # The GEMM sees only rank-local activations; the serialized projections
        # still have full widths and must be split using their physical layout.
        self._gdn_widths = {
            "q": num_k_heads * self.head_k_dim, "k": num_k_heads * self.head_k_dim,
            "v": num_v_heads * self.head_v_dim, "z": num_v_heads * self.head_v_dim,
        }
        for child, projections, dim in (
            (self.in_proj_qkvz, self._gdn_widths, 1),
            (self.out_proj, {"": hidden_size}, 0),
        ):
            declare = getattr(child.quant_method, "configure_checkpoint_shards", None)
            if declare is not None:
                declare(child, projections, dim, attn_tp_size, attn_tp_rank)

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
        local, offset = self._split_projection_component(component, parameter, tensor)
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

    def _split_projection_component(self, component, parameter, tensor):
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

        return self._split_qkv_or_z_rows(tensor, component, unit)

    def _load_split_ba(self, component: str, tensor: torch.Tensor) -> None:
        if self._ba_layout is not None and self._ba_layout != "split":
            raise RuntimeError("Cannot mix fused and split checkpoint layouts for B/A")
        self._ba_layout = "split"
        if component in self._ba_split_parts:
            raise RuntimeError(f"Duplicate split B/A tensor in_proj_{component}.weight")
        local, offset = self._split_ba_component(component, tensor)
        copy_weight_(
            self.in_proj_ba_w.data.narrow(1, offset, self.local_num_v_heads),
            local,
            f"in_proj_{component}.weight",
        )
        self._ba_split_parts.add(component)
        if self._ba_split_parts == {"a", "b"}:
            self._mark_weight_loaded("in_proj_ba_w")

    def _split_ba_component(self, component, tensor):
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
        return local, offset

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

    def _load_grouped_input(self, component, parameter, value):
        """Turn global HF QKV/Z into the Q,K,V,Z streams of the physical GEMM."""
        child = self.in_proj_qkvz
        names = tuple(component)
        if parameter == "g_idx":
            # Each input projection addresses the same K axis. Let QuantMethod
            # validate the original identity map separately for each projection.
            pieces = [value] * len(names)
        else:
            source = child._groupwise_checkpoint_layout
            if parameter not in ("qweight", "qzeros", "scales", "bias"):
                raise ValueError(f"Unsupported grouped GDN input parameter {parameter}")
            packed_n = parameter == "qzeros" or (parameter == "qweight" and source.method == "awq")
            pack = source.pack_factor if packed_n else 1
            widths = [self._gdn_widths[n] // pack for n in names]
            if value.ndim != (1 if parameter == "bias" else 2) or value.shape[-1] != sum(widths):
                raise ValueError(f"Invalid grouped GDN {component}.{parameter} shape: {tuple(value.shape)}")
            if component == "qkvz":
                # Fused Next exports group Q,K,V,Z by key head. Restrict the
                # head interval before any contiguous copy; never unpack codes.
                heads = self.linear_attn_config.linear_num_key_heads
                if (source.method == "awq" and self.head_k_dim % 8) or self.head_k_dim % pack:
                    raise ValueError("GDN head boundary splits a quantizer packing group")
                if parameter == "bias" and torch.count_nonzero(value).item():
                    raise ValueError("GPTQ checkpoint bias for GDN must be all zero")
                grouped = value.unflatten(-1, (heads, sum(widths) // heads))
                grouped = grouped.narrow(-2, self.attn_tp_rank * self.local_num_k_heads, self.local_num_k_heads)
                blocks = grouped.split([w // heads for w in widths], dim=-1)
                pieces = []
                for name, width, block in zip(names, widths, blocks):
                    shape = (*value.shape[:-1], width)
                    key = f"{name}.{parameter}"
                    spec = child.quant_method.checkpoint_read_spec(child, key, shape)
                    # Validation-only GPTQ bias requires the full source buffer;
                    # it has already been checked above across all heads.
                    if parameter == "bias":
                        pieces.append(value[:width])
                    else:
                        pieces.append(spec.wrap_sharded_tensor(block.flatten(-2)))
            else:
                # Qwen3.5 exports Q,K,V contiguously and Z separately. Delegate
                # TP selection, scale checks and completeness to the common leaf.
                pieces = value.split(widths, dim=-1)
        self._claim_layout(self._qkvz_layouts, parameter, "fused" if component == "qkvz" else "split")
        self._load_child_weights(child, {
            f"{name}.{parameter}": piece for name, piece in zip(names, pieces)
        })

    def _split_projection_update(self, module, incoming):
        from rtp_llm.models_py.quant_methods.unquantized import UnquantizedLinearMethod

        fields = {}
        dense = type(module.quant_method) is UnquantizedLinearMethod
        for key, value in incoming.items():
            source, _, field = key.partition(".")
            if source not in ("in_proj_qkv", "in_proj_z"):
                raise ValueError("Cannot update split and fused QKVZ in one request")
            if dense and (field != "weight" or value.dtype != module.weight.dtype):
                raise ValueError(f"Expected dense GDN weight with dtype {module.weight.dtype}")
            component = source.removeprefix("in_proj_")
            fields.setdefault(field, {})[component] = self._split_projection_component(
                component, field, value
            )
        local = {}
        for field, parts in fields.items():
            if parts.keys() == {"qkv", "z"}:
                qkv, z = parts["qkv"][0], parts["z"][0]
                if qkv.dtype != z.dtype:
                    raise TypeError("QKV/Z components must have the same dtype")
                local[field] = torch.cat((qkv, z), dim=0)
            elif dense:
                # Preserve the untouched component in logical checkpoint layout;
                # the leaf re-encodes it if the live weight is swizzled.
                full = module.weight.detach().to(device="cpu", copy=True)
                if getattr(module, "_dense_swizzled", False):
                    from rtp_llm.models_py.quant_methods.dense_layout import decode_dense_weight
                    full = decode_dense_weight(full)
                part, offset = next(iter(parts.values()))
                full.narrow(0, offset, part.shape[0]).copy_(part)
                local[field] = full
            else:
                raise ValueError("Quantized GDN updates require complete QKV and Z components")
        return module.prepare_weight_updates(local)

    def prepare_child_weight_updates(self, module, incoming):
        if module is self.in_proj_qkvz and any(k.startswith("in_proj_") for k in incoming):
            return self._split_projection_update(module, incoming)
        local = {
            name: self._projection_tensor(module, name, tensor)
            for name, tensor in incoming.items()
        }
        return module.prepare_weight_updates(local)

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        for name, tensor in weights.items():
            projection, _, parameter = name.partition(".")
            if self._grouped_controls is not None and projection in ("in_proj_a", "in_proj_b", "in_proj_ba"):
                tensor = self._grouped_controls.consume(projection.removeprefix("in_proj_"), parameter, tensor)
                if tensor is None:
                    continue
                name = projection + ".weight"
            if projection in ("in_proj_qkvz", "in_proj_qkv", "in_proj_z") and hasattr(
                self.in_proj_qkvz.quant_method, "configure_checkpoint_shards"
            ):
                self._load_grouped_input(projection.removeprefix("in_proj_"), parameter, tensor)
                continue
            if projection == "out_proj" and hasattr(self.out_proj.quant_method, "configure_checkpoint_shards"):
                self._load_child_weights(self.out_proj, {parameter: tensor})
                continue
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
            elif name in self._state_parameters:
                if name == "in_proj_ba.weight":
                    if self._ba_layout is not None and self._ba_layout != "fused":
                        raise RuntimeError(
                            "Cannot mix split and fused checkpoint layouts for B/A"
                        )
                    self._ba_layout = "fused"
                if not self._assign_weight(
                    self, self._state_parameters[name], self._local_state_tensor(name, tensor)
                ):
                    raise RuntimeError(f"Failed to load {name}")
            elif name == "in_proj_b.weight":
                self._load_split_ba("b", tensor)
            elif name == "in_proj_a.weight":
                self._load_split_ba("a", tensor)
            else:
                raise RuntimeError(f"Unsupported Qwen3-Next linear tensor {name!r}")
