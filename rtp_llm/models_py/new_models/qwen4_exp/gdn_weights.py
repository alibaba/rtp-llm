"""Direct Qwen4 GDN checkpoint ownership, using main runtime for execution.

The loading/reordering protocol is reused from the verified NewLoader GDN
owner. No GDN inference implementation or legacy weight-name dictionary lives
here. CPU construction/loading is supported independently of CUDA execution.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Any, Dict, List, Optional, Set, Tuple
import torch
from torch import nn
from rtp_llm.models_py.module_base import RtpModule, copy_weight_
from rtp_llm.models_py.new_models.qwen4_exp.hyper_connection import Qwen4TensorWeight


def _projection(*, input_size, output_size, quant_config, prefix, params_dtype, fp8_storage=False, **kwargs):
    # BF16 checkpoint ownership does not require distributed/native runtime
    # imports. Quantized projections keep the shared quantization owner.
    if fp8_storage:
        if quant_config is not None:
            raise ValueError("Qwen4 block storage cannot be combined with another linear quant method")
        from rtp_llm.models_py.new_models.qwen4_exp.dense_weights import Qwen4BlockProjection
        layer = Qwen4BlockProjection((output_size, input_size), dtype=params_dtype)
        layer.prefix = prefix
        layer.quant_config = None
        return layer
    if quant_config is None:
        layer = Qwen4TensorWeight((output_size, input_size), dtype=params_dtype)
        layer.prefix = prefix
        layer.quant_config = None
        return layer
    from rtp_llm.models_py.layers.linear import ColumnParallelLinear
    return ColumnParallelLinear(input_size=input_size, output_size=output_size,
        quant_config=quant_config, prefix=prefix, params_dtype=params_dtype, **kwargs)

def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value

def _finite_positive_real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive real")
    return float(value)

def reorder_ba(ts: List[torch.Tensor], la_cfg: Any):
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


def reorder_qkvz(ts: List[torch.Tensor], la_cfg: Any):
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
    la_cfg: Any,
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


class Qwen4GDNWeights(RtpModule):
    """Qwen4 linear-attention checkpoint parameters.

    Weight loading handles:
      - reorder_qkvz: fused per-head Q/K/V/Z rows
      - reorder_ba: fused per-head B/A rows
      - transpose: linear-attention projections into their runtime layouts
    Execution is bound separately to main's GDN runtime.
    """

    def __init__(
        self,
        linear_attn_config: Any,
        hidden_size: int,
        rms_norm_eps: float,
        attn_tp_size: int,
        attn_tp_rank: int,
        params_dtype: torch.dtype,
        quant_config: Optional[Any],
        prefix: str,
        fp8_storage: bool = False,
    ):
        super().__init__()
        self.fp8_storage = fp8_storage
        self._storage_ready = False
        self.register_buffer("in_proj_ba_storage_scale", None)
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
        self.in_proj_qkvz = _projection(
            fp8_storage=fp8_storage,
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
        self.out_proj = _projection(
            fp8_storage=fp8_storage,
            input_size=self.local_num_v_heads * self.head_v_dim,
            output_size=hidden_size,
            tp_size=1,
            tp_rank=0,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
            bias=False,
            params_dtype=params_dtype,
        )
        # Checkpoint permutation and backend storage belong to this GDN, not
        # the rank-local linears. Do not bypass them via the manager's exact
        # leaf fallback until the bound runtime supports a staged refresh.
        for projection in (self.in_proj_qkvz, self.out_proj):
            projection.supports_independent_weight_updates = False
        self._qkvz_layouts: Dict[str, str] = {}
        self._qkvz_split_parts: Dict[str, Set[str]] = {}
        self._ba_layout: Optional[str] = None
        self._ba_split_parts: Set[str] = set()

    def process_weights_after_loading(self):
        if self._storage_ready:
            return
        for projection in (self.in_proj_qkvz, self.out_proj):
            hook = getattr(projection, "process_weights_after_loading", None)
            if hook is not None:
                hook()
        if self.fp8_storage and all(size % 16 == 0 for size in self.in_proj_ba_w.shape):
            values = self.in_proj_ba_w.float()
            scale = values.abs().amax().clamp_min(1e-12).reshape(1) / 448.0
            self.in_proj_ba_w.data = (values / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            self.in_proj_ba_storage_scale = scale
        self._storage_ready = True

    @staticmethod
    def _block_size(layer: RtpModule) -> Tuple[int, int]:
        value = getattr(layer.quant_config, "weight_block_size", [128, 128])
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"Invalid FP8 weight_block_size {value!r}")
        block_n = _positive_int(value[0], "FP8 output block size")
        block_k = _positive_int(value[1], "FP8 input block size")
        return block_n, block_k

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
            if hasattr(self.in_proj_qkvz, "_record_parameter_loaded"):
                self.in_proj_qkvz._record_parameter_loaded(parameter)
            else:
                self.in_proj_qkvz._mark_weight_loaded(parameter)
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

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        for name, tensor in weights.items():
            if name == "in_proj_qkvz.weight":
                self._claim_layout(self._qkvz_layouts, "weight", "fused")
                reordered = reorder_qkvz([tensor], self.linear_attn_config)
                self._load_child_weights(
                    self.in_proj_qkvz, {"weight": self._split_qkvz_rows(reordered, 1)}
                )
            elif name == "in_proj_qkvz.weight_scale_inv":
                self._claim_layout(self._qkvz_layouts, "weight_scale_inv", "fused")
                block_n, _ = self._block_size(self.in_proj_qkvz)
                reordered = reorder_qkvz_scale(tensor, self.linear_attn_config, block_n)
                self._load_child_weights(
                    self.in_proj_qkvz,
                    {"weight_scale_inv": self._split_qkvz_rows(reordered, block_n)},
                )
            elif name == "in_proj_qkvz.weight_scale":
                self._claim_layout(self._qkvz_layouts, "weight_scale", "fused")
                if tensor.numel() == 1:
                    local = tensor
                else:
                    local = self._split_qkvz_rows(
                        reorder_qkvz([tensor], self.linear_attn_config), 1
                    )
                self._load_child_weights(self.in_proj_qkvz, {"weight_scale": local})
            elif name == "in_proj_qkvz.input_scale":
                self._claim_layout(self._qkvz_layouts, "input_scale", "fused")
                self._load_child_weights(self.in_proj_qkvz, {"input_scale": tensor})
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
            elif name == "out_proj.weight":
                self._load_child_weights(
                    self.out_proj, {"weight": self._split_out_weight(tensor)}
                )
            elif name == "out_proj.weight_scale_inv":
                self._load_child_weights(
                    self.out_proj,
                    {"weight_scale_inv": self._split_out_block_scale(tensor)},
                )
            elif name in ("out_proj.weight_scale", "out_proj.input_scale"):
                self._load_child_weights(
                    self.out_proj, {name.rsplit(".", 1)[-1]: tensor}
                )
            else:
                raise RuntimeError(f"Unsupported Qwen3-Next linear tensor {name!r}")
