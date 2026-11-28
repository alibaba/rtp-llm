"""Checkpoint-owned Kimi KDA parameters, independent of the attention scheduler."""
from typing import Any, Dict

import torch
from torch import nn

from rtp_llm.models_py.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from rtp_llm.models_py.module_base import RtpModule, copy_weight_
from rtp_llm.models_py.utils.typed_storage_view import LinearCacheConverter


class _MergedKimiConv1d(RtpModule):
    """Own the rank-local fused Q/K/V convolution kernel weight."""

    shard_names = ("q_conv1d", "k_conv1d", "v_conv1d")

    def __init__(
        self,
        global_channels: int,
        kernel_size: int,
        tp_size: int,
        tp_rank: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        if global_channels % tp_size:
            raise ValueError("KDA convolution channels must be divisible by TP")
        self.global_channels = global_channels
        self.local_channels = global_channels // tp_size
        self.kernel_size = kernel_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.prefix = prefix
        self.weight = nn.Parameter(
            torch.empty(
                len(self.shard_names) * self.local_channels,
                kernel_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        self._loaded_shards = set()

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        for name, tensor in weights.items() if isinstance(weights, dict) else weights:
            projection, separator, suffix = name.partition(".")
            if (
                not separator
                or suffix != "weight"
                or projection not in self.shard_names
            ):
                raise RuntimeError(f"Unsupported Kimi convolution tensor {name!r}")
            if projection in self._loaded_shards:
                raise RuntimeError(f"Duplicate Kimi convolution tensor {name!r}")
            if tensor.dim() == 3:
                if tensor.shape[1] != 1:
                    raise ValueError(
                        f"{name} middle dimension must be 1, got {tuple(tensor.shape)}"
                    )
                tensor = tensor.squeeze(1)
            expected = (self.global_channels, self.kernel_size)
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, got {tuple(tensor.shape)}"
                )
            source_offset = self.tp_rank * self.local_channels
            local = tensor.narrow(0, source_offset, self.local_channels).contiguous()
            shard_index = self.shard_names.index(projection)
            target_offset = shard_index * self.local_channels
            copy_weight_(
                self.weight.data.narrow(0, target_offset, self.local_channels),
                local,
                f"{self.prefix}.{name}",
            )
            self._loaded_shards.add(projection)
        if self._loaded_shards == set(self.shard_names):
            self._mark_weight_loaded("weight")


class KimiKDAParameters(RtpModule):
    """Kimi Delta Attention with direct checkpoint-owned NewLoader modules."""

    def __init__(self, cfg: Dict[str, Any], prefix: str) -> None:
        super().__init__()
        hidden_size = cfg["hidden_size"]
        head_dim = cfg["kda_head_dim"]
        num_heads = cfg["kda_num_heads"]
        tp_size = cfg["attn_tp_size"]
        tp_rank = cfg["attn_tp_rank"]
        params_dtype = cfg["params_dtype"]
        quant_config = cfg["quant_config"]
        self.head_dim = head_dim
        self.local_num_heads = num_heads // tp_size
        self.qkv_size = 3 * self.local_num_heads * head_dim
        self.ssm_state_dtype = cfg["ssm_state_dtype"]
        self.conv_state_dtype = cfg["conv_state_dtype"]
        self.conv_kernel = cfg["conv_kernel"]
        self.cache_converter = LinearCacheConverter(
            local_num_v_heads=self.local_num_heads,
            head_v_dim=self.head_dim,
            head_k_dim=self.head_dim,
            ssm_state_dtype=self.ssm_state_dtype,
            linear_conv_kernel_dim=self.conv_kernel,
            qkv_size=self.qkv_size,
            conv_state_dtype=self.conv_state_dtype,
        )

        self.qkv_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_size=3 * num_heads * head_dim,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            bias=False,
            shard_names=["q_proj", "k_proj", "v_proj"],
            params_dtype=params_dtype,
        )
        self.b_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=num_heads,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.b_proj",
            bias=False,
            params_dtype=params_dtype,
        )
        self.f_a_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=head_dim,
            tp_size=1,
            tp_rank=0,
            quant_config=quant_config,
            prefix=f"{prefix}.f_a_proj",
            bias=False,
            params_dtype=params_dtype,
        )
        self.f_b_proj = ColumnParallelLinear(
            input_size=head_dim,
            output_size=num_heads * head_dim,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.f_b_proj",
            bias=False,
            params_dtype=params_dtype,
        )
        self.g_a_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=head_dim,
            tp_size=1,
            tp_rank=0,
            quant_config=quant_config,
            prefix=f"{prefix}.g_a_proj",
            bias=False,
            params_dtype=params_dtype,
        )
        self.g_b_proj = ColumnParallelLinear(
            input_size=head_dim,
            output_size=num_heads * head_dim,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.g_b_proj",
            bias=False,
            params_dtype=params_dtype,
        )
        self.conv1d = _MergedKimiConv1d(
            global_channels=num_heads * head_dim,
            kernel_size=self.conv_kernel,
            tp_size=tp_size,
            tp_rank=tp_rank,
            params_dtype=params_dtype,
            prefix=prefix,
        )
        self.dt_bias = nn.Parameter(
            torch.empty(self.local_num_heads * head_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.A_log = nn.Parameter(
            torch.empty(self.local_num_heads, dtype=torch.float32),
            requires_grad=False,
        )
        from rtp_llm.models_py.triton_kernels.common.layernorm_gated import RmsNormGated

        self.o_norm = RmsNormGated(
            nn.Parameter(
                torch.empty(head_dim, dtype=params_dtype), requires_grad=False
            ),
            eps=cfg["rms_norm_eps"],
            group_size=head_dim,
            activation="sigmoid",
        )
        self.o_proj = RowParallelLinear(
            input_size=num_heads * head_dim,
            output_size=hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            bias=False,
            reduce_output=True,
            params_dtype=params_dtype,
        )

    def _split_dt_bias(self, tensor: torch.Tensor) -> torch.Tensor:
        global_heads = self.local_num_heads * self.qkv_proj.tp_size
        expected = global_heads * self.head_dim
        if tensor.numel() != expected:
            raise ValueError(
                f"dt_bias must contain {expected} values, got {tensor.numel()}"
            )
        view = tensor.reshape(global_heads, self.head_dim)
        start = self.qkv_proj.tp_rank * self.local_num_heads
        return view.narrow(0, start, self.local_num_heads).reshape(-1).contiguous()

    def _apply(self, fn, recurse=True):
        # A_log and dt_bias are FP32 KDA recurrence/gate parameters, even when
        # checkpoint projections move to BF16. Do not round then re-promote.
        scalars = {name: getattr(self, name).detach() for name in ("A_log", "dt_bias")}
        result = super()._apply(fn, recurse)
        for name, source in scalars.items():
            target = getattr(self, name)
            if target.dtype != torch.float32:
                with torch.no_grad():
                    target.data = source.to(device=target.device, dtype=torch.float32)
        return result

    def _split_a_log(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.reshape(-1)
        global_heads = self.local_num_heads * self.qkv_proj.tp_size
        if tuple(tensor.shape) != (global_heads,):
            raise ValueError(
                f"A_log must squeeze to {(global_heads,)}, got {tuple(tensor.shape)}"
            )
        start = self.qkv_proj.tp_rank * self.local_num_heads
        return tensor.narrow(0, start, self.local_num_heads).contiguous()

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        for name, tensor in weights.items() if isinstance(weights, dict) else weights:
            if name == "dt_bias":
                if not self._assign_weight(self, name, self._split_dt_bias(tensor)):
                    raise RuntimeError("Failed to load Kimi dt_bias")
            elif name == "A_log":
                if not self._assign_weight(self, name, self._split_a_log(tensor)):
                    raise RuntimeError("Failed to load Kimi A_log")
            else:
                # Native FST's staging buffer belongs to the iterator and may
                # be released on its next step. Copy before advancing.
                super().load_weights({name: tensor})
