"""Unified dense MLP implementation supporting multiple activation types."""

from typing import Dict, Optional, Type

import torch
from torch import nn

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.modules.factory.linear.parallel import (
    row_parallel_linear,
    sequence_parallel_column_weight,
    sequence_parallel_row_weight,
)
from rtp_llm.models_py.modules.base import FusedSiluAndMul
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.ops import ActivationType, HWKernelConfig, ParallelismConfig
from rtp_llm.utils.model_weight import W

_ACTIVATION_FUNC_MAP: Dict[ActivationType, Type[nn.Module]] = {
    ActivationType.Swiglu: FusedSiluAndMul,
    ActivationType.Gelu: nn.GELU,
}

_GATED_ACTIVATION_TYPE_LIST = [ActivationType.Swiglu]


class DenseMLPParallelExecutor(nn.Module):
    """Provide gate/up/down projections from raw TP shards."""

    def __init__(
        self,
        weights: Dict[str, torch.Tensor],
        parallelism_config: ParallelismConfig,
        *,
        gated: bool,
    ) -> None:
        super().__init__()
        self.weights = weights
        self.gated = bool(gated)
        self.tp_size = int(parallelism_config.get_ffn_tp_size())
        self.tp_rank = int(parallelism_config.get_ffn_tp_rank())
        self._full_column_weights: dict[str, torch.Tensor] = {}
        self._full_row_weights: dict[str, torch.Tensor] = {}
        self._validate_weights()

    def _validate_weights(self) -> None:
        required = (W.ffn_w1, W.ffn_w3, W.ffn_w2) if self.gated else (
            W.ffn_w3,
            W.ffn_w2,
        )
        missing = [key for key in required if key not in self.weights]
        unsupported = [
            key
            for key in required
            if key in self.weights and not self.weights[key].is_floating_point()
        ]
        if missing:
            raise ValueError(
                "parallel DenseMLP requires separate TP weight shards; "
                f"missing {missing}"
            )
        if unsupported:
            raise ValueError(
                "parallel DenseMLP currently supports floating-point weights "
                f"only; got {unsupported}"
            )

    def _column_weight(
        self,
        weight_name: str,
        cache_key: str,
        sequence_parallel: bool,
    ) -> torch.Tensor:
        return sequence_parallel_column_weight(
            self.weights,
            weight_name,
            self.tp_size,
            self.tp_rank,
            self._full_column_weights,
            cache_key,
            sequence_parallel=sequence_parallel,
        )

    def _sp_active(self, x: torch.Tensor, requested: bool) -> bool:
        return requested and self.tp_size > 1 and x.is_cuda

    def gate(
        self,
        x: torch.Tensor,
        *,
        sequence_parallel: bool,
    ) -> Optional[torch.Tensor]:
        if not self.gated:
            return None
        return torch.matmul(
            x,
            self._column_weight(
                W.ffn_w1,
                "dense_gate",
                self._sp_active(x, sequence_parallel),
            ),
        )

    def up(self, x: torch.Tensor, *, sequence_parallel: bool) -> torch.Tensor:
        return torch.matmul(
            x,
            self._column_weight(
                W.ffn_w3,
                "dense_up",
                self._sp_active(x, sequence_parallel),
            ),
        )

    def down(
        self,
        activated: torch.Tensor,
        *,
        sequence_parallel: bool,
    ) -> torch.Tensor:
        sp_active = self._sp_active(activated, sequence_parallel)
        down_weight = sequence_parallel_row_weight(
            self.weights,
            W.ffn_w2,
            self.tp_size,
            self.tp_rank,
            self._full_row_weights,
            "dense_down",
            sequence_parallel=sp_active,
        )
        if sp_active:
            return torch.matmul(activated, down_weight)
        return row_parallel_linear(activated, down_weight, self.tp_size)


class DenseMLP(nn.Module):
    """Unified dense MLP with injectable gated activations.

    - For SiGLU (Swiglu): Uses gate_up_proj + fused silu_and_mul + down_proj
    - For GELU (Gelu): Uses intermediate_proj + GELU activation + output_proj
    - Model-specific gated activations can choose merged or split gate/up
      projections without reimplementing the MLP and TP output collective.
    """

    def __init__(
        self,
        activation_type: ActivationType,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        quant_config: object,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        *,
        gated_activation: Optional[nn.Module] = None,
        merge_gate_up: bool = True,
        parallel_executor: Optional[DenseMLPParallelExecutor] = None,
    ):
        super().__init__()

        self.activation_type = activation_type
        self.parallelism_config = parallelism_config
        self.parallel_executor = parallel_executor
        if gated_activation is None:
            if self.activation_type not in _ACTIVATION_FUNC_MAP:
                raise ValueError(f"Unsupported activation type: {activation_type}")
            self.act_fn = _ACTIVATION_FUNC_MAP[activation_type]()
            self.is_gated = activation_type in _GATED_ACTIVATION_TYPE_LIST
        else:
            # Some architectures use the standard gated-MLP projection layout
            # with a model-specific gated activation. Keep the projection and
            # TP orchestration here and inject only that mathematical primitive.
            self.act_fn = gated_activation
            self.is_gated = True
        self.merge_gate_up = bool(merge_gate_up)

        if self.parallel_executor is not None:
            if self.parallel_executor.gated != self.is_gated:
                raise ValueError("DenseMLP activation and parallel executor disagree")
            return

        if self.is_gated:
            if not self.merge_gate_up:
                self.gate_proj = LinearFactory.create_linear_from_weights(
                    weights,
                    W.ffn_w1,
                    W.ffn_s1,
                    W.ffn_b1,
                    quant_config=quant_config,
                    hw_kernel_config=hw_kernel_config,
                    weight_scale_2_key=W.ffn_w1_s2,
                    input_scale_key=W.ffn_w1_i_s,
                )
                self.up_proj = LinearFactory.create_linear_from_weights(
                    weights,
                    W.ffn_w3,
                    W.ffn_s3,
                    W.ffn_b3,
                    quant_config=quant_config,
                    hw_kernel_config=hw_kernel_config,
                    weight_scale_2_key=W.ffn_w3_s2,
                    input_scale_key=W.ffn_w3_i_s,
                )
            elif W.ffn_w13 not in weights:
                self.up_proj = LinearFactory.create_merged_linear(
                    weights,
                    weight_keys=[W.ffn_w1, W.ffn_w3],
                    scale_keys=[W.ffn_s1, W.ffn_s3],
                    bias_keys=[W.ffn_b1, W.ffn_b3],
                    quant_config=quant_config,
                    dim=-1,
                    hw_kernel_config=hw_kernel_config,
                    scale2_keys=[W.ffn_w1_s2, W.ffn_w3_s2],
                    input_scale_keys=[W.ffn_w1_i_s, W.ffn_w3_i_s],
                )
            else:
                self.up_proj = LinearFactory.create_linear_from_weights(
                    weights, W.ffn_w13, W.ffn_s13, W.ffn_b13,
                    quant_config=quant_config, hw_kernel_config=hw_kernel_config,
                    weight_scale_2_key=W.ffn_w13_s2,
                    input_scale_key=W.ffn_w13_i_s,
                )

        else:
            self.up_proj = LinearFactory.create_linear_from_weights(
                weights, W.ffn_w3, W.ffn_s3, W.ffn_b3,
                quant_config=quant_config, hw_kernel_config=hw_kernel_config,
                weight_scale_2_key=W.ffn_w3_s2,
                input_scale_key=W.ffn_w3_i_s,
            )

        self.down_proj = LinearFactory.create_linear_from_weights(
            weights, W.ffn_w2, W.ffn_s2, W.ffn_b2,
            quant_config=quant_config, hw_kernel_config=hw_kernel_config,
            weight_scale_2_key=W.ffn_w2_s2,
            input_scale_key=W.ffn_w2_i_s
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        sequence_parallel: bool = False,
        valid_token_count: Optional[int] = None,
    ) -> torch.Tensor:
        del valid_token_count
        if self.parallel_executor is not None:
            gate = self.parallel_executor.gate(
                x, sequence_parallel=sequence_parallel
            )
            up = self.parallel_executor.up(x, sequence_parallel=sequence_parallel)
            return self.parallel_executor.down(
                self._activate(gate, up),
                sequence_parallel=sequence_parallel,
            )
        if sequence_parallel:
            raise RuntimeError("DenseMLP has no sequence-parallel executor")

        gate = self.gate_proj(x) if self.is_gated and not self.merge_gate_up else None
        output = self.down_proj(self._activate(gate, self.up_proj(x)))
        if self.parallelism_config.get_ffn_tp_size() > 1:
            output = all_reduce(output, group=Group.TP)
        return output

    def _activate(
        self,
        gate: Optional[torch.Tensor],
        up: torch.Tensor,
    ) -> torch.Tensor:
        if gate is None:
            return self.act_fn(up)
        if self.merge_gate_up:
            return self.act_fn(torch.cat((gate, up), dim=-1))
        return self.act_fn(gate, up)
