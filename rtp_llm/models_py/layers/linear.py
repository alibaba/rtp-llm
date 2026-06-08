from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.quant_methods.base import QuantizationConfig, QuantizeMethodBase


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        bias: bool = False,
        params_dtype: torch.dtype = torch.float16,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.prefix = prefix
        self.params_dtype = params_dtype

        if quant_config is None:
            quant_config = QuantizationConfig(quant_type="none")
        self.quant_config = quant_config
        self.quant_method = quant_config.get_quant_method(self, prefix)

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(output_size, dtype=params_dtype), requires_grad=False
            )
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.quant_method.apply(self, x, self.bias)
        return output


class ColumnParallelLinear(LinearBase):

    shard_names: List[str] = []

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        bias: bool = False,
        gather_output: bool = False,
        params_dtype: torch.dtype = torch.float16,
        **kwargs,
    ):
        self.output_size_per_partition = output_size // tp_size
        self.gather_output = gather_output

        super().__init__(
            input_size=input_size,
            output_size=self.output_size_per_partition,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=prefix,
            bias=bias,
            params_dtype=params_dtype,
            **kwargs,
        )

        self.quant_method.create_weights(
            layer=self,
            input_size=input_size,
            output_size=self.output_size_per_partition,
            params_dtype=params_dtype,
        )

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        for name, tensor in weights.items():
            param_name = self._get_param_name(name)
            if param_name is None:
                continue

            param = getattr(self, param_name, None)
            if param is None or not isinstance(param, nn.Parameter):
                continue

            if param_name == "weight":
                tensor = self._split_weight(tensor, dim=0)
            elif param_name == "bias":
                tensor = self._split_weight(tensor, dim=0)
            elif param_name in ("weight_scale", "input_scale"):
                if (
                    tensor.numel() > 1
                    and tensor.shape[0] == self.output_size_per_partition * self.tp_size
                ):
                    tensor = self._split_weight(tensor, dim=0)

            if tensor.shape != param.shape:
                raise ValueError(
                    f"Shape mismatch for {self.prefix}.{param_name}: "
                    f"weight {tensor.shape} vs param {param.shape}"
                )
            param.data.copy_(tensor)

        self.quant_method.process_weights_after_loading(self)

    def _split_weight(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        if self.tp_size <= 1:
            return tensor
        size_per_partition = tensor.shape[dim] // self.tp_size
        start = self.tp_rank * size_per_partition
        return tensor.narrow(dim, start, size_per_partition).contiguous()

    def _get_param_name(self, weight_name: str) -> Optional[str]:
        parts = weight_name.rsplit(".", 1)
        name = parts[-1] if len(parts) > 1 else weight_name
        return name


class RowParallelLinear(LinearBase):

    shard_names: List[str] = []

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        bias: bool = False,
        reduce_output: bool = True,
        params_dtype: torch.dtype = torch.float16,
        **kwargs,
    ):
        self.input_size_per_partition = input_size // tp_size
        self.reduce_output = reduce_output

        super().__init__(
            input_size=self.input_size_per_partition,
            output_size=output_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=prefix,
            bias=bias,
            params_dtype=params_dtype,
            **kwargs,
        )

        self.quant_method.create_weights(
            layer=self,
            input_size=self.input_size_per_partition,
            output_size=output_size,
            params_dtype=params_dtype,
        )

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        for name, tensor in weights.items():
            param_name = self._get_param_name(name)
            if param_name is None:
                continue

            param = getattr(self, param_name, None)
            if param is None or not isinstance(param, nn.Parameter):
                continue

            if param_name == "weight":
                tensor = self._split_weight(tensor, dim=1)
            elif param_name in ("weight_scale", "input_scale"):
                if (
                    tensor.numel() > 1
                    and tensor.shape[-1] == self.input_size_per_partition * self.tp_size
                ):
                    tensor = self._split_weight(tensor, dim=-1)

            if tensor.shape != param.shape:
                raise ValueError(
                    f"Shape mismatch for {self.prefix}.{param_name}: "
                    f"weight {tensor.shape} vs param {param.shape}"
                )
            param.data.copy_(tensor)

        self.quant_method.process_weights_after_loading(self)

    def _split_weight(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        if self.tp_size <= 1:
            return tensor
        size_per_partition = tensor.shape[dim] // self.tp_size
        start = self.tp_rank * size_per_partition
        return tensor.narrow(dim, start, size_per_partition).contiguous()

    def _get_param_name(self, weight_name: str) -> Optional[str]:
        parts = weight_name.rsplit(".", 1)
        return parts[-1] if len(parts) > 1 else weight_name


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        bias: bool = False,
        shard_names: Optional[List[str]] = None,
        params_dtype: torch.dtype = torch.float16,
        **kwargs,
    ):
        if shard_names:
            self.shard_names = shard_names

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=prefix,
            bias=bias,
            params_dtype=params_dtype,
            **kwargs,
        )

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        num_shards = len(self.shard_names) if self.shard_names else 1
        shard_size = self.output_size_per_partition // num_shards

        # Collect per-tensor (scalar) shard values to apply max-merge after the
        # main loop — needed for already-quantized FP8 per-tensor ckpts where
        # gate_proj/up_proj have separate scalar weight_scale.
        per_tensor_collect: Dict[str, List[float]] = {}

        for full_name, tensor in weights.items():
            param_name = self._get_param_name(full_name)
            shard_id = self._get_shard_id(full_name)

            if param_name != "weight" and param_name != "bias":
                param = getattr(self, param_name, None)
                if param is None or not isinstance(param, nn.Parameter):
                    continue

                if shard_id < 0:
                    # Un-sharded param (e.g. already-merged ckpt) — full copy.
                    param.data.copy_(tensor)
                    continue

                # Per-shard non-weight param (e.g. per-channel weight_scale).
                if param.numel() == 1:
                    # Per-tensor scalar: collect, will pick max across shards.
                    per_tensor_collect.setdefault(param_name, []).append(
                        float(tensor.flatten()[0].item())
                    )
                    continue

                # Per-channel/per-shard tensor: TP-slice + offset copy along dim 0.
                # Convention: ckpt tensor's dim-0 covers one shard's full output
                # (= shard_size * tp_size).
                if (
                    self.tp_size > 1
                    and tensor.shape[0] == shard_size * self.tp_size
                ):
                    start = self.tp_rank * shard_size
                    tensor = tensor.narrow(0, start, shard_size).contiguous()
                offset = shard_id * shard_size
                if tensor.shape[0] != shard_size:
                    raise ValueError(
                        f"Shape mismatch for merged {self.prefix}.{param_name} "
                        f"shard={shard_id}: got dim-0={tensor.shape[0]}, "
                        f"expected {shard_size}"
                    )
                param.data[offset : offset + shard_size].copy_(tensor)
                continue

            if shard_id < 0:
                if param_name == "weight":
                    split_tensor = self._split_weight(tensor, dim=0)
                    self.weight.data.copy_(split_tensor)
                continue

            if self.tp_size > 1:
                shard_output_size = tensor.shape[0] // self.tp_size
                start = self.tp_rank * shard_output_size
                tensor = tensor.narrow(0, start, shard_output_size).contiguous()

            if param_name == "weight":
                offset = shard_id * shard_size
                self.weight.data[offset : offset + shard_size, :].copy_(tensor)
            elif param_name == "bias" and self.bias is not None:
                offset = shard_id * shard_size
                self.bias.data[offset : offset + shard_size].copy_(tensor)

        # Resolve per-tensor scalar params via max-merge.
        # TODO(phase-5): rescale weight shards so they all share max_val to
        # avoid the small accuracy loss from picking max without rescaling.
        for param_name, vals in per_tensor_collect.items():
            param = getattr(self, param_name, None)
            if param is None:
                continue
            if len(set(round(v, 6) for v in vals)) > 1:
                import logging
                logging.warning(
                    "[MergedColumn %s] %s shards have differing scalars %s; "
                    "using max — small accuracy loss until phase-5 rescaling.",
                    self.prefix, param_name, vals,
                )
            param.data.fill_(max(vals))

        self.quant_method.process_weights_after_loading(self)

    def _get_shard_id(self, weight_name: str) -> int:
        for idx, shard_name in enumerate(self.shard_names):
            if shard_name in weight_name:
                return idx
        return -1


class QKVParallelLinear(ColumnParallelLinear):

    shard_names: List[str] = ["q_proj", "k_proj", "v_proj"]

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        bias: bool = False,
        params_dtype: torch.dtype = torch.float16,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.num_heads_per_partition = num_heads // tp_size
        self.num_kv_heads_per_partition = max(1, num_kv_heads // tp_size)

        self.q_size = self.num_heads_per_partition * head_dim
        self.kv_size = self.num_kv_heads_per_partition * head_dim

        total_output = self.q_size + 2 * self.kv_size

        LinearBase.__init__(
            self,
            input_size=hidden_size,
            output_size=total_output,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=prefix,
            bias=bias,
            params_dtype=params_dtype,
            **kwargs,
        )
        self.output_size_per_partition = total_output
        self.gather_output = False

        self.quant_method.create_weights(
            layer=self,
            input_size=hidden_size,
            output_size=total_output,
            params_dtype=params_dtype,
        )

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        q_weight = k_weight = v_weight = None
        q_bias = k_bias = v_bias = None
        # Collect per-q/k/v tensors for non-weight/bias params (e.g.
        # weight_scale per channel). Keyed by param_name.
        qkv_extra: Dict[str, Dict[str, torch.Tensor]] = {}
        other_weights: Dict[str, torch.Tensor] = {}

        for full_name, tensor in weights.items():
            param_name = self._get_param_name(full_name)
            qkv_key = None
            if "q_proj" in full_name:
                qkv_key = "q"
            elif "k_proj" in full_name:
                qkv_key = "k"
            elif "v_proj" in full_name:
                qkv_key = "v"

            if qkv_key is not None:
                if param_name == "weight":
                    if qkv_key == "q":
                        q_weight = tensor
                    elif qkv_key == "k":
                        k_weight = tensor
                    else:
                        v_weight = tensor
                elif param_name == "bias":
                    if qkv_key == "q":
                        q_bias = tensor
                    elif qkv_key == "k":
                        k_bias = tensor
                    else:
                        v_bias = tensor
                else:
                    qkv_extra.setdefault(param_name, {})[qkv_key] = tensor
            else:
                other_weights[full_name] = tensor

        if q_weight is not None and k_weight is not None and v_weight is not None:
            q_split = self._split_qkv(q_weight, self.num_heads, self.head_dim)
            k_split = self._split_qkv(k_weight, self.num_kv_heads, self.head_dim)
            v_split = self._split_qkv(v_weight, self.num_kv_heads, self.head_dim)

            combined = torch.cat([q_split, k_split, v_split], dim=0)
            self.weight.data.copy_(combined)
        elif other_weights:
            for full_name, tensor in other_weights.items():
                param_name = self._get_param_name(full_name)
                if param_name == "weight":
                    split_tensor = self._split_weight(tensor, dim=0)
                    self.weight.data.copy_(split_tensor)
                elif param_name in ("weight_scale", "input_scale"):
                    param = getattr(self, param_name, None)
                    if param is not None:
                        param.data.copy_(tensor)

        # Merge q/k/v extra tensors (per-channel weight_scale, etc.) the same
        # way as weights: TP-slice along the head/kv-head dim, then cat in qkv
        # order.
        for param_name, qkv_tensors in qkv_extra.items():
            param = getattr(self, param_name, None)
            if param is None or not isinstance(param, nn.Parameter):
                continue
            if not all(k in qkv_tensors for k in ("q", "k", "v")):
                # Partial — fall back to first tensor's full copy (legacy).
                continue

            if param.numel() == 1:
                # Per-tensor scalar across q/k/v: max-merge.
                vals = [
                    float(qkv_tensors[k].flatten()[0].item()) for k in ("q", "k", "v")
                ]
                if len(set(round(v, 6) for v in vals)) > 1:
                    import logging
                    logging.warning(
                        "[QKV %s] %s q/k/v have differing scalars %s; "
                        "using max — small accuracy loss until phase-5 rescaling.",
                        self.prefix, param_name, vals,
                    )
                param.data.fill_(max(vals))
            else:
                # Per-channel: TP-slice each piece along output dim, then cat.
                q_t = self._split_qkv(qkv_tensors["q"], self.num_heads, self.head_dim)
                k_t = self._split_qkv(
                    qkv_tensors["k"], self.num_kv_heads, self.head_dim
                )
                v_t = self._split_qkv(
                    qkv_tensors["v"], self.num_kv_heads, self.head_dim
                )
                cat = torch.cat([q_t, k_t, v_t], dim=0)
                if cat.shape != param.shape:
                    raise ValueError(
                        f"Shape mismatch for merged QKV {self.prefix}.{param_name}: "
                        f"got {cat.shape}, expected {param.shape}"
                    )
                param.data.copy_(cat)

        if self.bias is not None:
            present = {
                "q": q_bias is not None,
                "k": k_bias is not None,
                "v": v_bias is not None,
            }
            if all(present.values()):
                q_b_split = self._split_qkv(q_bias, self.num_heads, self.head_dim)
                k_b_split = self._split_qkv(k_bias, self.num_kv_heads, self.head_dim)
                v_b_split = self._split_qkv(v_bias, self.num_kv_heads, self.head_dim)
                combined_bias = torch.cat([q_b_split, k_b_split, v_b_split], dim=0)
                self.bias.data.copy_(combined_bias)
            elif any(present.values()):
                # Partial bias is almost always a ckpt/mapping issue: e.g.
                # Qwen2 qkv_bias=True but only q_proj.bias was provided. Surface
                # it loudly so it doesn't get diluted into a silent zero bias.
                missing = [k for k, v in present.items() if not v]
                import logging

                logging.warning(
                    "[QKV %s] partial bias loaded — missing %s; bias for those "
                    "projections will stay zero-initialized. Check ckpt or "
                    "WeightsMapper.",
                    self.prefix,
                    missing,
                )

        self.quant_method.process_weights_after_loading(self)

    def _split_qkv(
        self, tensor: torch.Tensor, num_heads: int, head_dim: int
    ) -> torch.Tensor:
        if self.tp_size <= 1:
            return tensor
        heads_per_partition = max(1, num_heads // self.tp_size)
        size_per_partition = heads_per_partition * head_dim
        start = self.tp_rank * size_per_partition
        return tensor.narrow(0, start, size_per_partition).contiguous()
