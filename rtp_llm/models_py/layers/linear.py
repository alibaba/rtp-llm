import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather, all_reduce
from rtp_llm.models_py.quant_methods.base import QuantizationConfig, QuantizeMethodBase


def _weight_name_segments(weight_name: str, prefix: str) -> List[str]:
    segments = [segment for segment in weight_name.split(".") if segment]
    prefix_segments = [segment for segment in prefix.split(".") if segment]
    if prefix_segments and segments[: len(prefix_segments)] == prefix_segments:
        return segments[len(prefix_segments) :]
    return segments


def _find_shard_segment(weight_name: str, prefix: str, shard_names: List[str]) -> int:
    segments = _weight_name_segments(weight_name, prefix)
    for idx, shard_name in enumerate(shard_names):
        if shard_name in segments:
            return idx
    return -1


class LinearBase(nn.Module):

    # FP8 per-block (DeepSeek-style) quantization block size, used when
    # TP-slicing / shard-merging `weight_scale_inv` block grids.
    _FP8_BLOCK = 128

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
        if tp_size <= 0:
            raise ValueError(f"tp_size must be positive, got {tp_size}")
        if tp_rank < 0 or tp_rank >= tp_size:
            raise ValueError(
                f"tp_rank must satisfy 0 <= tp_rank < tp_size, got "
                f"tp_rank={tp_rank}, tp_size={tp_size}, prefix={prefix!r}"
            )
        self.input_size = input_size
        self.output_size = output_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.prefix = prefix
        self.params_dtype = params_dtype
        self._loaded_weight_keys = set()

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

    def process_weights_after_loading(self):
        # Called once by NewModelLoader._run_post_load_hooks after model.to(device)
        # and after every weight shard is in place. Doing the work here (rather
        # than at the end of load_weights) is required for streaming dispatch:
        # MergedColumnParallelLinear / QKVParallelLinear receive their shards on
        # separate RtpModule.load_weights ticks, so a per-tick call would quantize
        # before all shards arrived (corrupt scale) and then re-quantize an
        # already-fp8 weight on the next tick (kernel dispatch failure).
        if getattr(self, "_post_load_done", False):
            return
        self._check_load_complete()
        self.quant_method.process_weights_after_loading(self)
        self._post_load_done = True

    def _mark_loaded(self, key):
        self._loaded_weight_keys.add(key)

    def _required_load_keys(self):
        keys = set()
        # Online quant methods create runtime scales during post-load; already
        # quantized methods must receive their ckpt scales explicitly.
        quant_type = getattr(self.quant_config, "quant_type", "none")
        online = quant_type in (
            "fp8_online",
            "fp8_per_channel_online",
            "fp8_block_online",
        )
        if isinstance(getattr(self, "weight", None), nn.Parameter):
            keys.add("weight")
        if isinstance(getattr(self, "bias", None), nn.Parameter):
            keys.add("bias")
        for name in ("qweight", "qzeros", "scales"):
            if isinstance(getattr(self, name, None), nn.Parameter):
                keys.add(name)
        if not online and isinstance(getattr(self, "weight_scale", None), nn.Parameter):
            keys.add("weight_scale")
        if isinstance(getattr(self, "weight_scale_inv", None), nn.Parameter):
            keys.add("weight_scale_inv")
        return keys

    def _check_load_complete(self):
        required = self._required_load_keys()
        missing = sorted(required - self._loaded_weight_keys)
        if missing:
            raise RuntimeError(
                f"{type(self).__name__} {self.prefix!r} missing required "
                f"checkpoint tensors: {missing}; loaded={sorted(self._loaded_weight_keys)}"
            )

    def _fp8_scale_block_size(self):
        block_size = getattr(self.quant_config, "weight_block_size", None)
        if block_size is None:
            block_size = [self._FP8_BLOCK, self._FP8_BLOCK]
        if len(block_size) != 2:
            raise ValueError(f"weight_block_size must have 2 values, got {block_size}")
        block_n, block_k = int(block_size[0]), int(block_size[1])
        if block_n <= 0 or block_k <= 0:
            raise ValueError(
                f"weight_block_size values must be positive, got {block_size}"
            )
        return [block_n, block_k]

    @staticmethod
    def _check_fp8_block_aligned(start: int, size: int, block: int, what: str):
        if start % block != 0 or size % block != 0:
            raise ValueError(
                f"{what} requires block-aligned shard boundaries, got "
                f"start={start}, size={size}, block={block}. Non-aligned "
                f"FP8 block scales would overlap checkpoint blocks and cannot "
                f"be copied safely without requantization."
            )

    @staticmethod
    def _ceil_div(x: int, y: int) -> int:
        return (x + y - 1) // y

    @staticmethod
    def _check_divisible(value: int, divisor: int, what: str, prefix: str):
        if divisor <= 0:
            raise ValueError(f"{what} requires positive tp_size, got {divisor}")
        if value % divisor != 0:
            raise ValueError(
                f"{what} ({value}) must be divisible by tp_size ({divisor}) "
                f"for prefix={prefix!r}; non-divisible TP sharding would drop weights"
            )


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
        self._check_divisible(output_size, tp_size, "ColumnParallelLinear output_size", prefix)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = super().forward(x)
        if self.gather_output and self.tp_size > 1:
            original_shape = output.shape
            output_2d = output.reshape(-1, original_shape[-1])
            m, n = output_2d.shape
            gathered = all_gather(output_2d, group=Group.TP)
            output = (
                gathered.reshape(self.tp_size, m, n)
                .transpose(0, 1)
                .contiguous()
                .reshape(*original_shape[:-1], n * self.tp_size)
            )
        return output

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
            elif param_name == "weight_scale_inv":
                if self.tp_size > 1:
                    block_n, _ = self._fp8_scale_block_size()
                    start = self.tp_rank * self.output_size_per_partition
                    self._check_fp8_block_aligned(
                        start,
                        self.output_size_per_partition,
                        block_n,
                        f"FP8 block ColumnParallel scale {self.prefix}",
                    )
                    tensor = self._split_weight(tensor, dim=0)
            elif param_name in ("scales", "zeros"):
                if (
                    tensor.dim() == 1
                    and tensor.shape[0] == self.output_size_per_partition * self.tp_size
                ):
                    if self.tp_size > 1:
                        tensor = self._split_weight(tensor, dim=0)
                elif (
                    tensor.dim() > 1
                    and tensor.shape[-1]
                    == self.output_size_per_partition * self.tp_size
                ):
                    if self.tp_size > 1:
                        tensor = self._split_weight(tensor, dim=-1)
            elif param_name in ("qweight", "qzeros"):
                # AWQ(w4a16):qweight[in, out//8] / qzeros[g, out//8] / scales[g, out]
                # —— 输出维都是 dim1,ColumnParallel 切输出 → 切 dim1。
                if self.tp_size > 1:
                    tensor = self._split_weight(tensor, dim=1)

            if tensor.shape != param.shape:
                if tensor.dim() == 1 and param.dim() == 2 and tensor.numel() == param.numel():
                    tensor = tensor.view_as(param)
                elif (
                    param_name == "weight"
                    and tensor.dim() == 2
                    and tensor.shape[0] != tensor.shape[1]
                    and tensor.t().shape == param.shape
                ):
                    tensor = tensor.t().contiguous()
                else:
                    raise ValueError(
                        f"Shape mismatch for {self.prefix}.{param_name}: "
                        f"weight {tensor.shape} vs param {param.shape}"
                    )
            param.data.copy_(tensor)
            self._mark_loaded(param_name)

        # process_weights_after_loading is invoked by NewModelLoader's
        # post-load hook after every shard has landed; see LinearBase.

    def _split_weight(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        if self.tp_size <= 1:
            return tensor
        self._check_divisible(
            tensor.shape[dim],
            self.tp_size,
            f"{type(self).__name__} tensor dim {dim}",
            self.prefix,
        )
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
        self._check_divisible(input_size, tp_size, "RowParallelLinear input_size", prefix)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reduce_output and self.tp_size > 1:
            output = self.quant_method.apply(self, x, bias=None)
            output = all_reduce(output, group=Group.TP)
            if self.bias is not None:
                output = output + self.bias
            return output
        return super().forward(x)

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
            elif param_name == "weight_scale_inv":
                if self.tp_size > 1:
                    _, block_k = self._fp8_scale_block_size()
                    start = self.tp_rank * self.input_size_per_partition
                    self._check_fp8_block_aligned(
                        start,
                        self.input_size_per_partition,
                        block_k,
                        f"FP8 block RowParallel scale {self.prefix}",
                    )
                    tensor = self._split_weight(tensor, dim=1)
            elif param_name == "g_idx":
                if self.tp_size > 1:
                    tensor = self._split_weight(tensor, dim=0)
            elif param_name in ("scales", "zeros"):
                if tensor.dim() > 1 and tensor.shape[0] > 1:
                    if self.tp_size > 1:
                        tensor = self._split_weight(tensor, dim=0)
            elif param_name in ("qweight", "qzeros"):
                # AWQ(w4a16):qweight[in, out//8] / qzeros[g, out//8] / scales[g, out]
                # —— 输入维都是 dim0,RowParallel 切输入 → 切 dim0。
                if self.tp_size > 1:
                    tensor = self._split_weight(tensor, dim=0)

            if tensor.shape != param.shape:
                if tensor.dim() == 1 and param.dim() == 2 and tensor.numel() == param.numel():
                    tensor = tensor.view_as(param)
                elif (
                    param_name == "weight"
                    and tensor.dim() == 2
                    and tensor.shape[0] != tensor.shape[1]
                    and tensor.t().shape == param.shape
                ):
                    tensor = tensor.t().contiguous()
                else:
                    raise ValueError(
                        f"Shape mismatch for {self.prefix}.{param_name}: "
                        f"weight {tensor.shape} vs param {param.shape}"
                    )
            param.data.copy_(tensor)
            self._mark_loaded(param_name)

        # process_weights_after_loading is invoked by NewModelLoader's
        # post-load hook after every shard has landed; see LinearBase.

    def _split_weight(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        if self.tp_size <= 1:
            return tensor
        self._check_divisible(
            tensor.shape[dim],
            self.tp_size,
            f"{type(self).__name__} tensor dim {dim}",
            self.prefix,
        )
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
        num_shards = len(self.shard_names) if self.shard_names else 1
        if num_shards <= 0:
            raise ValueError("MergedColumnParallelLinear requires at least one shard")
        partition = output_size // tp_size
        if partition % num_shards != 0:
            raise ValueError(
                f"MergedColumnParallelLinear output partition ({partition}) "
                f"must be divisible by num_shards ({num_shards}) for prefix={prefix!r}"
            )

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
                    self._mark_loaded(param_name)
                    continue

                if param_name in ("qweight", "qzeros", "scales"):
                    # AWQ(w4a16):输出维是 dim1;qweight/qzeros 以 pack_factor 压缩,
                    # scales 不压缩。每个 shard(gate/up)沿 dim1 TP 切 + 按 shard 偏移拼入。
                    pf = (
                        1
                        if param_name == "scales"
                        else getattr(self.quant_method, "PACK_FACTOR", 8)
                    )
                    shard_cols = shard_size // pf
                    if (
                        self.tp_size > 1
                        and tensor.shape[1] == shard_cols * self.tp_size
                    ):
                        start = self.tp_rank * shard_cols
                        tensor = tensor.narrow(1, start, shard_cols).contiguous()
                    if tensor.shape[1] != shard_cols:
                        raise ValueError(
                            f"[AWQ merged] {self.prefix}.{param_name} shard={shard_id}: "
                            f"dim1={tensor.shape[1]} != expected {shard_cols}"
                        )
                    off = shard_id * shard_cols
                    param.data[:, off : off + shard_cols].copy_(tensor)
                    self._mark_loaded((param_name, shard_id))
                    continue

                if param_name == "weight_scale_inv":
                    # FP8 per-block scale grid [out_blocks, in_blocks]. Copying
                    # a gate/up shard is only valid when the shard boundary is
                    # aligned to the source block grid. Otherwise one source
                    # scale block overlaps two local shards and must be
                    # reconstructed by dequant+requant, which this path does not do.
                    block_n, _ = self._fp8_scale_block_size()
                    shard_row_start = shard_id * shard_size
                    self._check_fp8_block_aligned(
                        shard_row_start,
                        shard_size,
                        block_n,
                        f"FP8 block merged scale {self.prefix}.{self.shard_names[shard_id]}",
                    )
                    blocks_per_shard = shard_size // block_n
                    if (
                        self.tp_size > 1
                        and tensor.shape[0] == blocks_per_shard * self.tp_size
                    ):
                        start = self.tp_rank * blocks_per_shard
                        tensor = tensor.narrow(0, start, blocks_per_shard).contiguous()
                    if tensor.shape[0] != blocks_per_shard:
                        raise ValueError(
                            f"Shape mismatch for merged {self.prefix}.{param_name} "
                            f"shard={shard_id}: got block rows={tensor.shape[0]}, "
                            f"expected {blocks_per_shard}"
                        )
                    offset = shard_id * blocks_per_shard
                    param.data[offset : offset + blocks_per_shard].copy_(tensor)
                    self._mark_loaded((param_name, shard_id))
                    continue

                # Per-shard non-weight param (e.g. per-channel weight_scale).
                if param.numel() == 1:
                    # Streaming dispatch calls load_weights once per tensor, so
                    # per-shard scalar scales must be accumulated on the instance
                    # and merged only after every shard has landed.
                    val = float(tensor.flatten()[0].item())
                    if not hasattr(self, "_merged_per_tensor_scales"):
                        self._merged_per_tensor_scales = {}
                    self._merged_per_tensor_scales.setdefault(param_name, {})[
                        shard_id
                    ] = val
                    self._mark_loaded((param_name, shard_id))
                    continue

                # Per-channel/per-shard tensor: TP-slice + offset copy along dim 0.
                # Convention: ckpt tensor's dim-0 covers one shard's full output
                # (= shard_size * tp_size).
                if self.tp_size > 1 and tensor.shape[0] == shard_size * self.tp_size:
                    start = self.tp_rank * shard_size
                    tensor = tensor.narrow(0, start, shard_size).contiguous()
                offset = shard_id * shard_size
                if tensor.shape[0] != shard_size:
                    raise ValueError(
                        f"Shape mismatch for merged {self.prefix}.{param_name} "
                        f"shard={shard_id}: got dim-0={tensor.shape[0]}, "
                        f"expected {shard_size}"
                    )
                if tensor.dim() == 1 and param.dim() == 2:
                    tensor = tensor.view(-1, 1)
                param.data[offset : offset + shard_size].copy_(tensor)
                self._mark_loaded((param_name, shard_id))
                continue

            if shard_id < 0:
                if param_name == "weight":
                    split_tensor = self._split_weight(tensor, dim=0)
                    if (
                        split_tensor.shape != self.weight.shape
                        and split_tensor.dim() == 2
                        and split_tensor.shape[0] != split_tensor.shape[1]
                        and split_tensor.t().shape == self.weight.shape
                    ):
                        split_tensor = split_tensor.t().contiguous()
                    self.weight.data.copy_(split_tensor)
                    self._mark_loaded("weight")
                continue

            if self.tp_size > 1:
                self._check_divisible(
                    tensor.shape[0],
                    self.tp_size,
                    f"{type(self).__name__} shard tensor dim 0",
                    self.prefix,
                )
                shard_output_size = tensor.shape[0] // self.tp_size
                start = self.tp_rank * shard_output_size
                tensor = tensor.narrow(0, start, shard_output_size).contiguous()

            if param_name == "weight":
                offset = shard_id * shard_size
                if (
                    tensor.dim() == 2
                    and self.weight.shape[0] == tensor.shape[1]
                    and self.weight.shape[1] >= offset + shard_size
                ):
                    self.weight.data[:, offset : offset + shard_size].copy_(
                        tensor.t().contiguous()
                    )
                else:
                    self.weight.data[offset : offset + shard_size, :].copy_(tensor)
                self._mark_loaded(("weight", shard_id))
            elif param_name == "bias" and self.bias is not None:
                offset = shard_id * shard_size
                self.bias.data[offset : offset + shard_size].copy_(tensor)
                self._mark_loaded(("bias", shard_id))

        # process_weights_after_loading is invoked by NewModelLoader's
        # post-load hook after every shard has landed; see LinearBase.

    def _required_load_keys(self):
        base = super()._required_load_keys()
        if not self.shard_names:
            return base
        shard_ids = range(len(self.shard_names))
        required = set()
        for key in base:
            direct_loaded = key in self._loaded_weight_keys
            shard_keys = {(key, shard_id) for shard_id in shard_ids}
            if direct_loaded or shard_keys.issubset(self._loaded_weight_keys):
                continue
            required.update(shard_keys)
        return required

    def process_weights_after_loading(self):
        # Merge per-tensor gate/up scales and rescale fp8 weight shards so both
        # shards share the same final scale. This must run before quant_method
        # post-load hooks, matching QKVParallelLinear's streaming-safe behavior.
        if getattr(self, "_post_load_done", False):
            return
        self._check_load_complete()
        self._merge_merged_per_tensor_scales()
        self.quant_method.process_weights_after_loading(self)
        self._post_load_done = True

    def _merge_merged_per_tensor_scales(self):
        scales_by_param = getattr(self, "_merged_per_tensor_scales", None)
        if not scales_by_param:
            return

        num_shards = len(self.shard_names) if self.shard_names else 1
        shard_size = self.output_size_per_partition // num_shards
        for param_name, scales in scales_by_param.items():
            param = getattr(self, param_name, None)
            if param is None or param.numel() != 1:
                continue
            max_scale = max(scales.values())
            param.data.fill_(max_scale)
            if param_name != "weight_scale":
                continue
            for shard_id, scale_val in scales.items():
                if scale_val == max_scale:
                    continue
                weight_slice = self._merged_shard_weight_slice(shard_id, shard_size)
                ratio = scale_val / max_scale
                rescaled = (weight_slice.float() * ratio).to(weight_slice.dtype)
                weight_slice.copy_(rescaled)

    def _get_shard_id(self, weight_name: str) -> int:
        return _find_shard_segment(weight_name, self.prefix, self.shard_names)

    def _merged_shard_weight_slice(self, shard_id: int, shard_size: int) -> torch.Tensor:
        offset = shard_id * shard_size
        if self.weight.shape[0] >= offset + shard_size:
            return self.weight.data[offset : offset + shard_size, :]
        if self.weight.dim() == 2 and self.weight.shape[1] >= offset + shard_size:
            return self.weight.data[:, offset : offset + shard_size]
        raise ValueError(
            f"Cannot locate merged shard {shard_id} for {self.prefix}.weight: "
            f"weight shape={tuple(self.weight.shape)}, shard_size={shard_size}"
        )



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

        if num_heads <= 0 or num_kv_heads <= 0 or head_dim <= 0:
            raise ValueError(
                f"QKVParallelLinear requires positive head counts and head_dim, got "
                f"num_heads={num_heads}, num_kv_heads={num_kv_heads}, "
                f"head_dim={head_dim}, prefix={prefix!r}"
            )
        if num_heads % tp_size != 0:
            raise ValueError(
                f"QKVParallelLinear requires num_heads ({num_heads}) to be "
                f"divisible by tp_size ({tp_size}) for prefix={prefix!r}"
            )
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"QKVParallelLinear requires num_heads ({num_heads}) to be "
                f"divisible by num_kv_heads ({num_kv_heads}) for GQA, "
                f"prefix={prefix!r}"
            )

        self.kv_tp_size = math.gcd(num_kv_heads, tp_size)
        if self.kv_tp_size <= 0 or num_kv_heads % self.kv_tp_size != 0:
            raise ValueError(
                f"QKVParallelLinear cannot form a valid KV TP group for "
                f"num_kv_heads={num_kv_heads}, tp_size={tp_size}, prefix={prefix!r}"
            )
        self.kv_replication_group_size = tp_size // self.kv_tp_size
        self.kv_tp_rank = tp_rank // self.kv_replication_group_size

        self.num_heads_per_partition = num_heads // tp_size
        self.num_kv_heads_per_partition = num_kv_heads // self.kv_tp_size
        if self.num_heads_per_partition % self.num_kv_heads_per_partition != 0:
            raise ValueError(
                f"QKVParallelLinear requires local Q heads "
                f"({self.num_heads_per_partition}) to be divisible by local KV "
                f"heads ({self.num_kv_heads_per_partition}) for prefix={prefix!r}"
            )

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
        # Streaming-dispatch contract: this method may be called many times
        # with a single q/k/v shard at a time. Each call writes the shard
        # into its own offset slice in self.weight (and self.bias, etc.) so
        # that partial calls are safe — there's no "wait for all three" gate.
        # NewModelLoader._run_post_load_hooks invokes
        # process_weights_after_loading after every shard has landed; see
        # LinearBase.
        for full_name, tensor in weights.items():
            param_name = self._get_param_name(full_name)
            if param_name is None:
                continue

            qkv_key = None
            qkv_shard_id = _find_shard_segment(full_name, self.prefix, self.shard_names)
            if qkv_shard_id == 0:
                qkv_key = "q"
            elif qkv_shard_id == 1:
                qkv_key = "k"
            elif qkv_shard_id == 2:
                qkv_key = "v"

            if qkv_key is None:
                if self._dispatch_fused_qkv_param(param_name, tensor):
                    continue
                param = getattr(self, param_name, None)
                if isinstance(param, nn.Parameter) and tensor.shape == param.shape:
                    param.data.copy_(tensor)
                    self._mark_loaded(param_name)
                continue

            self._dispatch_qkv_shard(qkv_key, param_name, tensor)

    def _qkv_partition(self, qkv_key: str):
        if qkv_key == "q":
            return self.num_heads, self.q_size, 0, self.tp_size, self.tp_rank
        if qkv_key == "k":
            return (
                self.num_kv_heads,
                self.kv_size,
                self.q_size,
                self.kv_tp_size,
                self.kv_tp_rank,
            )
        if qkv_key == "v":
            return (
                self.num_kv_heads,
                self.kv_size,
                self.q_size + self.kv_size,
                self.kv_tp_size,
                self.kv_tp_rank,
            )
        raise ValueError(f"Unknown QKV shard key {qkv_key!r}")

    def _split_fused_qkv_tensor(self, param_name: str, tensor: torch.Tensor):
        q_rows = self.num_heads * self.head_dim
        kv_rows = self.num_kv_heads * self.head_dim
        if param_name in ("qweight", "qzeros", "scales"):
            pf = 1 if param_name == "scales" else getattr(self.quant_method, "PACK_FACTOR", 8)
            q_cols = q_rows // pf
            kv_cols = kv_rows // pf
            if tensor.dim() < 2 or tensor.shape[1] != q_cols + 2 * kv_cols:
                return None
            q = tensor.narrow(1, 0, q_cols)
            k = tensor.narrow(1, q_cols, kv_cols)
            v = tensor.narrow(1, q_cols + kv_cols, kv_cols)
            return {"q": q, "k": k, "v": v}
        if param_name == "weight_scale_inv":
            block_n, _ = self._fp8_scale_block_size()
            self._check_fp8_block_aligned(0, q_rows, block_n, f"FP8 block fused QKV scale {self.prefix}.q")
            self._check_fp8_block_aligned(q_rows, kv_rows, block_n, f"FP8 block fused QKV scale {self.prefix}.k")
            self._check_fp8_block_aligned(q_rows + kv_rows, kv_rows, block_n, f"FP8 block fused QKV scale {self.prefix}.v")
            q_blocks = q_rows // block_n
            kv_blocks = kv_rows // block_n
            if tensor.shape[0] != q_blocks + 2 * kv_blocks:
                return None
            q = tensor.narrow(0, 0, q_blocks)
            k = tensor.narrow(0, q_blocks, kv_blocks)
            v = tensor.narrow(0, q_blocks + kv_blocks, kv_blocks)
            return {"q": q, "k": k, "v": v}
        total_rows = q_rows + 2 * kv_rows
        if tensor.shape[0] != total_rows:
            return None
        q = tensor.narrow(0, 0, q_rows)
        k = tensor.narrow(0, q_rows, kv_rows)
        v = tensor.narrow(0, q_rows + kv_rows, kv_rows)
        return {"q": q, "k": k, "v": v}

    def _dispatch_fused_qkv_param(self, param_name: str, tensor: torch.Tensor) -> bool:
        if param_name in ("weight_scale", "input_scale") and tensor.numel() == 1:
            param = getattr(self, param_name, None)
            if isinstance(param, nn.Parameter) and param.numel() == 1:
                param.data.copy_(tensor.view_as(param))
                self._mark_loaded(param_name)
                return True

        param = getattr(self, param_name, None)
        if (
            param_name == "weight_scale_inv"
            and isinstance(param, nn.Parameter)
            and tensor.shape == param.shape
            and self.tp_size == 1
        ):
            param.data.copy_(tensor)
            self._mark_loaded(param_name)
            return True

        if param_name not in (
            "weight",
            "bias",
            "weight_scale",
            "weight_scale_inv",
            "qweight",
            "qzeros",
            "scales",
        ):
            return False
        parts = self._split_fused_qkv_tensor(param_name, tensor)
        if parts is None:
            return False
        for qkv_key, part in parts.items():
            self._dispatch_qkv_shard(qkv_key, param_name, part.contiguous())
        return True

    def _dispatch_qkv_shard(self, qkv_key: str, param_name: str, tensor: torch.Tensor):
        """Write a single q/k/v shard into the merged parameter at its offset."""
        num_heads, size, offset, partition_world, partition_rank = self._qkv_partition(qkv_key)

        if param_name in ("qweight", "qzeros", "scales"):
            # AWQ(w4a16):输出维是 dim1;qweight/qzeros 以 pack_factor 压缩。
            # q/k/v 各自的 size、offset 在 dim1 上换算(压缩参数 //pf)。
            param = getattr(self, param_name, None)
            if param is None or not isinstance(param, nn.Parameter):
                return
            pf = (
                1
                if param_name == "scales"
                else getattr(self.quant_method, "PACK_FACTOR", 8)
            )
            cols = size // pf
            off_cols = offset // pf
            if partition_world > 1 and tensor.shape[1] == cols * partition_world:
                start = partition_rank * cols
                tensor = tensor.narrow(1, start, cols).contiguous()
            if tensor.shape[1] != cols:
                raise ValueError(
                    f"[AWQ QKV] {self.prefix}.{param_name}/{qkv_key}: "
                    f"dim1={tensor.shape[1]} != expected {cols}"
                )
            param.data[:, off_cols : off_cols + cols].copy_(tensor)
            self._mark_loaded((param_name, qkv_key))
            return

        if param_name == "weight":
            split = self._split_qkv(tensor, qkv_key)
            if (
                split.dim() == 2
                and self.weight.shape[0] >= offset + size
                and self.weight.shape[1] == split.shape[1]
            ):
                self.weight.data[offset : offset + size].copy_(split)
            elif (
                split.dim() == 2
                and self.weight.shape[0] == split.shape[1]
                and self.weight.shape[1] >= offset + size
            ):
                self.weight.data[:, offset : offset + size].copy_(
                    split.t().contiguous()
                )
            else:
                raise ValueError(
                    f"Shape mismatch for QKV {self.prefix}.weight/{qkv_key}: "
                    f"weight shape={tuple(self.weight.shape)}, split shape={tuple(split.shape)}, "
                    f"offset={offset}, size={size}"
                )
            self._mark_loaded(("weight", qkv_key))
            return

        if param_name == "bias":
            if self.bias is None:
                return
            split = self._split_qkv(tensor, qkv_key)
            self.bias.data[offset : offset + size].copy_(split)
            self._mark_loaded(("bias", qkv_key))
            return

        # Auxiliary per-shard params (per-channel weight_scale, input_scale, ...).
        param = getattr(self, param_name, None)
        if param is None or not isinstance(param, nn.Parameter):
            return

        if param_name == "weight_scale_inv":
            # FP8 per-block grid for this q/k/v shard: TP-slice along the
            # output(head)-block dim, then place at the shard's block offset
            # in the merged [total_blocks, in_blocks] grid. Copying is only
            # correct when both q/k/v boundaries and TP sub-shard boundaries
            # align to block_n; otherwise a single source scale block would be
            # shared by two local regions and must be rebuilt by requantization.
            block_n, _ = self._fp8_scale_block_size()
            self._check_fp8_block_aligned(
                offset,
                size,
                block_n,
                f"FP8 block QKV scale {self.prefix}.{qkv_key}",
            )
            expected_blocks = size // block_n
            if partition_world > 1:
                rows = size
                start_row = partition_rank * rows
                self._check_fp8_block_aligned(
                    start_row,
                    rows,
                    block_n,
                    f"FP8 block QKV TP scale {self.prefix}.{qkv_key}",
                )
                start_blk = start_row // block_n
                block_count = rows // block_n
                tensor = tensor.narrow(0, start_blk, block_count).contiguous()
                expected_blocks = block_count
            if tensor.shape[0] != expected_blocks:
                raise ValueError(
                    f"Shape mismatch for QKV {self.prefix}.{param_name}/{qkv_key}: "
                    f"got block rows={tensor.shape[0]}, expected {expected_blocks}"
                )
            off_blk = offset // block_n
            param.data[off_blk : off_blk + tensor.shape[0]].copy_(tensor)
            self._mark_loaded((param_name, qkv_key))
            return

        if param.numel() == 1:
            # Per-tensor scalar: collect per-shard scales now and merge in
            # process_weights_after_loading. Naive max-merge here is wrong for
            # `weight_scale` because each q/k/v shard's fp8 weights are stored
            # in the representation of its OWN ckpt scale; if the merged scale
            # is the max across shards, the smaller-scale shards' fp8 values
            # decode to magnitudes inflated by max/own — silently corrupt.
            # The original loader's `merge_qkv_hf_fp8_with_scale` rescales each
            # shard's weight to share max_scale; we mirror that in post-load.
            val = float(tensor.flatten()[0].item())
            if not hasattr(self, "_qkv_per_tensor_scales"):
                self._qkv_per_tensor_scales = {}
            self._qkv_per_tensor_scales.setdefault(param_name, {})[qkv_key] = val
            self._mark_loaded((param_name, qkv_key))
            return

        # Per-channel along output dim: TP-slice and write into [offset:offset+size).
        split = self._split_qkv(tensor, qkv_key)
        if split.shape[0] != size:
            raise ValueError(
                f"Shape mismatch for QKV shard {self.prefix}.{param_name}/{qkv_key}: "
                f"got dim-0={split.shape[0]}, expected {size}"
            )
        if split.dim() == 1 and param.dim() == 2:
            split = split.view(-1, 1)
        param.data[offset : offset + size].copy_(split)
        self._mark_loaded((param_name, qkv_key))

    def _required_load_keys(self):
        base = LinearBase._required_load_keys(self)
        shard_keys = ("q", "k", "v")
        required = set()
        for key in base:
            direct_loaded = key in self._loaded_weight_keys
            qkv_required = {(key, shard) for shard in shard_keys}
            if direct_loaded or qkv_required.issubset(self._loaded_weight_keys):
                continue
            required.update(qkv_required)
        return required

    def process_weights_after_loading(self):
        # Merge per-tensor q/k/v scales and rescale fp8 weights so all three
        # shards share the same final scale. Must run BEFORE the quant_method
        # post-load hook (which may rebind weight/scale Parameters and freeze
        # dtypes), so we override here instead of letting LinearBase handle it.
        if getattr(self, "_post_load_done", False):
            return
        self._check_load_complete()
        self._merge_qkv_per_tensor_scales()
        self.quant_method.process_weights_after_loading(self)
        self._post_load_done = True

    def _merge_qkv_per_tensor_scales(self):
        scales_by_param = getattr(self, "_qkv_per_tensor_scales", None)
        if not scales_by_param:
            return

        for param_name, scales in scales_by_param.items():
            param = getattr(self, param_name, None)
            if param is None or param.numel() != 1:
                continue
            max_scale = max(scales.values())
            param.data.fill_(max_scale)
            # input_scale doesn't index into self.weight; only weight_scale
            # requires per-shard fp8 rescale.
            if param_name != "weight_scale":
                continue
            for qkv_key, scale_val in scales.items():
                if scale_val == max_scale:
                    continue
                offset, size = self._qkv_shard_offset_size(qkv_key)
                weight_slice = self.weight.data[offset : offset + size]
                ratio = scale_val / max_scale
                rescaled = (weight_slice.float() * ratio).to(weight_slice.dtype)
                self.weight.data[offset : offset + size].copy_(rescaled)

        del self._qkv_per_tensor_scales

    def _qkv_shard_offset_size(self, qkv_key: str):
        if qkv_key == "q":
            return 0, self.q_size
        if qkv_key == "k":
            return self.q_size, self.kv_size
        return self.q_size + self.kv_size, self.kv_size

    def _split_qkv(self, tensor: torch.Tensor, qkv_key: str) -> torch.Tensor:
        num_heads, size_per_partition, _, partition_world, partition_rank = self._qkv_partition(qkv_key)
        if partition_world <= 1:
            return tensor
        if tensor.shape[0] == size_per_partition:
            return tensor
        full_size = num_heads * self.head_dim
        if tensor.shape[0] != full_size:
            raise ValueError(
                f"Shape mismatch for QKV shard {self.prefix}.{qkv_key}: "
                f"got dim-0={tensor.shape[0]}, expected local {size_per_partition} "
                f"or full {full_size}"
            )
        start = partition_rank * size_per_partition
        return tensor.narrow(0, start, size_per_partition).contiguous()
