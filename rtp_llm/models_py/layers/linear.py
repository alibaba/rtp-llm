from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.quant_methods.base import QuantizationConfig, QuantizeMethodBase


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
        self.quant_method.process_weights_after_loading(self)
        self._post_load_done = True

    def _fp8_scale_block_size(self):
        return getattr(
            self.quant_config, "weight_block_size", [self._FP8_BLOCK, self._FP8_BLOCK]
        )

    @staticmethod
    def _ceil_div(x: int, y: int) -> int:
        return (x + y - 1) // y


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
            elif param_name == "weight_scale_inv":
                if self.tp_size > 1:
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
            elif param_name in ("qweight", "qzeros", "scales"):
                # AWQ(w4a16):qweight[in, out//8] / qzeros[g, out//8] / scales[g, out]
                # —— 输出维都是 dim1,ColumnParallel 切输出 → 切 dim1。
                if self.tp_size > 1:
                    tensor = self._split_weight(tensor, dim=1)

            if tensor.shape != param.shape:
                if (
                    param_name == "weight"
                    and tensor.dim() == 2
                    and tensor.t().shape == param.shape
                ):
                    tensor = tensor.t().contiguous()
                else:
                    raise ValueError(
                        f"Shape mismatch for {self.prefix}.{param_name}: "
                        f"weight {tensor.shape} vs param {param.shape}"
                    )
            param.data.copy_(tensor)

        # process_weights_after_loading is invoked by NewModelLoader's
        # post-load hook after every shard has landed; see LinearBase.

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
            elif param_name == "weight_scale_inv":
                if self.tp_size > 1:
                    tensor = self._split_weight(tensor, dim=1)
            elif param_name == "g_idx":
                if self.tp_size > 1:
                    tensor = self._split_weight(tensor, dim=0)
            elif param_name in ("scales", "zeros"):
                if tensor.dim() > 1 and tensor.shape[0] > 1:
                    if self.tp_size > 1:
                        tensor = self._split_weight(tensor, dim=0)
            elif param_name in ("qweight", "qzeros", "scales"):
                # AWQ(w4a16):qweight[in, out//8] / qzeros[g, out//8] / scales[g, out]
                # —— 输入维都是 dim0,RowParallel 切输入 → 切 dim0。
                if self.tp_size > 1:
                    tensor = self._split_weight(tensor, dim=0)

            if tensor.shape != param.shape:
                if (
                    param_name == "weight"
                    and tensor.dim() == 2
                    and tensor.t().shape == param.shape
                ):
                    tensor = tensor.t().contiguous()
                else:
                    raise ValueError(
                        f"Shape mismatch for {self.prefix}.{param_name}: "
                        f"weight {tensor.shape} vs param {param.shape}"
                    )
            param.data.copy_(tensor)

        # process_weights_after_loading is invoked by NewModelLoader's
        # post-load hook after every shard has landed; see LinearBase.

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
                    continue

                if param_name == "weight_scale_inv":
                    # FP8 per-block scale grid [out_blocks, in_blocks]. Each
                    # shard (gate/up) ships its own grid; place this shard's
                    # block-rows at its block offset. shard_size is in weight
                    # rows, so convert through the quant config's N block.
                    block_n, _ = self._fp8_scale_block_size()
                    blocks_per_shard = self._ceil_div(shard_size, block_n)
                    if (
                        self.tp_size > 1
                        and tensor.shape[0] == blocks_per_shard * self.tp_size
                    ):
                        start = self.tp_rank * blocks_per_shard
                        tensor = tensor.narrow(0, start, blocks_per_shard).contiguous()
                    offset = shard_id * blocks_per_shard
                    param.data[offset : offset + blocks_per_shard].copy_(tensor)
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
                param.data[offset : offset + shard_size].copy_(tensor)
                continue

            if shard_id < 0:
                if param_name == "weight":
                    split_tensor = self._split_weight(tensor, dim=0)
                    if split_tensor.shape != self.weight.shape and split_tensor.t().shape == self.weight.shape:
                        split_tensor = split_tensor.t().contiguous()
                    self.weight.data.copy_(split_tensor)
                continue

            if self.tp_size > 1:
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
                    self.prefix,
                    param_name,
                    vals,
                )
            param.data.fill_(max(vals))

        # process_weights_after_loading is invoked by NewModelLoader's
        # post-load hook after every shard has landed; see LinearBase.

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
            if "q_proj" in full_name:
                qkv_key = "q"
            elif "k_proj" in full_name:
                qkv_key = "k"
            elif "v_proj" in full_name:
                qkv_key = "v"

            if qkv_key is None:
                # Non-q/k/v param (e.g. an already-merged ckpt or auxiliary
                # tensor that maps directly onto a self.* param).
                if param_name == "weight":
                    self.weight.data.copy_(self._split_weight(tensor, dim=0))
                else:
                    param = getattr(self, param_name, None)
                    if isinstance(param, nn.Parameter) and tensor.shape == param.shape:
                        param.data.copy_(tensor)
                continue

            self._dispatch_qkv_shard(qkv_key, param_name, tensor)

    def _dispatch_qkv_shard(self, qkv_key: str, param_name: str, tensor: torch.Tensor):
        """Write a single q/k/v shard into the merged parameter at its offset."""
        # Resolve per-shard size and offset in the merged QKV layout.
        if qkv_key == "q":
            num_heads, size, offset = self.num_heads, self.q_size, 0
        elif qkv_key == "k":
            num_heads, size, offset = self.num_kv_heads, self.kv_size, self.q_size
        else:  # "v"
            num_heads, size, offset = (
                self.num_kv_heads,
                self.kv_size,
                self.q_size + self.kv_size,
            )

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
            if self.tp_size > 1 and tensor.shape[1] == cols * self.tp_size:
                start = self.tp_rank * cols
                tensor = tensor.narrow(1, start, cols).contiguous()
            if tensor.shape[1] != cols:
                raise ValueError(
                    f"[AWQ QKV] {self.prefix}.{param_name}/{qkv_key}: "
                    f"dim1={tensor.shape[1]} != expected {cols}"
                )
            param.data[:, off_cols : off_cols + cols].copy_(tensor)
            return

        if param_name == "weight":
            split = self._split_qkv(tensor, num_heads, self.head_dim)
            if (
                split.dim() == 2
                and self.weight.shape[0] == split.shape[1]
                and self.weight.shape[1] >= offset + size
            ):
                self.weight.data[:, offset : offset + size].copy_(
                    split.t().contiguous()
                )
            else:
                self.weight.data[offset : offset + size].copy_(split)
            return

        if param_name == "bias":
            if self.bias is None:
                return
            split = self._split_qkv(tensor, num_heads, self.head_dim)
            self.bias.data[offset : offset + size].copy_(split)
            return

        # Auxiliary per-shard params (per-channel weight_scale, input_scale, ...).
        param = getattr(self, param_name, None)
        if param is None or not isinstance(param, nn.Parameter):
            return

        if param_name == "weight_scale_inv":
            # FP8 per-block grid for this q/k/v shard: TP-slice along the
            # output(head)-block dim, then place at the shard's block offset
            # in the merged [total_blocks, in_blocks] grid.
            blk = self._FP8_BLOCK
            if self.tp_size > 1:
                heads_pp = max(1, num_heads // self.tp_size)
                rows = heads_pp * self.head_dim
                start_blk = (self.tp_rank * rows) // blk
                tensor = tensor.narrow(0, start_blk, rows // blk).contiguous()
            off_blk = offset // blk
            param.data[off_blk : off_blk + tensor.shape[0]].copy_(tensor)
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
            return

        # Per-channel along output dim: TP-slice and write into [offset:offset+size).
        split = self._split_qkv(tensor, num_heads, self.head_dim)
        if split.shape[0] != size:
            raise ValueError(
                f"Shape mismatch for QKV shard {self.prefix}.{param_name}/{qkv_key}: "
                f"got dim-0={split.shape[0]}, expected {size}"
            )
        param.data[offset : offset + size].copy_(split)

    def process_weights_after_loading(self):
        # Merge per-tensor q/k/v scales and rescale fp8 weights so all three
        # shards share the same final scale. Must run BEFORE the quant_method
        # post-load hook (which may rebind weight/scale Parameters and freeze
        # dtypes), so we override here instead of letting LinearBase handle it.
        if getattr(self, "_post_load_done", False):
            return
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

    def _split_qkv(
        self, tensor: torch.Tensor, num_heads: int, head_dim: int
    ) -> torch.Tensor:
        if self.tp_size <= 1:
            return tensor
        heads_per_partition = max(1, num_heads // self.tp_size)
        size_per_partition = heads_per_partition * head_dim
        start = self.tp_rank * size_per_partition
        return tensor.narrow(0, start, size_per_partition).contiguous()
