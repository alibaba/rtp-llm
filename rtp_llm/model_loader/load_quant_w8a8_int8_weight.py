"""Load floating-point checkpoints through the existing INT8 weight layouts."""

import copy
import logging
import time
from typing import Optional

import torch
from rtp_llm.config.quant_config import CompressedW8A8Int8PerChannelQuantConfig
from rtp_llm.model_loader.compressed_w8a8_int8_per_channel_weight import (
    CompressedW8A8Int8PerChannelWeight,
)
from rtp_llm.model_loader.tensor_source import DatabaseTensorSource
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.model_weight import W
from rtp_llm.utils.w8a8_int8_quant import (
    is_recipe_weight,
    quantize_weight_per_output_channel,
    scale_name_for,
)

logger = logging.getLogger(__name__)


def is_load_time_w8a8(quant_config) -> bool:
    return (
        isinstance(quant_config, CompressedW8A8Int8PerChannelQuantConfig)
        and not quant_config.is_quanted()
    )


def validate_w8a8_source(database: CkptDatabase) -> dict[str, str]:
    """Metadata-only validation, before allocating model weights on the device."""
    if not isinstance(database, CkptDatabase) or not database.is_safetensor:
        raise ValueError(
            "load-time W8A8 requires an unquantized safetensors checkpoint"
        )
    scales = {}
    for name in database.get_pretrain_tensor_names():
        if not is_recipe_weight(name):
            continue
        dtype = database.get_tensor_type(name)
        shape = database.get_tensor_shape(name)
        if dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError(
                f"load-time W8A8 source {name} must be floating point, got {dtype}"
            )
        if len(shape) not in (2, 3) or any(dim <= 0 for dim in shape):
            raise ValueError(
                f"load-time W8A8 source {name} has unsupported shape {shape}"
            )
        scale_name = scale_name_for(name)
        if database.has_tensor(scale_name):
            raise ValueError(f"load-time W8A8 source already contains {scale_name}")
        scales[scale_name] = name
    if not scales:
        raise ValueError(
            "load-time W8A8 found no weights covered by the Qwen MoE recipe"
        )
    return scales


class W8A8QuantizingDatabase(CkptDatabase):
    """A read-only INT8 view over a floating-point safetensors database.

    CkptDatabase's slicing contract lets the existing MoE pre-shard path keep
    working. Each requested output row is quantized over its COMPLETE source K
    dimension before taking the requested K slice. This preserves the offline
    quantize-then-TP-split semantics for every TP rank.

    Only scales are cached. BF16/FP32 temporaries are bounded by chunk_rows;
    returned INT8 tensors still occupy their requested shape. One view is used
    per composite weight, so the small scale cache cannot grow across layers.
    """

    def __init__(self, source: CkptDatabase, chunk_rows: int, scale_sources=None):
        # Deliberately do not open another set of checkpoint handles.
        self._source = source
        if (
            isinstance(chunk_rows, bool)
            or not isinstance(chunk_rows, int)
            or chunk_rows <= 0
        ):
            raise ValueError("W8A8_QUANT_CHUNK_ROWS must be a positive integer")
        self.chunk_rows = chunk_rows
        self._scale_sources = (
            validate_w8a8_source(source) if scale_sources is None else scale_sources
        )
        self._weight_names = set(self._scale_sources.values())
        self._scales = {}
        self._valid_rows = {}
        self.quantized_rows = 0

    @property
    def is_safetensor(self):
        return True

    def has_tensor(self, name):
        return name in self._scale_sources or self._source.has_tensor(name)

    def get_tensor_shape(self, name):
        weight_name = self._scale_sources.get(name, name)
        shape = self._source.get_tensor_shape(weight_name)
        return torch.Size((*shape[:-1], 1)) if name in self._scale_sources else shape

    def get_tensor_type(self, name):
        if name in self._scale_sources:
            return torch.float32
        if name in self._weight_names:
            return torch.int8
        return self._source.get_tensor_type(name)

    def load_tensor(self, name, data_type: Optional[torch.dtype] = None):
        if name not in self._weight_names and name not in self._scale_sources:
            return self._source.load_tensor(name, data_type)
        return [self.load_tensor_slice(name, (), data_type)]

    @staticmethod
    def _normalize_slice(tensor_slice, shape):
        if len(tensor_slice) > len(shape):
            raise ValueError("too many slice dimensions for W8A8 tensor")
        cuts = tuple(tensor_slice) + (slice(None),) * (len(shape) - len(tensor_slice))
        result = []
        squeeze = []
        for dim, (cut, size) in enumerate(zip(cuts, shape)):
            if isinstance(cut, int):
                index = cut + size if cut < 0 else cut
                if index < 0 or index >= size:
                    raise IndexError(
                        f"W8A8 tensor index {cut} outside dimension {size}"
                    )
                result.append(slice(index, index + 1))
                squeeze.append(dim)
            elif isinstance(cut, slice):
                start, stop, step = cut.indices(size)
                if step != 1:
                    raise ValueError("W8A8 loading supports contiguous slices only")
                result.append(slice(start, max(start, stop)))
            else:
                raise TypeError(f"unsupported W8A8 tensor slice: {cut!r}")
        return result, squeeze

    def load_tensor_slice(self, name, tensor_slice, data_type=None):
        is_scale = name in self._scale_sources
        weight_name = self._scale_sources.get(name, name)
        if weight_name not in self._weight_names:
            return self._source.load_tensor_slice(name, tensor_slice, data_type)
        dtype = torch.float32 if is_scale else torch.int8
        if data_type is not None and data_type != dtype:
            raise ValueError(
                f"W8A8 tensor {name} requires {dtype}, requested {data_type}"
            )
        shape = self._source.get_tensor_shape(weight_name)
        cuts, squeeze = self._normalize_slice(tensor_slice, self.get_tensor_shape(name))
        output = torch.empty([cut.stop - cut.start for cut in cuts], dtype=dtype)
        if weight_name not in self._scales:
            self._scales[weight_name] = torch.empty(
                (*shape[:-1], 1), dtype=torch.float32
            )
            self._valid_rows[weight_name] = torch.zeros(shape[:-1], dtype=torch.bool)
        cached_scale = self._scales[weight_name]
        valid = self._valid_rows[weight_name]
        expert_ids = range(cuts[0].start, cuts[0].stop) if len(shape) == 3 else [None]
        row_cut, col_cut = cuts[-2:]
        for expert_slot, expert in enumerate(expert_ids):
            prefix = () if expert is None else (expert,)
            out_prefix = () if expert is None else (expert_slot,)
            for start in range(row_cut.start, row_cut.stop, self.chunk_rows):
                stop = min(start + self.chunk_rows, row_cut.stop)
                rows = (*prefix, slice(start, stop))
                destination = (
                    *out_prefix,
                    slice(start - row_cut.start, stop - row_cut.start),
                )
                if is_scale and valid[rows].all().item():
                    output[destination] = cached_scale[rows][..., col_cut]
                    continue
                raw = self._source.load_tensor_slice(
                    weight_name, (*rows, slice(None)), torch.float32
                )
                if not torch.isfinite(raw).all().item():
                    raise ValueError(
                        f"non-finite W8A8 source values in {weight_name} at {rows}"
                    )
                quantized, scale = quantize_weight_per_output_channel(raw)
                cached_scale[rows] = scale
                valid[rows] = True
                self.quantized_rows += stop - start
                output[destination] = (scale if is_scale else quantized)[..., col_cut]
                del raw, quantized, scale
        for dim in reversed(squeeze):
            output = output.squeeze(dim)
        return output


class LoadQuantW8A8Int8PerChannelWeight(CompressedW8A8Int8PerChannelWeight):
    """Generate checkpoint-format INT8/scale, then reuse prequantized layouts."""

    @classmethod
    def support(cls, quant_config, src_weight_info):
        if not is_load_time_w8a8(quant_config):
            return False
        if not src_weight_info.weights or not all(
            is_recipe_weight(weight.name) for weight in src_weight_info.weights
        ):
            if src_weight_info.name in (W.moe_w1, W.moe_w2):
                raise ValueError(
                    "load-time W8A8 requires fused gate_up_proj/down_proj MoE "
                    "checkpoint tensors; split-expert checkpoint layouts are not supported"
                )
            return False
        checkpoint_config = copy.copy(quant_config)
        checkpoint_config._is_quanted = True
        return super().support(checkpoint_config, src_weight_info)

    def __init__(self, src_weight_info, quant_config, *args, **kwargs):
        self._source_weight_info = src_weight_info
        self._chunk_rows = quant_config.load_chunk_rows
        super().__init__(src_weight_info, quant_config, *args, **kwargs)

    def get_tensor_names(self, layer_id, load_config):
        if getattr(self._source_weight_info, "stacked_ckpt_keys", False):
            return {
                weight.tensor_name(layer_id)
                for weight in self._source_weight_info.weights
            }
        return self._source_weight_info.get_tensor_names(layer_id, load_config)

    def _load_raw_tensor(self, tensor_source, layer_id, device, load_config):
        if load_config.merge_lora:
            raise ValueError("load-time W8A8 does not support merging LoRA weights")
        if torch.device(device).type != "cpu":
            raise ValueError("load-time W8A8 quantization must run on CPU")
        source = tensor_source.get_database()
        # Build the per-weight view using only this descriptor's checkpoint keys.
        # ModelLoader has already checked all source metadata before loading.
        names = self.get_tensor_names(layer_id, load_config)
        scales = {
            scale_name_for(name): name for name in names if is_recipe_weight(name)
        }
        database = W8A8QuantizingDatabase(source, self._chunk_rows, scales)
        started = time.perf_counter()
        result = super()._load_raw_tensor(
            DatabaseTensorSource(database), layer_id, device, load_config
        )
        logger.info(
            "w8a8_load_quant weight=%s layer=%s tp=%s rank=%s rows=%s seconds=%.3f",
            self.name,
            layer_id,
            load_config.tp_size,
            load_config.tp_rank,
            database.quantized_rows,
            time.perf_counter() - started,
        )
        return result
