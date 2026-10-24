"""GPU-resident FP8 vocabulary lookup with BF16 output and TP ordering."""

import torch
from torch import nn

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather


class Fp8VocabEmbedding(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        tp_size: int,
        device: torch.device,
        weight_scale=None,
    ):
        super().__init__()
        if (
            not weight.is_cuda
            or weight.dtype != torch.float8_e4m3fn
            or weight.ndim != 2
            or not weight.is_contiguous()
            or min(weight.shape) <= 0
            or tp_size < 1
            or weight.device != device
        ):
            raise ValueError(
                "FP8 vocabulary requires contiguous CUDA E4M3 rows on its execution device"
            )
        if weight_scale is None or weight_scale.numel() != 1:
            raise ValueError("FP8 vocabulary requires one scale")
        self.weight_scale = weight_scale.to(device=device, dtype=torch.float32).reshape(
            1
        )
        self.rocm_decode_table = None
        if torch.version.hip is not None:
            from .rocm_fp8 import fn_decode_table

            self.rocm_decode_table = fn_decode_table(device)
        self.column_scales = None
        self.tp_size = tp_size
        self.hidden_size = weight.shape[1]
        self.vocab_size = weight.shape[0]
        self.device = device
        self.tile_width = next(
            width
            for width in (256, 128, 64, 32, 16, 8, 4, 2, 1)
            if self.hidden_size % width == 0
        )
        self.column_tiles = self.hidden_size // self.tile_width
        flat = weight.view(-1, self.tile_width)
        self.shards = [flat]
        self.pointer_table = torch.tensor(
            [flat.data_ptr()], dtype=torch.int64, device=weight.device
        )
        self.device = weight.device
        self.weight = self.shards[0].view_as(weight)
        self.weight._rtp_requires_restart_for_update = True
        self.columns = torch.arange(
            self.column_tiles, dtype=torch.int64, device=self.device
        )

    def prepare_byte_gather(self, *, role="PDFUSION"):
        """Collect immutable column scales once, before serving or capture.

        FP8 vocabulary storage and its quantization scales already require a
        model restart for replacement. Keep each rank's original scale; TP
        shards are independently quantized and must not share a scalar.
        """
        # Engine ParallelismConfig supplies RoleType enums. Standalone callers
        # may supply their string names; normalize both at this public boundary.
        if str(role).rsplit(".", 1)[-1] not in ("PREFILL", "PDFUSION"):
            return
        if self.column_scales is not None:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare vocabulary column scales before capture")
        if not (
            self.tp_size == 2
            and self.hidden_size == 1280
            and torch.version.hip is None
            and torch.cuda.get_device_capability(self.device) == (12, 0)
        ):
            return
        scales = all_gather(self.weight_scale, group=Group.TP)
        if not (
            scales.device == self.device
            and scales.dtype == torch.float32
            and scales.numel() == self.tp_size
            and scales.is_contiguous()
        ):
            raise ValueError("invalid TP vocabulary column scales")
        self.column_scales = scales.reshape(self.tp_size)
        self.weight_scale._rtp_requires_restart_for_update = True

    def forward(
        self,
        input: torch.Tensor,
        position_ids=None,
        token_types=None,
        text_tokens_mask=None,
    ) -> torch.Tensor:
        if (
            input.device != self.device
            or input.ndim != 1
            or input.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("FP8 vocabulary lookup requires packed CUDA token ids")
        if text_tokens_mask is not None and text_tokens_mask.numel():
            if (
                text_tokens_mask.device != input.device
                or text_tokens_mask.shape != input.shape
            ):
                raise ValueError("vocabulary text mask must match packed token ids")
        from rtp_llm.models_py.triton_kernels.common.fp8_vocab_lookup import vocab_rows

        if self.rocm_decode_table is not None:
            from .rocm_fp8 import vocab_rows as rocm_vocab_rows

            output = rocm_vocab_rows(
                input,
                self.weight,
                self.weight_scale,
                self.rocm_decode_table,
                text_tokens_mask,
            )
        else:
            output = vocab_rows(input, self.weight, self.weight_scale, text_tokens_mask)
        if self.tp_size > 1:
            rows, width = output.shape
            output = all_gather(output, group=Group.TP)
            output = (
                output.reshape(self.tp_size, rows, width)
                .transpose(0, 1)
                .contiguous()
                .reshape(rows, -1)
            )
        return output

    def forward_hyper(self, input, *, branches, text_tokens_mask=None):
        """Target-only embedding consumer; keep MTP's ordinary hidden width."""
        if (
            input.device != self.device
            or input.ndim != 1
            or input.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("FP8 vocabulary lookup requires packed CUDA token ids")
        if (
            text_tokens_mask is not None
            and text_tokens_mask.numel()
            and (
                text_tokens_mask.device != input.device
                or text_tokens_mask.shape != input.shape
            )
        ):
            raise ValueError("vocabulary text mask must match packed token ids")
        if self.rocm_decode_table is not None:
            value = self.forward(input, text_tokens_mask=text_tokens_mask)
            return value.repeat(1, branches)
        if (
            self.column_scales is not None
            and branches == 4
            and self.tp_size == 2
            and self.hidden_size == 1280
            and 1024 <= input.numel() <= 12288
        ):
            from rtp_llm.models_py.triton_kernels.common.fp8_vocab_gather import (
                dequant_hyper,
                raw_rows,
            )

            local = raw_rows(input, self.weight)
            gathered = all_gather(local, group=Group.TP)
            return dequant_hyper(
                gathered,
                self.column_scales,
                input.numel(),
                self.hidden_size,
                self.tp_size,
                branches,
                text_tokens_mask,
            )
        from rtp_llm.models_py.triton_kernels.common.fp8_vocab_lookup import (
            gather_hyper_rows,
            vocab_rows,
        )

        local = vocab_rows(input, self.weight, self.weight_scale, text_tokens_mask)
        gathered = all_gather(local, group=Group.TP) if self.tp_size > 1 else local
        return gather_hyper_rows(
            gathered, local.shape[0], local.shape[1], self.tp_size, branches
        )
