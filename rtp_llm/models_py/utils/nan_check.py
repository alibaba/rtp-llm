from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from rtp_llm.ops.compute_ops import PyModelOutputs


class NanCheckStage(IntEnum):
    ATTENTION = 1
    FFN = 2
    MODEL_OUTPUT = 3
    RAW_LOGITS = 4


@dataclass(frozen=True)
class NanCheckMetadata:
    records: torch.Tensor
    layer_id: int
    stage_id: int


def detect_nan(tensor: torch.Tensor, metadata: Optional[NanCheckMetadata]) -> None:
    """Record each token's first nonfinite value, without synchronizing to CPU.

    Records are int64 [tokens, 4]: layer, stage, flattened feature index, kind.
    Kind is 0 for clean, 1 for NaN, 2 for +Inf, and 3 for -Inf. Calls sharing
    records must execute in order on the tensor's producer stream.
    """
    if metadata is None:
        return
    detect_nan_tensor(tensor, metadata.records, metadata.layer_id, metadata.stage_id)


def detect_nan_tensor(
    tensor: torch.Tensor,
    records: Optional[torch.Tensor],
    layer_id: int,
    stage_id: int,
) -> None:
    """Tensor-only entry point shared with the C++ model wrapper."""
    if records is None or records.numel() == 0 or tensor.numel() == 0:
        return
    if not tensor.is_floating_point():
        return
    tokens = records.shape[0]
    values = tensor.reshape(tokens, -1)
    if values.is_cuda:
        from rtp_llm.models_py.triton_kernels.common.nan_check import detect_nan_cuda

        detect_nan_cuda(values, records, layer_id, int(stage_id))
        return

    # CPU reference also supports the CPU model path and inexpensive unit tests.
    bad = ~torch.isfinite(values)
    first = bad.to(torch.int8).argmax(dim=1)
    value = values.gather(1, first[:, None]).squeeze(1)
    kind = torch.where(torch.isnan(value), 1, torch.where(value > 0, 2, 3))
    candidate = torch.stack(
        (
            torch.full_like(first, layer_id),
            torch.full_like(first, int(stage_id)),
            first,
            kind,
        ),
        dim=1,
    )
    write = (records[:, 3] == 0) & bad.any(dim=1)
    records.copy_(torch.where(write[:, None], candidate, records))


def finish_nan_check(
    outputs: "PyModelOutputs", records: Optional[torch.Tensor], num_layers: int
) -> "PyModelOutputs":
    if records is not None and records.numel() != 0:
        detect_nan_tensor(
            outputs.hidden_states, records, num_layers, NanCheckStage.MODEL_OUTPUT
        )
        outputs.nan_metadata = records
    return outputs
