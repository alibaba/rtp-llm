"""Canonical model-input views and fixed graph buffers for V4.1 components."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class V41ModelRows:
    token_ids: torch.Tensor
    token_types: torch.Tensor
    valid: torch.Tensor
    history_ids: torch.Tensor
    history_valid: torch.Tensor

    @classmethod
    def from_model_inputs(cls, inputs):
        result = cls(
            inputs.input_ids,
            inputs.v41_token_types,
            inputs.v41_token_valid,
            inputs.engram_history_ids,
            inputs.engram_history_valid,
        )
        result.validate()
        return result

    def validate(self):
        count = self.token_ids.numel()
        if (
            self.token_ids.ndim != 1
            or self.token_ids.dtype != torch.int32
            or not self.token_ids.is_contiguous()
        ):
            raise ValueError("V4.1 canonical input IDs must be one-dimensional int32")
        for tensor, shape, dtype in (
            (self.token_types, (count,), torch.int32),
            (self.valid, (count,), torch.bool),
            (self.history_ids, (count, 3), torch.int32),
            (self.history_valid, (count, 3), torch.bool),
        ):
            if (
                tensor.shape != shape
                or tensor.dtype != dtype
                or tensor.device != self.token_ids.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    "V4.1 metadata must have exact row shape, dtype, contiguity and device"
                )
        torch._assert_async(
            (((self.token_ids >= 0) & (self.token_ids < 129280)) | ~self.valid).all(),
            "V4.1 canonical ID outside vocabulary",
        )
        torch._assert_async(
            ((self.token_types >= -1) & (self.token_types <= 3)).all(),
            "invalid V4.1 image type",
        )
        torch._assert_async(
            (~self.image_mask | (self.token_ids == 129264)).all(),
            "image span lost canonical ID 129264",
        )
        torch._assert_async(
            ((self.token_types == -1) | self.valid).all(),
            "padding cannot retain image token types",
        )
        torch._assert_async(
            (~self.history_valid | self.valid[:, None]).all(),
            "padding cannot retain Engram history",
        )
        torch._assert_async(
            (
                ((self.history_ids >= 0) & (self.history_ids < 129280))
                | ~self.history_valid
            ).all(),
            "Engram history ID outside vocabulary",
        )

    @property
    def image_mask(self):
        return self.valid & (self.token_types >= 0)

    @property
    def text_mask(self):
        return self.valid & (self.token_types == -1)

    def engram_hashes(self, hasher):
        self.validate()
        return hasher(
            self.token_ids.to(torch.int64)[:, None],
            self.history_ids.to(torch.int64),
            self.history_valid,
            self.text_mask[:, None],
        )[:, 0]


class V41GraphInputBuffers:
    """Fixed input storage; callers bind its lifetime to their actual graph owner."""

    def __init__(self, capacity, *, device):
        if type(capacity) is not int or capacity < 0:
            raise ValueError("V4.1 graph capacity must be a nonnegative integer")
        self.rows = V41ModelRows(
            torch.zeros(capacity, dtype=torch.int32, device=device),
            torch.full((capacity,), -1, dtype=torch.int32, device=device),
            torch.zeros(capacity, dtype=torch.bool, device=device),
            torch.zeros((capacity, 3), dtype=torch.int32, device=device),
            torch.zeros((capacity, 3), dtype=torch.bool, device=device),
        )

    def update(self, rows):
        rows.validate()
        count = rows.token_ids.numel()
        if (
            count > self.rows.token_ids.numel()
            or rows.token_ids.device != self.rows.token_ids.device
        ):
            raise ValueError(
                "V4.1 graph input exceeds its capacity or uses another device"
            )
        for destination, source, padding in (
            (self.rows.token_ids, rows.token_ids, 0),
            (self.rows.token_types, rows.token_types, -1),
            (self.rows.valid, rows.valid, False),
            (self.rows.history_ids, rows.history_ids, 0),
            (self.rows.history_valid, rows.history_valid, False),
        ):
            destination[:count].copy_(source)
            destination[count:].fill_(padding)
        return self.rows
