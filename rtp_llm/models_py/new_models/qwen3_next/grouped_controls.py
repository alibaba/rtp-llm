"""Qwen3.5 checkpoint importer for GDN's two small, floating B/A control vectors."""

import torch

from rtp_llm.models_py.quant_methods.groupwise import groupwise_settings
from rtp_llm.models_py.quant_methods.groupwise_checkpoint import (
    GroupwiseCheckpointLayout,
    PackedGroupwiseWeights,
)
from rtp_llm.models_py.read_spec import unwrap_checkpoint_tensor


class GroupedControlSource:
    def __init__(self, quant, prefix, hidden, heads, dtype):
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("GDN controls require a floating compute dtype")
        self.method, self.bits, self.group = groupwise_settings(quant)
        self.hidden, self.heads, self.dtype = hidden, heads, dtype
        self.ignored = {
            p
            for p in ("a", "b", "ba")
            if quant.is_layer_ignored(f"{prefix}.in_proj_{p}")
        }
        self.layout = None
        self.encoding = {}
        self.seen = set()
        self.pending = {}

    def consume(self, component, field, tensor):
        """Return [N,K] only once weight data is complete; validate late metadata too."""
        tensor, spec = unwrap_checkpoint_tensor(tensor)
        if spec is not None:
            raise ValueError(
                "GDN control projection requires a complete physical source"
            )
        layout = "fused" if component == "ba" else "split"
        encoding = "float" if field == "weight" else "packed"
        if (
            self.layout not in (None, layout)
            or self.encoding.setdefault(component, encoding) != encoding
        ):
            raise ValueError("Mixed physical layouts for the same GDN B/A projection")
        self.layout = layout
        if (component, field) in self.seen:
            raise ValueError(f"Duplicate GDN in_proj_{component}.{field}")
        self.seen.add((component, field))
        n = self.heads * (2 if component == "ba" else 1)
        if encoding == "float":
            if tensor.dtype not in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
            ) or tensor.shape != (n, self.hidden):
                raise ValueError("Invalid floating GDN control projection")
            return tensor
        # A fused packed archive cannot override even one explicitly floating half.
        if (
            "ba" in self.ignored
            or component in self.ignored
            or (component == "ba" and self.ignored)
        ):
            raise ValueError(f"Packed source for excluded GDN in_proj_{component}")
        physical = GroupwiseCheckpointLayout(
            self.method, self.bits, self.group, self.hidden, n
        )
        if field == "g_idx":
            identity = (
                torch.arange(self.hidden, dtype=torch.int32, device=tensor.device)
                // self.group
            )
            if (
                self.method != "gptq"
                or tensor.dtype != torch.int32
                or not torch.equal(tensor, identity)
            ):
                raise ValueError("Invalid or activation-ordered GDN g_idx")
            return None
        if field == "bias":
            if (
                self.method != "gptq"
                or tensor.shape != (n,)
                or tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32)
                or torch.count_nonzero(tensor)
            ):
                raise ValueError(
                    "Grouped bias for a biasless GDN projection must be GPTQ zero bias"
                )
            return None
        if field not in physical.shapes or tensor.shape != physical.shapes[field]:
            raise ValueError(
                f"Invalid grouped GDN {component}.{field} shape: {tuple(tensor.shape)}"
            )
        if field == "scales":
            if (
                tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32)
                or not torch.isfinite(tensor).all()
                or (tensor < 0).any()
            ):
                raise ValueError("Invalid GDN group scales")
        elif tensor.dtype != torch.int32:
            raise ValueError(f"GDN {field} requires int32 storage")
        # The file handle can close between these three fields. Keep only small
        # control-projection shards, never the larger GDN QKVZ/out projections.
        state = self.pending.setdefault(component, {})
        state[field] = tensor.detach().clone()
        if len(state) != 3:
            return None
        packed = PackedGroupwiseWeights.from_tensors(
            **self.pending.pop(component),
            method=self.method,
            bits=self.bits,
            group_size=self.group,
        )
        return packed.decode_control_weight(self.dtype).T
