"""Stage-local, named-only loading of Engram WKV and gate weights."""

from dataclasses import dataclass

import torch

from rtp_llm.models_py.modules.dsv4.v41_engram import V41Engram
from rtp_llm.models_py.modules.dsv4.v41_engram_host import (
    HostEngramEmbedding,
    V41EngramTableSource,
    plan_stage_engram_tables,
)
from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import V41MXFP8Linear, _require_sm120
from rtp_llm.utils.safetensors_header import SafetensorsSpan, named_safetensors_spans


@dataclass(frozen=True)
class V41EngramLayerSource:
    table: V41EngramTableSource
    weight: SafetensorsSpan
    scales: SafetensorsSpan
    q_weight: SafetensorsSpan
    k_weight: SafetensorsSpan
    norm_eps: float

    @property
    def layer_id(self):
        return self.table.layer_id

    @property
    def spans(self):
        return (self.weight, self.scales, self.q_weight, self.k_weight)

    @property
    def weight_bytes(self):
        return sum(span.size for span in self.spans)

    @property
    def packed_scale_bytes(self):
        n, k = self.weight.shape
        return ((n + 127) // 128 * 128) * ((k // 32 + 3) // 4 * 4)

    def check_sources(self):
        self.table.check_sources()
        for span in self.spans:
            span.check_source()


def plan_stage_engram_sources(config, checkpoint, owned_layer_ids, checkpoint_sha256):
    """Validate all owned table/projection headers before allocating any weights."""
    tables = plan_stage_engram_tables(
        config, checkpoint, owned_layer_ids, checkpoint_sha256
    )
    if not tables:
        return {}
    text = config.text
    dim, lanes = text["hidden_size"], text["hc_mult"]
    k = (
        (text["engram_max_ngram_size"] - 1)
        * text["engram_n_heads"]
        * text["engram_head_dim"]
    )
    n = (lanes + 1) * dim
    expected = {}
    for layer_id in tables:
        prefix = f"layers.{layer_id}.engram."
        expected.update(
            {
                prefix + "wkv.weight": ((n, k), "F8_E4M3"),
                prefix + "wkv.scale": (((n + 31) // 32, k // 32), "F8_E8M0"),
                prefix + "q_weight": ((lanes, dim), "BF16"),
                prefix + "k_weight": ((lanes, dim), "BF16"),
            }
        )
    spans = named_safetensors_spans(checkpoint, expected)
    result = {}
    for layer_id, table in tables.items():
        prefix = f"layers.{layer_id}.engram."
        result[layer_id] = V41EngramLayerSource(
            table,
            *(
                spans[prefix + suffix]
                for suffix in ("wkv.weight", "wkv.scale", "q_weight", "k_weight")
            ),
            norm_eps=text["rms_norm_eps"],
        )
        result[layer_id].check_sources()
    return result


def _load_tensor(span: SafetensorsSpan, device):
    # Read directly into the only CPU staging allocation, never mmap a full shard.
    raw = torch.empty(span.size, dtype=torch.uint8, device="cpu")
    span.read_into(raw.numpy())
    dtype = {
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E8M0": torch.uint8,
        "BF16": torch.bfloat16,
    }[span.dtype]
    return raw.view(dtype).reshape(span.shape).to(device=device, non_blocking=False)


def load_engram_weights(source: V41EngramLayerSource, *, max_bytes: int, device="cpu"):
    """Load exactly four tensors, retaining UE8M0 bytes and native checkpoint dtypes.

    max_bytes admits stored weight allocations; FlashInfer preparation and provider
    workspace are separate. CPU staging holds at most one tensor during CUDA load.
    """
    if type(max_bytes) is not int or max_bytes < source.weight_bytes:
        raise ValueError("Engram projection exceeds the admitted weight byte budget")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Engram weights require a CPU or CUDA device")
    if device.type == "cuda":
        _require_sm120(device)
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Engram weight loading must precede graph capture")
    source.check_sources()
    loaded = {
        name: _load_tensor(getattr(source, name), device)
        for name in ("weight", "scales", "q_weight", "k_weight")
    }
    source.check_sources()
    return loaded


def load_engram_module(
    source, embedding, *, device, max_bytes: int, chunk_tokens=128, backend="flashinfer"
):
    """Attach small weights only to their matching table owner and checkpoint."""
    if (
        not isinstance(embedding, HostEngramEmbedding)
        or embedding._table.source != source.table
    ):
        raise ValueError("Engram weights and embedding must share one admitted source")
    if backend not in ("flashinfer", "eager"):
        raise ValueError("Unknown Engram projection backend")
    device = torch.device(device)
    if (backend == "eager") != (device.type == "cpu"):
        raise ValueError("Engram eager backend is CPU-only; FlashInfer requires CUDA")
    if type(chunk_tokens) is not int or chunk_tokens <= 0:
        raise ValueError("Engram chunk_tokens must be a positive integer")
    required = source.weight_bytes + (
        source.packed_scale_bytes if backend == "flashinfer" else 0
    )
    if type(max_bytes) is not int or max_bytes < required:
        raise ValueError(
            "Engram module exceeds the admitted weight and packed-scale byte budget"
        )
    with embedding._table._mutex:
        embedding._table._check_open()
        weights = load_engram_weights(source, max_bytes=max_bytes, device=device)
        projection = V41MXFP8Linear(
            weights["weight"], weights["scales"], backend=backend
        )
        return V41Engram(
            embedding,
            projection,
            weights["q_weight"],
            weights["k_weight"],
            eps=source.norm_eps,
            chunk_tokens=chunk_tokens,
        )
