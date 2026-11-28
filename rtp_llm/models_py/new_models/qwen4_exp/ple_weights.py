"""Module-owned Qwen4 PLE checkpoint storage.

The n-gram table is host-resident and sharded by *checkpoint tensor name*.
Unlike ordinary linear TP, this must never concatenate the global table or
move it with the enclosing model. Runtime kernels remain the existing PLE
consumer; this module owns only checkpoint parameters and validated bindings.
"""

import re
from typing import Optional

import torch
from torch import nn

from rtp_llm.models_py.borrowed_parameters import FixedHostCheckpointStorage
from rtp_llm.models_py.module_base import RtpModule, copy_weight_
from rtp_llm.models_py.read_spec import ReadSpec, unwrap_checkpoint_tensor


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


class Qwen4PLEHostTable(FixedHostCheckpointStorage, RtpModule):
    """Own this TP rank's independent n-gram shards, never a combined table.

    Shapes are discovered as each local shard arrives. Zero-sized placeholders
    preserve Parameter identity without guessing the checkpoint's prime-padded
    number of rows. The first accepted shape is checked against every later
    shard; a caller that has inspected metadata can also supply ``shard_rows``.
    """

    _SHARD = re.compile(r"shard_(0|[1-9][0-9]*)\.weight\Z")

    def __init__(
        self,
        shard_count: int,
        embedding_dim: int,
        *,
        tp_size: int = 1,
        tp_rank: int = 0,
        dtype: torch.dtype = torch.bfloat16,
        source_fp8: bool = False,
        preserve_fp8: bool = False,
        shard_rows: Optional[int] = None,
    ):
        super().__init__()
        self.shard_count = _positive_int(shard_count, "shard_count")
        self.embedding_dim = _positive_int(embedding_dim, "embedding_dim")
        self.tp_size = _positive_int(tp_size, "tp_size")
        if type(tp_rank) is not int or not 0 <= tp_rank < tp_size:
            raise ValueError("tp_rank must lie in [0, tp_size)")
        if shard_count % tp_size:
            raise ValueError("PLE checkpoint shard_count must be divisible by TP")
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("PLE compute dtype must be FP16, BF16 or FP32")
        if type(source_fp8) is not bool or type(preserve_fp8) is not bool:
            raise TypeError("PLE source_fp8 and preserve_fp8 must be bool")
        if preserve_fp8 and not source_fp8:
            raise ValueError("Preserving FP8 PLE requires an FP8 checkpoint source")
        self.compute_dtype = dtype
        self.source_fp8 = source_fp8
        self.preserve_fp8 = preserve_fp8
        self._ft_predequantized = False
        self.shard_rows = (
            None if shard_rows is None else _positive_int(shard_rows, "shard_rows")
        )
        per_rank = shard_count // tp_size
        self.shard_indices = tuple(range(tp_rank * per_rank, (tp_rank + 1) * per_rank))
        self._seen = set()
        self._scaled = set()
        # FP8->BF16 conversion needs the checkpoint's scalar before publication.
        # Stream raw FP8 into its eventual local owner first, regardless of key
        # ordering. Only one local shard is expanded at a time in postprocessing.
        storage_dtype = torch.float8_e4m3fn if source_fp8 else dtype
        for index in self.shard_indices:
            self.register_parameter(
                f"shard_{index}",
                nn.Parameter(
                    torch.empty(0, embedding_dim, dtype=storage_dtype, device="cpu"),
                    requires_grad=False,
                ),
            )
        if source_fp8:
            self.register_buffer(
                "weight_scale", torch.empty((), dtype=torch.float32, device="cpu")
            )
        else:
            self.register_buffer("weight_scale", None)

    def owns_checkpoint_name(self, name: str) -> bool:
        match = self._SHARD.fullmatch(name)
        if match:
            index = int(match.group(1))
            if index >= self.shard_count:
                raise ValueError(f"PLE shard index {index} outside configured table")
            return index in self.shard_indices
        if name == "weight_scale":
            return self.source_fp8
        raise ValueError(f"Unexpected PLE table checkpoint key {name!r}")

    def checkpoint_read_spec(self, name, shape):
        if not self.owns_checkpoint_name(name):
            return ReadSpec.skip(name, tuple(shape))
        if name == "weight_scale":
            if tuple(shape) not in ((), (1,)):
                raise ValueError("PLE global weight scale must be a scalar")
        else:
            self._validate_shape(shape)
        return ReadSpec.full(name, tuple(shape))

    def _validate_shape(self, shape):
        if len(shape) != 2 or shape[0] <= 0 or shape[1] != self.embedding_dim:
            raise ValueError(f"Invalid PLE shard shape {tuple(shape)}")
        if self.shard_rows is not None and shape[0] != self.shard_rows:
            raise ValueError("PLE shards must have equal row counts")

    def load_weights(self, weights):
        iterator = weights.items() if isinstance(weights, dict) else weights
        for name, value in iterator:
            if not self.owns_checkpoint_name(name):
                raise ValueError(
                    f"Non-local or disabled PLE tensor reached loader: {name}"
                )
            if name in self._seen:
                raise ValueError(f"Duplicate PLE checkpoint tensor {name!r}")
            source, spec = unwrap_checkpoint_tensor(value)
            if spec is not None and spec.is_local_shard:
                raise ValueError("PLE table shards must not be split a second time")
            if source.is_meta:
                raise ValueError("PLE source must be materialized")
            if name == "weight_scale":
                # The released FP8 checkpoint stores this global scale as BF16.
                # OldLoader requested load_tensor(..., torch.float32) before
                # validation; native FST and scratch now deliver the raw source
                # dtype. Normalize at the owning module, not in the reader (the
                # neighboring FP8 table shards must retain their original bytes).
                if (
                    source.dtype not in (
                        torch.float16,
                        torch.bfloat16,
                        torch.float32,
                        torch.float64,
                    )
                    or source.numel() != 1
                    or source.ndim > 1
                ):
                    raise ValueError(
                        "PLE scale requires one finite positive FP32-representable "
                        f"floating scalar; got {source.dtype} shape {tuple(source.shape)}"
                    )
                scale = source.reshape(()).to(device="cpu", dtype=torch.float32)
                if not bool(torch.isfinite(scale)) or not bool(scale > 0):
                    raise ValueError(
                        "PLE scale requires one finite positive FP32-representable "
                        f"floating scalar; got {source.dtype} value {source.tolist()}"
                    )
                self.weight_scale.copy_(scale)
                self._mark_weight_loaded("weight_scale")
            else:
                self._validate_shape(source.shape)
                if self.source_fp8:
                    if source.dtype != torch.float8_e4m3fn:
                        raise TypeError(
                            "FP8 PLE checkpoint shards must retain E4M3 bytes"
                        )
                elif source.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    raise TypeError(
                        "Unquantized PLE shard requires floating checkpoint values"
                    )
                parameter_name = name.removesuffix(".weight")
                target = self._parameters[parameter_name]
                owned = target.data if tuple(target.shape) == tuple(source.shape) else torch.empty(source.shape, dtype=target.dtype, device="cpu")
                copy_weight_(owned, source, name)
                # Parameters stay stable, but the initial zero-sized allocation
                # becomes a single correctly sized local shard. No shared FST
                # backing buffer is retained after its reader closes.
                target.data = owned
                self.shard_rows = source.shape[0]
                self._mark_weight_loaded(parameter_name)
            self._seen.add(name)

    def load_ft_shards(self, checkpoint, layer_id):
        """Import this rank's OldLoader runtime shards without touching peer ranks.

        The old exporter drops the scale after conversion to BF16. Import that
        table as a materialized runtime value; applying an HF scale again corrupts it.
        Return consumed names for the enclosing model adapter's completeness check.
        """
        if self._seen:
            raise RuntimeError("PLE FT restore requires a fresh table")
        if type(layer_id) is not int or layer_id < 0:
            raise ValueError("invalid PLE FT layer id")
        from rtp_llm.utils.model_weight import W

        prefix = f"layers.{layer_id}."
        tables = {
            f"{prefix}{W.qwen4_ple_ngram_shards}.{i}": i for i in self.shard_indices
        }
        scale = prefix + W.qwen4_ple_ngram_scale
        owned = set(tables) | ({scale} if self.preserve_fp8 else set())
        actual = {
            k
            for k in checkpoint.entries
            if k.startswith(prefix + W.qwen4_ple_ngram_shards + ".") or k == scale
        }
        if actual != owned:
            raise ValueError(
                f"PLE FT shard/scale ownership mismatch: missing={owned - actual}, unexpected={actual - owned}"
            )
        if self.preserve_fp8:
            self.load_weights({"weight_scale": checkpoint.read(scale)})
        self._ft_predequantized = self.source_fp8 and not self.preserve_fp8
        if self._ft_predequantized:
            self.weight_scale = None
        for key, index in tables.items():
            tensor = checkpoint.read(key)
            name = f"shard_{index}"
            dtype = torch.float8_e4m3fn if self.preserve_fp8 else self.compute_dtype
            if tensor.device.type != "cpu" or tensor.dtype != dtype:
                raise ValueError(
                    f"Unexpected PLE FT storage for {key}: {tensor.device}/{tensor.dtype}"
                )
            if self._ft_predequantized:
                self._validate_shape(tensor.shape)
                self._parameters[name].data = tensor.detach().clone()
                self.shard_rows = tensor.shape[0]
                self._mark_weight_loaded(name)
                self._seen.add(name + ".weight")
                self._scaled.add(index)
            else:
                self.load_weights({name + ".weight": tensor})
        self.validate_weights_loaded()
        return owned

    def validate_weights_loaded(self, loaded_tensor_ids=None):
        expected = {f"shard_{index}.weight" for index in self.shard_indices}
        if self.source_fp8 and not self._ft_predequantized:
            expected.add("weight_scale")
        missing = expected - self._seen
        if missing:
            raise RuntimeError(f"Missing PLE checkpoint tensors: {sorted(missing)}")
        super().validate_weights_loaded(loaded_tensor_ids)

    def process_weights_after_loading(self):
        self.validate_weights_loaded()
        if self.source_fp8 and not self.preserve_fp8 and not self._ft_predequantized:
            # Match OldLoader exactly: cast values AND scalar to compute dtype
            # before multiplication, not FP32 dequant followed by a final cast.
            scale = self.weight_scale.to(dtype=self.compute_dtype)
            for index in self.shard_indices:
                if index not in self._scaled:
                    target = self._parameters[f"shard_{index}"]
                    target.data = target.to(dtype=self.compute_dtype) * scale
                    self._scaled.add(index)

    def runtime_shards(self):
        self.validate_weights_loaded()
        if self.source_fp8 and not self.preserve_fp8:
            if len(self._scaled) != len(self.shard_indices):
                raise RuntimeError(
                    "PLE dequantization must finish before runtime binding"
                )
        return [self._parameters[f"shard_{index}"] for index in self.shard_indices]

    def adopt_runtime_shards(self, shards):
        """Publish the existing PLE UVA preparer's host allocations atomically.

        CUDA PLE preparation may pin/copy the pageable table. The owning module
        must adopt those exact returned allocations so it does not retain a
        second complete local table. Parameter objects remain stable; bindings
        are validated before any data pointer changes. This is startup-only,
        not permission to update graph-bound PLE weights during inference.
        """
        parameters = self.runtime_shards()
        if len(shards) != len(parameters):
            raise ValueError("PLE runtime must return every local shard")
        for target, source in zip(parameters, shards):
            if (
                not isinstance(source, torch.Tensor)
                or source.device.type != "cpu"
                or source.dtype != target.dtype
                or source.shape != target.shape
                or not source.is_contiguous()
            ):
                raise ValueError(
                    "PLE runtime shard must preserve CPU shape/dtype/layout"
                )
        for target, source in zip(parameters, shards):
            target.data = source.detach()
            target._rtp_requires_restart_for_update = True


class Qwen4PLEEmbeddingWeights(RtpModule):
    """Checkpoint-shaped PLE embedding subtree; integer metadata never casts."""

    def __init__(
        self,
        *,
        ngram_size,
        heads_per_ngram,
        shard_count,
        embedding_dim,
        tp_size=1,
        tp_rank=0,
        dtype=torch.bfloat16,
        source_fp8=False,
        preserve_fp8=False,
        shard_rows=None,
    ):
        super().__init__()
        if _positive_int(ngram_size, "ngram_size") <= 1:
            raise ValueError("ngram_size must exceed one")
        self.ngram_size = ngram_size
        heads = (ngram_size - 1) * _positive_int(heads_per_ngram, "heads_per_ngram")
        self.register_buffer(
            "layer_multipliers", torch.empty(ngram_size, dtype=torch.int64)
        )
        self.register_buffer(
            "ngram_heads_offsets", torch.empty(heads, dtype=torch.int64)
        )
        self.register_buffer(
            "ngram_heads_vocab_sizes", torch.empty(heads, dtype=torch.int64)
        )
        self.ngram_embedding = Qwen4PLEHostTable(
            shard_count,
            embedding_dim,
            tp_size=tp_size,
            tp_rank=tp_rank,
            dtype=dtype,
            source_fp8=source_fp8,
            preserve_fp8=preserve_fp8,
            shard_rows=shard_rows,
        )

    def load_weights(self, weights):
        # A custom owning subtree participates in NewModelLoader's pre-migration
        # completeness pass. It validates the coupled integer metadata/table
        # geometry, rather than waiting for a runtime constructor to notice.
        super().load_weights(weights)

    def checkpoint_read_spec(self, name, shape):
        if name.startswith("ngram_embedding."):
            return self.ngram_embedding.checkpoint_read_spec(
                name[len("ngram_embedding.") :], shape
            )
        return None

    def validate_weights_loaded(self, loaded_tensor_ids=None):
        super().validate_weights_loaded(loaded_tensor_ids)
        self.ngram_embedding.validate_weights_loaded(loaded_tensor_ids)
        sizes = self.ngram_heads_vocab_sizes
        offsets = self.ngram_heads_offsets
        if not bool((sizes > 0).all()) or not bool((offsets >= 0).all()):
            raise ValueError("PLE vocab sizes must be positive and offsets nonnegative")
        if bool((offsets[1:] < offsets[:-1] + sizes[:-1]).any()):
            raise ValueError("PLE n-gram head ranges must not overlap")
        capacity = self.ngram_embedding.shard_rows * self.ngram_embedding.shard_count
        if int(offsets[-1]) + int(sizes[-1]) > capacity:
            raise ValueError("PLE checkpoint shards do not cover hashed vocabulary")

    def runtime_arguments(self, *, eos_token_id, prefetch_budget_mb=256):
        """Bind directly to the existing Qwen4ExpNGramEmbedding constructor."""
        self.validate_weights_loaded()
        table = self.ngram_embedding
        return dict(
            shards=table.runtime_shards(),
            vocab_sizes=self.ngram_heads_vocab_sizes,
            offsets=self.ngram_heads_offsets,
            multipliers=self.layer_multipliers,
            ngram_size=self.ngram_size,
            eos_token_id=eos_token_id,
            shard_indices=list(table.shard_indices),
            total_shards=table.shard_count,
            distributed_reduce=table.tp_size > 1,
            weight_scale=table.weight_scale if table.preserve_fp8 else None,
            prefetch_budget_mb=prefetch_budget_mb,
        )

    def checkpoint_name_filter(self, prefix: str):
        """Compose into the enclosing model's pre-materialization name filter.

        ``prefix`` ends at this embedding subtree (for example
        ``model.layers.0.ple.ple_embedding.``). Other model tensors pass through;
        rank-external table keys are excluded before file/payload selection.
        """
        if not isinstance(prefix, str) or not prefix or not prefix.endswith("."):
            raise ValueError("PLE checkpoint prefix must be nonempty and end in '.'")
        table_prefix = prefix + "ngram_embedding."

        def should_load(name):
            if name.startswith(table_prefix):
                return self.ngram_embedding.owns_checkpoint_name(
                    name[len(table_prefix) :]
                )
            return True

        return should_load
