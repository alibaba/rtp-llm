"""Owned V4.1 Engram weights, reusing the current-main hashing/gating runtime."""

import torch
from torch import nn

from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules.dsv41.engram import (
    Engram,
    EngramLayout,
    HostEngramEmbedding,
    _PinnedSharedTable,
    _ReferenceWKV,
    dequantize_block_weight,
)
from rtp_llm.models_py.new_models.deepseek_v41.engram_source import EngramSource
from rtp_llm.models_py.read_spec import ReadSpec, unwrap_checkpoint_tensor


class OwnedEngramHostTable(RtpModule):
    """Raw host Parameters keep the source mapping alive, without a table copy.

    The pre-materialization source planner must route these leaves to CPU.
    Keeping a destination on CPU alone cannot prevent FST GPU file staging.
    """

    def __init__(self, rows, width):
        super().__init__()
        if (
            type(rows) is not int
            or rows <= 0
            or type(width) is not int
            or width <= 0
            or width % 32
        ):
            raise ValueError(
                "Engram table requires positive rows and width divisible by 32"
            )
        self.shape = (rows, width)
        self.weight = nn.Parameter(
            torch.empty(0, dtype=torch.float8_e4m3fn, device="cpu"), requires_grad=False
        )
        self.scale = nn.Parameter(
            torch.empty(0, dtype=torch.float8_e8m0fnu, device="cpu"),
            requires_grad=False,
        )

    def _apply(self, fn, recurse=True):
        return self

    def checkpoint_source_device(self, name):
        if name not in ("weight", "scale"):
            raise ValueError(f"Unknown Engram table checkpoint leaf: {name}")
        return "cpu"

    def checkpoint_read_spec(self, name, shape):
        expected = {
            "weight": self.shape,
            "scale": (self.shape[0], self.shape[1] // 32),
        }
        if name not in expected or tuple(shape) != expected[name]:
            raise ValueError(f"Invalid Engram table {name} shape {tuple(shape)}")
        return ReadSpec.full(name, tuple(shape))

    def load_weights(self, weights):
        items = weights.items() if isinstance(weights, dict) else weights
        for name, value in items:
            tensor, applied = unwrap_checkpoint_tensor(value)
            self.checkpoint_read_spec(name, tuple(tensor.shape))
            if applied is not None and tuple(applied.source_shape) != tuple(
                tensor.shape
            ):
                raise ValueError("Engram host table cannot accept a sliced checkpoint")
            if tensor.device.type != "cpu":
                raise RuntimeError("Engram source tables must be materialized on CPU")
            if not tensor.is_contiguous():
                raise ValueError("Engram host table storage must be contiguous")
            parameter = getattr(self, name)
            if tensor.dtype != parameter.dtype:
                raise TypeError(
                    f"Engram {name} requires {parameter.dtype}, got {tensor.dtype}"
                )
            if name in getattr(self, "_rtp_loaded_weight_names", set()):
                raise RuntimeError(f"Duplicate Engram table {name}")
            # Own the mmap/storage through the Parameter; preserve identity for
            # aliases without materializing a second ~94GiB table allocation.
            parameter.data = tensor
            self._mark_weight_loaded(name)


class OwnedEngramProjection(RtpModule):
    """Native 32x32 E4M3/UE8M0 projection; CPU uses main's tiny reference."""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.scale = nn.Parameter(
            torch.empty(
                (output_size + 31) // 32,
                (input_size + 31) // 32,
                dtype=torch.float8_e8m0fnu,
            ),
            requires_grad=False,
        )
        object.__setattr__(self, "_executor", None)

    @property
    def v41_runtime_linear(self):
        # The native WKV is deliberately not a registered child. Expose its
        # warmup contract without registering a second owner or advertising
        # the CPU reference as a group32 CUDA executor.
        return getattr(self._executor, "v41_runtime_linear", None)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse)
        object.__setattr__(self, "_executor", None)
        return result

    def validate_runtime_device(self, device):
        device = torch.device(device)
        if device.type == "cpu":
            return
        if device.type != "cuda" or getattr(torch.version, "hip", None) is not None:
            raise RuntimeError("Engram native MXFP8 requires CUDA")
        if torch.cuda.get_device_capability(device)[0] not in (10, 12):
            raise RuntimeError("Engram native MXFP8 requires DeepGEMM E8M0 device mode")
        try:
            import deep_gemm
            from deep_gemm.utils.layout import (
                get_mn_major_tma_aligned_packed_ue8m0_tensor,
            )
        except (ImportError, OSError, AttributeError) as exc:
            raise RuntimeError(
                "Engram cannot import its required DeepGEMM runtime"
            ) from exc
        if not callable(getattr(deep_gemm, "fp8_fp4_gemm_nt", None)) or not callable(
            get_mn_major_tma_aligned_packed_ue8m0_tensor
        ):
            raise RuntimeError(
                "Engram requires native MXFP8 GEMM and UE8M0 layout support"
            )

    def process_weights_after_loading(self):
        self.validate_weights_loaded()
        self.validate_runtime_device(self.weight.device)
        if self.weight.device.type == "cpu":
            executor = _ReferenceWKV(dequantize_block_weight(self.weight, self.scale))
        else:
            from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear

            executor = V41MXFP8Linear(self.weight, self.scale)
        object.__setattr__(self, "_executor", executor)

    def forward(self, value):
        if self._executor is None:
            raise RuntimeError(
                "Engram projection must be finalized after checkpoint loading"
            )
        return self._executor(value)


class OwnedEngram(Engram, RtpModule):
    """Own the checkpoint, but inherit the serving gate's dispatch for every layout."""

    def __init__(self, config, layer_id, *, checkpoint_path=None):
        RtpModule.__init__(self)
        self.layout = EngramLayout(config)
        self.layer_hash_index = self.layout.layer_ids.index(layer_id)
        self.layer_id = layer_id
        hidden = int(config["hidden_size"])
        hc = int(config["hc_mult"])
        self.embed = OwnedEngramHostTable(
            self.layout.num_embeddings[self.layer_hash_index], self.layout.head_dim
        )
        self.wkv = OwnedEngramProjection(
            self.layout.n_hash_cols * self.layout.head_dim, (hc + 1) * hidden
        )
        self.q_weight = nn.Parameter(
            torch.empty(hc, hidden, dtype=torch.bfloat16), requires_grad=False
        )
        self.k_weight = nn.Parameter(
            torch.empty(hc, hidden, dtype=torch.bfloat16), requires_grad=False
        )
        self.eps = float(config["rms_norm_eps"])
        self.checkpoint_path = checkpoint_path
        self.embed_tokens = None
        self._pinned = None

    @classmethod
    def from_checkpoint(cls, config, layer_id, checkpoint_path, device, *, load_method, resident_weights=None):
        """Initialize only this local layer, after the backbone resolves its loader.

        The backbone excludes Engram keys from its general shard stream and
        invokes this at the existing per-stage Engram binding point.
        """
        model = cls(config, layer_id, checkpoint_path=checkpoint_path)
        device = torch.device(device)
        model.wkv.validate_runtime_device(device)
        rows, width = model.embed.shape
        specs = {
            "embed.weight": ((rows, width), "F8_E4M3", torch.float8_e4m3fn),
            "embed.scale": ((rows, width // 32), "F8_E8M0", torch.float8_e8m0fnu),
            "q_weight": (tuple(model.q_weight.shape), "BF16", model.q_weight.dtype),
            "k_weight": (tuple(model.k_weight.shape), "BF16", model.k_weight.dtype),
            "wkv.weight": (
                tuple(model.wkv.weight.shape),
                "F8_E4M3",
                model.wkv.weight.dtype,
            ),
            "wkv.scale": (
                tuple(model.wkv.scale.shape),
                "F8_E8M0",
                model.wkv.scale.dtype,
            ),
        }
        # Header validation and tensor reads belong to cold builders only. On a
        # hit use the retained host table and the four small resident GPU views.
        from rtp_llm.resident_weights import resident_init_scalar, resident_init_tensors
        source = None

        def checkpoint():
            nonlocal source
            if source is None:
                source = EngramSource(checkpoint_path, layer_id, specs, load_method)
            return source

        group = f"native.engram.{layer_id}"

        def table_pair():
            return checkpoint().load("embed.weight"), checkpoint().load("embed.scale")

        model.to(device=device)
        if device.type == "cuda":
            # Main's flock winner alone reads/stages the host table on a miss.
            # Attaching ranks bind the resulting pages, never allocating or
            # reading their own ~94 GiB copy.
            identity = resident_init_scalar(
                resident_weights, group + ".host", lambda: checkpoint().cache_identity()
            )
            model._pinned = _PinnedSharedTable(
                None,
                None,
                identity,
                device,
                loader=table_pair if load_method == "scratch" else None,
                load_into=(
                    (lambda mapping: checkpoint().fill_shared_table(mapping))
                    if load_method == "fastsafetensors"
                    else None
                ),
                table_shape=model.embed.shape,
                require_existing=resident_weights is not None and source is None,
                resident_manager=getattr(resident_weights, "manager", None),
            )
            pair = model._pinned.weight, model._pinned.scales.view(torch.float8_e8m0fnu)
        else:
            pair = table_pair()
        model.embed.load_weights(zip(("weight", "scale"), pair))
        names = ("q_weight", "k_weight", "wkv.weight", "wkv.scale")
        if resident_weights is None:
            model.load_weights((name, checkpoint().load(name)) for name in names)
        else:
            values = resident_init_tensors(
                resident_weights, group + ".small", tuple(group + "." + n for n in names), (),
                lambda: tuple(checkpoint().load(n).to(device) for n in names)
            )
            for name, value in zip(names, values):
                param = model.get_parameter(name)
                if param.shape != value.shape or param.dtype != value.dtype or value.device != param.device:
                    raise ValueError(f"Incompatible resident Engram tensor {name}")
                param.data = value
                prefix, _, key = name.rpartition(".")
                model.get_submodule(prefix)._mark_weight_loaded(key)
        model.wkv.process_weights_after_loading()
        model.process_weights_after_loading()
        return model

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse)
        self.embed_tokens = None
        return result

    def process_weights_after_loading(self):
        self.validate_weights_loaded()
        self.embed.validate_weights_loaded()
        self.wkv.validate_weights_loaded()
        if self.q_weight.device.type == "cuda" and self._pinned is None:
            raise RuntimeError(
                "V4.1 Engram requires its shared host source before CUDA execution"
            )
        self.embed_tokens = HostEngramEmbedding(
            self.embed.weight, self.embed.scale, pinned=self._pinned
        )

    def forward(self, hidden, hash_ids, token_mask=None):
        if self.embed_tokens is None:
            raise RuntimeError(
                "Engram host lookup must be finalized after checkpoint loading"
            )
        return super().forward(hidden, hash_ids, token_mask)
