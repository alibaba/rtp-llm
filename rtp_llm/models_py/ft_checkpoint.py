"""Boundary reader for the rank-local format emitted by weights_convert.

FT checkpoints contain *runtime* tensors, not HF weights. This reader never
routes them through HF sharding or quantization post-processing. A model must
explicitly declare each supported boundary conversion before using it.
"""

import json
import re
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import torch
from safetensors import safe_open

from rtp_llm.utils.checkpoint_shard_path import resolve_safetensors_shard

_KEY = re.compile(r"^rank_(\d+)_(\d+)_(\d+)\.(global\..+|layers\.\d+\..+)$")


def validate_format_parameter_ownership(model: torch.nn.Module, *, ft=False):
    """Only FT adapters declaring the exact skipped source/target may keep loans."""
    borrowed = [
        f"{path}.{name}" if path else name
        for path, module in model.named_modules()
        for name in getattr(module, "_rtp_borrowed_parameter_names", ())
    ]
    if not borrowed:
        return {}
    preserve = getattr(model, "preserved_ft_parameter_loans", {}) if ft else {}
    bindings = getattr(model, "_rtp_borrowed_parameter_bindings", {})
    if set(borrowed) != set(bindings) or not set(borrowed) <= preserve.keys():
        raise PermissionError(
            "FT/Meta checkpoint adapters cannot overwrite borrowed parameters: "
            + ", ".join(sorted(borrowed))
        )
    from rtp_llm.models_py.borrowed_parameters import validate_borrowed_parameters

    validate_borrowed_parameters(model)
    return {target: preserve[target] for target in borrowed}


def _unique_json_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate FT index key: {name!r}")
        result[name] = value
    return result


@dataclass(frozen=True)
class CheckpointTopology:
    tp_size: int = 1
    dp_size: int = 1
    ep_size: int = 1
    tp_rank: int = 0
    dp_rank: int = 0
    ep_rank: int = 0

    def __post_init__(self):
        for axis in ("tp", "dp", "ep"):
            size, rank = getattr(self, axis + "_size"), getattr(self, axis + "_rank")
            if type(size) is not int or size < 1:
                raise ValueError(f"{axis}_size must be a positive integer")
            if type(rank) is not int or not 0 <= rank < size:
                raise ValueError(f"{axis}_rank={rank!r} is outside size={size}")


@dataclass(frozen=True)
class RuntimeTensorBinding:
    """An ephemeral import conversion, owned by the model's format adapter.

    Names in ``sources`` are the historical index suffixes (global./layers.).
    ``target`` names an actual model parameter/buffer, not another weight map.
    Converters must be pure and return the destination layout, already local.
    """

    sources: tuple[str, ...]
    target: str
    convert: Callable[[tuple[torch.Tensor, ...]], torch.Tensor]
    read: Callable[[], tuple[torch.Tensor, ...]] | None = None
    install: Callable[[tuple[torch.Tensor, ...]], None] | None = None
    # Some legacy FFNs use a physically padded pair of TP projections. Their
    # owner supplies both the exact validated geometry and its installer.
    install_shape: tuple[int, ...] | None = None


def ft_layer_bindings(model):
    """Pair source global layer ids with *actual* registered destination names."""
    ids = tuple(
        model.layer_ids() if hasattr(model, "layer_ids") else range(len(model.layers))
    )
    children = tuple(model.layers.named_children())
    if len(children) != len(ids):
        raise ValueError("FT pipeline range disagrees with local decoder layers")
    if isinstance(model.layers, torch.nn.ModuleDict):
        # main PP owners key their modules by global id, not stage-local offset.
        if tuple(key for key, _ in children) != tuple(map(str, ids)):
            raise ValueError("FT pipeline layer keys disagree with source global ids")
    return tuple((index, key, child) for index, (key, child) in zip(ids, children))


def ft_lm_head_source(checkpoint):
    """Provenance of a serialized output head; a later vocabulary patch invalidates it."""
    source = checkpoint.index_metadata.get("ft_lm_head_source")
    if source not in (None, "embedding", "lm_head"):
        raise ValueError("Invalid FT LM-head provenance")
    if any(
        key.endswith((".global.embedding", ".global.lm_head"))
        for key in getattr(checkpoint, "_overlay", {})
    ):
        return None
    return source


class FTCheckpoint:
    def __init__(
        self,
        root: Path,
        topology: CheckpointTopology,
        entries: Mapping[str, tuple[str, Path]],
    ):
        self.root = root
        self.topology = topology
        self.entries = dict(entries)
        self.read_backend = "scratch"
        self._overlay = {}

    def apply_overlay(self, overlay):
        # Preserve the database's exact checkpoint-name override semantics.
        # Keep mmap-backed references, never a second runtime weight namespace.
        self._overlay = dict(overlay.weights())

    @classmethod
    def open(cls, model_path: str, topology: CheckpointTopology, *, num_layers=None):
        root = Path(model_path).resolve()
        index = root / "model.safetensors.index.json"
        if not index.is_file():
            return None
        with index.open(encoding="utf-8") as stream:
            metadata = json.load(stream, object_pairs_hook=_unique_json_object)
        if not isinstance(metadata, dict):
            raise ValueError("Checkpoint index must be an object")
        if metadata.get("is_ft_style_weight") is not True:
            return None
        env = metadata.get("__env__params__", {})
        if not isinstance(env, dict):
            raise ValueError("FT __env__params__ must be an object")
        raw_pp = env.get("PP_SIZE", 1)
        if not re.fullmatch(r"[1-9]\d*", str(raw_pp)):
            raise ValueError(f"Invalid FT PP_SIZE: {raw_pp!r}")
        # Match legacy default semantics when older exports omit these fields.
        for axis in ("tp", "dp", "ep"):
            expected = getattr(topology, axis + "_size")
            raw = env.get(axis.upper() + "_SIZE", expected)
            # WeightConverter preserves the caller's raw EP_SIZE=0 in the
            # index, although its exporter resolves it to TP_SIZE * DP_SIZE.
            if axis == "ep" and str(raw) == "0":
                raw = topology.tp_size * topology.dp_size
            if isinstance(raw, bool) or not re.fullmatch(r"[1-9]\d*", str(raw)):
                raise ValueError(f"Invalid FT {axis.upper()}_SIZE: {raw!r}")
            if int(raw) != expected:
                raise ValueError(
                    f"FT {axis.upper()} topology mismatch: checkpoint={raw}, requested={expected}"
                )
        weight_map = metadata.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("FT checkpoint requires a nonempty weight_map")
        preshuffled = metadata.get("moe_weights_preshuffled", [])
        if not isinstance(preshuffled, list) or any(
            not isinstance(key, str) or key not in weight_map for key in preshuffled
        ):
            raise ValueError("FT preshuffle metadata must name indexed tensors")
        if preshuffled:
            # Only the ROCm producer emits AITER-tiled expert matrices. Its
            # F8_E4M3FN bytes have the same dtype/shape as plain CUDA weights.
            # Let the format adapters undo the producer layout exactly once,
            # before the target device applies its own runtime preparation.
            if metadata.get("ft_export_backend", "rocm") != "rocm":
                raise ValueError("FT preshuffle metadata conflicts with producer ABI")
            metadata["ft_export_backend"] = "rocm"
        requested = (topology.tp_rank, topology.dp_rank, topology.ep_rank)
        sizes = (topology.tp_size, topology.dp_size, topology.ep_size)
        by_rank = {}
        paths = {}
        for key, filename in weight_map.items():
            match = _KEY.fullmatch(key)
            if match is None:
                raise ValueError(f"Malformed FT runtime tensor key: {key!r}")
            ranks = tuple(int(match[i]) for i in (1, 2, 3))
            if any(rank >= size for rank, size in zip(ranks, sizes)):
                raise ValueError(f"FT tensor rank outside requested topology: {key}")
            if not isinstance(filename, str) or not filename:
                raise ValueError(f"Invalid FT shard for {key}")
            path = paths.get(filename)
            if path is None:
                path = resolve_safetensors_shard(root, filename)
                paths[filename] = path
            rank_entries = by_rank.setdefault(ranks, {})
            suffix = match[4]
            if suffix in rank_entries:
                raise ValueError(f"Duplicate logical FT tensor on {ranks}: {suffix}")
            rank_entries[suffix] = (key, path)
        if requested not in by_rank:
            raise ValueError(f"FT checkpoint has no tensors for rank {requested}")
        if int(raw_pp) != 1:
            # WeightConverter always sets pc.pp_size=1 and dumps every layer,
            # but serializes the caller's original env_params *unmodified*.
            # Don't misclassify those files as stage-local based on PP_SIZE.
            # A genuinely stage-local archive does not identify its source
            # stage in the key contract; don't infer its offset from WORLD.
            if type(num_layers) is not int or num_layers <= 0:
                raise ValueError(
                    "FT PP_SIZE requires proof of full-model layer coverage"
                )
            expected_ids = set(range(num_layers))
            for ranks, entries in by_rank.items():
                available = {
                    int(name.split(".", 2)[1])
                    for name in entries
                    if name.startswith("layers.")
                }
                if available != expected_ids:
                    raise ValueError(
                        f"Unproven FT full-model layer coverage on {ranks}"
                    )
        checkpoint = cls(root, topology, by_rank[requested])
        checkpoint._by_rank = by_rank
        checkpoint.index_metadata = metadata
        checkpoint.env_params = env
        return checkpoint

    def select_stage(self, layer_ids: range, num_layers: int):
        """Project a full-model FT export onto the runtime's actual layer range.

        The legacy writer only emits full-model files. The legacy *reader*
        permits PP consumers, whose layer ids are global and whose first/last
        globals follow ModelLoader.is_global_weight_local. Drafts can own the
        entire layer range even when the physical topology has multiple stages.
        """
        if (
            not isinstance(layer_ids, range)
            or layer_ids.step != 1
            or type(num_layers) is not int
            or num_layers <= 0
            or not 0 <= layer_ids.start < layer_ids.stop <= num_layers
        ):
            raise ValueError(
                f"Invalid FT stage range {layer_ids!r} for {num_layers} layers"
            )

        def belongs(name):
            if name.startswith("layers."):
                index = int(name.split(".", 2)[1])
                if index >= num_layers:
                    raise ValueError(f"FT tensor layer id outside model: {name}")
                return index in layer_ids
            if name in ("global.embedding", "global.position_encoding.weight"):
                return 0 in layer_ids
            if name in ("global.lm_head", "global.lm_head_b") or name.startswith(
                "global.final_layernorm."
            ):
                return num_layers - 1 in layer_ids
            return True

        self.entries = {k: v for k, v in self.entries.items() if belongs(k)}

    def _shape(self, key, path):
        if key in self._overlay:
            return tuple(self._overlay[key].shape)
        with safe_open(str(path), framework="pt", device="cpu") as shard:
            return tuple(shard.get_slice(key).get_shape())

    def _export_slot_ids(self, ranks, key, layer_id):
        manifests = self.index_metadata.get("module_export_metadata")
        if manifests is None:
            return None
        rank_key = "rank_%02d_%02d_%02d" % ranks
        manifest = manifests.get(rank_key) if isinstance(manifests, dict) else None
        if (
            not isinstance(manifest, dict)
            or manifest.get("binding_schema_version") != 1
        ):
            raise ValueError(f"Missing or invalid FT binding manifest for {rank_key}")
        for axis, rank in zip(("tp", "dp", "ep"), ranks):
            if manifest.get(axis + "_rank") != rank or manifest.get(
                axis + "_size"
            ) != getattr(self.topology, axis + "_size"):
                raise ValueError(f"FT module manifest topology mismatch for {rank_key}")
        leaves = [
            leaf
            for leaf in manifest["modules"].values()
            if key in leaf["parameter_bindings"].values()
        ]
        if len(leaves) != 1 or leaves[0].get("layer_id") != layer_id:
            raise ValueError(f"Ambiguous FT expert module binding for {key}")
        ids = leaves[0].get("logical_expert_ids")
        if not isinstance(ids, list):
            raise ValueError(
                f"FT expert manifest must declare logical_expert_ids for {key}"
            )
        return tuple(ids)

    def _expert_row_sources(self, name, layer_id, placement):
        """Locate rows of an already-sharded projection in target physical order.

        Main's CLI exports either pure TP (all EP=0, distinct expert TP slices)
        or one worker per EP, with no redundant experts and canonical slot order.
        Module-export manifests additionally describe noncanonical source slots.
        Never reuse a payload from a different expert TP stripe.
        """
        if (
            type(layer_id) is not int
            or not name.startswith(f"layers.{layer_id}.")
            or name not in self.entries
        ):
            raise ValueError(f"FT expert tensor not owned by this layer/stage: {name}")
        t = self.topology
        if (t.ep_rank, t.ep_size) != (placement.ep_rank, placement.ep_size):
            raise ValueError("FT expert target EP coordinate mismatch")
        if t.ep_size == 1:
            ranks = [(t.tp_rank, t.dp_rank, t.ep_rank)]
        elif t.ep_size == t.tp_size * t.dp_size:
            ranks = sorted(self._by_rank, key=lambda xyz: xyz[2])
            if [xyz[2] for xyz in ranks] != list(range(t.ep_size)):
                raise ValueError("Missing or ambiguous FT source EP rank")
        else:
            raise ValueError("FT expert TP stripes require an explicit source topology")

        logical_count = placement.logical_expert_count
        sources = {}
        expected_tail = None
        for xyz in ranks:
            key, path = self._by_rank[xyz][name]
            shape = self._shape(key, path)
            if (
                not shape
                or shape[0] == 0
                or (expected_tail is not None and shape[1:] != expected_tail)
            ):
                raise ValueError(
                    f"FT expert rows have incompatible shapes: {key} {shape}"
                )
            expected_tail = shape[1:]
            ids = self._export_slot_ids(xyz, key, layer_id)
            if ids is None:
                if shape[0] * t.ep_size != logical_count:
                    raise ValueError(
                        f"Ambiguous legacy FT expert order for {key}; export logical_expert_ids"
                    )
                ids = range(xyz[2] * shape[0], (xyz[2] + 1) * shape[0])
            if len(ids) != shape[0] or any(
                type(i) is not int or not 0 <= i < logical_count for i in ids
            ):
                raise ValueError(
                    f"Invalid FT physical-to-logical expert rows for {key}"
                )
            for slot, logical in enumerate(ids):
                # Replicated experts in explicit placements have identical
                # semantic values; prefer the rank-local source when available.
                if logical not in sources or xyz[2] == t.ep_rank:
                    sources[logical] = (xyz, key, path, slot)
        if len(sources) != logical_count:
            raise ValueError(f"FT source placement omits logical experts for {name}")
        return [sources[i] for i in placement.local_physical_to_logical], expected_tail

    def read_expert_rank_scalars(self, name, weight_name, layer_id, placement):
        """Restore a rank-wide legacy scalar for each selected expert slot."""
        if not name.startswith(f"layers.{layer_id}.") or name not in self.entries:
            raise ValueError(f"FT calibration not owned by layer/stage: {name}")
        rows, _ = self._expert_row_sources(weight_name, layer_id, placement)
        cache = {}
        for xyz, *_ in rows:
            if xyz in cache:
                continue
            key, path = self._by_rank[xyz][name]
            scalar = self._read_tensor(key, path)
            if scalar.numel() != 1 or scalar.dtype != torch.float32:
                raise ValueError(f"Expected one float32 FT calibration scalar: {key}")
            cache[xyz] = scalar.reshape(())
        return torch.stack([cache[xyz] for xyz, *_ in rows])

    def read_expert_rows(self, name, layer_id, placement):
        """Read only rows belonging to the selected destination expert slots."""
        rows, expected_tail = self._expert_row_sources(name, layer_id, placement)
        runs = []
        for _, key, path, slot in rows:
            if runs and runs[-1][:2] == (key, path) and runs[-1][3] == slot:
                runs[-1] = (key, path, runs[-1][2], slot + 1)
            else:
                runs.append((key, path, slot, slot + 1))
        parts = [
            self._read_tensor(
                k, p, (slice(start, end),) + (slice(None),) * len(expected_tail)
            )
            for k, p, start, end in runs
        ]
        if any(part.dtype != parts[0].dtype for part in parts[1:]):
            raise ValueError(f"FT source EP ranks use inconsistent dtype for {name}")
        if len(parts) == 1:
            return parts[0]
        dtype = parts[0].dtype
        if dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            # A changed placement commonly needs several noncontiguous runs.
            # Not all supported CPU builds implement cat for either FP8 dtype.
            return torch.cat([part.view(torch.uint8) for part in parts], dim=0).view(dtype)
        return torch.cat(parts, dim=0)

    def reload_expert_module(self, live, staging):
        """Rebuild the exact import bindings on a private one-layer staging tree."""
        layout = getattr(live, "_ft_expert_binding_layout", None)
        if layout is None:
            raise ValueError("FT owner did not declare an expert reload layout")
        factory, source_prefix, source_layer = layout
        root = torch.nn.Module()
        root.add_module("experts", staging)
        # Keep the original peer-rank catalog, overlays and chosen reader. Only
        # the scope checked by load_parameters is restricted; never mutate the
        # startup catalog (callbacks on other layers may run concurrently).
        view = copy(self)
        bindings = factory(staging, source_prefix, "experts", view, source_layer)
        keys = {name for binding in bindings for name in binding.sources}
        view.entries = {name: self.entries[name] for name in keys}
        view.load_parameters(root, tuple(bindings))

    def read(self, name: str) -> torch.Tensor:
        key, path = self.entries[name]
        return self._read_tensor(key, path)

    def _read_tensor(self, key, path, slices=None):
        if key in getattr(self, "_overlay", {}):
            tensor = self._overlay[key]
            return tensor if slices is None else tensor[slices]
        if self.read_backend == "fastsafetensors":
            from rtp_llm.models_py.fst_range_reader import iter_fst_range_tensors
            from rtp_llm.models_py.read_spec import ReadSpec, unwrap_checkpoint_tensor

            def resolve(name, shape):
                return ReadSpec(
                    source_name=name,
                    output_name=name,
                    source_shape=tuple(shape),
                    slices=slices,
                )

            values = list(
                iter_fst_range_tensors(
                    [str(path)],
                    name_filter=lambda name: name == key,
                    read_spec_resolver=resolve,
                )
            )
            if len(values) != 1 or values[0][0] != key:
                raise ValueError(f"FT index tensor {key!r} missing from {path.name}")
            # The FST reader has already applied the requested local slice.
            return unwrap_checkpoint_tensor(values[0][1])[0]
        if self.read_backend != "scratch":
            raise ValueError(f"Unknown FT read backend: {self.read_backend}")
        with safe_open(str(path), framework="pt", device="cpu") as shard:
            if key not in shard.keys():
                raise ValueError(f"FT index tensor {key!r} missing from {path.name}")
            return (
                shard.get_tensor(key)
                if slices is None
                else shard.get_slice(key)[slices]
            )

    def read_rank_slice(self, name: str, tp_rank: int, slices=None):
        """Read a bounded slice from a peer TP file without a collective."""
        if name not in self.entries:
            raise KeyError(name)
        if type(tp_rank) is not int or not 0 <= tp_rank < self.topology.tp_size:
            raise ValueError(f"Invalid FT TP peer {tp_rank!r}")
        if tp_rank == self.topology.tp_rank:
            key, path = self.entries[name]
        else:
            # FT producer emits one EP coordinate per (TP, DP) worker. A
            # peer in the same DP replica generally has a DIFFERENT EP rank.
            # Resolve its actual indexed tuple, never infer it from WORLD.
            peers = [
                entries
                for (tp, dp, _), entries in self._by_rank.items()
                if tp == tp_rank and dp == self.topology.dp_rank
            ]
            if len(peers) != 1:
                raise ValueError(
                    f"Expected one indexed FT TP peer {tp_rank} in DP {self.topology.dp_rank}, got {len(peers)}"
                )
            key, path = peers[0][name]
        if slices is not None:
            return self._read_tensor(key, path, slices)
        return self._shape(key, path)

    @torch.no_grad()
    def load_parameters(
        self,
        model: torch.nn.Module,
        bindings: tuple[RuntimeTensorBinding, ...],
        source_validators: Mapping[str, Callable[[torch.Tensor], None]] | None = None,
        *,
        generated_parameters: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        """Validate the entire declared layout, then copy into existing storage.

        This is deliberately not a generic model loading entry point. The model
        adapter must prove quantization/device layouts and declare all bindings.
        Unknown source tensors or unbound destination parameters are errors.
        """
        preserved = validate_format_parameter_ownership(model, ft=True)
        targets = dict(model.named_parameters(remove_duplicate=False))
        loans = [targets[path] for path in preserved]
        for path in preserved:
            del targets[path]
        for path, module in model.named_modules():
            for name, value in module._buffers.items():
                if value is not None and name not in module._non_persistent_buffers_set:
                    targets[f"{path}.{name}" if path else name] = value
        from rtp_llm.models_py.module_base import checkpoint_byte_span, validate_checkpoint_storage

        if loans:
            spans = [(loan.device, *checkpoint_byte_span(loan)) for loan in loans if loan.numel()]
            loan_ids = {id(loan) for loan in loans}
            for name, value in targets.items():
                overlaps = id(value) in loan_ids
                if not overlaps and value.numel():
                    start, end = checkpoint_byte_span(value)
                    overlaps = any(
                        dev == value.device and start < hi and lo < end
                        for dev, lo, hi in spans
                    )
                if overlaps:
                    raise PermissionError(
                        f"FT adapter cannot write an alias of a borrowed parameter: {name}"
                    )
        # Include generated targets: their bytes must also survive the import.
        validate_checkpoint_storage(targets.items())
        # Raw format adapters may build an online quantizer whose scale is
        # generated after import. Require the exact declared parameter; do not
        # silently exempt any other missing binding. Runtime FT snapshots
        # cannot use this exception to omit a serialized scale.
        if generated_parameters and isinstance(self, FTCheckpoint):
            raise ValueError("FT runtime snapshots cannot omit generated parameters")
        for name, parameter in (generated_parameters or {}).items():
            if name not in targets or targets[name] is not parameter:
                raise ValueError(f"Unknown generated parameter {name}")
            del targets[name]
        used_sources, used_targets = self.entries.keys() & preserved.values(), set()
        for name, validate in (source_validators or {}).items():
            if name in preserved.values():
                raise ValueError(f"FT validator cannot consume borrowed source {name}")
            validate(self.read(name))
            used_sources.add(name)
        target_owners = {}
        for binding in bindings:
            if set(binding.sources).intersection(preserved.values()):
                raise ValueError(f"FT binding cannot consume a borrowed source: {binding.target}")
            if not binding.sources:
                raise ValueError(f"FT binding has no source: {binding.target}")
            if binding.target in used_targets or binding.target not in targets:
                raise ValueError(f"Duplicate or unknown FT target: {binding.target}")
            if any(name not in self.entries for name in binding.sources):
                raise ValueError(
                    f"Missing FT sources for {binding.target}: {binding.sources}"
                )
            values = (
                binding.read()
                if binding.read
                else tuple(self.read(name) for name in binding.sources)
            )
            tensor = binding.convert(values)
            target = targets[binding.target]
            if id(target) in target_owners:
                raise ValueError(
                    "FT tied parameter bindings require an explicit model adapter: "
                    f"{target_owners[id(target)]}, {binding.target}"
                )
            target_owners[id(target)] = binding.target
            shape = target.shape
            if binding.install_shape is not None:
                if binding.install is None:
                    raise ValueError(
                        f"No installer for FT storage layout: {binding.target}"
                    )
                shape = binding.install_shape
            if not isinstance(tensor, torch.Tensor) or tensor.shape != shape:
                raise ValueError(f"FT layout mismatch for {binding.target}")
            # Runtime snapshots must not be silently requantized/cast.
            if tensor.dtype != target.dtype:
                raise ValueError(
                    f"FT dtype mismatch for {binding.target}: {tensor.dtype} vs {target.dtype}"
                )
            used_sources.update(binding.sources)
            used_targets.add(binding.target)
            del tensor, values
        if used_sources != self.entries.keys():
            raise ValueError(
                f"Unbound FT tensors: {sorted(self.entries.keys() - used_sources)}"
            )
        if used_targets != targets.keys():
            raise ValueError(
                f"Unbound model tensors: {sorted(targets.keys() - used_targets)}"
            )
        # Validate before mutation, but do not retain a model-sized second set of
        # runtime weights. This conservative boundary implementation reads twice.
        # Exported checkpoint files must remain immutable throughout loading.
        for binding in bindings:
            path = binding.target
            target = targets[path]
            values = (
                binding.read()
                if binding.read
                else tuple(self.read(name) for name in binding.sources)
            )
            if binding.install is None:
                target.copy_(binding.convert(values))
            else:
                # Module-owned loaders also own checkpoint provenance flags and
                # tied-parameter fan-out. Reuse them after all layout validation.
                binding.install(values)
            del values
            owner_name, _, name = path.rpartition(".")
            owner = model.get_submodule(owner_name) if owner_name else model
            marker = getattr(owner, "_record_parameter_loaded", None)
            if marker is None:
                marker = getattr(owner, "_mark_weight_loaded", None)
            if marker is not None:
                marker(name)
            else:
                # Ordinary nn.Embedding, nn.Linear and nn.LayerNorm owners do
                # not have RtpModule methods. Record the same provenance as
                # the direct checkpoint dispatch for their registered leaves.
                from rtp_llm.models_py.module_base import _mark_loaded

                _mark_loaded(owner, name)
