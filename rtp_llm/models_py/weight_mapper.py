import errno
import glob
import json
import mmap
import ntpath
import os
import re
from dataclasses import dataclass
from functools import cached_property
from typing import (
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

import torch

from rtp_llm.models_py.read_spec import ReadSpec

_SAFETENSORS_INDEX = "model.safetensors.index.json"
_PYTORCH_INDEX = "pytorch_model.bin.index.json"
_CONSOLIDATED_RANK_RE = re.compile(r"^consolidated[._-](\d+)(?:[._-]|$)", re.IGNORECASE)
_EXCLUDED_WEIGHT_FILES = {
    "adapter_model.bin",
    "adapter_model.safetensors",
    "optimizer.bin",
    "pytorch_model.bin.index.json",
    "model.safetensors.index.json",
    "training_args.bin",
}


@dataclass(frozen=True)
class CheckpointCatalog:
    """Validated tensor-to-shard metadata from one checkpoint index.

    The catalog is intentionally data-only: it does not open tensor payloads or
    choose a loading strategy.  NewLoader can share one instance across shard
    discovery, preflight checks, filtering, and its future load planner instead
    of parsing and resolving the same index repeatedly.
    """

    index_path: str
    format: str
    entries: Tuple[Tuple[str, str], ...]
    shard_files: Tuple[str, ...]

    @cached_property
    def _keys_by_file(self):
        # Physical paths can be symlinks (e.g. a shared HF blob cache).
        canonical = {p: os.path.realpath(p) for p in {path for _, path in self.entries}}
        result = {}
        for name, path in self.entries:
            result.setdefault(canonical[path], set()).add(name)
        return result

    @cached_property
    def _declared_names(self):
        return frozenset(name for name, _ in self.entries)

    def shard_keys(self, path, physical_keys):
        """Validate the declared source and expose only indexed tensor names.

        A reduced checkpoint may reference a genuine full-model shard. Names
        absent from the index are not part of this checkpoint. An indexed name
        in the wrong shard must *not* silently substitute for its declared copy.
        """
        expected = self._keys_by_file[os.path.realpath(path)]
        physical = set(physical_keys)
        missing = expected - physical
        if missing:
            raise ValueError(
                f"Checkpoint index declares tensor {min(missing)!r} in {path!r}, but it is missing"
            )
        extras = physical - expected
        misplaced = extras & self._declared_names
        if misplaced:
            raise ValueError(
                f"Checkpoint tensor {min(misplaced)!r} found outside its indexed shard in {path!r}"
            )
        return expected


class WeightsMapper:
    def __init__(
        self,
        exact_mapping: Optional[Dict[str, str]] = None,
        prefix_mapping: Optional[Dict[str, str]] = None,
        regex_mapping: Optional[List[Tuple[str, str]]] = None,
    ):
        self.exact_mapping = exact_mapping or {}
        self.prefix_mapping = dict(
            sorted(
                (prefix_mapping or {}).items(),
                key=lambda item: len(item[0]),
                reverse=True,
            )
        )
        self.regex_mapping = regex_mapping or []

    def map_name(self, name: str) -> str:
        if name in self.exact_mapping:
            return self.exact_mapping[name]
        for old_prefix, new_prefix in self.prefix_mapping.items():
            if name.startswith(old_prefix):
                return new_prefix + name[len(old_prefix) :]
        for pattern, replacement in self.regex_mapping:
            mapped = re.sub(pattern, replacement, name)
            if mapped != name:
                return mapped
        return name

    def apply(
        self, weights: Iterator[Tuple[str, torch.Tensor]]
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        for name, tensor in weights:
            yield self.map_name(name), tensor


class WeightsFilter:
    def __init__(
        self,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ):
        self.include_patterns = [re.compile(p) for p in (include_patterns or [])]
        self.exclude_patterns = [re.compile(p) for p in (exclude_patterns or [])]

    def should_load(self, name: str) -> bool:
        if self.include_patterns and not any(
            p.search(name) for p in self.include_patterns
        ):
            return False
        return not any(p.search(name) for p in self.exclude_patterns)

    def apply(
        self, weights: Iterator[Tuple[str, torch.Tensor]]
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        for name, tensor in weights:
            if self.should_load(name):
                yield name, tensor


class SafetensorSliceExpander(Protocol):
    """Expand one safetensors key into first-dimension slices."""

    def handles_safetensor(self, name: str) -> bool: ...

    def expand_safetensor(
        self, name: str, shape: Tuple[int, ...]
    ) -> Sequence[Tuple[str, int]]: ...


class ReadSpecResolver(Protocol):
    def __call__(self, name: str, shape: Tuple[int, ...]) -> ReadSpec: ...


def _unwrap_pytorch_state_dict(payload: object) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"PyTorch checkpoint payload must be a mapping, got {type(payload).__name__}"
        )
    for key in ("state_dict", "model"):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            return nested
    return payload


def _load_safetensors(
    path: str,
    device: str,
    name_filter: Optional[Callable[[str], bool]] = None,
    slice_expander: Optional[SafetensorSliceExpander] = None,
    read_spec_resolver: Optional[ReadSpecResolver] = None,
    catalog: Optional[CheckpointCatalog] = None,
    scratch_expert_reader=None,
    w8a8_source=None,
) -> Iterator[Tuple[str, torch.Tensor]]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device=device) as handle:
        keys = handle.keys()
        allowed = catalog.shard_keys(path, keys) if catalog is not None else None
        if w8a8_source is not None:
            handle = w8a8_source.shard(handle, allowed)
            keys, allowed = handle.keys(), None
        for name in keys:
            if allowed is not None and name not in allowed:
                continue
            if name_filter is not None and not name_filter(name):
                continue
            if slice_expander is not None and slice_expander.handles_safetensor(name):
                source = handle.get_slice(name)
                shape = tuple(source.get_shape())
                for expanded_name, index in slice_expander.expand_safetensor(
                    name, shape
                ):
                    if not 0 <= index < shape[0]:
                        raise ValueError(
                            f"Invalid first-dimension slice {index} for {name!r} "
                            f"with shape {shape}"
                        )
                    # Preserve the leading dimension during the file read so the
                    # safetensors backend materializes exactly one expert.
                    yield expanded_name, source[index : index + 1][0]
                continue
            if read_spec_resolver is not None:
                source = handle.get_slice(name)
                shape = tuple(source.get_shape())
                read_spec = read_spec_resolver(name, shape)
                if not read_spec.load:
                    continue
                if read_spec.is_local_shard:
                    yield read_spec.output_name, read_spec.materialize(source)
                    continue
                if scratch_expert_reader is not None:
                    parts = scratch_expert_reader(name, source)
                    if parts is not None:
                        yield from parts
                        continue
            yield name, handle.get_tensor(name)


def _can_map_cpu_archive(path: str) -> bool:
    # torch.load uses a process-wide mmap policy. Never opt a checkpoint into
    # a writable shared mapping; owners may update source views in place.
    if (
        os.name != "posix"
        or torch.serialization.get_default_mmap_options() != mmap.MAP_PRIVATE
    ):
        return False
    with open(path, "rb") as handle:
        if handle.read(4) != b"PK\x03\x04":  # pre-ZIP torch.save format
            return False
        try:
            # Probe the same whole-file mapping without faulting tensor pages.
            # Some file systems support reads but not private maps. Do not
            # retry a real I/O, disconnection or memory error on a second path.
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_COPY):
                return True
        except OSError as error:
            if error.errno in (errno.ENODEV, errno.ENOSYS, errno.EOPNOTSUPP):
                return False
            raise


def _load_pytorch(
    path: str,
    device: str,
    name_filter: Optional[Callable[[str], bool]] = None,
    read_spec_resolver: Optional[ReadSpecResolver] = None,
    catalog: Optional[CheckpointCatalog] = None,
) -> Iterator[Tuple[str, torch.Tensor]]:
    # Each CPU reader otherwise materializes the entire ZIP shard in anonymous
    # memory before yielding even a single rank-local tensor.
    options = (
        {"mmap": _can_map_cpu_archive(path)}
        if torch.device(device).type == "cpu"
        else {}
    )
    payload = torch.load(path, map_location=device, weights_only=True, **options)
    state_dict = _unwrap_pytorch_state_dict(payload)
    allowed = (
        catalog.shard_keys(path, (key for key, value in state_dict.items() if isinstance(value, torch.Tensor)))
        if catalog is not None else None
    )
    for name, tensor in state_dict.items():
        if (
            isinstance(name, str)
            and isinstance(tensor, torch.Tensor)
            and (allowed is None or name in allowed)
            and (name_filter is None or name_filter(name))
        ):
            if read_spec_resolver is not None:
                read_spec = read_spec_resolver(name, tuple(tensor.shape))
                if not read_spec.load:
                    continue
                if read_spec.is_local_shard:
                    yield read_spec.output_name, read_spec.materialize(tensor)
                    continue
            yield name, tensor


def get_all_weights(
    ckpt_paths: List[str],
    device: str = "cpu",
    name_filter: Optional[Callable[[str], bool]] = None,
    safetensor_slice_expander: Optional[SafetensorSliceExpander] = None,
    read_spec_resolver: Optional[ReadSpecResolver] = None,
    catalog: Optional[CheckpointCatalog] = None,
    advise_scratch_cache: bool = False,
    scratch_expert_reader=None,
    w8a8_source=None,
) -> Iterator[Tuple[str, torch.Tensor]]:
    seen = set()
    for path in ckpt_paths:
        if path.endswith(".safetensors"):
            weights = _load_safetensors(
                path,
                device,
                name_filter,
                safetensor_slice_expander,
                read_spec_resolver,
                catalog,
                scratch_expert_reader,
                w8a8_source,
            )
            if advise_scratch_cache:
                from rtp_llm.utils.ckpt_file_info import ShardPageCache

                if ShardPageCache.enabled():
                    weights = _advise_after_consumption(ShardPageCache(path), weights)
        elif path.endswith((".bin", ".pt", ".pth")):
            if safetensor_slice_expander is not None:
                raise ValueError(
                    "Safetensor slice expansion cannot be used with a PyTorch "
                    f"checkpoint: {path}"
                )
            weights = _load_pytorch(path, device, name_filter, read_spec_resolver, catalog)
        else:
            raise ValueError(f"Unsupported checkpoint format: {path}")
        try:
            for name, tensor in weights:
                if name in seen:
                    raise RuntimeError(
                        f"Checkpoint tensor {name!r} appears in more than one shard"
                    )
                seen.add(name)
                yield name, tensor
        finally:
            close = getattr(weights, "close", None)
            if close is not None:
                close()


def _advise_after_consumption(cache, weights):
    """Count yielded slices, after their synchronous consumer has copied them."""
    from rtp_llm.models_py.read_spec import unwrap_checkpoint_tensor
    consumed = False
    try:
        for name, value in weights:
            tensor, _ = unwrap_checkpoint_tensor(value)
            nbytes = tensor.numel() * tensor.element_size()
            consumed |= nbytes > 0
            try:
                yield name, value
            finally:
                cache._note_bytes_read(nbytes)
    finally:
        weights.close()
        if consumed:
            cache.finish_page_cache()


def _read_checkpoint_index(
    model_path: str, index_name: str
) -> Optional[Tuple[str, Mapping[str, object]]]:
    index_path = os.path.join(model_path, index_name)
    if not os.path.isfile(index_path):
        return None
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate key {key!r} in checkpoint index {index_path}")
            result[key] = value
        return result

    try:
        with open(index_path, encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=unique_keys)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Failed to read checkpoint index {index_path}: {exc}"
        ) from exc
    weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Checkpoint index {index_path} has no non-empty weight_map")
    return index_path, weight_map


def _resolve_index_shard_path(
    model_root: str,
    index_path: str,
    shard_name: object,
    expected_suffix: str,
) -> str:
    if not isinstance(shard_name, str) or not shard_name.endswith(expected_suffix):
        raise ValueError(
            f"Checkpoint index {index_path} contains invalid shard {shard_name!r}"
        )
    path_parts = re.split(r"[\\/]", shard_name)
    if os.path.isabs(shard_name) or ntpath.isabs(shard_name) or os.pardir in path_parts:
        raise ValueError(
            f"Checkpoint index {index_path} references a path outside model "
            f"directory: {shard_name!r}"
        )

    normalized_name = os.path.normpath(shard_name)
    shard_path = os.path.join(model_root, normalized_name)
    try:
        inside_model = os.path.commonpath([model_root, shard_path]) == model_root
    except ValueError:
        inside_model = False
    if not inside_model:
        raise ValueError(
            f"Checkpoint index {index_path} references a path outside model "
            f"directory: {shard_name!r}"
        )
    if not _is_model_weight_file(normalized_name, allow_consolidated=True):
        raise ValueError(
            f"Checkpoint index {index_path} references non-model file {shard_name!r}"
        )
    if not os.path.isfile(shard_path):
        raise FileNotFoundError(
            f"Checkpoint index {index_path} references missing shard {shard_name!r}"
        )
    return shard_path


def _checkpoint_catalog_from_index(
    model_path: str, index_name: str
) -> Optional[CheckpointCatalog]:
    index = _read_checkpoint_index(model_path, index_name)
    if index is None:
        return None
    index_path, weight_map = index
    expected_suffix = ".safetensors" if index_name == _SAFETENSORS_INDEX else ".bin"
    model_root = os.path.realpath(model_path)
    entries = []
    shard_paths = {}
    shard_files = []
    shard_identities = set()
    for name, shard_name in weight_map.items():
        if not isinstance(name, str):
            raise ValueError(
                f"Checkpoint index {index_path} contains non-string tensor name"
            )
        if not isinstance(shard_name, str):
            raise ValueError(
                f"Checkpoint index {index_path} contains invalid shard "
                f"{shard_name!r}"
            )
        if shard_name not in shard_paths:
            shard_path = _resolve_index_shard_path(
                model_root,
                index_path,
                shard_name,
                expected_suffix,
            )
            shard_paths[shard_name] = shard_path
            shard_identity = os.path.realpath(shard_path)
            if shard_identity not in shard_identities:
                shard_identities.add(shard_identity)
                shard_files.append(shard_path)
        entries.append((name, shard_paths[shard_name]))
    return CheckpointCatalog(
        index_path=index_path,
        format="safetensors" if index_name == _SAFETENSORS_INDEX else "pytorch",
        entries=tuple(entries),
        shard_files=tuple(shard_files),
    )


def safetensor_catalog(model_path: str) -> Optional[CheckpointCatalog]:
    """Build the validated safetensors index catalog, if one exists."""
    return _checkpoint_catalog_from_index(model_path, _SAFETENSORS_INDEX)


def checkpoint_catalog(model_path: str) -> Optional[CheckpointCatalog]:
    return safetensor_catalog(model_path) or _checkpoint_catalog_from_index(model_path, _PYTORCH_INDEX)


def safetensor_index_entries(
    model_path: str,
    catalog: Optional[CheckpointCatalog] = None,
) -> Optional[List[Tuple[str, str]]]:
    """Return validated ``(tensor_name, shard_path)`` index entries.

    Callers that only need checkpoint metadata can use the Hugging Face index
    without opening every safetensors shard header again.  ``None`` means the
    checkpoint has no safetensors index and the caller must inspect the files.
    """
    if catalog is None:
        catalog = safetensor_catalog(model_path)
    if catalog is None:
        return None
    if catalog.format != "safetensors":
        raise ValueError(
            f"Expected a safetensors catalog, got format={catalog.format!r}"
        )
    return list(catalog.entries)


def select_safetensor_files(
    model_path: str,
    checkpoint_files: Sequence[str],
    name_filter: Optional[Callable[[str], bool]],
    catalog: Optional[CheckpointCatalog] = None,
    *,
    allow_empty: bool = False,
) -> List[str]:
    """Select shards containing at least one accepted checkpoint tensor.

    Model predicates can be rank-local (e.g. fixed-host PLE tables). The FST
    transport reconciles these file and key schedules across its actual group;
    rank-specific expert slicing is applied separately.
    """
    files = list(checkpoint_files)
    if name_filter is None or not files:
        return files
    if not callable(name_filter):
        raise TypeError("name_filter must be callable")
    if not all(path.lower().endswith(".safetensors") for path in files):
        return files

    discovered = {os.path.realpath(path): path for path in files}
    selected_identities = set()
    index_entries = safetensor_index_entries(model_path, catalog)
    if index_entries is not None:
        catalog_shards = tuple(dict.fromkeys(path for _, path in index_entries))
        identity_by_shard = {path: os.path.realpath(path) for path in catalog_shards}
        for name, shard_path in index_entries:
            identity = identity_by_shard[shard_path]
            if name_filter(name):
                if identity not in discovered:
                    raise ValueError(
                        "Checkpoint index references an undiscovered shard for "
                        f"tensor {name!r}: {shard_path!r}"
                    )
                selected_identities.add(identity)
    else:
        from safetensors import safe_open

        for path in files:
            with safe_open(path, framework="pt", device="cpu") as handle:
                if any(name_filter(name) for name in handle.keys()):
                    selected_identities.add(os.path.realpath(path))

    selected = [path for path in files if os.path.realpath(path) in selected_identities]
    if not selected and not allow_empty:
        raise ValueError("Checkpoint filter did not match any safetensors shard")
    return selected


def _files_from_index(model_path: str, index_name: str) -> Optional[List[str]]:
    catalog = _checkpoint_catalog_from_index(model_path, index_name)
    return None if catalog is None else list(catalog.shard_files)


def _is_consolidated_weight_file(path: str) -> bool:
    return os.path.basename(path).lower().startswith("consolidated")


def is_rank_local_checkpoint(files: List[str]) -> bool:
    return any(
        _CONSOLIDATED_RANK_RE.match(os.path.basename(path)) is not None
        for path in files
    )


def _is_model_weight_file(path: str, allow_consolidated: bool = False) -> bool:
    name = os.path.basename(path).lower()
    if name in _EXCLUDED_WEIGHT_FILES:
        return False
    excluded_prefixes = [
        "adapter_model",
        "optimizer",
        "rng_state",
        "scheduler",
        "training_args",
    ]
    if not allow_consolidated:
        excluded_prefixes.append("consolidated")
    return not name.startswith(tuple(excluded_prefixes))


def _select_consolidated_files(
    files: List[str], tp_rank: int, tp_size: int
) -> List[str]:
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (tp_rank, tp_size)
    ):
        raise TypeError("tp_rank and tp_size must be integers")
    if tp_size <= 0 or not 0 <= tp_rank < tp_size:
        raise ValueError(
            f"Invalid TP partition for consolidated checkpoint: "
            f"rank={tp_rank}, size={tp_size}"
        )

    ranked_files = {}
    unranked_files = []
    for path in files:
        match = _CONSOLIDATED_RANK_RE.match(os.path.basename(path))
        if match is None:
            unranked_files.append(path)
            continue
        rank = int(match.group(1))
        if rank in ranked_files:
            raise ValueError(f"Duplicate consolidated checkpoint rank {rank}")
        ranked_files[rank] = path

    if not ranked_files:
        if len(unranked_files) != 1:
            raise ValueError("Multiple unranked consolidated checkpoints are ambiguous")
        return unranked_files
    if unranked_files:
        raise ValueError("Ranked and unranked consolidated checkpoints cannot be mixed")
    if len(ranked_files) != tp_size:
        raise ValueError(
            f"Found {len(ranked_files)} consolidated rank files, but tp_size={tp_size}; "
            "all checkpoint ranks are required"
        )

    expected_ranks = set(range(tp_size))
    if set(ranked_files) != expected_ranks:
        raise ValueError(
            f"Consolidated checkpoint ranks must be contiguous {sorted(expected_ranks)}, "
            f"got {sorted(ranked_files)}"
        )
    return [ranked_files[tp_rank]]


def discover_ckpt_files(
    model_path: str, tp_rank: int = 0, tp_size: int = 1
) -> List[str]:
    if not os.path.isdir(model_path):
        raise NotADirectoryError(f"Model path is not a directory: {model_path}")
    for index_name in (_SAFETENSORS_INDEX, _PYTORCH_INDEX):
        indexed_files = _files_from_index(model_path, index_name)
        if indexed_files is not None:
            return indexed_files
    consolidated_by_format = []
    for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth"):
        candidates = sorted(glob.glob(os.path.join(model_path, pattern)))
        files = [path for path in candidates if _is_model_weight_file(path)]
        if files:
            return files
        consolidated = [
            path
            for path in candidates
            if _is_consolidated_weight_file(path)
            and _is_model_weight_file(path, allow_consolidated=True)
        ]
        if consolidated:
            consolidated_by_format.append(consolidated)
    if consolidated_by_format:
        return _select_consolidated_files(consolidated_by_format[0], tp_rank, tp_size)
    return []
