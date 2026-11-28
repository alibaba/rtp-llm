"""Meta Llama consolidated checkpoint boundary; source TP is not runtime TP."""

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

from rtp_llm.models_py.ft_checkpoint import FTCheckpoint

_SHARD = re.compile(r"^consolidated(?:[._-](\d+))?\.(pth|pt|bin|safetensors)$")


class MetaLlamaCheckpoint:
    # Reuse only the storage-preserving, complete-binding import protocol.
    # This format has no FT runtime keys and no FT metadata semantics.
    load_parameters = FTCheckpoint.load_parameters

    @classmethod
    def open(cls, model_path):
        root = Path(model_path)
        params_path = root / "params.json"
        if not params_path.is_file():
            return None
        shards = {}
        for path in root.iterdir():
            match = _SHARD.fullmatch(path.name)
            if match:
                rank = int(match[1] or 0)
                if rank in shards:
                    raise ValueError(f"Duplicate Meta source shard rank {rank}")
                shards[rank] = path
        if not shards:
            return None
        if set(shards) != set(range(len(shards))):
            raise ValueError("Meta source shards must be contiguous starting at zero")
        result = cls()
        result.root = root
        result.files = [str(shards[rank]) for rank in range(len(shards))]
        with params_path.open() as stream:
            result.params = json.load(stream)
        if not isinstance(result.params, dict):
            raise ValueError("Meta params.json must be an object")
        result._torch_shards = {}
        result._overlay = {}
        result.read_backend = "scratch"
        result.entries = {}
        for rank, path in enumerate(result.files):
            names = result._keys(path)
            if rank and set(names) != set(result.entries):
                raise ValueError(
                    "Meta source shards must contain the same tensor names"
                )
            for name in names:
                result.entries.setdefault(name, []).append(path)
        return result

    def apply_overlay(self, overlay):
        # A finetune tensor replaces the raw logical tensor once, not once per
        # base source-TP shard. Unrelated checkpoint names remain unused exactly
        # as in the legacy database; model-owned conversion still validates shape.
        self._overlay = dict(overlay.weights(lambda name: name in self.entries))

    def _torch_state(self, path):
        if path not in self._torch_shards:
            try:
                state = torch.load(
                    path, map_location="cpu", weights_only=True, mmap=True
                )
            except RuntimeError as error:
                if "mmap" not in str(error):
                    raise
                # Pre-ZIP PyTorch archives cannot mmap; keep safe weights-only
                # decoding for this genuine historical format (never pickle).
                state = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(state, dict) or not all(
                isinstance(k, str) and isinstance(v, torch.Tensor)
                for k, v in state.items()
            ):
                raise ValueError("Meta checkpoint must be a flat tensor dictionary")
            self._torch_shards[path] = state
        return self._torch_shards[path]

    def _keys(self, path):
        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as handle:
                return list(handle.keys())
        return list(self._torch_state(path))

    def _shape(self, path, name):
        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as handle:
                return tuple(handle.get_slice(name).get_shape())
        return tuple(self._torch_state(path)[name].shape)

    def _slice(self, path, name, slices):
        if path.endswith(".safetensors"):
            if self.read_backend == "fastsafetensors":
                return FTCheckpoint._read_tensor(self, name, Path(path), slices)
            with safe_open(path, framework="pt", device="cpu") as handle:
                return handle.get_slice(name)[slices]
        if self.read_backend != "scratch":
            raise ValueError(
                "FST requires safetensors; Meta .pth archives require scratch"
            )
        return self._torch_state(path)[name][slices]

    def shape(self, name, concat_dim):
        if name in self._overlay:
            return tuple(self._overlay[name].shape)
        shapes = [self._shape(path, name) for path in self.entries[name]]
        first = shapes[0]
        if any(
            len(shape) != len(first)
            or any(
                value != first[axis]
                for axis, value in enumerate(shape)
                if axis != concat_dim
            )
            for shape in shapes
        ):
            raise ValueError(f"Meta inconsistent source layout for {name}")
        if concat_dim is None:
            return first
        result = list(first)
        result[concat_dim] = sum(shape[concat_dim] for shape in shapes)
        return tuple(result)

    def read_slice(self, name, concat_dim, slices=None):
        shape = self.shape(name, concat_dim)
        slices = (slice(None),) * len(shape) if slices is None else slices
        if len(slices) != len(shape) or any(
            not isinstance(item, slice) or item.step not in (None, 1) for item in slices
        ):
            raise ValueError(
                "Meta reader requires contiguous, dimension-preserving slices"
            )
        if name in self._overlay:
            return self._overlay[name][slices]
        if concat_dim is None:
            values = [self._slice(path, name, slices) for path in self.entries[name]]
            if any(not torch.equal(value, values[0]) for value in values[1:]):
                raise ValueError(f"Meta replicated tensors disagree: {name}")
            return values[0]
        begin, end, _ = slices[concat_dim].indices(shape[concat_dim])
        if begin == end:
            # TP may assign only padded FFN channels to this rank. Preserve
            # the checkpoint dtype and all other requested dimensions.
            local = list(slices)
            local[concat_dim] = slice(0, 0)
            return self._slice(self.entries[name][0], name, tuple(local))
        chunks, offset = [], 0
        for path in self.entries[name]:
            size = self._shape(path, name)[concat_dim]
            left, right = max(begin, offset), min(end, offset + size)
            if left < right:
                local = list(slices)
                local[concat_dim] = slice(left - offset, right - offset)
                chunks.append(self._slice(path, name, tuple(local)))
            offset += size
        if not chunks:
            raise ValueError(f"Empty Meta tensor selection for {name}")
        return torch.cat(chunks, dim=concat_dim)

    def read(self, name):
        raise ValueError(
            f"Meta tensor {name} needs an explicit module-owned source layout"
        )

    def read_rows(self, name, concat_dim, rows):
        """Gather global vocabulary rows without materializing peer tensors.

        A vocabulary ReadSpec may request rows in a different order from storage,
        including no rows on a padding-only rank. Keep that order and preserve
        the source backend even for the empty selection.
        """
        shape = self.shape(name, concat_dim)
        if len(shape) != 2 or any(
            type(row) is not int or not 0 <= row < shape[0] for row in rows
        ):
            raise ValueError("Meta vocabulary rows must be valid global row indices")
        if not rows:
            if name in self._overlay:
                return self._overlay[name][:0]
            sample = self._slice(
                self.entries[name][0], name, (slice(0, 1), slice(0, 1))
            )
            return sample.new_empty((0, shape[1]))
        spans = []
        begin = previous = rows[0]
        for row in rows[1:]:
            if row != previous + 1:
                spans.append((begin, previous + 1))
                begin = row
            previous = row
        spans.append((begin, previous + 1))
        return torch.cat(
            [
                self.read_slice(name, concat_dim, (slice(begin, end), slice(None)))
                for begin, end in spans
            ],
            dim=0,
        )
