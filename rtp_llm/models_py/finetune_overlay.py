"""Legacy ptuning_path checkpoint overlays, not a prefix-tuning runtime.

OldLoader's database gives tensors in ptuning_path/pytorch_model.bin priority
over identically named base tensors. Reuse checkpoint names and ReadSpecs so
this behavior does not require an intermediate runtime-weight namespace.
"""

import os
import re

import torch

from rtp_llm.models_py.read_spec import ReadSpec
from rtp_llm.models_py.weight_mapper import _unwrap_pytorch_state_dict

_PREFIX_ENCODER = re.compile(r"(?:^|\.)prefix_encoder(?:\.|$)")


class FinetuneCheckpointOverlay:
    def __init__(self, path: str):
        self.path = path
        metadata = _unwrap_pytorch_state_dict(
            torch.load(path, map_location="meta", weights_only=True, mmap=True)
        )
        self.names = frozenset(
            name
            for name, value in metadata.items()
            if isinstance(name, str) and isinstance(value, torch.Tensor)
        )
        if any(_PREFIX_ENCODER.search(name) for name in self.names):
            raise ValueError(
                "prefix_encoder checkpoint tensors have no executable prefix-tuning "
                "runtime; ptuning_path supports ordinary finetune weight overlays only"
            )

    @classmethod
    def discover(cls, directory, base_files):
        # Match the legacy database: no path, a missing directory, or a directory
        # containing only safetensors does not add a finetune checkpoint.
        if not directory:
            return None
        path = os.path.join(directory, "pytorch_model.bin")
        if not os.path.isfile(path):
            return None
        if os.path.realpath(path) in {os.path.realpath(p) for p in base_files}:
            return None
        return cls(path)

    def base_filter(self, name_filter):
        return lambda name: name not in self.names and (
            name_filter is None or name_filter(name)
        )

    def base_read_spec_resolver(self, resolver):
        def resolve(name, shape):
            # Native FST resolves metadata before yielding to name_filter.
            # Do not construct a destination plan using a shadowed base shape.
            return (
                ReadSpec.skip(name, shape)
                if name in self.names
                else resolver(name, shape)
            )

        return resolve

    def base_slice_expander(self, expander):
        if expander is None:
            return None
        names = self.names

        class BaseExpander:
            def __getattr__(self, name):
                return getattr(expander, name)

            def handles_safetensor(self, name):
                # Native FST expands stacked experts before the ordinary
                # ReadSpec callback. Shadowed sources must take the skip path.
                return name not in names and expander.handles_safetensor(name)

        return BaseExpander()

    def weights(self, name_filter=None, read_spec_resolver=None, slice_expander=None):
        # mmap keeps an expert overlay from eagerly copying all expert payloads
        # into host memory. The same model-owned ReadSpecs select local pieces.
        payload = torch.load(
            self.path, map_location="cpu", weights_only=True, mmap=True
        )
        state = _unwrap_pytorch_state_dict(payload)
        for name in state:
            if name not in self.names or (
                name_filter is not None and not name_filter(name)
            ):
                continue
            tensor = state[name]
            if slice_expander is not None and slice_expander.handles_safetensor(name):
                for expanded_name, index in slice_expander.expand_safetensor(
                    name, tuple(tensor.shape)
                ):
                    yield expanded_name, tensor[index]
                continue
            if read_spec_resolver is not None:
                spec = read_spec_resolver(name, tuple(tensor.shape))
                if not spec.load:
                    continue
                if spec.is_local_shard:
                    yield spec.output_name, spec.materialize(tensor)
                    continue
            yield name, tensor

    def prepend(
        self,
        base_weights,
        name_filter=None,
        read_spec_resolver=None,
        slice_expander=None,
    ):
        try:
            yield from self.weights(name_filter, read_spec_resolver, slice_expander)
            yield from base_weights
        finally:
            close = getattr(base_weights, "close", None)
            if callable(close):
                close()
