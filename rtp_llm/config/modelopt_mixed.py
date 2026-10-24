"""Per-module ModelOpt recipes; module names refer to checkpoint tensors."""

import fnmatch
import re

import torch

from rtp_llm.config.quant_config import QuantizationConfig


class ModelOptMixedPrecisionConfig(QuantizationConfig):
    def __init__(
        self, quantized_layers, ignore=None, num_layers=None, layer_types=None, **kwargs
    ):
        # A mixed checkpoint has no model-wide bit width/group size. Individual
        # module descriptors select FP8 or delegate NVFP4 to a local FP4 config.
        super().__init__(bits=0, group_size=0, is_quanted=True)
        self.quantized_layers = dict(quantized_layers)
        self.ignore = list(ignore or [])
        self._recipes = {}
        layer_modules = {}
        domains = {}
        for name in self.quantized_layers:
            ignored = any(fnmatch.fnmatchcase(name, pattern) for pattern in self.ignore)
            template_ignored = any(
                fnmatch.fnmatchcase(self._canonical(name), pattern)
                for pattern in self.ignore
            )
            if ignored != template_ignored:
                raise ValueError(
                    f"Layer-specific ignore is unsupported for normalized recipes: {name}"
                )
            if ignored:
                continue
            match = re.fullmatch(r"(.*?layers)\.(\d+)\.(.+)", name)
            if match:
                domain, index, suffix = match.groups()
                domains.setdefault(domain, set()).add(int(index))
                layer_modules.setdefault(self._canonical(name), set()).add(int(index))
        for pattern in self.ignore:
            match = re.match(r"(.*?layers)\.(\d+)(?:\.|\*|$)", pattern)
            if match and any(
                fnmatch.fnmatchcase(domain, match.group(1)) for domain in domains
            ):
                raise ValueError(
                    f"Layer-specific ignore is unsupported for normalized recipes: {pattern}"
                )
        for name, recipe in self.quantized_layers.items():
            algo = recipe.get("quant_algo")
            if algo not in ("FP8", "NVFP4") or (
                algo == "NVFP4" and recipe.get("group_size", 16) != 16
            ):
                raise ValueError(f"Unsupported ModelOpt recipe for {name}: {recipe}")
            key = self._canonical(name)
            if key in self._recipes and self._recipes[key] != recipe:
                raise ValueError(f"Layer-dependent ModelOpt recipe for {name}")
            self._recipes[key] = dict(recipe)
        if layer_types is not None:
            layer_types = list(layer_types)
            if num_layers is None:
                num_layers = len(layer_types)
            if len(layer_types) != num_layers:
                raise ValueError("ModelOpt layer_types must cover num_hidden_layers")
        for key, present in layer_modules.items():
            domain, suffix = key.split(".{i}.", 1)
            expected = (
                set(range(num_layers)) if num_layers is not None else domains[domain]
            )
            if layer_types is not None and "linear_attention" in layer_types:
                if "linear_attn" in suffix.split("."):
                    expected = {
                        i
                        for i, kind in enumerate(layer_types)
                        if kind == "linear_attention"
                    }
                elif "self_attn" in suffix.split("."):
                    expected = {
                        i
                        for i, kind in enumerate(layer_types)
                        if kind == "full_attention"
                    }
            if present != expected:
                raise ValueError(
                    f"Incomplete ModelOpt recipe for {key}: "
                    f"missing layers {sorted(expected - present)}, "
                    f"unexpected layers {sorted(present - expected)}"
                )

    @staticmethod
    def _canonical(name):
        module = name.removesuffix(".weight")
        module = re.sub(r"(?<=layers\.)(?:\d+|\{i\})", "{i}", module)
        return re.sub(r"(?<=experts\.)(?:\d+|\{expert_id\})", "{expert_id}", module)

    def recipe_for(self, tensor_name):
        module = tensor_name.removesuffix(".weight")
        if any(fnmatch.fnmatchcase(module, pattern) for pattern in self.ignore):
            return None
        return self._recipes.get(self._canonical(module))

    @classmethod
    def get_method(cls):
        return "modelopt_mixed"

    @classmethod
    def get_algo(cls):
        # C++ runs no homogeneous quantization path for per-module recipes.
        return "none"

    def get_supported_compute_dtypes(self):
        # The installed SM120 rowwise FP8 GEMM writes BF16 output encoding.
        # Reject FP16 before loading instead of returning corrupted FP16 values.
        return [torch.bfloat16]

    def get_supported_kv_cache_dtypes(self):
        return [torch.float16, torch.bfloat16, torch.float8_e4m3fn]

    @classmethod
    def _from_config(cls, config):
        return cls(**config)
