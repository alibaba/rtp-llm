"""Per-module ModelOpt recipes; module names refer to checkpoint tensors."""

import fnmatch
import re

from rtp_llm.config.quant_config import ModelOptFp4Config


class ModelOptMixedPrecisionConfig(ModelOptFp4Config):
    def __init__(self, quantized_layers, ignore=None, **kwargs):
        super().__init__(bits=4, group_size=16, is_quanted=True)
        self.quantized_layers = dict(quantized_layers)
        self.ignore = list(ignore or [])
        self._recipes = {}
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

    @staticmethod
    def _canonical(name):
        return re.sub(r"(?<=layers\.)(?:\d+|\{i\})", "{i}", name)

    def recipe_for(self, tensor_name):
        module = tensor_name.removesuffix(".weight")
        if any(fnmatch.fnmatchcase(module, pattern) for pattern in self.ignore):
            return None
        return self._recipes.get(self._canonical(module))

    @classmethod
    def get_method(cls):
        return "modelopt_mixed"

    @classmethod
    def _from_config(cls, config):
        return cls(**config)
