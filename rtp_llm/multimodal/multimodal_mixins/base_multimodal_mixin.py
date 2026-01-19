import gc
import os
import re
from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.model_config import ModelConfig, VitParameters
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.model_loader.loader import ModelLoader, get_model_loader
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import CustomAtomicWeight
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    MultiModalEmbeddingInterface,
)
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.model_weight import CkptWeightInfo, identity, sp_id
from rtp_llm.utils.time_util import timer_wrapper


class BaseVitWeights:
    def __init__(self, vit_part: Dict[str, Any], with_prefix: bool = False):
        self.weight_names: List[str] = []
        self._set_weight_prefix()
        self._get_vit_params(vit_part, with_prefix)

    def _set_weight_prefix(self):
        self._ckpt_prefix = "model."
        self._ft_prefix = "self.mm_part."

    @property
    def ckpt_prefix(self) -> str:
        return self._ckpt_prefix

    @property
    def ft_prefix(self) -> str:
        return self._ft_prefix

    @ft_prefix.setter
    def ft_prefix(self, prefix: str) -> None:
        self._ft_prefix = prefix

    def _get_vit_params(self, vit_part: Dict[str, Any], with_prefix: bool = False):
        for vit_name, vit in vit_part.items():
            if isinstance(vit, torch.nn.Module):
                if len(vit_part) >= 2 or with_prefix:
                    self.weight_names.extend(
                        [vit_name + "." + w for w in vit.state_dict().keys()]
                    )
                else:
                    self.weight_names.extend(list(vit.state_dict().keys()))
            elif isinstance(vit, torch.nn.Parameter):
                self.weight_names.append(vit_name)
            else:
                raise Exception("Unknown vit part type")


class BaseMultiModalDeployWeightInfo(ModelDeployWeightInfo):
    def __init__(self, vit_config: VitConfig, vit_weights: BaseVitWeights, **kwargs):
        super().__init__(**kwargs)
        self.vit_config = vit_config
        self.vit_weights = vit_weights

    def _get_weight_info(self):
        weights = []
        weight_names = self.vit_weights.weight_names
        ckpt_prefix = self.vit_weights.ckpt_prefix
        for w in weight_names:
            weights.append(
                CustomAtomicWeight(
                    w, [CkptWeightInfo(ckpt_prefix + w, identity)], identity
                )
            )
        return ModelWeightInfo(layer_weights=[], weights=weights)


class BaseMultiModalMixin:
    mm_part: MultiModalEmbeddingInterface

    def __init__(
        self,
        model_config: ModelConfig,
        engine_config: EngineConfig,
        vit_config: VitConfig,
    ) -> None:
        self.model_config = model_config
        self.engine_config = engine_config
        self.vit_config = vit_config
        self.load_method = engine_config.load_config.load_method
        device = "cuda:0"

        with torch.device(device):
            torch_default_dtype = torch.get_default_dtype()
            torch.set_default_dtype(model_config.compute_dtype)
            self._init_multimodal()
            torch.set_default_dtype(torch_default_dtype)

        if self.model_config.mm_related_params.vit_weights is None:
            return

        self.mm_mixin_loader = self.create_mm_mixin_loader()
        self.weights = self.mm_mixin_loader.load_weights(device=device)

        dtype_str = self.model_config.data_type
        self.load_mm_weight(
            ctype=dtype_str,
            device=device,
        )

        self.mm_mixin_loader.force_clean_cuda_memory()

    def create_mm_mixin_loader(self) -> ModelLoader:
        database = CkptDatabase(self.model_config.ckpt_path)

        weights_info: ModelDeployWeightInfo = self.get_multimodal_mixin_weight_info()(
            model_config=self.model_config,
            parallelism_config=self.engine_config.parallelism_config,
            hw_kernel_config=self.engine_config.hw_kernel_config,
            kv_cache_config=self.engine_config.kv_cache_config,
            merge_lora=False,
            load_method=self.engine_config.load_config.load_method,
            vit_weights=self.model_config.mm_related_params.vit_weights,
            vit_config=self.vit_config,
        )

        return get_model_loader(
            self.model_config,
            weights_info,
            None,
            database,
            load_method=self.load_method,
        )

    @classmethod
    def get_multimodal_mixin_weight_info(cls) -> ModelDeployWeightInfo:
        return BaseMultiModalDeployWeightInfo

    def _init_multimodal(self) -> None:
        raise NotImplementedError

    @classmethod
    def _get_mm_module(cls, config: ModelConfig):
        raise NotImplementedError

    @classmethod
    def eval_mm_model_size(cls, config: ModelConfig):
        mm_part = cls._get_mm_module(config)
        return sum([t.numel() for t in mm_part.parameters()]) * 2

    @classmethod
    def eval_mm_model_param_count(cls, config: ModelConfig):
        mm_part = cls._get_mm_module(config)
        return sum([t.numel() for t in mm_part.parameters()])

    @timer_wrapper(description="load mm weight")
    def load_mm_weight(
        self,
        ctype: str,
        device: str,
    ):
        vit_weights = self.model_config.mm_related_params.vit_weights
        ft_prefix = vit_weights.ft_prefix
        weight_names = vit_weights.weight_names

        def _safe_load_from_module(param: torch.nn.Parameter, fname: str, ctype):
            from rtp_llm.utils.util import to_torch_dtype

            t = self.weights.get_global_weight_or_none(fname)
            if t is None:
                raise Exception(f"failed to get tensor from name {fname}")
            # Convert ctype (which may be DataType enum or string) to torch.dtype
            torch_dtype = to_torch_dtype(ctype)
            param.data = t.reshape(param.data.shape).to(torch_dtype).to(device)

        for w in weight_names:
            w_name = ft_prefix + w
            w_name = re.sub(r"\.\d+\.", lambda x: "[" + x.group(0)[1:-1] + "].", w_name)
            param = eval(w_name)
            _safe_load_from_module(param, w, ctype)

    def mm_gather_batch(self):
        raise NotImplementedError("MultiModalMixin.mm_gather_batch is not implemented")
