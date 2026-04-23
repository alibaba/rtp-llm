import gc
import os
import re
from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.model_loader.load_config import LoadMethod
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.multimodal_mixin_loader import MultimodalMixinLoader
from rtp_llm.model_loader.weight_module import CustomAtomicWeight
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    MultiModalEmbeddingInterface,
)
from rtp_llm.utils.base_model_datatypes import VitParameters
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
    def __init__(self, vit_config: VitConfig, vit_weights: BaseVitWeights):
        self.vit_config = vit_config
        self.vit_weights = vit_weights

    def get_weight_info(self):
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
        compute_dtype: torch.dtype,
        device: str,
        mm_related_params: VitParameters,
        load_method: LoadMethod,
        vit_config: VitConfig,
        ckpt_path: str,
    ) -> None:
        self.compute_dtype = compute_dtype
        self.mm_related_params = mm_related_params
        self.vit_config = vit_config
        self.load_method = load_method
        self.ckpt_path = ckpt_path

        with torch.device(device):
            torch_default_dtype = torch.get_default_dtype()
            torch.set_default_dtype(compute_dtype)
            try:
                self._init_multimodal()
            finally:
                # Always restore: leaking compute_dtype as the process-wide
                # default poisons every subsequent tensor allocation.
                torch.set_default_dtype(torch_default_dtype)

        if self.mm_related_params.vit_weights is None:
            return

        self.mm_mixin_loader = self.create_mm_mixin_loader()
        self.weights = self.mm_mixin_loader.load_weights(device=device)

        self.load_mm_weight(
            ctype=compute_dtype,
            device=device,
        )

        self.mm_mixin_loader.force_clean_cuda_memory()

    def create_mm_mixin_loader(self) -> MultimodalMixinLoader:
        database = CkptDatabase(self.ckpt_path)

        weights_info: ModelDeployWeightInfo = self.get_multimodal_mixin_weight_info()(
            vit_weights=self.mm_related_params.vit_weights,
            vit_config=self.vit_config,
        )

        return MultimodalMixinLoader(
            weights_info,
            database,
            load_method=self.load_method,
        )

    @classmethod
    def get_multimodal_mixin_weight_info(cls) -> ModelDeployWeightInfo:
        return BaseMultiModalDeployWeightInfo

    def _init_multimodal(self) -> None:
        raise NotImplementedError

    @classmethod
    def _get_mm_module(cls, mm_related_params: VitParameters, vit_config: VitConfig):
        raise NotImplementedError

    @classmethod
    def eval_mm_model_size(
        cls,
        mm_related_params: VitParameters,
        extra_data_path: str,
        local_extra_data_path: str,
    ):
        vit_config = VitConfig()
        vit_config.extra_data_path = extra_data_path
        vit_config.local_extra_data_path = local_extra_data_path
        mm_part = cls._get_mm_module(mm_related_params, vit_config)
        return sum([t.numel() for t in mm_part.parameters()]) * 2

    @classmethod
    def eval_mm_model_param_count(
        cls,
        mm_related_params: VitParameters,
        extra_data_path: str,
        local_extra_data_path: str,
    ):
        vit_config = VitConfig()
        vit_config.extra_data_path = extra_data_path
        vit_config.local_extra_data_path = local_extra_data_path
        mm_part = cls._get_mm_module(mm_related_params, vit_config)
        return sum([t.numel() for t in mm_part.parameters()])

    @timer_wrapper(description="load mm weight")
    def load_mm_weight(
        self,
        ctype: str,
        device: str,
    ):
        # Fail fast on empty ckpt: if load_weights() above returned nothing
        # (wrong ckpt path, corrupt safetensors, mixin misconfig), the loop
        # below silently leaves every ViT parameter at its random init and
        # inference produces garbage at runtime with no error. Surface the
        # root cause at load time instead.
        if not self.weights:
            raise RuntimeError(
                f"No multimodal weights loaded from {self.ckpt_path!r}; "
                f"check checkpoint path and mixin configuration."
            )
        vit_weights = self.mm_related_params.vit_weights
        ft_prefix = vit_weights.ft_prefix
        weight_names = vit_weights.weight_names

        def _safe_load_from_module(param: torch.nn.Parameter, fname: str, ctype):
            from rtp_llm.utils.util import to_torch_dtype

            t = self.weights.get(fname)
            if t is None:
                raise Exception(f"failed to get tensor from name {fname}")
            # Convert ctype (which may be DataType enum or string) to torch.dtype
            torch_dtype = to_torch_dtype(ctype)
            param.data = t.reshape(param.data.shape).to(torch_dtype).to(device)

        # ft_prefix is a hard-coded class attribute starting with "self.", weight_names come
        # from PyTorch state_dict().keys() loaded from the model checkpoint — both are
        # trusted framework-internal strings (not user input), so there is no injection risk.
        # We use a token walker instead of eval() for explicit path resolution.
        path_token = re.compile(r"([A-Za-z_]\w*)|\[(\d+)\]")

        def _resolve_path(root, path: str):
            obj = root
            for m in path_token.finditer(path):
                obj = getattr(obj, m.group(1)) if m.group(1) else obj[int(m.group(2))]
            return obj

        assert ft_prefix.startswith("self."), f"unexpected ft_prefix: {ft_prefix!r}"
        rel_prefix = ft_prefix[len("self.") :]
        for w in weight_names:
            w_name = re.sub(r"\.\d+\.", lambda x: "[" + x.group(0)[1:-1] + "].", w)
            param = _resolve_path(self, rel_prefix + w_name)
            _safe_load_from_module(param, w, ctype)

    def mm_gather_batch(self):
        raise NotImplementedError("MultiModalMixin.mm_gather_batch is not implemented")
