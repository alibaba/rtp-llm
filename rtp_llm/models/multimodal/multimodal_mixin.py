import gc
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from rtp_llm.config.model_config import ModelConfig, VitParameters
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import MMAtomicWeight
from rtp_llm.models.downstream_modules.plugin_loader import (  # Import load_module
    load_module,
)
from rtp_llm.models.multimodal.multimodal_common import MultiModalEmbeddingInterface
from rtp_llm.models.multimodal.multimodal_trt_engine import MultiModalTRTEngine
from rtp_llm.models_py.distributed.collective_torch import Group, barrier
from rtp_llm.ops import ParallelismConfig, VitSeparation
from rtp_llm.utils.custommodal_util import MethodType, load_custom_modal_class
from rtp_llm.utils.model_weight import CkptWeightInfo, identity, sp_id
from rtp_llm.utils.multimodal_util import MultimodalInput, get_vit_compute_dtype


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


class BaseMultiModalWeightInfo(ABC):
    def __init__(
        self,
        vit_weights: Optional[BaseVitWeights],
        **kwargs,
    ):
        self.vit_weights: Optional[BaseVitWeights] = vit_weights

    def _get_vit_info(self, llm_weights: "ModelWeightInfo") -> "ModelWeightInfo":
        if self.vit_separation == VitSeparation.VIT_SEPARATION_ROLE:
            llm_weights = ModelWeightInfo(layer_weights=[], weights=[])

        if self.vit_separation != VitSeparation.VIT_SEPARATION_REMOTE:
            if self.vit_weights is not None and self.tp_rank == 0:
                weight_names = self.vit_weights.weight_names
                ckpt_prefix = self.vit_weights.ckpt_prefix
                for w in weight_names:
                    w_name = ckpt_prefix + w
                    llm_weights.weights.append(
                        MMAtomicWeight(
                            w,
                            [CkptWeightInfo(w_name, identity)],
                            identity,
                            split_func=sp_id,
                        )
                    )

        return llm_weights


# 继承MultiModalMixin时，需要把声明写在GPT前以正确顺序构造，详情看super().__init__含义
class MultiModalMixin:
    mm_part: MultiModalEmbeddingInterface

    @property
    def vit_data_type(self):
        return get_vit_compute_dtype(self.config.data_type)

    def init_multimodal(self, config: GptInitModelParameters, device: str) -> None:
        self.vit_config = config.py_env_configs.vit_config
        if config.vit_separation != 2:
            with torch.device(device):
                torch_default_dtype = torch.get_default_dtype()
                torch.set_default_dtype(self.vit_data_type)
                self._init_multimodal(config)
                # Dynamic injection of custom multimodal embedding
                if hasattr(config, "custom_modal") and config.custom_modal:
                    cls = load_custom_modal_class(
                        config.custom_modal, config.ckpt_path, MethodType.Embedding
                    )
                    if cls:
                        try:
                            # Assuming the custom class constructor signature is (config, tokenizer)
                            custom_mm_part = cls(config)

                            # If a native mm_part already exists (e.g. Qwen-VL), we need to support both.
                            if hasattr(self, "mm_part") and self.mm_part is not None:
                                original_mm_part = self.mm_part
                                original_method = custom_mm_part.mm_embedding
                                logging.info(
                                    f"Native mm_part found: {type(original_mm_part).__name__}. Creating composite router."
                                )

                                def _composite_mm_embedding(url, mm_type, **kwargs):
                                    # Check if it's a custom type (handle both single int and list)
                                    is_custom = False
                                    target_type = (
                                        mm_type[0]
                                        if isinstance(mm_type, list)
                                        and len(mm_type) > 0
                                        else mm_type
                                    )
                                    if isinstance(target_type, int):  # Enum is int
                                        is_custom = target_type == [MMUrlType.CUSTOM]

                                    if is_custom:
                                        return original_method(url, mm_type, **kwargs)
                                    else:
                                        return original_mm_part.mm_embedding(
                                            url, mm_type, **kwargs
                                        )

                                custom_mm_part.mm_embedding = _composite_mm_embedding

                            self.mm_part = custom_mm_part

                            logging.info(
                                f"Successfully replaced mm_part with custom implementation: {cls.__name__}"
                            )
                        except Exception as e:
                            logging.error(
                                f"Failed to instantiate custom embedding module: {e}"
                            )
                            raise RuntimeError(
                                f"Custom embedding module load failure: {e}"
                            )

                            logging.info(
                                f"Successfully replaced mm_part with custom implementation: {cls.__name__}"
                            )
                        except Exception as e:
                            logging.error(
                                f"Failed to instantiate custom embedding module: {e}"
                            )
                            raise RuntimeError(
                                f"Custom embedding module load failure: {e}"
                            )
                torch.set_default_dtype(torch_default_dtype)

    def _init_multimodal(self) -> None:
        raise NotImplementedError

    @classmethod
    def _get_mm_module(cls, config: ModelConfig):
        raise NotImplementedError

    def _load_mm_weight(self, vit_params: VitParameters, ctype: str, device: str):
        # Load weight only for self.mm_part

        vit_weight = vit_params.vit_weights
        ft_prefix = vit_weight.ft_prefix
        weight_names = vit_weight.weight_names

        def _safe_load_from_module(param: torch.nn.Parameter, fname: str, ctype):
            from rtp_llm.utils.util import to_torch_dtype

            t = self.weight.get_global_weight_or_none(fname)
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

    @classmethod
    def init_model_weight_evaluator(cls, config: ModelConfig):
        config.mm_related_params.eval_param_count = cls.eval_mm_model_param_count
        config.mm_related_params.eval_model_size = cls.eval_mm_model_size

    @classmethod
    def eval_mm_model_size(cls, config: ModelConfig):
        mm_part = cls._get_mm_module(config)
        return sum([t.numel() for t in mm_part.parameters()]) * 2

    @classmethod
    def eval_mm_model_param_count(cls, config: ModelConfig):
        mm_part = cls._get_mm_module(config)
        return sum([t.numel() for t in mm_part.parameters()])

    # def init_mm_trt(
    #     self,
    #     ckpt_path: str,
    #     vit_params: VitParameters,
    #     tp_size: str,
    #     tp_rank: int,
    #     device: Union[str, torch.device],
    #     dtype: torch.dtype,
    # ):
    #     # check whether VIT tensorrt exist
    #     try:
    #         pass
    #     except ImportError:
    #         raise RuntimeError("tensorrt library not fonnd")

    #     nccl_op_: Optional[NcclOp] = None
    #     if tp_size > 1:
    #         nccl_op_ = NcclOp()

    #     try:
    #         # TODO(xyz): currently model_name_path is ugly, we should let model_name_path passed by the frontend in
    #         # environment variable
    #         model_name_path = ckpt_path.replace("/", "_")

    #         visual_trt_engine = MultiModalTRTEngine(
    #             model_name_path, vit_params.config.get("image_size"), device, dtype
    #         )

    #         # TRT engine doesn't support TP, here we only generate trt engine on rank0 if trt engine is not cached
    #         if tp_rank == 0 and (
    #             (not MultiModalTRTEngine.trt_engine_cached(model_name_path, dtype))
    #             or self.vit_config.trt_cache_enabled == 0
    #         ):
    #             self._load_mm_weight(vit_params, dtype, device)

    #             # create cached dir if not exists
    #             output_dir = MultiModalTRTEngine.cache_path(model_name_path, dtype)
    #             if not os.path.exists(output_dir):
    #                 os.makedirs(output_dir)

    #             visual_trt_engine.export_onnx(self.mm_part.vit, tp_size)

    #             # eagerly gc VIT network, release GPU memory for generating trt engine
    #             self.gc_mm_part(vit_params)

    #             visual_trt_engine.generate_trt_engine()

    #             # create a completion file to mark that the trt engine has been generated and cached
    #             MultiModalTRTEngine.completion_file_path(model_name_path, dtype).touch()

    #         # for TP > 1, only rank0 will generate trt engine, other ranks will wait rank0 to generate trt engine
    #         if tp_size > 1:
    #             nccl_op_.barrier(torch.device(device))

    #         self.gc_mm_part(vit_params)
    #         # Currently, the multimodel network isn't split between devices. Only Rank 0 loads the weights.
    #         # After supporting TP mm network, we will remove the check here.
    #         if tp_rank == 0:
    #             visual_trt_engine.load_trt_engine()
    #             self.mm_part = visual_trt_engine

    #     except Exception as e:
    #         raise RuntimeError(f"init multimodal trt error: {e}")

    def gc_mm_part(self, vit_params: VitParameters):
        del self.mm_part
        self.mm_part = None
        vit_params.vit_weights = None
        gc.collect()
        torch.cuda.empty_cache()

    def load_mm_weight(
        self,
        model_config: ModelConfig,
        ctype: str,
        tp_size: int,
        tp_rank: int,
        device: str,
    ):
        vit_trt = self.vit_config.vit_trt

        if vit_trt == 1:
            # mm_related_params is in model_config, not mm_model_config
            # mm_related_params = model_config.mm_related_params
            # self.init_mm_trt(
            #     model_config.ckpt_path,
            #     mm_related_params,
            #     tp_size,
            #     tp_rank,
            #     device,
            # )
            raise Exception("trt engine is not supported")

        if tp_rank >= 1:
            return

        # For trt engine, we don't need to load weight since its weight is inside trt engine.
        if isinstance(self.mm_part, MultiModalTRTEngine):
            return

        # Call custom load_weight if available.
        if hasattr(self.mm_part, "load_weight") and callable(self.mm_part.load_weight):
            logging.info("Calling custom mm_part.load_weight() in load_mm_weight...")
            self.mm_part.load_weight(self.weight)

        self._load_mm_weight(self.config.mm_related_params, ctype, device)
