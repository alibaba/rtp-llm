import gc
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch

from rtp_llm.config.model_config import ModelConfig, VitParameters
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.model_loader.weight_module import MMAtomicWeight

if TYPE_CHECKING:
    from rtp_llm.model_loader.model_weight_info import ModelWeightInfo

from rtp_llm.models.multimodal.multimodal_common import MultiModalEmbeddingInterface
from rtp_llm.models.multimodal.multimodal_trt_engine import MultiModalTRTEngine
from rtp_llm.utils.model_weight import CkptWeightInfo, identity, sp_id
from rtp_llm.models_py.distributed.collective_torch import barrier, Group


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


class BaseMultiModalWeightInfo:
    def __init__(
        self,
        vit_weights: Optional[BaseVitWeights],
        **kwargs,
    ):
        self.vit_weights: Optional[BaseVitWeights] = vit_weights

    def _get_vit_info(self, llm_weights: "ModelWeightInfo") -> "ModelWeightInfo":
        if self.vit_weights is not None:
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

    def init_multimodal(
        self,
        mm_model_config: Any,  # MMModelConfig
        vit_config: VitConfig,
        device: str,
    ) -> None:
        self.vit_config = vit_config
        with torch.device(device):
            torch_default_dtype = torch.get_default_dtype()
            torch.set_default_dtype(self.model_config.compute_dtype)
            self._init_multimodal(
                mm_model_config=mm_model_config,
                vit_config=vit_config,
            )
            torch.set_default_dtype(torch_default_dtype)

    def _init_multimodal(
        self,
        mm_model_config: Any,  # MMModelConfig
        vit_config: VitConfig,
    ) -> None:
        raise NotImplementedError

    def _load_mm_weight(self, vit_params: VitParameters, ctype: str, device: str):
        # Load weight only for self.mm_part
        from rtp_llm.utils.util import to_torch_dtype
        torch_dtype = to_torch_dtype(ctype)

        vit_weight = vit_params.vit_weights
        ft_prefix = vit_weight.ft_prefix
        weight_names = vit_weight.weight_names
        mm = eval(ft_prefix[:-1])

        state_dict_to_load = {}
        for w in weight_names:
            t = self.weight.get_global_weight_or_none(w)
            assert t is not None, f"failed to get tensor from name {w}"
            target_shape = mm.state_dict()[w].shape
            state_dict_to_load[w] = t.reshape(target_shape).to(torch_dtype).to(device)

        missing_keys, unexpected_keys = mm.load_state_dict(state_dict_to_load, strict=False)

        if unexpected_keys:
            raise Exception(
                f"Unexpected keys when loading mm weights: {unexpected_keys}"
            )

    def init_mm_trt(
        self,
        ckpt_path: str,
        vit_params: VitParameters,
        tp_size: str,
        tp_rank: int,
        device: Union[str, torch.device],
        dtype: torch.dtype,
    ):
        # check whether VIT tensorrt exist
        try:
            pass
        except ImportError:
            raise RuntimeError("tensorrt library not fonnd")

        try:
            # TODO(xyz): currently model_name_path is ugly, we should let model_name_path passed by the frontend in
            # environment variable
            model_name_path = ckpt_path.replace("/", "_")

            visual_trt_engine = MultiModalTRTEngine(
                model_name_path,
                vit_params.config.get("image_size"),
                device,
                dtype,
                vit_config=self.vit_config,
            )

            # TRT engine doesn't support TP, here we only generate trt engine on rank0 if trt engine is not cached
            if tp_rank == 0 and (
                (not MultiModalTRTEngine.trt_engine_cached(model_name_path, dtype))
                or self.vit_config.trt_cache_enabled == 0
            ):
                self._load_mm_weight(vit_params, dtype, device)

                # create cached dir if not exists
                output_dir = MultiModalTRTEngine.cache_path(
                    model_name_path, dtype, self.vit_config
                )
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                visual_trt_engine.export_onnx(self.mm_part.vit, tp_size)

                # eagerly gc VIT network, release GPU memory for generating trt engine
                self.gc_mm_part(vit_params)

                visual_trt_engine.generate_trt_engine()

                # create a completion file to mark that the trt engine has been generated and cached
                MultiModalTRTEngine.completion_file_path(model_name_path, dtype).touch()

            # for TP > 1, only rank0 will generate trt engine, other ranks will wait rank0 to generate trt engine
            if tp_size > 1:
                barrier(group=Group.TP)

            self.gc_mm_part(vit_params)
            # Currently, the multimodel network isn't split between devices. Only Rank 0 loads the weights.
            # After supporting TP mm network, we will remove the check here.
            if tp_rank == 0:
                visual_trt_engine.load_trt_engine()
                self.mm_part = visual_trt_engine

        except Exception as e:
            raise RuntimeError(f"init multimodal trt error: {e}")

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
            mm_related_params = model_config.mm_related_params
            self.init_mm_trt(
                model_config.ckpt_path,
                mm_related_params,
                tp_size,
                tp_rank,
                device,
            )
            return

        # wait rank0 finish loading weight, otherwise gang_server will die
        if tp_size > 1:
            barrier(group=Group.TP)
        # Currently, the multimodel network isn't split between devices. Only Rank 0 loads the weights.
        # After supporting TP mm network, we will remove the check here.
        if tp_rank >= 1:
            return

        # For trt engine, we don't need to load weight since its weight is inside trt engine.
        if isinstance(self.mm_part, MultiModalTRTEngine):
            return

        # mm_related_params is in model_config, not mm_model_config
        mm_related_params = model_config.mm_related_params
        self._load_mm_weight(mm_related_params, ctype, device)
