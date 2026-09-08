import gc
from typing import Dict, Optional

import torch

from rtp_llm.model_loader.load_config import LoadMethod
from rtp_llm.model_loader.model_weight_info import ModelDeployWeightInfo
from rtp_llm.utils.database import BaseDatabase


class MultimodalMixinLoader:
    """
    Lightweight loader for multimodal mixin weights.

    It mirrors the constructor shape of ModelLoader but only retains
    the minimal pieces needed for multimodal mixin scenarios.
    """

    def __init__(
        self,
        weights_info: ModelDeployWeightInfo,
        database: BaseDatabase,
        load_method: LoadMethod = LoadMethod.AUTO,
    ):
        self.weights_info = weights_info
        self.database = database
        self.load_method = load_method

    def load_weights(
        self, device: str = "cpu", data_type: torch.dtype = torch.float16
    ) -> Dict[str, torch.Tensor]:
        """
        Load multimodal (ViT) weights from the given database.

        This is a lightweight loader that avoids ModelConfig dependency.
        Returns a flat dict mapping weight_name -> tensor moved to `device`.
        """
        # Prefer a direct weight info getter if provided by weights_info.
        weight_info = self.weights_info.get_weight_info()

        loaded: Dict[str, torch.Tensor] = {}
        for weight in weight_info.weights:
            tensors = []
            for ckpt in getattr(weight, "weights", []):
                name = ckpt.tensor_name(None)
                parts = self.database.load_tensor(name, data_type=data_type)
                merged = ckpt.merge_fun(parts)
                tensors.append(merged)
            processed = weight.process_fun(tensors)
            loaded[weight.name] = processed.to(device)
        return loaded

    def force_clean_cuda_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
