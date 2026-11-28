"""Direct checkpoint owners for Kimi-K2.5's MoonViT and projector."""

from collections.abc import Mapping

import torch

from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod,
    NewModelLoader,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.kimi_k25.vision_runtime import (
    KimiK25MultiModalProjector,
    KimiK25VisionConfig,
    MoonViT3dPretrainedModel,
    mm_projector_forward,
)
from rtp_llm.models_py.read_spec import unwrap_checkpoint_tensor


class KimiK25ForVisionEmbedding(RtpModule):
    fastsafetensors_load_isolated = True

    def resident_constructor_state(self):
        # Vision topology and nonpersistent rotary tables are entirely config-derived.
        return {}

    def __init__(self, model_config, load_config: NewLoaderConfig):
        super().__init__()
        if load_config.tp_size != 1 or load_config.tp_rank != 0:
            raise ValueError("Kimi-K2.5 vision is single-rank and replicated")
        if not isinstance(model_config, Mapping) or not isinstance(
            model_config.get("vision_config"), Mapping
        ):
            raise TypeError("Kimi-K2.5 requires model_config.vision_config")
        config = KimiK25VisionConfig(**model_config["vision_config"])
        self.vision_tower = MoonViT3dPretrainedModel(config)
        self.mm_projector = KimiK25MultiModalProjector(config)
        # Do not round non-persistent time/rotary tables to parameter dtype.
        for parameter in self.parameters():
            parameter.data = parameter.data.to(dtype=load_config.compute_dtype)

    def checkpoint_weight_name_filter(self):
        return lambda name: name.startswith(("vision_tower.", "mm_projector."))

    def load_weights(self, weights):
        iterator = weights.items() if isinstance(weights, dict) else weights
        dtype = self.vision_tower.patch_embed.proj.weight.dtype

        def staged_weights():
            for name, value in iterator:
                tensor, _ = unwrap_checkpoint_tensor(value)
                if tensor.is_floating_point():
                    tensor = tensor.half().to(dtype)
                yield name, tensor

        super().load_weights(staged_weights())

    def forward(self, pixel_values, grid_thws):
        return mm_projector_forward(
            self.mm_projector, self.vision_tower(pixel_values, grid_thws)
        )


def load_kimi_k25_vision(
    vision_config,
    model_path,
    compute_dtype,
    device,
    load_method=NewLoaderLoadMethod.SCRATCH,
):
    loader = NewModelLoader(
        model_config={"model_type": "kimi_k25_vision", "vision_config": vision_config},
        load_config=NewLoaderConfig(
            compute_dtype=compute_dtype, device=device, load_method=load_method
        ),
        model_path=model_path,
    )
    with torch.device("cpu"):
        return loader.load()
