"""Stream the service's SigLIP, projector and image delimiters as one replica."""

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.read_spec import unwrap_checkpoint_tensor
from rtp_llm.models_py.registry import register_model


@register_model("deepseek_vl2_vision")
class DeepSeekVLV2VisionOwner(RtpModule):
    fastsafetensors_load_isolated = True
    _parts = (
        "vision",
        "projector",
        "image_newline",
        "view_seperator",
        "tile_indicators",
    )

    def __init__(self, model_config, load_config):
        super().__init__()
        embedding = model_config["embedding"]
        for name in self._parts:
            if hasattr(embedding, name):
                setattr(self, name, getattr(embedding, name))
        self.to(dtype=load_config.compute_dtype)

    def checkpoint_weight_name_filter(self):
        return lambda name: name.startswith(("vision.", "projector.")) or name in (
            "image_newline",
            "view_seperator",
            "tile_indicators",
        )

    def load_weights(self, weights):
        items = weights.items() if isinstance(weights, dict) else weights

        def staged():
            for name, value in items:
                tensor, _ = unwrap_checkpoint_tensor(value)
                # The legacy ViT reader staged floating-point weights through fp16.
                yield name, tensor.half() if tensor.is_floating_point() else tensor

        super().load_weights(staged())


def load_vision(embedding, ckpt_path, compute_dtype, device, load_method):
    model = NewModelLoader(
        model_config={"model_type": "deepseek_vl2_vision", "embedding": embedding},
        model_path=ckpt_path,
        load_config=NewLoaderConfig(
            compute_dtype=compute_dtype,
            device=device,
            load_method=getattr(load_method, "value", load_method),
        ),
    ).load()
    # Module.to may replace standalone parameters on a device transition; the
    # serving interface must use the final copies, including the delimiters.
    for name in model._parts:
        if hasattr(model, name):
            setattr(embedding, name, getattr(model, name))
    return embedding
