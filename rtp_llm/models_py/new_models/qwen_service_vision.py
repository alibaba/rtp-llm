"""Direct checkpoint owner for main's Qwen service vision towers."""

import torch

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule, copy_weight_
from rtp_llm.models_py.read_spec import unwrap_checkpoint_tensor
from rtp_llm.models_py.registry import register_model


@register_model("qwen_service_vision")
class QwenServiceVisionOwner(RtpModule):
    # A service replica owns its whole visual tower even under TP/CP.
    fastsafetensors_load_isolated = True

    def __init__(self, model_config, load_config):
        super().__init__()
        self.visual = model_config["visual"]
        # A pretrained model constructor may select FP32 internally even when
        # the service requests BF16. Only checkpoint-owned tensors are cast;
        # leave non-persistent rotary constants in their constructor dtype.
        with torch.no_grad():
            for module in self.visual.modules():
                tensors = list(module.parameters(recurse=False))
                tensors.extend(
                    buf
                    for key, buf in module.named_buffers(recurse=False)
                    if key not in module._non_persistent_buffers_set
                )
                for tensor in tensors:
                    if (
                        tensor.is_floating_point()
                        and tensor.dtype != load_config.compute_dtype
                    ):
                        tensor.data = tensor.data.to(load_config.compute_dtype)
        self.prefix = model_config["vision_prefix"]
        self._observed_prefix = None

    def checkpoint_weight_name_filter(self):
        if self.prefix is not None:
            return lambda name: name.startswith(self.prefix)
        return lambda name: name.startswith("visual.") or ".visual." in name

    def load_weights(self, weights):
        seen = set()
        half_masks = {}
        params = dict(self.visual.named_parameters())

        def staged():
            for key, value in weights.items() if isinstance(weights, dict) else weights:
                prefix, _, name = key.rpartition("visual.")
                if self._observed_prefix is None:
                    self._observed_prefix = prefix
                if prefix != self._observed_prefix or name in seen:
                    raise ValueError(f"ambiguous visual checkpoint key: {key}")
                seen.add(name)
                tensor, _ = unwrap_checkpoint_tensor(value)
                # MultimodalMixinLoader stages floating weights through fp16.
                if tensor.is_floating_point():
                    tensor = tensor.to(torch.float16)
                # Main's Qwen2.5-VL MLP owns gate/up in one parameter. Copy each
                # half as it arrives; don't retain any shard-backed tensor.
                path, _, field = name.rpartition(".")
                block, _, proj = path.rpartition(".")
                fused = f"{block}.up_gate_proj.{field}"
                if proj in ("gate_proj", "up_proj") and fused in params:
                    index = int(proj == "up_proj")
                    if fused in seen:
                        raise ValueError(
                            f"both fused and split visual keys for {fused}"
                        )
                    target = params[fused]
                    if target.shape[0] % 2:
                        raise ValueError(f"invalid fused visual parameter {fused}")
                    copy_weight_(target.chunk(2, dim=0)[index], tensor, key)
                    half_masks[fused] = half_masks.get(fused, 0) | (1 << index)
                    if half_masks[fused] == 3:
                        # Dispatch the completed target to record its identity.
                        yield "visual." + fused, target.detach()
                else:
                    if name in half_masks:
                        raise ValueError(f"both fused and split visual keys for {name}")
                    yield "visual." + name, tensor

        super().load_weights(staged())


def load_service_vision(
    visual, model_path, compute_dtype, device, load_method, *, prefix
):
    loader = NewModelLoader(
        model_config={
            "model_type": "qwen_service_vision",
            "vision_prefix": prefix,
            "visual": visual,
        },
        load_config=NewLoaderConfig(
            compute_dtype=compute_dtype,
            device=device,
            load_method=getattr(load_method, "value", load_method),
        ),
        model_path=model_path,
    )
    return loader.load().visual
