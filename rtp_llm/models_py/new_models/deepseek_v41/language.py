"""V4.1 target ownership on the unmodified main target forward/cache runtime."""

import logging

import torch

from rtp_llm.models_py.model_desc.deepseek_v41_model import (
    DeepSeekV41Model,
    _V41ImageEmbedding,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.deepseek_v41.owner import OwnedV41Mixin


class _OwnedImageEmbedding(_V41ImageEmbedding):
    def __init__(self, base, owner):
        torch.nn.Module.__init__(self)
        self.base = base
        # A runtime callback is not a child module: registering the model here
        # would make model -> embed -> owner a cyclic parameter graph.
        object.__setattr__(self, "owner", owner)


class DeepSeekV41ForCausalLM(OwnedV41Mixin, DeepSeekV41Model, RtpModule):
    def _load_owned_extra_weights(self):
        from rtp_llm.models_py.modules.dsv41.engram import EngramLayout, NgramHashState
        from rtp_llm.models_py.new_models.deepseek_v41.engram import OwnedEngram

        if not isinstance(self.v4.embed, _OwnedImageEmbedding):
            self.v4.embed = _OwnedImageEmbedding(self.v4.embed, self)
        config = self._v4_args.v41_config
        layout = EngramLayout(config)
        if not layout.layer_ids:
            return
        if self._owned_source_method is None:
            raise RuntimeError(
                "NewLoader must publish its selected method before V4.1 Engram loading"
            )
        device = self.v4.embed.weight.device
        self._engram_hash_state = NgramHashState(
            layout, self._owned_checkpoint_path, device=device
        )
        loadable = [
            layer_id for layer_id in layout.layer_ids if layer_id < len(self.v4.layers)
        ]
        rotation = 0
        if (
            len(loadable) > 1
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            rotation = torch.distributed.get_rank() % len(loadable)
        for layer_id in loadable[rotation:] + loadable[:rotation]:
            self.v4.layers[layer_id].engram = OwnedEngram.from_checkpoint(
                config,
                layer_id,
                self._owned_checkpoint_path,
                device,
                load_method=self._owned_source_method,
                resident_weights=getattr(getattr(self, "_resident_bank", None), "global_weights", None),
            )
        self._engram_layers = tuple(
            (layer_id, layout.layer_ids.index(layer_id)) for layer_id in loadable
        )
        if self._engram_layers:
            logging.getLogger(__name__).info(
                "DeepSeek V4.1 Engram initialized on layers %s",
                [layer_id for layer_id, _ in self._engram_layers],
            )
