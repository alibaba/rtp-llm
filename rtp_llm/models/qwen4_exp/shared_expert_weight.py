"""Shared-expert TP sharding, independent of routed-expert EP."""

from rtp_llm.model_loader.ffn_weight import FfnWeight


class Qwen4SharedFfnWeight(FfnWeight):
    @staticmethod
    def _shared_load_config(load_config):
        return load_config.model_copy(
            update={
                "ffn_tp_size": load_config.tp_size,
                "ffn_tp_rank": load_config.tp_rank,
            }
        )

    def _split(self, tensor, load_config):
        return super()._split(tensor, self._shared_load_config(load_config))

    def _split_lora(self, tensor, load_config):
        return super()._split_lora(tensor, self._shared_load_config(load_config))

    def update(self, tensor, device, load_config, **kwargs):
        return super().update(
            tensor, device, self._shared_load_config(load_config), **kwargs
        )
