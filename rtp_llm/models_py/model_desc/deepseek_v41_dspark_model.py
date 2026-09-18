"""Built-in V4.1 DSpark with delayed mHC and three independent SWA caches."""

from rtp_llm.models_py.model_desc.deepseek_v4_dspark_model import DeepSeekV4DSparkModel
from rtp_llm.models_py.model_desc.deepseek_v41_model import configure_v41


class DeepSeekV41DSparkModel(DeepSeekV4DSparkModel):
    def __init__(self, model_config, *args, **kwargs) -> None:
        super().__init__(model_config, *args, **kwargs)
        configure_v41(self, model_config)
        config = self._v4_args.v41_config
        self._v4_args.n_routed_experts = int(config["dspark_n_routed_experts"])
        self._v4_args.n_activated_experts = int(config["dspark_num_experts_per_tok"])
        # Draft caches hold each draft layer's projection of shared target
        # features. Main-backbone KV sharing must never alias these caches.
        config["kv_source_layer_ids"] = []
        config["compress_ratios"] = [0] * self._v4_args.n_layers


__all__ = ["DeepSeekV41DSparkModel"]
