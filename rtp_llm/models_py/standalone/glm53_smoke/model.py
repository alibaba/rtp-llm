"""Select four real MoE layers without substituting their implementation."""

from __future__ import annotations

SOURCE_LAYERS = (4, 5, 6, 7)


def remap_weights(info, source_layers=SOURCE_LAYERS):
    if len(info.layer_weights) != len(source_layers):
        raise ValueError("four-layer checkpoint mapping does not match manifest")
    for modules, source in zip(info.layer_weights, source_layers):
        for module in modules:
            for component in module.get_components():
                for weight in getattr(component, "weights", None) or ():
                    if isinstance(weight.name, str):
                        weight.name = weight.name.replace("{i}", str(source))


def install():
    from rtp_llm.models import glm5_3_flash as module

    cls = module.Glm53Flash
    if getattr(cls, "_four_layer_smoke", False):
        return
    original_config = cls._create_config.__func__
    original_weight = module.Glm53FlashWeight

    class FourLayerWeight(original_weight):
        def _get_weight_info(self):
            info = super()._get_weight_info()
            remap_weights(info)
            return info

    def config(model_cls, path):
        result = original_config(model_cls, path)
        schedule = result.hybrid_attention_config.hybrid_attention_types
        if not all(i in result.moe_layer_index for i in SOURCE_LAYERS):
            raise ValueError("smoke requires four checkpoint MoE layers")
        selected = [schedule[i] for i in SOURCE_LAYERS]
        if not (selected[0] == selected[1] == selected[2] != selected[3]):
            raise ValueError("checkpoint does not contain the expected 3 KDA + 1 MLA")
        result.num_layers = 4
        result.moe_layer_index = list(range(4))
        result.hybrid_attention_config.hybrid_attention_types = selected
        result.attn_config.indexer_layer_ids = [3]
        result.indexer_types = [result.indexer_types[i] for i in SOURCE_LAYERS]
        return result

    cls._create_config = classmethod(config)
    cls.get_weight_cls = staticmethod(lambda: FourLayerWeight)
    cls._four_layer_smoke = True
