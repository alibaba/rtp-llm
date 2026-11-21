"""Qwen3-MoE target: biasless GQA, Q/K norm and only routed experts."""

from dataclasses import dataclass

from rtp_llm.models_py.new_models.glm4_moe.parameters import install_gqa_moe_parameters


@dataclass(frozen=True)
class _RoutedLayout:
    routed_layers: frozenset
    use_qk_norm = True
    correction_bias = False
    shared_width = 0
    dense_width = 0

    @staticmethod
    def target_owns(name):
        # Unlike GLM NextN, plain Qwen3 has no appended draft layers.
        return True


def install_qwen3_moe_parameters(owner, cfg, *, layer_ids, partitioned):
    config = cfg["model_config"]
    routed = frozenset(range(cfg["num_layers"]))
    if (set(config.moe_layer_index) & routed) != routed:
        raise ValueError("Qwen3-MoE requires all target layers to be routed")
    if config.n_shared_experts or not config.qk_norm:
        raise ValueError("plain Qwen3-MoE requires Q/K norm and no shared experts")
    if int(config.scoring_func) != 0:
        raise ValueError("Qwen3-MoE requires softmax expert routing")
    install_gqa_moe_parameters(
        owner,
        cfg,
        layer_ids=layer_ids,
        partitioned=partitioned,
        layout=_RoutedLayout(routed),
        qkv_bias=False,
    )
