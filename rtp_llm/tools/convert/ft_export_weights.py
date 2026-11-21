"""Convert original checkpoint layouts into FT records, for the offline tool only.

FT's on-disk tensor names and TP axes are a file-format contract. They need not
match NewLoader parameter names or owning devices. This adapter never creates a
serving model, chooses a loading backend, or installs weights into a model.
"""

from importlib import import_module

from rtp_llm.device import get_current_device
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.tensor_source import DatabaseTensorSource
from rtp_llm.ops import TaskType
from rtp_llm.tools.convert.ft_export_writer import write_ft_shards
from rtp_llm.utils.model_weight import W


class FTExportWeights:
    def __init__(
        self, model_config, weights_info, misc_weights_info, database, *,
        force_cpu_load_weights=False, moe_pure_tp_preshard=False,
    ):
        from rtp_llm.config.quant_config import (
            Fp8BlockWiseQuantConfig,
            Fp8PerChannelCompressedQuantConfig,
            Fp8PerChannelQuarkQuantConfig,
        )

        # The converter selects transforms from the descriptor registry when
        # constructing fields. Import the matching plugin before that point:
        # otherwise the unquantized field writes the kernel without its scales.
        quant = getattr(model_config, "quant_config", None)
        if isinstance(quant, Fp8BlockWiseQuantConfig):
            import_module("rtp_llm.model_loader.per_block_fp8_quant_weight")
        elif isinstance(
            quant, (Fp8PerChannelCompressedQuantConfig, Fp8PerChannelQuarkQuantConfig)
        ):
            import_module("rtp_llm.model_loader.per_channel_fp8_quant_weight")

        self._info = weights_info.create_model_weight_info(database)
        self._custom = tuple(misc_weights_info or ())
        self._task_type = model_config.task_type
        self._attention_only = weights_info.is_attn_model
        placement = LoadConfig.create_redundant_expert(
            layer_num=model_config.num_layers,
            expert_num=model_config.expert_num,
            ep_size=weights_info.ep_size,
            num_nodes=weights_info.num_nodes,
            phy_exp_num=weights_info.phy_exp_num_,
            phy2log_path=model_config.phy2log_path,
        )
        self._config = weights_info.create_load_config(
            compute_dtype=model_config.compute_dtype, database=database,
            phy2log=placement, exported_device=get_current_device(),
            force_cpu_load_weights=force_cpu_load_weights,
            moe_pure_tp_preshard=moe_pure_tp_preshard,
        )

    def _records(self, device):
        config = self._config
        source = DatabaseTensorSource(config.database)
        if not self._attention_only and self._info.layer_weights:
            for layer_id in config.layer_range:
                layer = {}
                for field in self._info.layer_weights[layer_id]:
                    layer.update(field.load(source, layer_id, device, config))
                for name, tensor in layer.items():
                    yield layer_id, name, tensor
        for field in self._info.weights:
            if self._task_type != TaskType.LANGUAGE_MODEL and field.name == W.lm_head:
                continue
            if field.name in (W.embedding, W.positional_embedding) and 0 not in config.layer_range:
                continue
            if (field.name in (W.lm_head, W.lm_head_b) or field.name.startswith("final_layernorm.")) and config.num_layers - 1 not in config.layer_range:
                continue
            for name, tensor in field.load(source, None, device, config).items():
                yield None, name, tensor
        for field in self._custom:
            for name, tensor in field.load(source, None, device, config).items():
                yield None, name, tensor

    def dump_weight_as_ft_style(self, device, output_dir):
        config = self._config
        if config.is_ft_style_weight:
            raise ValueError("FT export requires a non-FT input checkpoint")
        if config.layer_range != range(config.num_layers):
            raise ValueError("FT export requires full model depth, not a partial PP stage")
        write_ft_shards(
            self._records(device), output_dir,
            tp_rank=config.tp_rank, dp_rank=config.dp_rank, ep_rank=config.ep_rank,
        )
