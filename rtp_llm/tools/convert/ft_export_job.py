"""Build an FT-format writer, without constructing or loading a serving model."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from rtp_llm.model_loader.load_config import LoadMethod

if TYPE_CHECKING:
    from rtp_llm.config.model_config import ModelConfig
    from rtp_llm.ops import HWKernelConfig, KVCacheConfig, ParallelismConfig


@dataclass(frozen=True)
class DraftFTDescriptor:
    """Offline language draft: the target owns tokenizer/task metadata."""

    model_config: "ModelConfig"
    parallelism_config: "ParallelismConfig"
    hw_kernel_config: "HWKernelConfig"
    kv_cache_config: "KVCacheConfig"
    get_weight_cls: Callable[[], type]
    load_method: LoadMethod = LoadMethod.AUTO
    merge_lora: bool = False
    force_cpu_load_weights: bool = False
    loader_recycle_handles: bool = False
    moe_pure_tp_preshard: bool = False

    def _init_custom_module(self):
        return None


def create_ft_export_job(model):
    """Use tokenizer/vocab/task metadata from an *unloaded* model descriptor.

    The rank-local public FT file layout predates the serving parameter layout.
    Its descriptors and transformations belong to this serialization boundary,
    not to service startup. In particular, custom classifier weights must be
    discovered before export, without calling the handler's runtime init.
    """
    from rtp_llm.model_loader.weight_module import CustomAtomicWeight
    from rtp_llm.models_py.downstream_heads import unpack_handler_mapping
    from rtp_llm.tools.convert.ft_export_weights import FTExportWeights
    from rtp_llm.utils.database import CkptDatabase
    from rtp_llm.utils.model_weight import CkptWeightInfo

    config = model.model_config
    # Width can split evenly while the cut still lands inside a Q head.
    # Use attention TP (not global TP) just as the descriptors do.
    tp = model.parallelism_config.get_attn_tp_size()
    heads = config.attn_config.head_num
    if tp < 1 or heads % tp:
        raise ValueError(
            f"Invalid FT attention TP: head_num={heads} must be divisible by tp_size={tp}"
        )
    # These draft formats contain private layers a target's dense adapter
    # cannot import. Validate the registered *draft* before emitting shards.
    kind = getattr(config, "model_type", None)
    writer_factory = None
    if kind in ("qwen_3_dflash", "qwen_3_moe_eagle3", "qwen_2-mtp"):
        from rtp_llm.models_py.registry import get_model_class

        owner = get_model_class(kind)
        if not callable(vars(owner).get("load_ft_checkpoint")):
            raise ValueError(f"{kind} cannot import its draft FT format; use the original HF draft")
        if kind != "qwen_2-mtp" and getattr(owner, "preserved_ft_parameter_loans", {}) != {
            "embed_tokens.weight": "global.embedding", "lm_head.weight": "global.lm_head"
        }:
            raise ValueError(f"{kind} lacks its target vocabulary FT loan contract")
        writer_factory = getattr(owner, "ft_export_writer", None)
    database = CkptDatabase(
        config.ckpt_path,
        config.ptuning_path,
        recycle_handles=model.loader_recycle_handles,
    )
    if len(config.lora_infos) == 1:
        for name, path in config.lora_infos.items():
            database.load_lora(name, path)
        database.dump_lora_info()

    if writer_factory is not None:
        return writer_factory(config, model.parallelism_config, database)

    descriptors = model.get_weight_cls()(
        model_config=config,
        parallelism_config=model.parallelism_config,
        hw_kernel_config=model.hw_kernel_config,
        kv_cache_config=model.kv_cache_config,
        merge_lora=model.merge_lora,
        load_method=model.load_method,
    )
    custom_module = model._init_custom_module()
    custom_fields = []
    if custom_module is not None:
        custom_fields = [
            CustomAtomicWeight(role, [CkptWeightInfo(checkpoint_name)], data_type=dtype)
            for role, checkpoint_name, dtype in map(
                unpack_handler_mapping, custom_module.checkpoint_weight_mappings()
            )
        ]
    return FTExportWeights(
        config,
        descriptors,
        custom_fields,
        database,
        force_cpu_load_weights=model.force_cpu_load_weights,
        moe_pure_tp_preshard=model.moe_pure_tp_preshard,
    )
