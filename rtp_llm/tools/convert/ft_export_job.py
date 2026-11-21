"""Build an FT-format writer, without constructing or loading a serving model."""


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
    database = CkptDatabase(
        config.ckpt_path,
        config.ptuning_path,
        recycle_handles=model.loader_recycle_handles,
    )
    if len(config.lora_infos) == 1:
        for name, path in config.lora_infos.items():
            database.load_lora(name, path)
        database.dump_lora_info()

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
