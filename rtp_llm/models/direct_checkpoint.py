"""Service boundary for loading directly into the Python model's parameters."""

import logging
import threading
from dataclasses import dataclass

import torch

from rtp_llm.models_py.borrowed_parameters import aliases_from_runtime_exports
from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod,
    NewModelLoader,
)
from rtp_llm.models_py.quant_methods import QuantizationConfig
from rtp_llm.models_py.registry import get_model_class, is_model_registered


@dataclass
class RuntimeWeightView:
    """Non-owning tensor references used by main's engine and custom handlers.

    The engine's MTP constructor uses len(weights) for layer cardinality even
    when a direct Python module owns the layer. An interior PP stage need not
    export any global tensors. Nothing here maps checkpoint names to tensors.
    """

    weights: list
    global_weights: dict
    dtype: torch.dtype
    device: str

    def get_global_weight(self, name):
        return self.global_weights[name]

    def get_global_weight_or_none(self, name):
        return self.global_weights.get(name)


def _runtime_view(module, config, device):
    exports = module.runtime_weight_view()
    if not isinstance(exports, dict) or not all(
        isinstance(k, str) and k and isinstance(t, torch.Tensor)
        for k, t in exports.items()
    ):
        raise TypeError("runtime_weight_view must return named tensor references")
    return RuntimeWeightView(
        [{} for _ in range(config.num_layers)], exports, config.compute_dtype, device
    )


def _resolve_owner_loans(service, config, device, publish_runtime):
    aliases = {}
    prepared_loan = None
    admin_lock = None
    if service._weight_alias_names:
        owner_cls = get_model_class(config.model_type)
        source = service._weight_alias_owner
        if source is None or source.py_model is None or source.weight is None:
            raise RuntimeError("draft parameter loans require an already loaded target")
        # Both C++ wrappers read the same Parameter storage. A target update
        # must also exclude a draft forward which is still submitting kernels.
        if publish_runtime:
            admin_lock = getattr(source.py_model, "_weight_update_lock", None)
            if admin_lock is None:
                raise RuntimeError("draft parameter loans require the target's update lock")
        exports = {
            name: source.weight.get_global_weight(name)
            for name in service._weight_alias_names
        }
        capture = getattr(owner_cls, "capture_runtime_loan", None)
        if capture is not None:
            prepared_loan = capture(
                source.py_model, source.model_config, config, exports, device
            )
        if prepared_loan is None:
            aliases = aliases_from_runtime_exports(
                source.py_model, exports, owner_cls.runtime_parameter_paths
            )
    return aliases, prepared_loan, admin_lock


def load_owned_model(service, *, publish_runtime=True):
    """Build locally first; never expose a half-loaded module to serving."""
    config = service.model_config
    if not is_model_registered(config.model_type):
        raise ValueError(
            f"Cannot serve model_type={config.model_type!r}: no direct checkpoint "
            "owner is registered. Architecture detection alone does not provide "
            "a loader. Check rtp_llm.models_py.registry.list_models()."
        )
    device = service._get_device_str()
    if (
        not publish_runtime
        and torch.device(device).type == "cuda"
        and not torch.cuda.is_available()
    ):
        device = "cpu"
    admin_lock = None
    source_quant = config.quant_config
    quant_type = "none"
    if source_quant is not None:
        # Shared main quant config parser distinguishes prequantized vs online
        # representations for the same get_method() value.
        from rtp_llm.models_py.quant_methods.moe_quant_key import (
            moe_checkpoint_method_key,
        )

        quant_type = moe_checkpoint_method_key(source_quant)
    method = getattr(service.load_method, "value", service.load_method)
    load_config = NewLoaderConfig.from_runtime(
        service.parallelism_config,
        device=device,

        compute_dtype=config.compute_dtype,
        load_method=method,
        quant_config=QuantizationConfig(
            quant_type,
            source_config=source_quant,
            hw_kernel_config=service.hw_kernel_config,
        ),
        moe_config=service.moe_config,
        fmha_config=service.fmha_config,
        py_hw_kernel_config=service.hw_kernel_config,
        max_generate_batch_size=service.max_generate_batch_size,
        force_cpu_load_weights=getattr(service, "force_cpu_load_weights", False),
        reserve_runtime_mem_mb=getattr(service, "reserve_runtime_mem_mb", 0),
        moe_pure_tp_preshard=getattr(service, "moe_pure_tp_preshard", False),
        custom_weight_mappings=(
            service.custom_module.checkpoint_weight_mappings()
            if service.custom_module is not None
            else ()
        ),
        custom_projection_layers=(
            getattr(
                getattr(service.custom_module, "handler", None),
                "checkpoint_projection_layers",
                lambda: (),
            )()
        ),
    )
    loader = NewModelLoader(
        config, load_config, model_path=config.ckpt_path
    )
    logging.info(
        "Using NewModelLoader for model_type=%s requested=%s", config.model_type, method
    )
    adapters = config.lora_infos or {}
    merged_adapter = (
        next(iter(adapters.items()))
        if len(adapters) == 1 and service.merge_lora
        else None
    )
    if adapters and merged_adapter is None:
        logging.warning(
            "Starting with unmodified base weights: %d configured LoRA adapter(s), "
            "merge_lora=%s. Per-request LoRA selection is not supported by the "
            "native engine; requests naming an adapter will be rejected.",
            len(adapters),
            service.merge_lora,
        )
    def prepare_loans():
        nonlocal admin_lock
        aliases, prepared, admin_lock = _resolve_owner_loans(service, config, device, publish_runtime)
        return aliases, prepared

    def load(**resident_options):
        # The resident manager and load route were agreed by this loader's
        # group, even for owners whose checkpoint reads are isolated.
        if resident_options.get("resident") is not None:
            resident_options["coordinated_resident_load"] = True
        return loader.load(
            startup_lora=merged_adapter,
            prepare_loans=prepare_loans,
            **resident_options,
        )

    residency = getattr(service, "resident_weights_config", None)
    module = (
        service._load_with_residency(device, load, group=loader.checkpoint_group())
        if residency and residency.enable
        else load()
    )
    actual = loader.loaded_method
    requested = load_config.load_method
    if actual is None or (
        requested != NewLoaderLoadMethod.AUTO and actual != requested
    ):
        raise RuntimeError(
            f"Direct checkpoint backend mismatch: requested={requested}, loaded={actual}"
        )
    if not publish_runtime:
        # Offline conversion needs the loaded, postprocessed parameter owner,
        # not serving state, a second loader, or a second checkpoint read.
        service.checkpoint_owner = module
        service.model_weights_loader = loader
        service.device = device
        return
    finalize_input_vocab = getattr(module, "finalize_runtime_input_vocab", None)
    if callable(finalize_input_vocab):
        finalize_input_vocab(config, service.tokenizer)
    view = _runtime_view(module, config, device)
    module.weight = view
    controller = None
    if config.eplb_config.enable_eplb():
        from rtp_llm.models_py.eplb_controller import create_stage_expert_balancer

        controller = create_stage_expert_balancer(loader, module, config)
    if service.custom_module is not None:
        service.custom_module.init(view)
    from rtp_llm.models_py.lora_checkpoint import RankLocalCheckpointAdapters
    from rtp_llm.models_py.new_weight_manager import NewLoaderWeightManager

    # Live base updates and adapter restore/merge coordinate on the same lock.
    # Construct both before publishing anything that RPC workers can observe.
    if admin_lock is None:
        admin_lock = threading.RLock()
    # Native admission accepts the static adapter's name as an assertion of
    # already-merged weights, never as a request to change shared storage.
    module.startup_merged_lora_name = merged_adapter[0] if merged_adapter else ""
    rank_local_lora = RankLocalCheckpointAdapters(
        module,
        loader,
        startup_adapters=(merged_adapter[0],) if merged_adapter else (),
        admin_lock=admin_lock,
        resident_weights_enabled=bool(residency and residency.enable),
    )
    weight_manager = NewLoaderWeightManager(
        module, loader, adapters=rank_local_lora, admin_lock=admin_lock,
        resident_weights_enabled=bool(residency and residency.enable),
    )
    service.device = device
    service.weight = view
    service.py_model = module
    service.checkpoint_owner = module
    service.model_weights_loader = loader
    service.py_eplb = controller
    service.weight_manager = weight_manager
    service.rank_local_lora = rank_local_lora
    logging.info(
        "NewModelLoader: model loaded successfully: model_type=%s selected=%s",
        config.model_type,
        actual.value,
    )
