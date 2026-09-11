"""Generic worker preflight before weight loading and implementation groups."""

from rtp_llm.device.runtime import DeviceRuntimeContext
from rtp_llm.models_py.pluggable.control import verify_store_protocol
from rtp_llm.models_py.pluggable.factory import (
    ModuleBuildContext,
    ModuleSelectionContext,
)
from rtp_llm.models_py.pluggable.spec import canonical_json


def prepare_worker_model_context(
    model_config, engine_config, distributed_server, *, timeout_s
):
    config = engine_config.module_dispatch
    if config.mode == "legacy":
        return None
    from rtp_llm.model_factory import ModelFactory

    adapter = ModelFactory.get_model_cls(model_config.model_type).get_module_adapter()
    if adapter is None:
        raise ValueError(f"Model {model_config.model_type!r} has no module adapter")
    pc = engine_config.parallelism_config
    platform = DeviceRuntimeContext.detect(
        local_rank=int(pc.local_rank), requested=config.platform
    )
    ctx = ModuleBuildContext(
        adapter.registry(),
        ModuleSelectionContext(
            platform, canonical_json(adapter.metadata(model_config, engine_config))
        ),
        config,
        world_size=int(pc.world_size),
        model_adapter=adapter,
    )
    ctx.prepare([adapter.root_request(ctx.selection)])
    if int(pc.world_size) == 1:
        ctx.verify_protocol()
    else:
        generation = getattr(distributed_server, "_module_build_generation", 0) + 1
        distributed_server._module_build_generation = generation
        ctx.verify_protocol(
            lambda digest: verify_store_protocol(
                distributed_server.store,
                namespace=f"target-{generation}",
                rank=int(pc.world_rank),
                ranks=range(int(pc.world_size)),
                digest=digest,
                timeout_s=timeout_s,
            )
        )
    return ctx
