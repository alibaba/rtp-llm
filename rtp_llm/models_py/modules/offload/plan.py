from dataclasses import dataclass

from rtp_llm.models_py.modules.offload.layout import PagedRowLayout
from rtp_llm.models_py.modules.offload.runtime import SparseOffloadRuntime


@dataclass(frozen=True)
class LayerOffloadPlan:
    runtime: SparseOffloadRuntime
    storage_index: int
    begin: bool
    drain: bool

    @property
    def storage(self):
        return self.runtime.storages[self.storage_index]


def build_offload_plans(
    kv_cache, layer_descs
) -> dict[tuple[int, str], LayerOffloadPlan]:
    """Build one runtime per descriptor-declared selection group and cache tag."""
    if kv_cache is None:
        return {}
    groups = {}
    for layer, descs in enumerate(layer_descs):
        seen = set()
        for desc in descs:
            if desc.tag in seen:
                raise ValueError(f"duplicate cache tag {desc.tag} at layer {layer}")
            seen.add(desc.tag)
            storage = kv_cache.get_offload_storage(layer, desc.tag)
            enabled = desc.offload is not None and desc.offload.enabled
            if (storage is not None) != enabled:
                raise ValueError(
                    f"offload declaration/storage mismatch for {layer}:{desc.tag}"
                )
            if not enabled:
                continue
            if storage.owner_layer != layer or storage.tag != desc.tag:
                raise ValueError("offload storage does not match its cache group owner")
            shared = desc.offload_runtime_group
            key = (desc.tag, shared is not None, layer if shared is None else shared)
            groups.setdefault(key, []).append(storage)

    plans = {}
    for (tag, _, _), storages in groups.items():
        runtime = SparseOffloadRuntime(tuple(storages), PagedRowLayout())
        for index, storage in enumerate(storages):
            plans[storage.owner_layer, tag] = LayerOffloadPlan(
                runtime, index, index == 0, index == len(storages) - 1
            )
    return plans
