"""Consumable FP8/FP4 executor inputs, shared by direct and legacy loaders."""

from weakref import ref


class OwnerWeightSource:
    """Non-owning handle: executors must not retain their parent nn.Module."""

    def __init__(self, owner):
        self._owner = ref(owner)

    def take(self, name):
        owner = self._owner()
        if owner is None:
            raise RuntimeError(
                "FP8/FP4 weight owner was released before initialization"
            )
        return owner.take_executor_weight(name)


def take_weight(source, name):
    if isinstance(source, OwnerWeightSource):
        return source.take(name)

    # This side alone understands the OldLoader descriptor ABI. Both inputs
    # transfer each tensor when the existing executor is ready to transform it.
    from rtp_llm.utils.model_weight import W

    legacy_names = {
        "routed_w13": W.moe_w1,
        "routed_s13": W.moe_s1,
        "routed_w2": W.moe_w2,
        "routed_s2": W.moe_s2,
        "shared_w13": W.ffn_w13,
        "shared_s13": W.ffn_s13,
        "shared_w2": W.ffn_w2,
        "shared_s2": W.ffn_s2,
    }
    return source.pop(legacy_names[name])
