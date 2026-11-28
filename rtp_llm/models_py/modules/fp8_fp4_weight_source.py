"""Consumable FP8/FP4 executor inputs, shared by direct and legacy loaders."""

from weakref import ref


class OwnerWeightSource:
    """Non-owning handle: executors must not retain their parent nn.Module."""

    def __init__(self, owner):
        self._owner = ref(owner)

    @property
    def owner(self):
        owner = self._owner()
        if owner is None:
            raise RuntimeError(
                "FP8/FP4 weight owner was released before initialization"
            )
        return owner

    def take(self, name):
        return self.owner.take_executor_weight(name)


def take_weight(source, name):
    if isinstance(source, OwnerWeightSource):
        return source.take(name)

    # This side alone understands the OldLoader descriptor ABI. Both inputs
    # transfer each tensor when the existing executor is ready to transform it.
    return source.pop(_descriptor_names()[name])


def _descriptor_names():
    from rtp_llm.utils.model_weight import W

    return {
        "routed_w13": W.moe_w1,
        "routed_s13": W.moe_s1,
        "routed_w2": W.moe_w2,
        "routed_s2": W.moe_s2,
        "shared_w13": W.ffn_w13,
        "shared_s13": W.ffn_s13,
        "shared_w2": W.ffn_w2,
        "shared_s2": W.ffn_s2,
    }

def resident_init_tensors(source, *, transform_key, output_names, consumed_names, builder):
    """Run the same executor transform against a descriptor or a direct owner."""
    from rtp_llm.resident_weights import resident_init_tensors as transform

    if not isinstance(source, OwnerWeightSource):
        keys = _descriptor_names()
        return transform(source, transform_key, output_names, tuple(keys[n] for n in consumed_names), builder)
    binding = getattr(source.owner, "_resident_expert_inputs", None)
    if binding is None:
        return builder()
    bank, prefix = binding
    qualified = tuple(prefix + name for name in consumed_names)
    def consume():
        # The recorder already captured the signatures; let each private raw
        # stack die as soon as the builder transfers it to the executor.
        for key in qualified:
            bank.global_weights.pop(key)
        return builder()
    result = transform(
        bank.global_weights, prefix + transform_key, tuple(prefix + n for n in output_names),
        qualified, consume
    )
    bank.consumed.update(qualified)
    for name in consumed_names:
        source.owner._parameters[name] = None  # replay bypasses the consuming builder
    return result
