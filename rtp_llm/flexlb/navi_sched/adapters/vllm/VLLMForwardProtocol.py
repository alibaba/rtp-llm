from functools import lru_cache

import msgspec


@lru_cache(maxsize=8)
def forward_request_type(request_type):
    return msgspec.defstruct(
        "ForwardEngineCoreRequest",
        [],
        bases=(request_type,),
        array_like=False,
    )


@lru_cache(maxsize=8)
def forward_outputs_type(output_type, outputs_type):
    names = set(output_type.__struct_fields__)
    fields = [
        (name, int | None, None)
        for name in (
            "num_cached_tokens",
            "num_external_computed_tokens",
        )
    ]
    if "prefill_stats" not in names:
        fields.append(("prefill_stats", dict[str, int] | None, None))
    wire_output = msgspec.defstruct(
        "ForwardEngineCoreOutput",
        fields,
        bases=(output_type,),
        array_like=False,
    )
    return msgspec.defstruct(
        "ForwardEngineCoreOutputs",
        [("outputs", list[wire_output], [])],
        bases=(outputs_type,),
        array_like=False,
    )


def forward_request_fields(request):
    # Native vLLM arrays change field positions between releases. Only the
    # cross-node representation uses names; tensor auxiliary frames stay native.
    return msgspec.structs.asdict(request)


def forward_outputs_fields(outputs):
    fields = msgspec.structs.asdict(outputs)
    fields["outputs"] = [msgspec.structs.asdict(value) for value in outputs.outputs]
    return fields


def restore_cache_statistics(
    output, native_output_type, prompt_tokens, cached_tokens, external_cached_tokens
):
    stats = getattr(output, "prefill_stats", None)
    if stats is not None:
        if isinstance(stats, dict):
            cached_tokens = stats["num_cached_tokens"]
            external_cached_tokens = stats["num_external_cached_tokens"]
        else:
            cached_tokens = stats.num_cached_tokens
            external_cached_tokens = stats.num_external_cached_tokens
    elif getattr(output, "num_cached_tokens", None) is not None:
        cached_tokens = output.num_cached_tokens
        external_cached_tokens = output.num_external_computed_tokens
        if (
            "prefill_stats" in native_output_type.__struct_fields__
            and prompt_tokens is not None
        ):
            from vllm.v1.metrics.stats import PrefillStats

            output.prefill_stats = PrefillStats(
                num_prompt_tokens=prompt_tokens,
                num_computed_tokens=max(0, prompt_tokens - cached_tokens),
                num_cached_tokens=cached_tokens,
                num_local_cached_tokens=cached_tokens - external_cached_tokens,
                num_external_cached_tokens=external_cached_tokens,
            )
    if "num_cached_tokens" in native_output_type.__struct_fields__:
        output.num_cached_tokens = cached_tokens
        output.num_external_computed_tokens = external_cached_tokens
    return cached_tokens, external_cached_tokens
