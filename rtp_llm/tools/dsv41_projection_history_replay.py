"""Reconstruct admitted L0-L15 histories after block norm, WKV and W13 fixes.

Saved own-stream FFN outputs substitute experts. France's fourth consumed
history has diverged and is outside this diagnostic's scope.
"""

from rtp_llm.tools.dsv41_attention_replay import history_identities
from rtp_llm.tools.dsv41_corrected_history_replay import FLAGS, HASHES, KERNEL_SHA256
from rtp_llm.tools.dsv41_corrected_history_replay import PROFILE as PREVIOUS_PROFILE
from rtp_llm.tools.dsv41_corrected_history_replay import PROMPTS, PROVIDER_SHA256
from rtp_llm.tools.dsv41_corrected_history_replay import main as run_replay

NUM_LAYERS = 16
STEP_COUNTS = (3, 4)
FULL_SHA256 = "b8846d5478164e8e13b8863b247380165e82d78047a5a70df5dbf863403cbc4c"
PROFILE = PREVIOUS_PROFILE + "_block_norm_reference_kv_ref_j"
PROVIDERS = {
    "block_norm_provider_sha256": "e3b8166f62aa5ac46ff2db5205bddecc290055cb0fd27a125d7e1e6af6879d3e",
    "kv_projection_provider_sha256": "9a105d20b6b4507d299ce755bda6212f4886efba819e888853c76f02376e696d",
    "router_eager_provider_sha256": "92c10ee2c41bdf746883ea90c5b4e5a270e56b86ab3f13de604516c1a8c63f74",
    "expert_projection_provider_sha256": "e874a438f53ef8bc883d76e33848bf1edc1ca72fd270a3c85528d2bcc46fad09",
}


def validate_recipe(traces, *, kernel_sha256, provider_sha256):
    if (
        not traces
        or kernel_sha256 != KERNEL_SHA256
        or provider_sha256 != PROVIDER_SHA256
    ):
        raise ValueError("Projection history requires pinned providers and traces")
    recipe = {
        "precision_profile": PROFILE,
        "sparse_prefill_reference_scope": "short-single-initial",
        "mhc_collapse_reference": True,
        "decode_tilelang_reference": True,
        "decode_tilelang_reference_scope": "single-token-planar",
        "decode_tilelang_provider_sha256": provider_sha256,
        **dict.fromkeys(FLAGS, True),
        **dict.fromkeys(HASHES, kernel_sha256),
        **PROVIDERS,
        "block_norm_reference": True,
        "block_norm_reference_scope": "explicit-blocks",
        "block_norm_reference_dim": 5120,
        "kv_projection_reference": True,
        "kv_projection_reference_scope": "explicit-attentions",
        "kv_projection_reference_dim": 5120,
        "kv_projection_reference_kernel_sha256": kernel_sha256,
        "router_eager_reference": True,
        "router_eager_reference_scope": "explicit-gates",
        "router_eager_reference_dim": 5120,
        "router_eager_reference_owners": 40,
        "expert_projection_reference": True,
        "expert_projection_reference_scope": "explicit-shared-routed-w13",
        "expert_projection_reference_dim": 5120,
        "expert_projection_shared_owners": 40,
        "expert_projection_routed_owners": 40,
        "expert_projection_reference_kernel_sha256": kernel_sha256,
    }
    for trace in traces:
        for name, expected in recipe.items():
            actual = trace["metadata"].get(name)
            if actual != expected or type(actual) is not type(expected):
                raise ValueError("Projection history recipe differs: " + name)
    return recipe


def admit_traces(reference, native, comparisons, prompt_index, *, layers=NUM_LAYERS):
    if (
        type(layers) is not int
        or layers != NUM_LAYERS
        or type(prompt_index) is not int
        or prompt_index not in (0, 1)
    ):
        raise ValueError("Projection history requires its sixteen-layer prompt scope")
    count = STEP_COUNTS[prompt_index]
    if not all(len(values) == count for values in (reference, native, comparisons)):
        raise ValueError("Projection history requires every matching step only")
    recipe = validate_recipe(
        native, kernel_sha256=KERNEL_SHA256, provider_sha256=PROVIDER_SHA256
    )
    identities = history_identities(reference, native)
    if identities[0]["input_ids"] != PROMPTS[prompt_index]:
        raise ValueError("Projection history prompt differs")
    needed = {
        f"layer{layer:02d}.ffn.{kind}"
        for layer in range(NUM_LAYERS)
        for kind in ("input", "output")
    }
    consumed = []
    for step, (identity, record, trace) in enumerate(
        zip(identities, comparisons, native)
    ):
        if (
            record.get("profile") != PROFILE
            or type(record.get("prompt_index")) is not int
            or record.get("prompt_index") != prompt_index
            or type(record.get("step")) is not int
            or record.get("step") != step
            or record.get("matching_prefix") is not True
            or record.get("candidate_identity") != identity
            or record.get("reference_identity") != identity
        ):
            raise ValueError("Projection history requires audited matching prefixes")
        consumed.extend(identity["input_ids"])
        metadata = trace["metadata"]
        if (
            type(metadata.get("prompt_index")) is not int
            or metadata.get("prompt_index") != prompt_index
            or metadata.get("prompt_token_ids") != PROMPTS[prompt_index]
            or metadata.get("consumed_token_ids") != consumed
        ):
            raise ValueError("Projection history recorded token history differs")
        for current in (reference[step], trace):
            if not needed <= current["tensors"].keys():
                raise ValueError("Projection history lacks complete FFN boundaries")
    return identities, recipe


def validate_schedule(args, layout):
    schedules = (
        (args.compress_ratios[:NUM_LAYERS], (0, 0, *([2] * 14))),
        (args.kv_source_layers, (2, 8, 14, 20)),
        (args.index_source_layers, (2, 8, 14, 20, 24, 28, 32, 36)),
        (layout.layer_ids, (1, 14)),
    )
    if type(args.n_layers) is not int or args.n_layers != 40:
        raise ValueError("Projection history layer schedule differs")
    if any(
        tuple(actual) != expected or any(type(value) is not int for value in actual)
        for actual, expected in schedules
    ):
        raise ValueError("Projection history source/Engram schedule differs")


if __name__ == "__main__":
    run_replay(
        layers=NUM_LAYERS,
        step_counts=STEP_COUNTS,
        profile=PROFILE,
        full_sha256=FULL_SHA256,
        trace_admission=admit_traces,
        schedule_admission=validate_schedule,
        control_marker="DSV41_PROJECTION_HISTORY_CONTROLS",
        description=__doc__,
    )
