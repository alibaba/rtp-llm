"""Exact retained L0-L6 histories from the qualified scoped-prefill full run.

Adapted from the admitted three-layer official/native reconstructors. Each
stream substitutes its own saved FFN outputs; experts are never re-executed.
"""

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import AutoTokenizer

from rtp_llm.tools.dsv41_attention_history_replay import prepared_engram
from rtp_llm.tools.dsv41_attention_history_replay import (
    require_exact as numerical_exact,
)
from rtp_llm.tools.dsv41_attention_replay import (
    NASReader,
    history_identities,
    official_attention,
    run_official_attention,
)
from rtp_llm.tools.dsv41_query_boundary_replay import (
    KERNEL_SHA256,
    require_bytes,
    sha256,
)
from rtp_llm.tools.dsv41_query_owner_replay import tensor_digest
from rtp_llm.tools.dsv41_reference_probe import model_args

NUM_LAYERS = 7
FULL_SHA256 = "b88bb0061d6b583d43d2620b3c7788fb6a36f41e66c037fe0be9302b293f8c28"
PROFILE = "tp8_projection_mhc_fp32_compressor_recipe_sparse_sink_div_rn_expert_down_reference_mhc_post_reference_engram_reference_query_reference_output_reference_sparse_prefill_reference"
FLAGS = (
    "compressor_fused",
    "hca_bf16_projection",
    "sparse_sink_div_rn_prefill",
    "sparse_sink_div_rn_decode",
    "shared_down_reference",
    "routed_down_reference",
    "mhc_post_reference",
    "engram_projection_reference",
    "engram_gate_reference",
    "query_first_reference",
    "query_second_reference",
    "output_projection_reference",
    "sparse_prefill_reference",
)
HASHES = (
    "down_reference_kernel_sha256",
    "engram_reference_kernel_sha256",
    "query_reference_kernel_sha256",
    "output_reference_kernel_sha256",
    "sparse_prefill_reference_kernel_sha256",
)


def require_exact(expected, actual, label):
    result = numerical_exact(expected, actual, label)
    # Full engine and reconstructed decode expose different singleton batch
    # dimensions. Normalize only shape; retain every dtype and raw value bit.
    require_bytes(expected.cpu(), actual.reshape_as(expected).cpu(), label + ".bytes")
    result["normalized_bytes_equal"] = True
    result["capture_shape"] = list(actual.shape)
    return result


def validate_recipe(traces, *, kernel_sha256):
    if kernel_sha256 != KERNEL_SHA256 or not traces:
        raise ValueError("Deep history requires its pinned kernel and nonempty traces")
    recipe = dict(
        precision_profile=PROFILE, sparse_prefill_reference_scope="short-single-initial"
    )
    recipe.update({name: True for name in FLAGS})
    recipe.update({name: kernel_sha256 for name in HASHES})
    for trace in traces:
        metadata = trace["metadata"]
        for name, expected in recipe.items():
            actual = metadata.get(name)
            if actual != expected or type(actual) is not type(expected):
                raise ValueError("Deep history recipe differs: " + name)
    return recipe


def admit_traces(reference, native, comparisons, prompt_index, *, layers=NUM_LAYERS):
    if (
        type(layers) is not int
        or layers != NUM_LAYERS
        or type(prompt_index) is not int
        or prompt_index not in (0, 1)
    ):
        raise ValueError("Deep history requires its seven-layer prompt scope")
    count = 3 if prompt_index == 0 else 4
    if len(reference) != count or len(native) != count or len(comparisons) != count:
        raise ValueError("Deep history requires every qualified matching step")
    recipe = validate_recipe(native, kernel_sha256=KERNEL_SHA256)
    identities = history_identities(reference, native)
    needed = {
        f"layer{layer:02d}.ffn.{kind}"
        for layer in range(NUM_LAYERS)
        for kind in ("input", "output")
    }
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
            raise ValueError("Deep history requires audited matching consumed prefixes")
        metadata = trace["metadata"]
        consumed = sum((row["input_ids"] for row in identities[: step + 1]), [])
        if (
            type(metadata.get("prompt_index")) is not int
            or metadata.get("prompt_index") != prompt_index
            or metadata.get("prompt_token_ids") != identities[0]["input_ids"]
            or metadata.get("consumed_token_ids") != consumed
        ):
            raise ValueError("Deep history recorded token history differs")
        for current in (reference[step], trace):
            if not needed <= current["tensors"].keys():
                raise ValueError("Deep history lacks its complete FFN boundaries")
    return identities, recipe


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def reconstruct_official(
    module, converter, args, layout, reader, traces, identities, hashes
):
    shards = [
        official_attention(module, converter, args, reader, layer)
        for layer in range(NUM_LAYERS)
    ]
    block = SimpleNamespace(
        norm_eps=args.norm_eps,
        hc_mult=args.hc_mult,
        hc_sinkhorn_iters=args.hc_sinkhorn_iters,
        hc_eps=args.hc_eps,
    )
    steps, boundaries, rows_by_step = [], {}, []
    split, raw_mixes = module.hc_split_sinkhorn, []

    def record_split(mixes, *args, **kwargs):
        raw_mixes.append(mixes.clone())
        return split(mixes, *args, **kwargs)

    for step, (trace, identity, step_hashes) in enumerate(
        zip(traces, identities, hashes)
    ):
        tokens = torch.tensor([identity["input_ids"]], dtype=torch.long, device="cpu")
        hidden = (
            reader.rows("embed.weight", tokens)
            .cuda()
            .unsqueeze(2)
            .repeat(1, 1, args.hc_mult, 1)
        )
        pre_mix = module.make_identity_pre_mix(hidden, args.hc_mult)
        inputs, references, selected, mhc = [], [], {}, []
        for layer in range(NUM_LAYERS):
            captured = {"layer_input": hidden.clone()}
            if layer in layout.layer_ids:
                ids = step_hashes[:, :, layout.layer_ids.index(layer), :]
                engram, rows = prepared_engram(module, args, layout, reader, layer, ids)
                hidden = engram(hidden, ids.cuda())
                selected[layer] = rows
            captured["attention_residual"] = hidden.clone()
            captured["attn_read_mix"] = pre_mix.clone()
            with torch.device("cuda"), patch.object(
                module, "hc_split_sinkhorn", record_split
            ):
                attn_pre, attn_post, attn_comb = module.Block.hc_mixes(
                    block,
                    hidden,
                    reader.load(f"layers.{layer}.hc_attn_fn").float(),
                    reader.load(f"layers.{layer}.hc_attn_scale").float(),
                    reader.load(f"layers.{layer}.hc_attn_base").float(),
                )
                captured.update(
                    attn_raw_mixes=raw_mixes.pop(),
                    attn_pre_mix=attn_pre.clone(),
                    attn_post_mix=attn_post.clone(),
                    attn_comb_mix=attn_comb.clone(),
                )
                norm = module.RMSNorm(args.dim, args.norm_eps)
                norm.weight.copy_(reader.load(f"layers.{layer}.attn_norm.weight"))
                collapsed = module.Block.hc_pre(block, hidden, pre_mix)
                captured["attn_collapsed"] = collapsed.clone()
                x = norm(collapsed)
                captured["attn_norm"] = x.clone()
                reference = run_official_attention(
                    module, shards[layer], x, identity["start_pos"], history=True
                )
                if (
                    args.compress_ratios[layer]
                    and layer not in args.index_source_layers
                ):
                    source = max(
                        value for value in args.index_source_layers if value <= layer
                    )
                    reference["index_cache"] = references[source]["index_cache"].clone()
                mixed = module.Block.hc_post(
                    block, reference["output"], hidden, attn_post, attn_comb
                )
                captured.update(
                    attention_output=reference["output"].clone(),
                    attn_post=mixed.clone(),
                    ffn_read_mix=attn_pre.clone(),
                )
                ffn_pre, ffn_post, ffn_comb = module.Block.hc_mixes(
                    block,
                    mixed,
                    reader.load(f"layers.{layer}.hc_ffn_fn").float(),
                    reader.load(f"layers.{layer}.hc_ffn_scale").float(),
                    reader.load(f"layers.{layer}.hc_ffn_base").float(),
                )
                captured.update(
                    ffn_raw_mixes=raw_mixes.pop(),
                    ffn_pre_mix=ffn_pre.clone(),
                    ffn_post_mix=ffn_post.clone(),
                    ffn_comb_mix=ffn_comb.clone(),
                )
                norm.weight.copy_(reader.load(f"layers.{layer}.ffn_norm.weight"))
                collapsed = module.Block.hc_pre(block, mixed, attn_pre)
                captured["ffn_collapsed"] = collapsed.clone()
                ffn_input = norm(collapsed)
                captured["ffn_norm"] = ffn_input.clone()
            label = f"step{step}.layer{layer:02d}.ffn.input"
            boundaries[label] = require_exact(
                trace["tensors"][f"layer{layer:02d}.ffn.input"], ffn_input, label
            )
            inputs.append(x[0].clone())
            references.append(reference)
            if layer + 1 < NUM_LAYERS:
                ffn_output = (
                    trace["tensors"][f"layer{layer:02d}.ffn.output"]
                    .cuda()
                    .reshape_as(x)
                )
                hidden = module.Block.hc_post(
                    block, ffn_output, mixed, ffn_post, ffn_comb
                )
                captured.update(ffn_output=ffn_output.clone(), ffn_post=hidden.clone())
                pre_mix = ffn_pre
            mhc.append(captured)
        steps.append({"inputs": inputs, "reference": references, "mhc": mhc})
        rows_by_step.append(selected)
    return steps, boundaries, rows_by_step


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "qualification", "admission_manifest", "report"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--input-artifact", type=Path, required=True)
    parser.add_argument("--prompt-index", type=int, required=True, choices=(0, 1))
    cli = parser.parse_args()
    if cli.report.exists() or cli.input_artifact.exists():
        raise ValueError("Deep history artifacts must be fresh")
    if torch.cuda.get_device_capability() != (12, 0):
        raise ValueError("Deep history requires leased SM120 hardware")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_float32_matmul_precision("highest")
    repo, root = Path(__file__).resolve().parents[2], cli.checkpoint.resolve()
    if sha256(cli.qualification) != FULL_SHA256:
        raise ValueError("Deep history full qualification changed")
    qualification = json.loads(cli.qualification.read_text())
    if not (
        qualification["accepted"]
        and not qualification["changes"]
        and qualification["all_eight_gpus_idle"]
        and qualification["idle_no_signal_teardown"]
        and qualification["bazel_exit_code"] == qualification["analysis_exit_code"] == 0
    ):
        raise ValueError("Deep history requires accepted full execution")
    for name, expected in qualification["evidence_sha256"].items():
        if sha256(Path(name)) != expected:
            raise ValueError("Deep history full evidence changed: " + name)
    base = cli.qualification.name.removesuffix(".qualification.json")
    full = json.loads((cli.qualification.parent / (base + ".run.json")).read_text())
    analysis = json.loads(
        (cli.qualification.parent / (base + ".analysis.json")).read_text()
    )
    admission = json.loads(cli.admission_manifest.read_text())
    for name, expected in admission["source_sha256"].items():
        if sha256(repo / name) != expected:
            raise ValueError("Deep history source changed: " + name)
    for category in ("asset_sha256", "helper_sha256"):
        for name, expected in admission[category].items():
            if sha256(Path(name)) != expected:
                raise ValueError("Deep history asset/helper changed: " + name)
    if full["profiles"][-1] != PROFILE or full["mode"] != "full":
        raise ValueError("Deep history qualified profile differs")
    count = 3 if cli.prompt_index == 0 else 4
    reference_paths = [
        cli.qualification.parent
        / "reference-shared-eos-trace-2"
        / f"reference-prompt{cli.prompt_index}-step{step}.pt"
        for step in range(count)
    ]
    native_paths = [
        Path(full["trace_dir"])
        / PROFILE
        / f"native-prompt{cli.prompt_index}-step{step}.pt"
        for step in range(count)
    ]
    for path in reference_paths:
        if sha256(path) != admission["asset_sha256"][str(path)]:
            raise ValueError("Deep history reference trace changed")
    for path in native_paths:
        relative = str(path.relative_to(full["trace_dir"]))
        if sha256(path) != analysis["trace_metadata"][relative]["sha256"]:
            raise ValueError("Deep history native trace changed")
    reference = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in reference_paths
    ]
    native = [
        torch.load(path, map_location="cpu", weights_only=True) for path in native_paths
    ]
    comparisons = [
        row
        for row in analysis["new_profile_comparisons"]
        if row["prompt_index"] == cli.prompt_index
    ][:count]
    identities, recipe = admit_traces(reference, native, comparisons, cli.prompt_index)
    sys.path.insert(0, str(root / "inference"))
    official, converter, kernel = [
        importlib.import_module(name) for name in ("model", "convert", "kernel")
    ]
    for module in (official, converter, kernel):
        if Path(module.__file__).resolve() != root / "inference" / (
            module.__name__ + ".py"
        ):
            raise ValueError("Unexpected released deep history module")
    if sha256(Path(kernel.__file__)) != KERNEL_SHA256:
        raise ValueError("Deep history released kernel differs")
    args = model_args(official, root, 256)
    if args.n_layers != 40 or tuple(args.compress_ratios[:NUM_LAYERS]) != (
        0,
        0,
        2,
        2,
        2,
        2,
        2,
    ):
        raise ValueError("Deep history source schedule differs")
    reader = NASReader(root)
    tokenizer = AutoTokenizer.from_pretrained(
        root, trust_remote_code=False, local_files_only=True
    )
    layout = official.EngramLayout.from_args(args)
    if tuple(layout.layer_ids) != (1, 14):
        raise ValueError("Deep history Engram schedule differs")
    with torch.device("cpu"):
        hash_state = official.NgramHashState(args, layout, tokenizer)
        hashes = [
            hash_state(
                torch.tensor([identity["input_ids"]], device="cpu", dtype=torch.long),
                identity["start_pos"],
            )
            for identity in identities
        ]
    steps, official_controls, prepared_rows = reconstruct_official(
        official, converter, args, layout, reader, reference, identities, hashes
    )
    row_before = {
        f"{step}.{layer}": tensor_digest(value)
        for step, values in enumerate(prepared_rows)
        for layer, value in values.items()
    }
    from rtp_llm.tools.dsv41_deep_native_history_replay import reconstruct_native

    config = json.loads((root / "config.json").read_text())["text_config"]
    native_controls, records, captures, inputs, mhc = reconstruct_native(
        args,
        config,
        reader,
        steps,
        native,
        identities,
        hashes,
        prepared_rows,
        hash_state.token_map.tolist(),
        recipe,
        kernel,
    )
    row_after = {
        f"{step}.{layer}": tensor_digest(value)
        for step, values in enumerate(prepared_rows)
        for layer, value in values.items()
    }
    cached_weights = {
        name: tensor_digest(value) for name, value in reader.values.items()
    }
    if row_before != row_after or cached_weights != reader.hashes:
        raise AssertionError(
            "Deep history prepared rows or cached source weights changed"
        )
    if (
        len(official_controls) != count * NUM_LAYERS
        or len(native_controls) != count * NUM_LAYERS
    ):
        raise AssertionError("Deep history reconstruction coverage differs")
    raw = cpu_tree(
        {
            "identities": identities,
            "native_recipe": recipe,
            "layers": NUM_LAYERS,
            "official_inputs": [step["inputs"] for step in steps],
            "official_attention": [step["reference"] for step in steps],
            "official_mhc": [step["mhc"] for step in steps],
            "native_inputs": inputs,
            "native_attention": captures,
            "native_mhc": mhc,
            "engram_rows": prepared_rows,
            "engram_hashes": hashes,
        }
    )
    with cli.input_artifact.open("xb") as stream:
        torch.save(raw, stream)
    report = {
        "prompt_index": cli.prompt_index,
        "layers": NUM_LAYERS,
        "identities": identities,
        "native_recipe": recipe,
        "official_reconstruction": official_controls,
        "native_reconstruction": native_controls,
        "native_input_history": records,
        "weight_sha256": reader.hashes,
        "selected_row_reads": reader.selected_row_reads,
        "native_revision": subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={repo}",
                "-C",
                str(repo),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip(),
        "source_sha256": admission["source_sha256"],
        "full_qualification_sha256": FULL_SHA256,
        "input_artifact_sha256": sha256(cli.input_artifact),
        "trace_sha256": {
            str(path): sha256(path) for path in reference_paths + native_paths
        },
        "limitations": [
            "Seven short matching-prefix layers; saved own-stream FFN outputs substitute experts.",
            "No serving, performance, CP, graph or new full-model accuracy promotion.",
        ],
    }
    report.update(
        {
            "engram_block_size": official.fp8_block_size,
            "prepared_row_sha256": {"before": row_before, "after": row_after},
            "cached_weight_sha256": cached_weights,
        }
    )
    with cli.report.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        "DSV41_DEEP_HISTORY_CONTROLS "
        + json.dumps(
            {
                "layers": NUM_LAYERS,
                "steps": count,
                "ffn_inputs_exact": 2 * count * NUM_LAYERS,
                "weights": len(reader.hashes),
                "selected_row_reads": len(reader.selected_row_reads),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
