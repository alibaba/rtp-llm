"""Retain L0-L9 under the qualified mHC/TileLang recipe on matching prefixes.

Each stream substitutes its own saved FFN outputs. Exact saved FFN input bytes
are required before interpreting any internal operator comparison.
"""

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from rtp_llm.tools.dsv41_attention_replay import NASReader, history_identities
from rtp_llm.tools.dsv41_deep_attention_history_replay import (
    FLAGS,
    HASHES,
    KERNEL_SHA256,
)
from rtp_llm.tools.dsv41_deep_attention_history_replay import (
    PROFILE as PREVIOUS_PROFILE,
)
from rtp_llm.tools.dsv41_deep_attention_history_replay import (
    cpu_tree,
    reconstruct_official,
)
from rtp_llm.tools.dsv41_query_boundary_replay import sha256
from rtp_llm.tools.dsv41_query_owner_replay import tensor_digest
from rtp_llm.tools.dsv41_reference_probe import model_args

NUM_LAYERS = 10
NUM_STEPS = 4
FULL_SHA256 = "9bec58af8cebbe290fcbc03ac14add62a7f075fa7151edc3ace61f2dcf2016e6"
PROVIDER_SHA256 = "e60d8d50b2ef75f2cc88db5caf5e1e45ff6083d209a2ce6f574a2df87fdd6c77"
PROFILE = PREVIOUS_PROFILE + "_mhc_collapse_reference_decode_tilelang_reference"
PROMPTS = ([671, 6102, 294, 8760, 344], [20, 940, 223, 20, 438])


def validate_recipe(traces, *, kernel_sha256, provider_sha256):
    if (
        not traces
        or kernel_sha256 != KERNEL_SHA256
        or provider_sha256 != PROVIDER_SHA256
    ):
        raise ValueError("Corrected history requires its pinned providers and traces")
    recipe = {
        "precision_profile": PROFILE,
        "sparse_prefill_reference_scope": "short-single-initial",
        "mhc_collapse_reference": True,
        "decode_tilelang_reference": True,
        "decode_tilelang_reference_scope": "single-token-planar",
        "decode_tilelang_provider_sha256": provider_sha256,
        **dict.fromkeys(FLAGS, True),
        **dict.fromkeys(HASHES, kernel_sha256),
    }
    for trace in traces:
        for name, expected in recipe.items():
            actual = trace["metadata"].get(name)
            if actual != expected or type(actual) is not type(expected):
                raise ValueError("Corrected history recipe differs: " + name)
    return recipe


def admit_traces(reference, native, comparisons, prompt_index, *, layers=NUM_LAYERS):
    if (
        type(layers) is not int
        or layers != NUM_LAYERS
        or type(prompt_index) is not int
        or prompt_index not in (0, 1)
    ):
        raise ValueError("Corrected history requires its ten-layer prompt scope")
    if not all(len(values) == NUM_STEPS for values in (reference, native, comparisons)):
        raise ValueError("Corrected history requires every qualified matching step")
    recipe = validate_recipe(
        native, kernel_sha256=KERNEL_SHA256, provider_sha256=PROVIDER_SHA256
    )
    identities = history_identities(reference, native)
    if identities[0]["input_ids"] != PROMPTS[prompt_index]:
        raise ValueError("Corrected history prompt differs")
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
            raise ValueError("Corrected history requires audited matching prefixes")
        consumed.extend(identity["input_ids"])
        metadata = trace["metadata"]
        if (
            type(metadata.get("prompt_index")) is not int
            or metadata.get("prompt_index") != prompt_index
            or metadata.get("prompt_token_ids") != PROMPTS[prompt_index]
            or metadata.get("consumed_token_ids") != consumed
        ):
            raise ValueError("Corrected history recorded token history differs")
        for current in (reference[step], trace):
            if not needed <= current["tensors"].keys():
                raise ValueError("Corrected history lacks complete FFN boundaries")
    return identities, recipe


def validate_schedule(args, layout):
    schedules = (
        (args.compress_ratios[:NUM_LAYERS], (0, 0, *([2] * 8))),
        (args.kv_source_layers, (2, 8, 14, 20)),
        (args.index_source_layers, (2, 8, 14, 20, 24, 28, 32, 36)),
        (layout.layer_ids, (1, 14)),
    )
    if type(args.n_layers) is not int or args.n_layers != 40:
        raise ValueError("Corrected history layer schedule differs")
    if any(
        tuple(actual) != expected or any(type(value) is not int for value in actual)
        for actual, expected in schedules
    ):
        raise ValueError("Corrected history source/Engram schedule differs")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "qualification", "admission_manifest", "report"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--input-artifact", type=Path, required=True)
    parser.add_argument("--prompt-index", type=int, choices=(0, 1), required=True)
    cli = parser.parse_args()
    if cli.report.exists() or cli.input_artifact.exists():
        raise ValueError("Corrected history artifacts must be fresh")
    if torch.cuda.get_device_capability() != (12, 0):
        raise ValueError("Corrected history requires leased SM120 hardware")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_float32_matmul_precision("highest")
    repo, root = Path(__file__).resolve().parents[2], cli.checkpoint.resolve()
    if sha256(cli.qualification) != FULL_SHA256:
        raise ValueError("Corrected history full qualification changed")
    qualification = json.loads(cli.qualification.read_text())
    if not (
        qualification["accepted"]
        and not qualification["changes"]
        and qualification["all_eight_gpus_idle"]
        and qualification["idle_no_signal_teardown"]
        and qualification["bazel_exit_code"] == qualification["analysis_exit_code"] == 0
    ):
        raise ValueError("Corrected history requires accepted full execution")
    for name, expected in qualification["evidence_sha256"].items():
        if sha256(Path(name)) != expected:
            raise ValueError("Corrected history full evidence changed: " + name)
    base = cli.qualification.name.removesuffix(".qualification.json")
    art = cli.qualification.parent
    full = json.loads((art / (base + ".run.json")).read_text())
    analysis = json.loads((art / (base + ".analysis.json")).read_text())
    admission = json.loads(cli.admission_manifest.read_text())
    for name, expected in admission["source_sha256"].items():
        if sha256(repo / name) != expected:
            raise ValueError("Corrected history source changed: " + name)
    for category in ("asset_sha256", "helper_sha256"):
        for name, expected in admission[category].items():
            if sha256(Path(name)) != expected:
                raise ValueError("Corrected history asset/helper changed: " + name)
    provider = repo / "rtp_llm/models_py/modules/dsv4/tilelang_kernels.py"
    if sha256(provider) != PROVIDER_SHA256:
        raise ValueError("Corrected history maintained provider changed")
    if full["mode"] != "full" or full["profiles"][-1] != PROFILE:
        raise ValueError("Corrected history qualified profile differs")
    reference_paths = [
        art
        / "reference-shared-eos-trace-2"
        / f"reference-prompt{cli.prompt_index}-step{step}.pt"
        for step in range(NUM_STEPS)
    ]
    native_paths = [
        Path(full["trace_dir"])
        / PROFILE
        / f"native-prompt{cli.prompt_index}-step{step}.pt"
        for step in range(NUM_STEPS)
    ]
    for path in reference_paths + native_paths:
        expected = admission["asset_sha256"][str(path)]
        if sha256(path) != expected:
            raise ValueError("Corrected history source trace changed")
        if path in native_paths:
            relative = str(path.relative_to(full["trace_dir"]))
            if analysis["trace_metadata"][relative]["sha256"] != expected:
                raise ValueError("Corrected history audited trace hash differs")
    reference, native = [
        [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
        for paths in (reference_paths, native_paths)
    ]
    comparisons = [
        row
        for row in analysis["new_profile_comparisons"]
        if row["profile"] == PROFILE and row["prompt_index"] == cli.prompt_index
    ]
    identities, recipe = admit_traces(reference, native, comparisons, cli.prompt_index)
    sys.path.insert(0, str(root / "inference"))
    official, converter, kernel = [
        importlib.import_module(name) for name in ("model", "convert", "kernel")
    ]
    for module in (official, converter, kernel):
        if Path(module.__file__).resolve() != root / "inference" / (
            module.__name__ + ".py"
        ):
            raise ValueError("Unexpected released corrected history module")
    if sha256(Path(kernel.__file__)) != KERNEL_SHA256:
        raise ValueError("Corrected history released kernel differs")
    args = model_args(official, root, 256)
    layout = official.EngramLayout.from_args(args)
    validate_schedule(args, layout)
    reader = NASReader(root)
    tokenizer = AutoTokenizer.from_pretrained(
        root, trust_remote_code=False, local_files_only=True
    )
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
        official,
        converter,
        args,
        layout,
        reader,
        reference,
        identities,
        hashes,
        layers=NUM_LAYERS,
    )
    before = {
        f"{step}.{layer}": tensor_digest(value)
        for step, values in enumerate(prepared_rows)
        for layer, value in values.items()
    }
    from rtp_llm.tools.dsv41_deep_native_history_replay import reconstruct_native

    config = json.loads((root / "config.json").read_text())["text_config"]
    controls, records, captures, inputs, mhc = reconstruct_native(
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
        layers=NUM_LAYERS,
    )
    after = {
        f"{step}.{layer}": tensor_digest(value)
        for step, values in enumerate(prepared_rows)
        for layer, value in values.items()
    }
    cached = {name: tensor_digest(value) for name, value in reader.values.items()}
    if before != after or cached != reader.hashes:
        raise AssertionError(
            "Corrected history prepared rows or cached weights changed"
        )
    if len(controls) != NUM_STEPS * NUM_LAYERS or len(official_controls) != len(
        controls
    ):
        raise AssertionError("Corrected history reconstruction coverage differs")
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
        "native_reconstruction": controls,
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
        "engram_block_size": official.fp8_block_size,
        "prepared_row_sha256": {"before": before, "after": after},
        "cached_weight_sha256": cached,
        "limitations": [
            "Ten matching-prefix layers; saved own-stream FFN outputs substitute experts.",
            "No serving, CP, graph, performance or full-model accuracy promotion.",
        ],
    }
    with cli.report.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        "DSV41_CORRECTED_HISTORY_CONTROLS "
        + json.dumps(
            {
                "layers": NUM_LAYERS,
                "steps": NUM_STEPS,
                "ffn_inputs_exact": 2 * len(controls),
                "weights": len(reader.hashes),
                "selected_row_reads": len(reader.selected_row_reads),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
