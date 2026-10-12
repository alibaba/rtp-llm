"""Hold corrected L0-L9 weighted norms and validate explicit-instance selection."""

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch

from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_block_norm_reference import (
    block_norm_reference,
)
from rtp_llm.tools.dsv41_attention_replay import NASReader, difference
from rtp_llm.tools.dsv41_corrected_history_replay import (
    FULL_SHA256,
    KERNEL_SHA256,
    NUM_LAYERS,
    NUM_STEPS,
    PROVIDER_SHA256,
    validate_recipe,
)
from rtp_llm.tools.dsv41_mhc_collapse_replay import exact
from rtp_llm.tools.dsv41_query_boundary_replay import sha256
from rtp_llm.tools.dsv41_query_owner_replay import tensor_digest


def admit(cli):
    repo = Path(__file__).resolve().parents[2]
    manifest = json.loads(cli.admission_manifest.read_text())
    for name, expected in manifest["source_sha256"].items():
        if sha256(repo / name) != expected:
            raise ValueError("Block norm source changed: " + name)
    for field in ("asset_sha256", "helper_sha256"):
        for name, expected in manifest[field].items():
            if sha256(Path(name)) != expected:
                raise ValueError("Block norm evidence/helper changed: " + name)
    qualification = json.loads(cli.history_qualification.read_text())
    if (
        sha256(cli.history_qualification) != manifest["history_qualification_sha256"]
        or qualification["accepted"] is not True
        or qualification["replay_exit_code"] != 0
        or qualification["audit_exit_code"] != 0
        or qualification["changes"]
        or not qualification["all_eight_gpus_idle"]
        or not qualification["idle_no_signal_teardown"]
    ):
        raise ValueError("Block norm requires the qualified corrected history")
    for name, expected in qualification["evidence_sha256"].items():
        if sha256(Path(name)) != expected:
            raise ValueError("Corrected history evidence changed: " + name)
    history = json.loads(cli.history_report.read_text())
    raw = torch.load(cli.history_artifact, map_location="cpu", weights_only=True)
    if (
        history["input_artifact_sha256"] != sha256(cli.history_artifact)
        or history["full_qualification_sha256"] != FULL_SHA256
        or raw["native_recipe"] != history["native_recipe"]
        or raw["identities"] != history["identities"]
        or len(raw["identities"]) != NUM_STEPS
        or raw["layers"] != history["layers"]
        or raw["layers"] != NUM_LAYERS
        or history["prompt_index"] != manifest["prompt_index"]
    ):
        raise ValueError("Block norm corrected layers or histories differ")
    recipe = validate_recipe(
        [{"metadata": raw["native_recipe"]}],
        kernel_sha256=KERNEL_SHA256,
        provider_sha256=PROVIDER_SHA256,
    )
    if recipe != raw["native_recipe"]:
        raise ValueError("Block norm corrected recipe differs")
    for field in ("official_reconstruction", "native_reconstruction"):
        if len(history[field]) != NUM_STEPS * NUM_LAYERS or not all(
            row["normalized_bytes_equal"] is True for row in history[field].values()
        ):
            raise ValueError("Block norm requires all corrected FFN byte controls")
    return manifest, history, raw


def graph_controls(norms, released_norms, raw):
    cases, records = [], []
    for step, layer, kind, mode in (
        (0, 7, "ffn", "allocated"),
        (0, 7, "ffn", "inplace"),
        (1, 0, "attn", "output"),
    ):
        norm, released = norms[layer, kind], released_norms[layer, kind]
        saved = raw["native_mhc"][step][layer][kind + "_collapsed"]
        seed = saved.cuda().reshape(-1, 5120).contiguous().clone()
        static = seed.clone()
        buffer = (
            static
            if mode == "inplace"
            else torch.empty_like(static) if mode == "output" else None
        )
        before_weight = norm.weight.cpu().clone()
        with block_norm_reference(True, norms=[norm]):
            norm(static, output=buffer)
            static.copy_(seed)
            norm(static, output=buffer)
            static.copy_(seed)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = norm(static, output=buffer)
            if buffer is not None and output is not buffer:
                raise AssertionError("Graph block norm lost its output buffer")
            replays = []
            for repeat in range(3):
                changed = (
                    torch.roll(seed, shifts=repeat + 1, dims=-1).float()
                    * (1 + (repeat + 1) / 16)
                ).bfloat16()
                static.copy_(changed)
                expected = released(changed).clone()
                graph.replay()
                exact(expected, output, "block norm graph output")
                exact(
                    expected if mode == "inplace" else changed,
                    static,
                    "block norm graph input",
                )
                replays.append(
                    {
                        "input": changed.cpu().clone(),
                        "input_after": static.cpu().clone(),
                        "released": expected.cpu().clone(),
                        "candidate": output.cpu().clone(),
                    }
                )
        after_weight = norm.weight.cpu().clone()
        exact(before_weight, after_weight, "block norm graph weight")
        identity = {"step": step, "layer": layer, "kind": kind, "mode": mode}
        cases.append(
            dict(
                identity,
                seed=seed.cpu().clone(),
                weight_before=before_weight,
                weight_after=after_weight,
                replays=replays,
            )
        )
        records.append(dict(identity, replays=3, output_buffer_preserved=True))
    return cases, records


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "checkpoint",
        "history_qualification",
        "history_report",
        "history_artifact",
        "admission_manifest",
        "report",
    ):
        parser.add_argument(name, type=Path)
    parser.add_argument("--tensor-artifact", type=Path, required=True)
    cli = parser.parse_args()
    if cli.report.exists() or cli.tensor_artifact.exists():
        raise ValueError("Block norm artifacts must be fresh")
    if torch.cuda.get_device_capability() != (12, 0):
        raise ValueError("Block norm requires leased SM120 hardware")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_float32_matmul_precision("highest")
    manifest, history, raw = admit(cli)
    sys.path.insert(0, str(cli.checkpoint / "inference"))
    official = importlib.import_module("model")
    if Path(official.__file__).resolve() != cli.checkpoint / "inference/model.py":
        raise ValueError("Unexpected released block norm module")
    config = json.loads((cli.checkpoint / "config.json").read_text())["text_config"]
    eps = config["rms_norm_eps"]
    reader, norms, released_norms = NASReader(cli.checkpoint), {}, {}
    for layer in range(NUM_LAYERS):
        for kind in ("attn", "ffn"):
            name = f"layers.{layer}.{kind}_norm.weight"
            weight = reader.load(name)
            if reader.hashes[name] != history["weight_sha256"][name]:
                raise ValueError("Block norm weight differs from corrected history")
            norms[layer, kind] = RMSNorm(weight, eps)
            with torch.device("cuda"):
                released = official.RMSNorm(5120, eps)
                released.weight.copy_(weight)
            released_norms[layer, kind] = released
    cases, records = [], []
    native_controls = official_controls = source_controls = 0
    original_forward = RMSNorm.forward
    for stream in ("official", "native"):
        if len(raw[stream + "_mhc"]) != NUM_STEPS:
            raise ValueError("Block norm step coverage differs")
        for step, layers in enumerate(raw[stream + "_mhc"]):
            if len(layers) != NUM_LAYERS:
                raise ValueError("Block norm layer coverage differs")
            for layer, values in enumerate(layers):
                for kind in ("attn", "ffn"):
                    norm, released = norms[layer, kind], released_norms[layer, kind]
                    saved = values[kind + "_norm"]
                    before = {
                        "input": values[kind + "_collapsed"].clone(),
                        "weight": norm.weight.cpu().clone(),
                    }
                    value = (
                        before["input"].cuda().reshape(-1, 5120).contiguous().clone()
                    )
                    label = f"{stream}.step{step}.layer{layer:02d}.{kind}"
                    baseline = norm(value).clone()
                    reference = released(value).clone()
                    if stream == "native":
                        exact(saved, baseline, label + ".native")
                        native_controls += 1
                    else:
                        exact(saved, reference, label + ".official")
                        official_controls += 1
                    with block_norm_reference(True, norms=[norm]):
                        candidate = norm(value).clone()
                    exact(reference, candidate, label + ".candidate")
                    restored = norm(value).clone()
                    exact(baseline, restored, label + ".restored")
                    if RMSNorm.forward is not original_forward:
                        raise AssertionError(
                            "Block norm original forward was not restored"
                        )
                    after = {
                        "input": value.cpu().reshape_as(before["input"]).clone(),
                        "weight": norm.weight.cpu().clone(),
                    }
                    for name, operand in before.items():
                        exact(operand, after[name], label + ".source." + name)
                        source_controls += 1
                    outputs = {
                        name: output.cpu().reshape_as(saved).clone()
                        for name, output in (
                            ("baseline", baseline),
                            ("released", reference),
                            ("candidate", candidate),
                            ("restored", restored),
                        )
                    }
                    identity = {
                        "stream": stream,
                        "step": step,
                        "layer": layer,
                        "kind": kind,
                        "identity": raw["identities"][step],
                    }
                    records.append(
                        dict(
                            identity,
                            metrics={
                                name: difference(outputs["released"], output)
                                for name, output in outputs.items()
                            },
                        )
                    )
                    cases.append(
                        dict(
                            identity,
                            operands_before=before,
                            operands_after=after,
                            outputs=outputs,
                        )
                    )
    graphs, graph_records = graph_controls(norms, released_norms, raw)
    cached = {name: tensor_digest(value) for name, value in reader.values.items()}
    if cached != reader.hashes or (
        len(cases),
        native_controls,
        official_controls,
        source_controls,
    ) != (160, 80, 80, 320):
        raise AssertionError("Block norm coverage or cached weights changed")
    if RMSNorm.forward is not original_forward:
        raise AssertionError("Block norm graph original forward was not restored")
    with cli.tensor_artifact.open("xb") as stream:
        torch.save(
            {
                "identities": raw["identities"],
                "native_recipe": raw["native_recipe"],
                "cases": cases,
                "graphs": graphs,
            },
            stream,
        )
    report = {
        "revision": manifest["revision"],
        "prompt_index": history["prompt_index"],
        "identities": raw["identities"],
        "native_recipe": raw["native_recipe"],
        "cases": records,
        "graphs": graph_records,
        "native_controls": native_controls,
        "official_controls": official_controls,
        "candidate_controls": len(cases),
        "restored_controls": len(cases),
        "source_controls": source_controls,
        "graph_controls": 9,
        "eps": eps,
        "weight_sha256": reader.hashes,
        "cached_weight_sha256": cached,
        "source_sha256": manifest["source_sha256"],
        "history_qualification_sha256": manifest["history_qualification_sha256"],
        "tensor_artifact_sha256": sha256(cli.tensor_artifact),
        "numerical_promotion": False,
    }
    with cli.report.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        "DSV41_BLOCK_NORM_CONTROLS "
        + json.dumps(
            {
                key: report[key]
                for key in (
                    "native_controls",
                    "official_controls",
                    "candidate_controls",
                    "restored_controls",
                    "source_controls",
                    "graph_controls",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
