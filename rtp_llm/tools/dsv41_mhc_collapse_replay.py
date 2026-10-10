"""Compare held deep mHC collapse against saved native/released controls."""

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv4.fp8.test.v41_mhc_collapse_reference import (
    mhc_collapse_reference,
    reference_collapse,
)
from rtp_llm.models_py.modules.dsv4.hc import delayed
from rtp_llm.tools.dsv41_attention_replay import difference
from rtp_llm.tools.dsv41_deep_attention_history_replay import (
    FULL_SHA256,
    NUM_LAYERS,
    PROFILE,
)
from rtp_llm.tools.dsv41_query_boundary_replay import require_bytes, sha256


def admit(cli):
    repo = Path(__file__).resolve().parents[2]
    manifest = json.loads(cli.admission_manifest.read_text())
    for name, expected in manifest["source_sha256"].items():
        if sha256(repo / name) != expected:
            raise ValueError("Collapse source changed: " + name)
    for field in ("asset_sha256", "helper_sha256"):
        for name, expected in manifest[field].items():
            if sha256(Path(name)) != expected:
                raise ValueError("Collapse evidence/helper changed: " + name)
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
        raise ValueError("Collapse requires the qualified deep history")
    for name, expected in qualification["evidence_sha256"].items():
        if sha256(Path(name)) != expected:
            raise ValueError("Deep history evidence changed: " + name)
    history = json.loads(cli.history_report.read_text())
    raw = torch.load(cli.history_artifact, map_location="cpu", weights_only=True)
    count = 3 if history["prompt_index"] == 0 else 4
    if (
        history["input_artifact_sha256"] != sha256(cli.history_artifact)
        or history["full_qualification_sha256"] != FULL_SHA256
        or history["native_recipe"]["precision_profile"] != PROFILE
        or raw["native_recipe"] != history["native_recipe"]
        or raw["identities"] != history["identities"]
        or len(raw["identities"]) != count
        or raw["layers"] != history["layers"]
        or raw["layers"] != NUM_LAYERS
    ):
        raise ValueError("Collapse deep recipe, layers or histories differ")
    for field in ("official_reconstruction", "native_reconstruction"):
        if len(history[field]) != count * NUM_LAYERS or not all(
            row["normalized_bytes_equal"] is True for row in history[field].values()
        ):
            raise ValueError("Collapse requires all deep FFN byte controls")
    return manifest, history, raw


def exact(expected, actual, label):
    return require_bytes(expected.cpu(), actual.cpu().reshape_as(expected), label)


def scope_controls():
    original = delayed.collapse_delayed
    with mhc_collapse_reference(False):
        assert delayed.collapse_delayed is original
    with mhc_collapse_reference(True):
        assert delayed.collapse_delayed is reference_collapse
        with mhc_collapse_reference(False), mhc_collapse_reference(True):
            assert delayed.collapse_delayed is reference_collapse
        try:
            with mhc_collapse_reference(True):
                raise RuntimeError("collapse-scope-control")
        except RuntimeError as error:
            assert str(error) == "collapse-scope-control"
        assert delayed.collapse_delayed is reference_collapse
    assert delayed.collapse_delayed is original
    return {
        "disabled": True,
        "nested": True,
        "exception_restoration": True,
        "restored": True,
    }


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
        raise ValueError("Collapse artifacts must be fresh")
    if torch.cuda.get_device_capability() != (12, 0):
        raise ValueError("Collapse requires leased SM120 hardware")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    manifest, history, raw = admit(cli)
    sys.path.insert(0, str(cli.checkpoint / "inference"))
    official = importlib.import_module("model")
    if Path(official.__file__).resolve() != cli.checkpoint / "inference/model.py":
        raise ValueError("Unexpected released collapse module")
    scope = scope_controls()
    cases, records, native_controls, official_controls, sources = [], [], 0, 0, 0
    for stream in ("official", "native"):
        if len(raw[stream + "_mhc"]) != len(raw["identities"]):
            raise ValueError("Collapse step coverage differs")
        for step, layers in enumerate(raw[stream + "_mhc"]):
            if len(layers) != NUM_LAYERS:
                raise ValueError("Collapse layer coverage differs")
            for layer, values in enumerate(layers):
                for kind in ("attn", "ffn"):
                    saved = values[kind + "_collapsed"]
                    before = {
                        "residual": values[
                            "attention_residual" if kind == "attn" else "attn_post"
                        ].clone(),
                        "pre_mix": values[kind + "_read_mix"].clone(),
                    }
                    residual = before["residual"].cuda().reshape(1, -1, 4, 5120)
                    pre_mix = before["pre_mix"].cuda().reshape(1, -1, 4)
                    entry = layer == 0 and kind == "attn"
                    label = f"{stream}.step{step}.layer{layer:02d}.{kind}"

                    def maintained():
                        return (
                            residual[..., 0, :].clone()
                            if entry
                            else delayed.collapse_delayed(residual, pre_mix)
                        )

                    baseline = maintained().clone()
                    released = official.Block.hc_pre(None, residual, pre_mix).clone()
                    if stream == "native":
                        exact(saved, baseline, label + ".native_control")
                        native_controls += 1
                    else:
                        exact(saved, released, label + ".official_control")
                        official_controls += 1
                    with mhc_collapse_reference(True):
                        candidate = maintained().clone()
                    exact(released, candidate, label + ".candidate")
                    restored = maintained().clone()
                    exact(baseline, restored, label + ".restored")
                    after = {
                        "residual": residual.cpu()
                        .reshape_as(before["residual"])
                        .clone(),
                        "pre_mix": pre_mix.cpu().reshape_as(before["pre_mix"]).clone(),
                    }
                    for name, value in before.items():
                        exact(value, after[name], label + ".source." + name)
                        sources += 1
                    outputs = {
                        name: value.cpu().reshape_as(saved).clone()
                        for name, value in (
                            ("baseline", baseline),
                            ("released", released),
                            ("candidate", candidate),
                            ("restored", restored),
                        )
                    }
                    metrics = {
                        name: difference(outputs["released"], outputs[name])
                        for name in outputs
                    }
                    identity = {
                        "stream": stream,
                        "step": step,
                        "layer": layer,
                        "kind": kind,
                        "entry": entry,
                        "identity": raw["identities"][step],
                    }
                    records.append(dict(identity, metrics=metrics))
                    cases.append(
                        dict(
                            identity,
                            operands_before=before,
                            operands_after=after,
                            outputs=outputs,
                        )
                    )
    count = len(raw["identities"])
    if (len(cases), native_controls, official_controls, sources) != (
        28 * count,
        14 * count,
        14 * count,
        56 * count,
    ):
        raise AssertionError("Collapse held coverage differs")
    with cli.tensor_artifact.open("xb") as stream:
        torch.save(
            {
                "identities": raw["identities"],
                "native_recipe": raw["native_recipe"],
                "cases": cases,
            },
            stream,
        )
    report = {
        "revision": manifest["revision"],
        "prompt_index": history["prompt_index"],
        "identities": raw["identities"],
        "native_recipe": raw["native_recipe"],
        "scope_controls": scope,
        "cases": records,
        "native_controls": native_controls,
        "official_controls": official_controls,
        "candidate_controls": len(cases),
        "restored_controls": len(cases),
        "source_controls": sources,
        "source_sha256": manifest["source_sha256"],
        "history_qualification_sha256": manifest["history_qualification_sha256"],
        "tensor_artifact_sha256": sha256(cli.tensor_artifact),
        "numerical_promotion": False,
    }
    with cli.report.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        "DSV41_MHC_COLLAPSE_CONTROLS "
        + json.dumps(
            {
                key: report[key]
                for key in (
                    "native_controls",
                    "official_controls",
                    "candidate_controls",
                    "restored_controls",
                    "source_controls",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
