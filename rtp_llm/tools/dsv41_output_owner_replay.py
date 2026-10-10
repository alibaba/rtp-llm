"""Propagate scoped output GEMMs through qualified retained attention owners.

Inputs stay held at the accepted official/native histories. Each profile has
an independent retained cache; changed outputs do not execute later layers or
experts. Every preceding profile must reproduce its saved bytes and metrics.
"""

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv4.fp8.test.v41_compressor_precision import (
    compressor_bf16,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_output_precision import (
    output_projection_reference,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_query_precision import (
    query_projection_reference,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_sparse_precision import (
    sparse_sink_div_rn,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_tp8_projection import tp8_projection
from rtp_llm.tools.dsv41_attention_history_replay import (
    compare_capture,
    cpu_snapshot,
    execute_native,
    fresh_cache,
    require_exact,
)
from rtp_llm.tools.dsv41_attention_replay import NASReader, native_attention
from rtp_llm.tools.dsv41_output_projection_replay import admit_owner
from rtp_llm.tools.dsv41_query_boundary_replay import require_bytes, sha256
from rtp_llm.tools.dsv41_query_owner_replay import sources, tensor_digest
from rtp_llm.tools.dsv41_reference_probe import model_args

PROFILES = (
    ("baseline", False, False, False),
    ("query", True, False, False),
    ("query_sparse_decode", True, True, False),
    ("query_sparse_decode_output", True, True, True),
)


def admit_projection(cli, history, owner_report, repo):
    report = json.loads(cli.projection_report.read_text())
    if report["tensor_artifact_sha256"] != sha256(cli.projection_artifact):
        raise ValueError("Output owner requires the qualified projection tensors")
    tensor = torch.load(cli.projection_artifact, map_location="cpu", weights_only=True)
    if (
        report["identities"] != history["identities"]
        or tensor["identities"] != history["identities"]
        or report["native_recipe"] != history["native_recipe"]
        or report["weight_sha256"] != owner_report["weight_sha256"]
    ):
        raise ValueError("Output owner projection history or weights differ")
    for field in (
        "history_report",
        "history_artifact",
        "query_report",
        "query_artifact",
        "owner_report",
        "owner_artifact",
    ):
        if report["evidence_sha256"][str(getattr(cli, field))] != sha256(
            getattr(cli, field)
        ):
            raise ValueError(f"Output owner projection evidence changed: {field}")
    for name, expected in report["source_sha256"].items():
        if sha256(repo / name) != expected:
            raise ValueError(f"Qualified projection source changed: {name}")
    wanted = {
        (stream, layer, step)
        for stream in ("official", "native")
        for layer in range(3)
        for step in range(len(history["identities"]))
    }
    cases = {(row["stream"], row["layer"], row["step"]): row for row in tensor["cases"]}
    if set(cases) != wanted or len(tensor["cases"]) != len(wanted):
        raise ValueError("Output owner requires every held projection case")
    for (_, _, step), case in cases.items():
        if case["identity"] != history["identities"][step]:
            raise ValueError("Output owner projection consumed prefix differs")
    return report, cases


def compare_snapshot(expected, actual, label, *, output_changed=False):
    if expected.keys() != actual.keys():
        raise AssertionError(f"Output owner snapshot fields changed: {label}")
    for field in ("capture", "cache"):
        if expected[field].keys() != actual[field].keys():
            raise AssertionError(f"Output owner {field} tags changed: {label}")
        for name, value in actual[field].items():
            if output_changed and field == "capture" and name == "output":
                continue
            require_bytes(expected[field][name], value, f"{label}.{field}.{name}")
    for field in ("input_before", "input_after"):
        require_bytes(expected[field], actual[field], f"{label}.{field}")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in (
        "checkpoint",
        "history_report",
        "history_artifact",
        "query_report",
        "query_artifact",
        "owner_report",
        "owner_artifact",
        "projection_report",
        "projection_artifact",
        "report",
    ):
        parser.add_argument(field, type=Path)
    parser.add_argument("--tensor-artifact", type=Path, required=True)
    cli = parser.parse_args()
    if cli.report.exists() or cli.tensor_artifact.exists():
        raise ValueError("Output owner replay artifacts must be fresh")
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("Output owner replay requires leased SM120 hardware")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_float32_matmul_precision("highest")
    root, repo = cli.checkpoint.resolve(), Path(__file__).resolve().parents[2]
    history, inputs, owner_report, preceding, _ = admit_owner(cli, root, repo)
    projection_report, projections = admit_projection(cli, history, owner_report, repo)
    sys.path.insert(0, str(root / "inference"))
    official, kernel = [importlib.import_module(name) for name in ("model", "kernel")]
    for module in (official, kernel):
        if Path(module.__file__).resolve() != root / "inference" / (
            module.__name__ + ".py"
        ):
            raise ValueError("Unexpected released output-owner module")
    args = model_args(official, root, 256)
    config = json.loads((root / "config.json").read_text())["text_config"]
    identities = history["identities"]
    reader, results, raw, source_controls = NASReader(root), {}, {}, {}
    totals = dict(
        preceding_records=0,
        candidate_records=0,
        projection_outputs=0,
        unchanged_capture_records=0,
        physical_cache_records=0,
        immutable_sources=0,
    )
    for stream in ("official", "native"):
        results[stream], raw[stream], source_controls[stream] = {}, {}, {}
        for layer in range(3):
            owner = native_attention(args, config, reader, layer)
            immutable = sources(owner)
            immutable.update(
                wo_a_weight=owner.wo_a_w,
                wo_a_scales=owner.wo_a_s,
                wo_a_bf16=owner._wo_a_bf16_v41,
            )
            initial = {name: value.cpu().clone() for name, value in immutable.items()}
            initial_hashes = {
                name: tensor_digest(value) for name, value in initial.items()
            }
            results[stream][str(layer)], raw[stream][str(layer)] = {}, {}
            source_controls[stream][str(layer)] = {}
            for profile, corrected_query, sparse_decode, corrected_output in PROFILES:
                cache, tables = fresh_cache(owner)
                records, snapshots = [], []
                with tp8_projection(True), compressor_bf16(
                    True, fused=True, hca_projection=True
                ), sparse_sink_div_rn(
                    True, decode=sparse_decode
                ), query_projection_reference(
                    corrected_query, kernel=kernel
                ), output_projection_reference(
                    corrected_output, kernel=kernel
                ):
                    for step, identity in enumerate(identities):
                        value = (
                            inputs[f"{stream}_inputs"][step][layer].cuda().contiguous()
                        )
                        before = value.clone()
                        reference = {
                            name: value.cuda()
                            for name, value in inputs["official_attention"][step][
                                layer
                            ].items()
                        }
                        _, capture = execute_native(
                            owner,
                            cache,
                            tables,
                            value,
                            identity["start_pos"],
                            reference,
                        )
                        snapshot = {
                            "capture": cpu_snapshot(capture),
                            "cache": {
                                name: value.cpu().clone()
                                for name, value in cache.storage.items()
                            },
                            "input_before": before.cpu(),
                            "input_after": value.cpu(),
                        }
                        label = f"{stream}.{layer}.{step}.{profile}"
                        input_control = require_bytes(before, value, label + ".input")
                        measured = compare_capture(owner, reference, capture, identity)
                        if not corrected_output:
                            expected = preceding["results"][stream][str(layer)][
                                profile
                            ][step]
                            compare_snapshot(expected, snapshot, label)
                            record = owner_report["results"][stream][str(layer)][
                                profile
                            ]["records"][step]
                            if measured != record["reference"]:
                                raise AssertionError(
                                    f"Preceding metrics changed: {label}"
                                )
                            totals["preceding_records"] += 1
                        else:
                            expected = raw[stream][str(layer)]["query_sparse_decode"][
                                step
                            ]
                            compare_snapshot(
                                expected, snapshot, label, output_changed=True
                            )
                            held = projections[(stream, layer, step)]
                            require_bytes(
                                held["attention_before"],
                                snapshot["capture"]["attention"],
                                label + ".held_attention",
                            )
                            checks = {
                                "input": input_control,
                                "projection_output": require_exact(
                                    held["outputs"]["candidate"],
                                    capture["output"],
                                    label + ".qualified_output",
                                ),
                                "unchanged_sources": sorted(
                                    snapshot["capture"].keys() - {"output"}
                                ),
                                "physical_cache_bytes": {
                                    str(name): value.numel() * value.element_size()
                                    for name, value in snapshot["cache"].items()
                                },
                            }
                            record = {"controls": checks, "reference": measured}
                            for name in (
                                "candidate_records",
                                "projection_outputs",
                                "unchanged_capture_records",
                                "physical_cache_records",
                            ):
                                totals[name] += 1
                        records.append(record)
                        snapshots.append(snapshot)
                        print(
                            json.dumps(
                                {
                                    "stream": stream,
                                    "layer": layer,
                                    "step": step,
                                    "profile": profile,
                                    "reference": measured,
                                },
                                allow_nan=False,
                            ),
                            flush=True,
                        )
                after = {}
                for name, value in immutable.items():
                    require_bytes(initial[name], value.cpu(), f"{profile}.{name}")
                    after[name] = tensor_digest(value)
                    totals["immutable_sources"] += 1
                source_controls[stream][str(layer)][profile] = {
                    "before": initial_hashes,
                    "after": after,
                }
                if not corrected_output:
                    group = owner_report["results"][stream][str(layer)][profile]
                    for name, expected in group["immutable_before"].items():
                        if initial_hashes[name] != expected or after[name] != expected:
                            raise AssertionError(
                                "Preceding owner source identity changed"
                            )
                    results[stream][str(layer)][profile] = group
                else:
                    results[stream][str(layer)][profile] = {"records": records}
                raw[stream][str(layer)][profile] = snapshots
            del owner, immutable, cache
    cases = 6 * len(identities)
    expected = dict(
        preceding_records=3 * cases,
        candidate_records=cases,
        projection_outputs=cases,
        unchanged_capture_records=cases,
        physical_cache_records=cases,
        immutable_sources=360,
    )
    if totals != expected or reader.hashes != owner_report["weight_sha256"]:
        raise AssertionError("Output owner coverage or checkpoint identities differ")
    payload = {
        "results": raw,
        "identities": identities,
        "native_recipe": history["native_recipe"],
    }
    with cli.tensor_artifact.open("xb") as output:
        torch.save(payload, output)
    source_names = set(projection_report["source_sha256"]) | {
        "rtp_llm/tools/dsv41_output_owner_replay.py",
        "rtp_llm/models_py/modules/dsv4/fp8/test/v41_output_precision.py",
    }
    report = {
        "results": results,
        "controls_exact": totals,
        "immutable_sources": source_controls,
        "identities": identities,
        "native_recipe": history["native_recipe"],
        "prompt_index": history["prompt_index"],
        "weight_sha256": reader.hashes,
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
        "source_sha256": {name: sha256(repo / name) for name in sorted(source_names)},
        "evidence_sha256": {
            str(getattr(cli, name)): sha256(getattr(cli, name))
            for name in (
                "history_report",
                "history_artifact",
                "query_report",
                "query_artifact",
                "owner_report",
                "owner_artifact",
                "projection_report",
                "projection_artifact",
            )
        },
        "tensor_artifact_sha256": sha256(cli.tensor_artifact),
        "limitations": [
            "Matching consumed prefixes only: three France or four arithmetic steps, L0-L2.",
            "Official/native inputs are held and each profile has an independent retained cache.",
            "Changed outputs do not propagate to later inputs and experts do not execute.",
            "Full-model accuracy, live collectives, owner graphs and performance are separate gates.",
        ],
    }
    with cli.report.open("x") as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
