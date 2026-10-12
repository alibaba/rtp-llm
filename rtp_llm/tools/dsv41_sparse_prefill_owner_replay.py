"""Propagate scoped sparse prefill through admitted retained output owners."""

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
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_sparse_prefill_reference import (
    sparse_prefill_reference,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_tp8_projection import tp8_projection
from rtp_llm.tools.dsv41_attention_history_replay import (
    compare_capture,
    cpu_snapshot,
    execute_native,
    fresh_cache,
)
from rtp_llm.tools.dsv41_attention_replay import NASReader, native_attention
from rtp_llm.tools.dsv41_output_owner_replay import PROFILES as PRECEDING_PROFILES
from rtp_llm.tools.dsv41_output_owner_replay import admit_projection, compare_snapshot
from rtp_llm.tools.dsv41_output_projection_replay import admit_owner
from rtp_llm.tools.dsv41_query_boundary_replay import require_bytes, sha256
from rtp_llm.tools.dsv41_query_owner_replay import sources, tensor_digest
from rtp_llm.tools.dsv41_reference_probe import model_args

PROFILE = "query_sparse_decode_output_sparse_prefill_reference"
PROFILES = tuple((*row, False) for row in PRECEDING_PROFILES) + (
    (PROFILE, True, True, True, True),
)
EVIDENCE = (
    "history_report",
    "history_artifact",
    "query_report",
    "query_artifact",
    "owner_report",
    "owner_artifact",
    "projection_report",
    "projection_artifact",
    "output_owner_report",
    "output_owner_artifact",
    "order_report",
    "order_artifact",
)


def admit_preceding(cli, history, query_owner, repo):
    report = json.loads(cli.output_owner_report.read_text())
    tensor = torch.load(
        cli.output_owner_artifact, map_location="cpu", weights_only=True
    )
    order_report = json.loads(cli.order_report.read_text())
    order = torch.load(cli.order_artifact, map_location="cpu", weights_only=True)
    for current in (report, tensor, order_report, order):
        if current["identities"] != history["identities"]:
            raise ValueError("Sparse owner requires matching held prefix identities")
    if (
        report["native_recipe"] != history["native_recipe"]
        or tensor["native_recipe"] != history["native_recipe"]
        or report["weight_sha256"] != query_owner["weight_sha256"]
    ):
        raise ValueError("Sparse owner recipe or weights differ")
    for current, path in (
        (report, cli.output_owner_artifact),
        (order_report, cli.order_artifact),
    ):
        if current["tensor_artifact_sha256"] != sha256(path):
            raise ValueError("Sparse owner admitted raw tensor digest differs")
        for name, expected in current["source_sha256"].items():
            if sha256(repo / name) != expected:
                raise ValueError(f"Sparse owner admitted source changed: {name}")
        for name, expected in current["evidence_sha256"].items():
            if sha256(Path(name)) != expected:
                raise ValueError(f"Sparse owner admitted evidence changed: {name}")
        for name in EVIDENCE[:8]:
            path = getattr(cli, name)
            if current["evidence_sha256"][str(path)] != sha256(path):
                raise ValueError("Sparse owner preceding evidence chain differs")
    wanted = {
        (stream, layer, 0) for stream in ("official", "native") for layer in range(3)
    }
    cases = {(row["stream"], row["layer"], row["step"]): row for row in order["cases"]}
    if set(cases) != wanted or len(order["cases"]) != 6:
        raise ValueError("Sparse owner requires all held prefill order cases")
    for stream, layer, _ in wanted:
        groups = tensor["results"][stream][str(layer)]
        if set(groups) != {row[0] for row in PRECEDING_PROFILES} or any(
            len(rows) != len(history["identities"]) for rows in groups.values()
        ):
            raise ValueError("Sparse owner preceding retained coverage differs")
    return report, tensor, order_report, cases


def candidate_controls(previous, current, *, prefill):
    skipped = {"attention", "native_attention", "output"} if prefill else set()
    if previous.keys() != current.keys():
        raise AssertionError("Sparse owner snapshot fields differ")
    for category in ("capture", "cache"):
        if previous[category].keys() != current[category].keys():
            raise AssertionError("Sparse owner snapshot tags differ")
        for name, value in current[category].items():
            if category == "capture" and name in skipped:
                continue
            require_bytes(
                previous[category][name], value, f"sparse_owner.{category}.{name}"
            )
    for name in ("input_before", "input_after"):
        require_bytes(previous[name], current[name], "sparse_owner." + name)
    unchanged = torch.equal(
        previous["capture"]["attention"].view(torch.uint8),
        current["capture"]["attention"].view(torch.uint8),
    )
    if unchanged:
        require_bytes(
            previous["capture"]["output"],
            current["capture"]["output"],
            "sparse_owner.unchanged_projection",
        )
    return {
        "unchanged_capture_tags": sorted(current["capture"].keys() - skipped),
        "physical_cache_bytes": {
            str(name): value.numel() * value.element_size()
            for name, value in current["cache"].items()
        },
        "attention_unchanged": unchanged,
        "projection_unchanged_required": unchanged,
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", *EVIDENCE, "report"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--tensor-artifact", type=Path, required=True)
    cli = parser.parse_args()
    if cli.report.exists() or cli.tensor_artifact.exists():
        raise ValueError("Sparse owner artifacts must be fresh")
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("Sparse owner replay requires leased SM120 hardware")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_float32_matmul_precision("highest")
    root, repo = cli.checkpoint.resolve(), Path(__file__).resolve().parents[2]
    history, inputs, query_owner, _, _ = admit_owner(cli, root, repo)
    admit_projection(cli, history, query_owner, repo)
    preceding_report, preceding, order_report, orders = admit_preceding(
        cli, history, query_owner, repo
    )
    sys.path.insert(0, str(root / "inference"))
    official, kernel = [importlib.import_module(name) for name in ("model", "kernel")]
    for module in (official, kernel):
        if Path(module.__file__).resolve() != root / "inference" / (
            module.__name__ + ".py"
        ):
            raise ValueError("Unexpected released sparse owner module")
    args = model_args(official, root, 256)
    config = json.loads((root / "config.json").read_text())["text_config"]
    reader, results, raw, immutable_controls = NASReader(root), {}, {}, {}
    totals = dict(
        preceding_records=0,
        candidate_records=0,
        prefill_records=0,
        decode_records=0,
        physical_cache_records=0,
        unchanged_projection_records=0,
        immutable_sources=0,
    )
    for stream in ("official", "native"):
        results[stream], raw[stream], immutable_controls[stream] = {}, {}, {}
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
            (
                results[stream][str(layer)],
                raw[stream][str(layer)],
                immutable_controls[stream][str(layer)],
            ) = ({}, {}, {})
            for profile, query, sparse_decode, output, sparse_prefill in PROFILES:
                cache, tables = fresh_cache(owner)
                records, snapshots = [], []
                with tp8_projection(True), compressor_bf16(
                    True, fused=True, hca_projection=True
                ), sparse_sink_div_rn(
                    True, decode=sparse_decode
                ), query_projection_reference(
                    query, kernel=kernel
                ), output_projection_reference(
                    output, kernel=kernel
                ), sparse_prefill_reference(
                    sparse_prefill
                ):
                    for step, identity in enumerate(history["identities"]):
                        value = (
                            inputs[stream + "_inputs"][step][layer].cuda().contiguous()
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
                        measured = compare_capture(owner, reference, capture, identity)
                        require_bytes(before, value, profile + ".input")
                        if not sparse_prefill:
                            compare_snapshot(
                                preceding["results"][stream][str(layer)][profile][step],
                                snapshot,
                                f"{stream}.{layer}.{step}.{profile}",
                            )
                            record = preceding_report["results"][stream][str(layer)][
                                profile
                            ]["records"][step]
                            if measured != record["reference"]:
                                raise AssertionError(
                                    "Preceding sparse owner metrics changed"
                                )
                            totals["preceding_records"] += 1
                        else:
                            parent = raw[stream][str(layer)][
                                "query_sparse_decode_output"
                            ][step]
                            checks = candidate_controls(
                                parent, snapshot, prefill=step == 0
                            )
                            if step == 0:
                                held = orders[(stream, layer, 0)]
                                for name in ("q", "kv", "indices", "lengths"):
                                    require_bytes(
                                        held["operands_before"][name],
                                        snapshot["capture"][name],
                                        "sparse_owner.held." + name,
                                    )
                                expected_attention = held["outputs"][
                                    "reference_compact"
                                ]["libdevice_exp_fusion_on"]
                                for name in ("attention", "native_attention"):
                                    require_bytes(
                                        expected_attention,
                                        snapshot["capture"][name],
                                        "sparse_owner.corrected." + name,
                                    )
                                totals["prefill_records"] += 1
                            else:
                                totals["decode_records"] += 1
                            totals["candidate_records"] += 1
                            totals["physical_cache_records"] += 1
                            totals["unchanged_projection_records"] += int(
                                checks["attention_unchanged"]
                            )
                            record = {"controls": checks, "reference": measured}
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
                    require_bytes(initial[name], value.cpu(), profile + "." + name)
                    after[name] = tensor_digest(value)
                    totals["immutable_sources"] += 1
                immutable_controls[stream][str(layer)][profile] = {
                    "before": initial_hashes,
                    "after": after,
                }
                if not sparse_prefill:
                    if (
                        immutable_controls[stream][str(layer)][profile]
                        != preceding_report["immutable_sources"][stream][str(layer)][
                            profile
                        ]
                    ):
                        raise AssertionError(
                            "Preceding immutable owner sources changed"
                        )
                    results[stream][str(layer)][profile] = preceding_report["results"][
                        stream
                    ][str(layer)][profile]
                else:
                    results[stream][str(layer)][profile] = {"records": records}
                raw[stream][str(layer)][profile] = snapshots
            del owner, immutable, cache
    cases = 6 * len(history["identities"])
    changed_prefills = sum(
        not torch.equal(
            orders[(stream, layer, 0)]["outputs"]["reference_compact"][
                "libdevice_exp_fusion_on"
            ].view(torch.uint8),
            preceding["results"][stream][str(layer)]["query_sparse_decode_output"][0][
                "capture"
            ]["attention"].view(torch.uint8),
        )
        for stream in ("official", "native")
        for layer in range(3)
    )
    expected = dict(
        preceding_records=4 * cases,
        candidate_records=cases,
        prefill_records=6,
        decode_records=cases - 6,
        physical_cache_records=cases,
        unchanged_projection_records=cases - changed_prefills,
        immutable_sources=450,
    )
    if totals != expected or reader.hashes != preceding_report["weight_sha256"]:
        raise AssertionError("Sparse owner coverage or weight identity differs")
    with cli.tensor_artifact.open("xb") as output:
        torch.save(
            {
                "results": raw,
                "identities": history["identities"],
                "native_recipe": history["native_recipe"],
            },
            output,
        )
    source_names = (
        set(order_report["source_sha256"])
        | set(preceding_report["source_sha256"])
        | {
            "rtp_llm/tools/dsv41_sparse_prefill_owner_replay.py",
            "rtp_llm/models_py/modules/dsv4/fp8/test/v41_sparse_prefill_reference.py",
        }
    )
    report = {
        "results": results,
        "controls_exact": totals,
        "immutable_sources": immutable_controls,
        "identities": history["identities"],
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
            str(getattr(cli, name)): sha256(getattr(cli, name)) for name in EVIDENCE
        },
        "tensor_artifact_sha256": sha256(cli.tensor_artifact),
        "limitations": [
            "Held matching-prefix inputs and retained caches; no experts or generation.",
            "Only short initial non-CP prefill is selected; all decode captures remain unchanged.",
            "No performance, serving or strict full-model accuracy promotion.",
        ],
    }
    with cli.report.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print("DSV41_SPARSE_OWNER_CONTROLS " + json.dumps(totals), flush=True)


if __name__ == "__main__":
    main()
