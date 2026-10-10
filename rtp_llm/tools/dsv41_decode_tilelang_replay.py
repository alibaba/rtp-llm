"""Qualify the maintained TileLang core through held planar decode caches."""

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv4 import tilelang_kernels
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_attention_triton
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_decode_tilelang_reference import (
    workspace_decode,
)
from rtp_llm.tools.dsv41_attention_replay import NASReader, difference
from rtp_llm.tools.dsv41_deep_decode_sparse_replay import (
    equal_bytes,
    paged_attention,
    reconstruct_pages,
)
from rtp_llm.tools.dsv41_mhc_collapse_replay import admit, exact
from rtp_llm.tools.dsv41_query_boundary_replay import KERNEL_SHA256, sha256
from rtp_llm.tools.dsv41_query_owner_replay import tensor_digest
from rtp_llm.tools.dsv41_sparse_precision_replay import released_attention


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
        raise ValueError("Decode sparse artifacts must be fresh")
    if torch.cuda.get_device_capability() != (12, 0):
        raise ValueError("Decode sparse requires leased SM120 hardware")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    manifest, history, raw = admit(cli)
    preceding_report = json.loads(Path(manifest["tile_parent_report"]).read_text())
    preceding_raw = torch.load(
        manifest["tile_parent_artifact"], map_location="cpu", weights_only=True
    )
    preceding_sha = sha256(Path(manifest["tile_parent_artifact"]))
    if (
        preceding_report["tensor_artifact_sha256"] != preceding_sha
        or preceding_raw["identities"] != raw["identities"]
    ):
        raise ValueError("TileLang preceding held dataset differs")
    preceding_cases = {
        (row["step"], row["layer"]): row for row in preceding_raw["cases"]
    }
    if len(preceding_cases) != 7 * (len(raw["identities"]) - 1):
        raise ValueError("TileLang preceding case coverage differs")
    provider_path = Path(tilelang_kernels.__file__).resolve()
    if (
        provider_path
        != Path(__file__).resolve().parents[2]
        / "rtp_llm/models_py/modules/dsv4/tilelang_kernels.py"
    ):
        raise ValueError("The maintained TileLang provider path differs")
    sys.path.insert(0, str(cli.checkpoint / "inference"))
    kernel = importlib.import_module("kernel")
    if (
        Path(kernel.__file__).resolve() != cli.checkpoint / "inference/kernel.py"
        or sha256(Path(kernel.__file__)) != KERNEL_SHA256
    ):
        raise ValueError("Decode sparse released kernel differs")
    reader, cases, records = NASReader(cli.checkpoint), [], []
    native_controls = restored_controls = source_controls = history_controls = 0
    workspace_controls = workspace_readonly = reconstructed_controls = (
        unchanged_page_controls
    ) = corrected = 0
    for step in range(1, len(raw["identities"])):
        identity = raw["identities"][step]
        if len(identity["input_ids"]) != 1:
            raise ValueError("Deep sparse requires single-token decode")
        end = identity["start_pos"] + 1
        for layer in range(7):
            ratio = 0 if layer < 2 else 2
            global_rows = end // ratio if ratio else 0
            width = 640 if ratio else 128
            saved = raw["native_attention"][step][layer]
            sink_name = f"layers.{layer}.attn.attn_sink"
            sink = reader.load(sink_name)
            if reader.hashes[sink_name] != history["weight_sha256"][sink_name]:
                raise ValueError("Decode sink weight differs from admitted history")
            operands = {
                "q": saved["q"].cuda().reshape(1, 64, 512).contiguous().clone(),
                "kv": saved["kv"].cuda().reshape(-1, 1, 512).contiguous().clone(),
                "indices": saved["indices"].cuda().contiguous().clone(),
                "lengths": saved["lengths"].cuda().contiguous().clone(),
                "sink": sink.contiguous().clone(),
            }
            if (
                tuple(operands["indices"].shape) != (1, 1, width)
                or operands["lengths"].tolist() != [width]
                or operands["kv"].shape[0] != 128 + global_rows
                or operands["indices"].dtype != torch.int32
                or operands["lengths"].dtype != torch.int32
            ):
                raise ValueError("Decode sparse plan scope differs")
            before = {name: value.cpu().clone() for name, value in operands.items()}
            label = f"step{step}.layer{layer:02d}"
            preceding = preceding_cases[step, layer]
            if preceding["identity"] != identity:
                raise ValueError("TileLang preceding identity differs")
            if not bool((before["indices"][..., :64] >= 0).any(-1).all()):
                raise ValueError("TileLang held scope requires a valid first SWA tile")
            for name, value in before.items():
                exact(
                    preceding["operands_before"][name],
                    value,
                    label + ".preceding_operands." + name,
                )
            caches, read_slots, decoded, plans = reconstruct_pages(
                operands, saved, global_rows, label
            )
            reconstructed_controls += len(caches)
            valid_rows = torch.ones(1, device="cuda", dtype=torch.bool)
            pages_before = {
                name: cache.raw.cpu().clone() for name, cache in caches.items()
            }
            plans_before = {name: value.cpu().clone() for name, value in plans.items()}

            for name in caches:
                for field, value in (
                    ("pages_before", pages_before[name]),
                    ("plans_before", plans_before[name]),
                    ("read_slots", read_slots[name]),
                    ("decoded_rows", decoded[name]),
                ):
                    exact(
                        preceding[field][name],
                        value,
                        label + ".preceding." + field + "." + name,
                    )

            baseline = paged_attention(
                _v41_attention_triton._attention, operands, caches, plans, valid_rows
            )
            exact(saved["attention"], baseline, label + ".native_control")
            native_controls += 1
            exact(
                preceding["outputs"]["baseline"],
                baseline,
                label + ".preceding_baseline",
            )
            candidate, workspace, workspace_indices = workspace_decode(
                operands["q"], operands["sink"], caches, plans
            )
            workspace_before = {
                "kv": workspace.cpu().clone(),
                "indices": workspace_indices.cpu().clone(),
            }
            selected = before["indices"].reshape(-1)
            expected = torch.zeros((width, 512), dtype=torch.bfloat16)
            live = selected >= 0
            expected[live] = before["kv"].reshape(-1, 512)[selected[live].long()]
            exact(expected, workspace, label + ".workspace_rows")
            expected_indices = torch.where(
                live, torch.arange(width, dtype=torch.int32), -1
            ).reshape(1, 1, width)
            exact(expected_indices, workspace_indices, label + ".workspace_indices")
            workspace_controls += 2
            released, masked = released_attention(kernel, operands)
            exact(
                preceding["outputs"]["released"],
                released,
                label + ".preceding_released",
            )
            workspace_operands = dict(
                operands, kv=workspace.reshape(-1, 1, 512), indices=workspace_indices
            )
            released_workspace, _ = released_attention(kernel, workspace_operands)
            exact(released, released_workspace, label + ".released_workspace")
            workspace_after = {
                "kv": workspace.cpu().clone(),
                "indices": workspace_indices.cpu().clone(),
            }
            for name in workspace_before:
                exact(
                    workspace_before[name],
                    workspace_after[name],
                    label + ".workspace_unchanged." + name,
                )
                workspace_readonly += 1
            restored = paged_attention(
                _v41_attention_triton._attention, operands, caches, plans, valid_rows
            )
            exact(baseline, restored, label + ".restored")
            restored_controls += 1
            after = {name: value.cpu().clone() for name, value in operands.items()}
            for name, value in before.items():
                exact(value, after[name], label + ".source." + name)
                source_controls += 1
            pages_after = {
                name: cache.raw.cpu().clone() for name, cache in caches.items()
            }
            plans_after = {name: value.cpu().clone() for name, value in plans.items()}
            for name in caches:
                exact(pages_before[name], pages_after[name], label + ".pages." + name)
                exact(plans_before[name], plans_after[name], label + ".plans." + name)
                unchanged_page_controls += 1
            reference = raw["official_attention"][step][layer]
            aligned_kv = torch.cat(
                (before["kv"][global_rows:], before["kv"][:global_rows])
            )
            mapped = torch.where(
                before["indices"] < global_rows,
                before["indices"] + 128,
                before["indices"] - global_rows,
            )
            mapped = torch.where(before["indices"] >= 0, mapped, -1)
            same_input = all(
                (
                    equal_bytes(reference["q"], before["q"]),
                    equal_bytes(reference["kv"], aligned_kv),
                    equal_bytes(reference["indices"], mapped),
                )
            )
            if same_input:
                exact(reference["attention"], released, label + ".released_history")
                history_controls += 1
            outputs = {
                name: value.cpu().clone()
                for name, value in (
                    ("baseline", baseline),
                    ("released", released),
                    ("candidate", candidate),
                    ("restored", restored),
                    ("released_workspace", released_workspace),
                )
            }
            metrics = {
                name: {
                    "released": difference(outputs["released"], value),
                    "history": difference(reference["attention"], value),
                }
                for name, value in outputs.items()
            }
            corrected += int(equal_bytes(outputs["released"], outputs["candidate"]))
            record = {
                "step": step,
                "layer": layer,
                "identity": identity,
                "global_rows": global_rows,
                "released_history_control_required": same_input,
                "metrics": metrics,
            }
            records.append(record)
            cases.append(
                {key: value for key, value in record.items() if key != "metrics"}
                | {
                    "operands_before": before,
                    "operands_after": after,
                    "outputs": outputs,
                    "released_masked_indices": masked.cpu().clone(),
                    "pages_before": pages_before,
                    "pages_after": pages_after,
                    "plans_before": plans_before,
                    "plans_after": plans_after,
                    "read_slots": {
                        name: value.cpu().clone() for name, value in read_slots.items()
                    },
                    "decoded_rows": {
                        name: value.cpu().clone() for name, value in decoded.items()
                    },
                    "valid_rows": valid_rows.cpu().clone(),
                    "workspace_before": workspace_before,
                    "workspace_after": workspace_after,
                }
            )
    count = 7 * (len(raw["identities"]) - 1)
    if (len(cases), native_controls, restored_controls, source_controls) != (
        count,
        count,
        count,
        5 * count,
    ):
        raise AssertionError("Deep sparse held coverage differs")
    if (
        workspace_controls != 2 * count
        or workspace_readonly != 2 * count
        or reconstructed_controls != 12 * (len(raw["identities"]) - 1)
        or unchanged_page_controls != reconstructed_controls
    ):
        raise AssertionError("Deep sparse reconstructed paged controls differ")
    if {
        name: tensor_digest(value) for name, value in reader.values.items()
    } != reader.hashes:
        raise AssertionError("Deep sparse cached sink weights changed")
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
        "cases": records,
        "native_controls": native_controls,
        "restored_controls": restored_controls,
        "source_controls": source_controls,
        "workspace_byte_controls": workspace_controls,
        "workspace_readonly_controls": workspace_readonly,
        "preceding_tensor_artifact_sha256": preceding_sha,
        "provider": "rtp_llm.models_py.modules.dsv4.tilelang_kernels.sparse_attn",
        "provider_sha256": sha256(provider_path),
        "first_swa_tile_valid_key_required": True,
        "reconstructed_row_controls": reconstructed_controls,
        "unchanged_page_controls": unchanged_page_controls,
        "cache_evidence": "reconstructed-pages-exact-decoded-rows",
        "released_history_controls": history_controls,
        "candidate_released_byte_matches": corrected,
        "candidate_exact_on_same_inputs": corrected == count,
        "weight_sha256": reader.hashes,
        "source_sha256": manifest["source_sha256"],
        "history_qualification_sha256": manifest["history_qualification_sha256"],
        "tensor_artifact_sha256": sha256(cli.tensor_artifact),
        "numerical_promotion": False,
    }
    with cli.report.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        "DSV41_DECODE_TILELANG_CONTROLS "
        + json.dumps(
            {
                key: report[key]
                for key in (
                    "native_controls",
                    "restored_controls",
                    "source_controls",
                    "workspace_byte_controls",
                    "workspace_readonly_controls",
                    "reconstructed_row_controls",
                    "unchanged_page_controls",
                    "released_history_controls",
                    "candidate_released_byte_matches",
                    "candidate_exact_on_same_inputs",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
