"""Verify frozen checkpoint CEP4PP2 prefill token and lifecycle smoke on CPU."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from rtp_llm.tools.dsv41_phase_smoke import verify_sources

TARGET = "//rtp_llm/models_py/modules/dsv4/test:test_v41_pp_checkpoint_engine_sm120"
ASSETS = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_identities(root):
    assets = {name: sha256(root / name) for name in ASSETS}
    return identities_from_assets(assets)


def identities_from_assets(assets):
    # Tokenization depends on both tokenizer files and checkpoint configuration.
    tokenizer = {
        name: assets[name] for name in ASSETS if name != "model.safetensors.index.json"
    }
    return {
        "checkpoint_sha256": assets["model.safetensors.index.json"],
        "config_sha256": assets["config.json"],
        "tokenizer_sha256": hashlib.sha256(
            json.dumps(tokenizer, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def records(log, marker):
    decoder = json.JSONDecoder()
    return [
        decoder.raw_decode(log[match.end() :].lstrip())[0]
        for match in re.finditer(re.escape(marker) + r"\s+", log)
    ]


def verify_assets(repo, checkpoint, manifest):
    pinned_path = Path(manifest["asset_manifest_path"])
    if sha256(pinned_path) != manifest["asset_manifest_sha256"]:
        raise ValueError("Pinned asset manifest changed")
    pinned = json.loads(pinned_path.read_text())
    if manifest["assets"] != {
        key: value for key, value in pinned.items() if key != "source_sha256"
    }:
        raise ValueError("Copied asset identities differ from the pinned manifest")
    reference_path = Path(manifest["reference_report_path"])
    reference = json.loads(reference_path.read_text())
    verify_sources(
        repo,
        checkpoint,
        reference_path,
        reference,
        {**pinned, "source_sha256": manifest["source_sha256"]},
    )
    required = {
        "rtp_llm/models_py/modules/dsv4/test/test_v41_pp_checkpoint_engine_sm120.py",
        "rtp_llm/tools/dsv41_pp_smoke.py",
    }
    if not required.issubset(manifest["source_sha256"]):
        raise ValueError("Frozen sources lack the CEP4PP2 harness and verifier")
    return reference


def verify_run(log, manifest, reference=None):
    full = reference is not None
    method = (
        "test_full_nas_cep4pp2_prefill" if full else "test_bounded_named_checkpoint"
    )
    if (
        manifest.get("bazel_exit_code") != 0
        or manifest.get("source_changes") != []
        or manifest.get("sources_unchanged") != len(manifest.get("source_sha256", {}))
        or manifest.get("targets") != [TARGET]
        or manifest.get("test_args") != ["V41PPCheckpointEngineSM120Test." + method]
    ):
        raise ValueError("Smoke requires a successful frozen single-method invocation")
    result = manifest.get("test_results", {}).get("pp_checkpoint", {})
    statuses = result.get("bazel_status_lines", [])
    if not any("PASSED" in line and "(cached)" not in line for line in statuses):
        raise ValueError("Smoke requires freshly executed target evidence")
    if (
        not result.get("gpu_lock_idle_teardown")
        or not re.search(r"Ran 1 test in [\d.]+s\s+OK\r?\n", log)
        or "exitcode: 0" not in log
        or "[GPU_LOCK_NO_KILL] leased GPUs verified idle; no signals" not in log
        or "release done" not in log
    ):
        raise ValueError("Smoke did not complete successful idle/no-signal teardown")
    executed = records(log, "DSV41_PP_CHECKPOINT_EXECUTION")
    outputs = records(log, "DSV41_PP_CHECKPOINT_OUTPUT")
    lifetimes = records(log, "DSV41_PP_CHECKPOINT_LIFETIME")
    for key, actual in (
        ("pp_checkpoint_execution_records", executed),
        ("pp_checkpoint_output_records", outputs),
        ("pp_checkpoint_lifetime_records", lifetimes),
    ):
        if manifest.get(key) != actual:
            raise ValueError("Preserved records differ from the frozen test log")
    if len(executed) != 32 or len(outputs) != 4 or len(lifetimes) != 8:
        raise ValueError(
            "Smoke requires 32 executions, four outputs and eight lifetimes"
        )
    by_execution = {(row["rank"], row["phase"]): row for row in executed}
    by_output = {row["phase"]: row for row in outputs}
    by_lifetime = {row["rank"]: row for row in lifetimes}
    if (
        set(by_execution) != {(rank, phase) for rank in range(8) for phase in range(4)}
        or set(by_output) != set(range(4))
        or set(by_lifetime) != set(range(8))
    ):
        raise ValueError("Smoke has duplicate or missing rank/phase coverage")
    fingerprints = {row["contract_fingerprint"] for row in executed}
    if len(fingerprints) != 1 or not re.fullmatch(
        r"[0-9a-f]{64}", next(iter(fingerprints))
    ):
        raise ValueError("Native PP contract fingerprints disagree")
    identities = executed[0]["identities"]
    if any(row["identities"] != identities for row in executed):
        raise ValueError("Native PP checkpoint identities disagree")
    if full and identities != identities_from_assets(
        manifest["assets"]["checkpoint_asset_sha256"]
    ):
        raise ValueError("Native contract differs from the frozen checkpoint assets")
    prompts = by_output[0]["prompts"]
    if (
        len(prompts) != 2
        or prompts[0] == prompts[1]
        or any(not prompt for prompt in prompts)
    ):
        raise ValueError("Smoke requires two distinct prompts")
    if full and prompts != manifest["assets"]["prompt_token_ids"]:
        raise ValueError("Smoke prompts differ from the independent reference")
    admitted = records(log, "DSV41_PP_CHECKPOINT_ADMISSION")
    if full:
        if reference.get("world_size") != 8 or len(reference.get("outputs", [])) != 2:
            raise ValueError("Smoke requires the independent TP8 two-prompt reference")
        if (
            len(admitted) != 1
            or manifest.get("pp_checkpoint_admission_records") != admitted
        ):
            raise ValueError("Full NAS smoke requires one preserved resource admission")
        admission = admitted[0]
        if (
            admission["full_nas"] is not True
            or admission["fresh_staging"] is not True
            or admission["engram_bytes"] != 202758032400
            or admission["reserve_gib"] < 32
            or admission["free_storage"]
            <= admission["engram_bytes"] + admission["reserve_gib"] * 2**30
            or admission["available_ram"] <= 2 * admission["engram_bytes"] + 128 * 2**30
            or admission["prompts"] != prompts
            or manifest["test_env"].get("DSV41_RUN_FULL_PP_ENGINE") != "1"
        ):
            raise ValueError("Full NAS resource admission is incomplete")
    elif admitted:
        raise ValueError("Bounded smoke cannot qualify full NAS admission")
    all_ids = []
    comparisons = []
    for phase in range(4):
        output = by_output[phase]
        count = 2 if phase < 2 else 1
        selected = prompts if phase < 2 else [prompts[phase - 2]]
        if (
            output["prompts"] != selected
            or output["full_prompt"] is not (phase < 2)
            or output["full_nas"] is not full
            or output["native_argmax_exact"] is not True
            or output["native_delivery_exact"] is not True
            or len(output["request_ids"]) != count
            or len(output["tokens"]) != count
            or any(type(token) is not int or token < 0 for token in output["tokens"])
        ):
            raise ValueError("Native output/request/profile record is incomplete")
        all_ids.extend(output["request_ids"])
        for key in ("hidden_sha256", "logits_sha256", "all_hidden_sha256"):
            hashes = output[key]
            if len(hashes) != count or any(
                (
                    digest is not None
                    if key == "all_hidden_sha256" and phase >= 2
                    else not isinstance(digest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                )
                for digest in hashes
            ):
                raise ValueError("Native output hashes are incomplete")
        for rank in range(8):
            row = by_execution[rank, phase]
            if (
                row["stage"] != rank // 4
                or row["max_sequence_length"] != 256
                or row["round"] != phase
                or row["full_prompt"] is not (phase < 2)
                or row["maintained_arithmetic"] is not True
                or len(row["spans"]) != count
                or [span[0] for span in row["spans"]] != output["request_ids"]
                or any(
                    len(span) != 6 or span[3:5] != [0, 0] or span[5] != len(prompt)
                    for span, prompt in zip(row["spans"], selected)
                )
            ):
                raise ValueError("Native stage execution/span record is incomplete")
            keys = {"hidden", "pre_mix"} if rank < 4 else {"hidden", "logits"}
            if set(row["hashes"]) != keys or any(
                re.fullmatch(r"[0-9a-f]{64}", digest) is None
                for digest in row["hashes"].values()
            ):
                raise ValueError("Native stage tensor hashes are incomplete")
        spans = [by_execution[rank, phase]["spans"] for rank in range(8)]
        if any(value != spans[0] for value in spans):
            raise ValueError("Stage request identities disagree")
        for rank in range(5, 8):
            if (
                by_execution[rank, phase]["hashes"]["logits"]
                != by_execution[4, phase]["hashes"]["logits"]
            ):
                raise ValueError("Gathered stage-one logits disagree")
        if full:
            wanted = {
                tuple(row["prompt_ids"]): row["generated_ids"][0]
                for row in reference["outputs"]
            }
            expected = [wanted[tuple(prompt)] for prompt in selected]
            comparisons.append(
                {
                    "phase": phase,
                    "native_ids": output["tokens"],
                    "reference_ids": expected,
                }
            )
            if output["tokens"] != expected:
                raise ValueError("Full NAS CEP4PP2 first-token reference mismatch")
    if len(set(all_ids)) != 6:
        raise ValueError("Six responses require six fresh request identities")
    for rank in range(8):
        if by_execution[rank, 0]["hashes"] != by_execution[rank, 1]["hashes"]:
            raise ValueError("Repeated stage arithmetic differs")
        lifetime = by_lifetime[rank]
        if (
            lifetime["cold_initializations"] != 1
            or lifetime["executions"] != 4
            or lifetime["shutdown"] is not True
            or lifetime["receiver_empty"] is not True
            or lifetime["engram_cleanup"] is not True
            or lifetime["active_requests"] != 0
            or lifetime["identity_count"] != 0
            or lifetime["retained_slots"] != 0
            or lifetime["free_before"] != lifetime["free_after"]
            or lifetime["cache_blocks"] != 8
        ):
            raise ValueError(
                "Native shutdown/capacity/receiver/Engram lifetime is incomplete"
            )
    for key in (
        "prompts",
        "tokens",
        "hidden_sha256",
        "logits_sha256",
        "all_hidden_sha256",
    ):
        if by_output[0][key] != by_output[1][key]:
            raise ValueError("Repeated delivered results differ")
    return {
        "phase": "prefill",
        "full_nas": full,
        "topology": {
            "pp_size": 2,
            "tp_size": 4,
            "cp_size": 4,
            "dp_size": 1,
            "ep_size": 4,
        },
        "execution_records": 32,
        "output_records": 4,
        "responses": 6,
        "lifetime_records": 8,
        "native_argmax_delivery_repetition_exact": True,
        "shutdown_capacity_receiver_engram_cleanup": True,
        "all_reference_first_ids_exact": full,
        "http_qualified": False,
        "pd_transfer_qualified": False,
        "reference_hidden_logit_accuracy_qualified": False,
        "contract_fingerprint": next(iter(fingerprints)),
        "identities": identities,
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--native-log", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--verify-assets-only", action="store_true")
    cli = parser.parse_args()
    manifest = json.loads(cli.run_manifest.read_text())
    reference = None
    if manifest.get("assets"):
        if cli.repo is None or cli.checkpoint is None:
            parser.error("Full NAS asset verification requires --repo and --checkpoint")
        reference = verify_assets(cli.repo, cli.checkpoint, manifest)
    if cli.verify_assets_only:
        if reference is None:
            parser.error("Asset-only verification requires a pinned asset manifest")
        print(
            "Checkpoint, tokenizer, released sources and independent reference identities verified"
        )
        return
    if cli.native_log is None or cli.report is None:
        parser.error("Smoke verification requires --native-log and --report")
    report = verify_run(cli.native_log.read_text(), manifest, reference)
    report["evidence_sha256"] = {
        str(path): sha256(path) for path in (cli.native_log, cli.run_manifest)
    }
    report["parent_revision"] = manifest["parent_revision"]
    with cli.report.open("x") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
