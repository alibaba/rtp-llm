"""Verify full-checkpoint native PREFILL/PDFUSION token smoke evidence on CPU."""

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path


TOPOLOGY = {"pp_size": 1, "tp_size": 1, "cp_size": 1, "dp_size": 8, "ep_size": 8}
OUTPUT = re.compile(
    r"rank=(\d+) phase=(\d+) precision=(\S+) "
    r"output_tokens=(\[[^\n]*?\]) hidden_sha256=(\[[^\n]*?\])"
)


def verify_run(phase, log, reference, manifest):
    if phase not in ("prefill", "decode"):
        raise ValueError("Phase must be prefill or decode")
    count = 1 if phase == "prefill" else 9
    role = "PREFILL" if phase == "prefill" else "PDFUSION"
    if (
        manifest.get("topology") != TOPOLOGY
        or manifest.get("new_tokens") != count
        or manifest.get("native_role") != role
        or manifest.get("profiles") != ["baseline"]
        or manifest.get("fake_balance_expert") is not False
        or manifest.get("eos_disabled") is not True
    ):
        raise ValueError("Smoke requires the native unpatched DP8EP8 phase profile")
    if reference.get("world_size") != 8:
        raise ValueError("Reference must be the independent TP8 run")
    outputs = reference.get("outputs", [])
    prompts = manifest.get("prompt_token_ids")
    if not isinstance(prompts, list) or len(prompts) != 2 or prompts[0] == prompts[1]:
        raise ValueError("Smoke requires two distinct recorded prompts")
    reference_by_prompt = {}
    for output in outputs:
        key = tuple(output["prompt_ids"])
        if key in reference_by_prompt:
            raise ValueError("Reference contains duplicate prompt records")
        reference_by_prompt[key] = output
    if set(reference_by_prompt) != {tuple(prompt) for prompt in prompts}:
        raise ValueError("Native and reference prompt IDs disagree")
    admitted = re.findall(r"full-engine admission [^\n]+ prompts=(\[[^\n]+\])", log)
    if len(admitted) != 1 or ast.literal_eval(admitted[0]) != prompts:
        raise ValueError("Native admission prompt IDs differ from the frozen run")
    completion = (
        f"checkpoint-to-native-engine full_nas=True new_tokens={count} "
        "context=256 precision=['baseline'] generation liveness PASS"
    )
    if (
        log.count(completion) != 1
        or not re.search(r"Ran 1 test in [\d.]+s\s+OK\r?\n", log)
        or "exitcode: 0" not in log
        or "[GPU_LOCK_NO_KILL] leased GPUs verified idle; no signals" not in log
        or "release done" not in log
    ):
        raise ValueError("Native phase test did not complete successful teardown")
    records = {}
    for rank, pattern, profile, ids, hashes in OUTPUT.findall(log):
        rank, pattern = int(rank), int(pattern)
        if profile != "baseline":
            raise ValueError("Diagnostic precision profiles cannot qualify this smoke")
        key = (rank, pattern)
        if key in records:
            raise ValueError("Duplicate native rank/phase record")
        ids, hashes = ast.literal_eval(ids), ast.literal_eval(hashes)
        if (
            len(ids) != count
            or any(type(token) is not int or token < 0 for token in ids)
            or len(hashes) != count
            or any(
                not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                for digest in hashes
            )
        ):
            raise ValueError("Native token/hidden-hash record is incomplete")
        records[key] = (ids, hashes)
    expected = {(0, 0), (7, 3)} | {
        (rank, pattern) for rank in range(8) for pattern in (1, 2)
    }
    if records.keys() != expected:
        raise ValueError("Native smoke requires all 18 rank/phase records")
    comparisons = []
    for prompt_index, prompt in enumerate(prompts):
        samples = [
            value
            for (_, pattern), value in records.items()
            if pattern % 2 == prompt_index
        ]
        if len(samples) != 9 or any(value != samples[0] for value in samples):
            raise ValueError("Repeated native IDs or hidden hashes disagree")
        expected_ids = reference_by_prompt[tuple(prompt)]["generated_ids"][:count]
        if len(expected_ids) != count or any(
            type(token) is not int or token < 0 for token in expected_ids
        ):
            raise ValueError("Reference does not cover every requested generation step")
        ids, hashes = samples[0]
        mismatch = next(
            (
                index
                for index, pair in enumerate(zip(expected_ids, ids))
                if pair[0] != pair[1]
            ),
            None,
        )
        eos = manifest.get("eos_token_id")
        if type(eos) is not int or eos < 0:
            raise ValueError("Smoke must record the checkpoint EOS token")
        eos_step = expected_ids.index(eos) + 1 if eos in expected_ids else None
        comparisons.append(
            {
                "prompt_token_ids": prompt,
                "reference_ids": expected_ids,
                "native_ids": ids,
                "hidden_sha256": hashes,
                "repetitions": len(samples),
                "all_requested_ids_exact": mismatch is None,
                "matching_prefix_tokens": count if mismatch is None else mismatch,
                "first_mismatch_step": None if mismatch is None else mismatch + 1,
                "reference_eos_step": eos_step,
                "through_eos_exact": ids[:eos_step] == expected_ids[:eos_step],
                "forced_steps_after_reference_eos": (
                    0 if eos_step is None else count - eos_step
                ),
            }
        )
    return {
        "phase": phase,
        "native_role": role,
        "topology": TOPOLOGY,
        "scope": "native engine token smoke",
        "http_qualified": False,
        "hidden_logit_accuracy_qualified": False,
        "cep4pp2_qualified": False,
        "liveness_and_teardown_pass": True,
        "record_count": len(records),
        "generation_steps_per_request": count,
        "real_decode_steps_per_request": count - 1,
        "all_requested_ids_exact": all(
            row["all_requested_ids_exact"] for row in comparisons
        ),
        "comparisons": comparisons,
    }


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(repo, checkpoint, reference_path, reference, manifest):
    if str(checkpoint.resolve()) != manifest["checkpoint"]:
        raise ValueError("Checkpoint path differs from the frozen run")
    if sha256(reference_path) != manifest["reference_report_sha256"]:
        raise ValueError("Independent reference report changed")
    sources = manifest["source_sha256"]
    required = {
        "rtp_llm/models_py/modules/dsv4/fp8/test/test_v41_checkpoint_engine_sm120.py",
        "rtp_llm/models_py/model_desc/deepseek_v41_model.py",
        "rtp_llm/cpp/normal_engine/test/V41NativeEngineTest.cc",
    }
    if not required.issubset(sources):
        raise ValueError("Manifest lacks native phase execution sources")
    for name, expected in sources.items():
        path = (repo / name).resolve()
        if not path.is_relative_to(repo.resolve()) or sha256(path) != expected:
            raise ValueError(f"Frozen native source changed: {name}")
    official_sources = reference["official_source_sha256"]
    if not {"model.py", "kernel.py", "engram.py", "convert.py", "config.json"}.issubset(
        official_sources
    ):
        raise ValueError("Reference lacks released source identities")
    for name, expected in official_sources.items():
        path = (checkpoint / "inference" / name).resolve()
        if (
            not path.is_relative_to((checkpoint / "inference").resolve())
            or sha256(path) != expected
        ):
            raise ValueError(f"Released reference source changed: {name}")
    if not {
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }.issubset(manifest.get("checkpoint_asset_sha256", {})):
        raise ValueError("Manifest lacks checkpoint config/tokenizer identities")
    for name, expected in manifest["checkpoint_asset_sha256"].items():
        path = (checkpoint / name).resolve()
        if not path.is_relative_to(checkpoint.resolve()) or sha256(path) != expected:
            raise ValueError(f"Checkpoint asset changed: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prefill", "decode"))
    parser.add_argument("--native-log", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    cli = parser.parse_args()
    manifest = json.loads(cli.run_manifest.read_text())
    reference = json.loads(cli.reference.read_text())
    verify_sources(cli.repo, cli.checkpoint, cli.reference, reference, manifest)
    report = verify_run(cli.phase, cli.native_log.read_text(), reference, manifest)
    report["evidence_sha256"] = {
        str(path): sha256(path)
        for path in (cli.native_log, cli.reference, cli.run_manifest)
    }
    report["revision"] = manifest["revision"]
    with cli.report.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("phase", "record_count", "all_requested_ids_exact")
            }
        )
    )
    if not report["all_requested_ids_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
