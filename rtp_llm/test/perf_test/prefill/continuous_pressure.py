"""Map an already active local EngineSession and frozen grid to one pressure epoch."""

import json
import subprocess
from pathlib import Path

from rtp_llm.test.continuous_pressure.adapters.rtp_llm_rpc import (
    RtpLlmPressureBackend,
    provider_settings,
)
from rtp_llm.test.perf_test.errors import CaseExecutionError, PartialRunError
from rtp_llm.test.perf_test.modes import ModeExecutionResult
from rtp_llm.test.perf_test.plan import thaw
from rtp_llm.test.perf_test.prefill.prompts import PrefixPromptFactory
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.storage import atomic_write_json


def pressure_epoch_valid(manifest, summary, certificate=None):
    result = summary.get("result", {})
    validity = summary.get("validity", {})
    if not (
        manifest.get("complete") is True
        and manifest.get("session_id")
        and result.get("session_id") == manifest["session_id"]
        and "failure" in result
        and result["failure"] is None
        and all(
            validity.get(k) is True
            for k in (
                "execution_valid",
                "full_pressure_valid",
                "bidirectional_migration_valid",
            )
        )
    ):
        return False
    start, end = result.get("measurement_start"), result.get("measurement_end")
    if (
        type(start) not in (int, float)
        or type(end) not in (int, float)
        or not start < end
        or not isinstance(summary.get("window"), dict)
    ):
        return False
    if (manifest.get("capture_requested") or "profile" in result) and not all(
        result.get("profile", {}).get(k) is True
        for k in ("trace_valid", "pressure_valid")
    ):
        return False
    if "remote_service_capture" in manifest:
        return bool(
            certificate
            and certificate.get("status") == "certified"
            and certificate.get("run_id") == manifest["session_id"]
            and certificate.get("scope") == "remote_service"
        )
    return True


def epoch_linkage(plan, run_manifest, directory):
    return dict(
        framework_run_id=run_manifest["run_id"],
        framework_epoch=Path(directory).name,
        code_hash=run_manifest["code_version"],
        profile_hash=plan.config_fingerprint,
        grid_hash=plan.workload_fingerprint,
    )


def _case_results(plan, directory, manifest, summary, certificate=None):
    complete = pressure_epoch_valid(manifest, summary, certificate)
    counts = [dict(sent=0, completed=0, cancelled=0, failed=0) for _ in plan.cases]
    requests = directory / "requests.jsonl"
    if requests.exists():
        with requests.open() as rows:
            for line in rows:
                row = json.loads(line)
                if (
                    row.get("phase") != "measurement"
                    or row.get("kind") != "formal"
                    or row.get("dispatch_s") is None
                ):
                    continue
                if row.get("session_id") != manifest.get("session_id"):
                    raise ValueError("pressure request belongs to another epoch")
                i = row["metadata"]["case_index"]
                if type(i) is not int or not 0 <= i < len(counts):
                    raise ValueError("unknown pressure case index")
                count = counts[i]
                count["sent"] += 1
                if row["terminal"] == "cancellation":
                    count["cancelled"] += 1
                elif row.get("success") is True and row["terminal"] == "completion":
                    count["completed"] += 1
                else:
                    count["failed"] += 1
    if sum(c["sent"] for c in counts) != (summary.get("window") or {}).get(
        "measurement_requests"
    ):
        complete = False
    results = []
    for case, count in zip(plan.cases, counts):
        valid = bool(complete and count["completed"] and not count["failed"])
        results.append(
            dict(
                case_id=case.case_id,
                identity=thaw(case.identity),
                runs=count["sent"],
                status="ok" if valid else "request_failed",
                errors=(
                    []
                    if valid
                    else [
                        dict(
                            code="pressure_evidence_incomplete",
                            message="Missing valid epoch or successful measurement requests",
                        )
                    ]
                ),
                metrics=dict(
                    epoch=directory.relative_to(plan.result_directory).as_posix(),
                    session_id=manifest["session_id"],
                    epoch_valid=complete,
                    **count,
                ),
            )
        )
    return tuple(results)


def _nsys_command(path, session_id, timeout):
    def command(record, action, *options):
        argv = [path or "nsys", action]
        if action != "export":
            argv.append("--session=" + session_id)
        argv.extend(options)
        entry = {"command": argv}
        record.setdefault("nsys_commands", []).append(entry)
        process = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        entry.update(
            returncode=process.returncode, stderr=process.stderr, stdout=process.stdout
        )
        process.check_returncode()

    return command


def run_continuous_pressure(context, *, tokenizer, directory, capture, nsys_session):
    """Borrow the engine, own the adapter's channels, and return a ModeExecutionResult."""
    plan, session = context.plan, context.session
    config = thaw(plan.resolved_config)
    workload, engine = config["workload"], config["engine"]
    settings = dict(workload["pressure"])
    concurrency, seed_count = settings.pop("concurrency"), settings.pop("seed_count")
    rate, poisson = settings.pop("request_rate"), settings.pop("poisson")
    settings.pop("capture")
    remote_manifest = settings.pop("remote_launch_manifest", None)
    settings["mock_cache_settings"] = provider_settings(
        engine["args"], engine["env"], settings
    )
    options = workload["profiling"]
    factory = PrefixPromptFactory(tokenizer)
    port = session.client.port
    run_manifest = json.loads(
        (Path(plan.result_directory) / "manifest.json").read_text()
    )
    linkage = epoch_linkage(plan, run_manifest, directory)
    try:
        from rtp_llm.config.py_config_modules import DASH_SC_GRPC_SERVER_PORT_OFFSET

        with RtpLlmPressureBackend.connect(
            dash_endpoint=f"127.0.0.1:{port+DASH_SC_GRPC_SERVER_PORT_OFFSET}",
            pressure_endpoint=f"127.0.0.1:{port+1}",
            ranks=tuple(range(config["model"]["topology"]["tp"])),
            timeout=workload["request_timeout_seconds"],
        ) as backend:
            session.assert_healthy()
            backend.run(
                remote_launch_manifest=remote_manifest,
                settings=settings,
                cases=[dict(thaw(c.identity), case_id=c.case_id) for c in plan.cases],
                encode_marker=lambda text: tokenizer.encode(
                    text, add_special_tokens=False
                ),
                decode_marker=lambda ids: tokenizer.decode(
                    ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
                ),
                make_ids=lambda n, marker: factory._exact_text_and_ids(n, marker)[1],
                directory=directory,
                model_tokens=engine["max_seq_len"],
                admission_requests=engine["concurrency_limit"],
                batch_tokens=engine["max_batch_tokens"],
                cache_alignment=workload["cache_alignment"],
                commit_tail_tokens=workload["commit_tail_tokens"],
                concurrency=concurrency,
                seed_count=seed_count,
                rate=rate,
                poisson=poisson,
                random_seed=workload["seed"],
                test_block_num=int(
                    engine["args"].get(
                        "test_block_num", engine["env"].get("TEST_BLOCK_NUM", 0)
                    )
                ),
                capture=capture,
                trace_directory=directory / "traces",
                nsys_command=(
                    _nsys_command(
                        options.get("nsys_path"),
                        nsys_session,
                        options["trace_timeout_seconds"],
                    )
                    if capture and options["backend"] == "nsys"
                    else None
                ),
                manifest=linkage,
            )
            native_session = backend.session_id
            session.assert_healthy()
        evidence = directory / "pressure"
        manifest = json.loads((evidence / "manifest.json").read_text())
        summary = json.loads((evidence / "summary.json").read_text())
        reference = dict(linkage, session_id=native_session)
        if not all(manifest.get(k) == v for k, v in reference.items()):
            raise ValueError(
                "completed pressure evidence differs from active run and native session"
            )
        atomic_write_json(directory / "epoch_reference.json", reference)
        results = _case_results(plan, evidence, manifest, summary)
    except (RuntimeError, ValueError, OSError, KeyError, ImportError) as error:
        raise CaseExecutionError(
            f"continuous pressure epoch {directory.name}: {error}"
        ) from error
    healthy = all(c["status"] == "ok" for c in results)
    return ModeExecutionResult(
        RunState.COMPLETED if healthy else RunState.PARTIAL,
        results,
        failure=(
            None
            if healthy
            else PartialRunError("continuous pressure evidence is incomplete")
        ),
    )
