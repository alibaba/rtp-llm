"""Single/multiple grid execution and unified request-list postprocessing."""

import argparse
import copy
import hashlib
import json
import os
import shlex
from pathlib import Path

from rtp_llm.test.perf_test.cache_grid.config.perf_profile import (
    load_profile,
    resolve_label,
)
from rtp_llm.test.perf_test.cache_grid.config.topology import profile_cards
from rtp_llm.test.perf_test.cache_grid.runner.archive_upload import (
    create_archive,
    upload_archive,
    validate_destination,
)
from rtp_llm.test.perf_test.cache_grid.runner.observations import collect

CAPACITY_KEYS = {
    "max_seq_len",
    "max_context_batch_size",
    "max_batch_tokens_size",
    "concurrency_limit",
    "workspace_tokens",
}


def validate_merge_inputs(paths):
    """Compare frozen configuration; capacity differences are retained in the audit."""
    signatures, records, profiles = [], [], []
    for path in paths:
        data = json.loads(path.read_text())
        snapshot = path.parent / "profile.snapshot.json"
        profile = load_profile(snapshot) if snapshot.exists() else data.get("profile")
        if not profile and len(paths) > 1:
            raise ValueError(
                f"cannot validate merged results: missing profile at {path}"
            )
        profile = profile or {}
        profiles.append(profile)
        signature = copy.deepcopy(profile)
        # Labels, reporting settings and compiler paths do not define the workload.
        signature = {
            k: signature.get(k) or {}
            for k in ("engine", "engine_args", "engine_env", "cache_grid")
        }
        capacities = {}
        for section in ("engine", "engine_args"):
            for key in list(signature[section]):
                if key.lstrip("-") in CAPACITY_KEYS:
                    capacities[f"{section}.{key}"] = signature[section].pop(key)
        signature["cache_grid"].pop("measure_runs", None)
        run_config = copy.deepcopy(data.get("run_config", {}))
        workload = run_config.get("workload", {})
        if isinstance(workload, dict) and "workspace_tokens" in workload:
            capacities["run_config.workload.workspace_tokens"] = workload[
                "workspace_tokens"
            ]
        # Record effective configuration too; differing keys are examined below.
        records.append(
            dict(
                path=str(path),
                capacities=capacities,
                run_config=run_config,
                profile_sha256=hashlib.sha256(
                    json.dumps(profile, sort_keys=True).encode()
                ).hexdigest(),
            )
        )
        signatures.append(signature)
    if any(s != signatures[0] for s in signatures[1:]):
        raise ValueError(
            "model/topology/precision/cache/engine profiles differ across results"
        )
    # Effective environment overrides must also agree when launch manifests exist.
    environments = []
    for path in paths:
        manifest = path.parent / "cache_perf_launch.json"
        if manifest.exists():
            env = json.loads(manifest.read_text()).get("env", {})
            environments.append(
                {
                    k: v
                    for k, v in env.items()
                    if k.startswith(("DSV4_", "PREFILL_", "PERF_"))
                    or k in ("WORLD_SIZE", "CUDA_VISIBLE_DEVICES")
                }
            )
    if environments and any(e != environments[0] for e in environments[1:]):
        raise ValueError("effective engine environments differ across results")
    configs = [r["run_config"] for r in records]
    # Unknown effective differences are rejected, rather than guessed equivalent.
    if len(configs) > 1 and any(c != configs[0] for c in configs[1:]):

        def strip_capacity(value):
            if isinstance(value, dict):
                return {
                    k: strip_capacity(v)
                    for k, v in value.items()
                    if k not in CAPACITY_KEYS | {"measure_runs", "cache_measure_runs"}
                }
            if isinstance(value, list) and all(isinstance(v, str) for v in value):
                cleaned, skip = [], False
                for v in value:
                    if skip:
                        skip = False
                        continue
                    flag = v.split("=", 1)[0].lstrip("-")
                    if flag in CAPACITY_KEYS:
                        skip = "=" not in v
                    else:
                        cleaned.append(v)
                return cleaned
            return value

        effective = [strip_capacity(c) for c in configs]
        if any(c != effective[0] for c in effective[1:]):
            raise ValueError("effective run_config differs across results")
    return records, profiles[0] if profiles else {}


def run(args):
    from rtp_llm.test.perf_test.cache_grid.runner import cache_perf as cli

    multi = args.result_root is not None
    if (args.result_dir is None) == (args.result_root is None):
        raise ValueError("supply exactly one of --result-dir and --result-root")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.cards is not None and args.cards <= 0:
        raise ValueError("--cards must be positive")
    if args.oss_destination:
        validate_destination(args.oss_destination)
        if args.partial:
            raise ValueError(
                "--oss-destination requires complete results, not --partial"
            )
    if args.cases or args.profile_backend != "kineto":
        raise ValueError(
            "pipeline requires formal measurements, not retest/profile options"
        )
    if args.partial and not args.skip_test:
        raise ValueError("--partial requires --skip-test")
    if args.skip_test and (
        args.profile
        or args.grid
        or args.grid_dir
        or args.env
        or args.runs is not None
        or args.config
        or args.output_base
        or args.bazel
        or args.skip_reuse_validation
        or args.test_mode != "run"
    ):
        raise ValueError(
            "--skip-test uses saved results; launch overrides are not allowed"
        )
    if args.test_mode == "resume" and (
        args.profile
        or args.grid
        or args.grid_dir
        or args.env
        or args.runs is not None
        or args.config
        or args.output_base
        or args.bazel
        or args.skip_reuse_validation
    ):
        raise ValueError(
            "resume uses frozen configuration; launch overrides are forbidden"
        )
    root = (args.result_root or args.result_dir).resolve()
    listing = root / "batch_runs.json"
    launches = []
    state = {}
    if multi:
        if args.grid:
            raise ValueError("--result-root requires --grid-dir, not --grid")
        if args.skip_test or args.test_mode == "resume":
            state = json.loads(listing.read_text())
            runs = state["runs"]
        else:
            if not args.grid_dir or not args.profile:
                raise ValueError("multi-grid run requires --grid-dir and --profile")
            if root.exists() and any(root.iterdir()):
                raise ValueError("run requires an empty/new result root")
            grids = sorted(args.grid_dir.resolve().glob("*.json"))
            if not grids:
                raise ValueError("no JSON grids found")
            if any(g.stem in {"report", "grids", "formula"} for g in grids):
                raise ValueError("grid names report/grids/formula are reserved")
            runs = [
                dict(grid_json=str(g), result_dir=str(root / g.stem), status="pending")
                for g in grids
            ]
    else:
        if args.grid_dir:
            raise ValueError("--grid-dir requires --result-root")
        runs = [dict(result_dir=str(root), status="pending")]
    destinations = [Path(r["result_dir"]).resolve() for r in runs]
    if len(set(destinations)) != len(destinations) or not runs:
        raise ValueError("run manifest requires distinct nonempty result directories")
    if multi and any(d.parent != root for d in destinations):
        raise ValueError("run manifest paths must be direct children of result-root")
    # Build every launch before starting any expensive model work.
    for entry, destination in zip(runs, destinations):
        if args.skip_test:
            continue
        local = argparse.Namespace(**vars(args))
        local.result_dir = destination
        local.mode = args.test_mode
        result = destination / "cache_grid_results.json"
        if args.test_mode == "resume":
            if result.exists() and json.loads(result.read_text()).get("complete"):
                cli.load_saved(destination)
                entry["status"] = "completed"
                continue
            if multi and not result.exists():
                local.mode = "run"
                local.allow_restart_without_checkpoint = True
                local.profile = root / "profile.snapshot.json"
                local.grid = Path(entry["grid_json"])
                if not state.get("launch_options"):
                    raise ValueError("run has no frozen launch options")
                if hashlib.sha256(local.profile.read_bytes()).hexdigest() != state.get(
                    "profile_sha256"
                ):
                    raise ValueError("root profile snapshot changed")
                if hashlib.sha256(local.grid.read_bytes()).hexdigest() != entry.get(
                    "grid_sha256"
                ):
                    raise ValueError("pending grid snapshot changed")
                for key, value in state["launch_options"].items():
                    setattr(local, key, value)
        elif multi:
            local.grid = Path(entry["grid_json"])
        inherited = None
        if multi and state.get("inherited_environment") is not None:
            inherited = {
                k: v
                for k, v in os.environ.items()
                if not cli.managed_env(k) or cli.is_sensitive_environment_name(k)
            }
            inherited.update(state["inherited_environment"])
        plan = cli.build_plan(local, inherited)
        launches.append((entry, plan))
        print(shlex.join(plan["command"]), flush=True)
    paths = [d / "cache_grid_results.json" for d in destinations]
    print("Unified postprocess inputs: " + ", ".join(map(str, paths)), flush=True)
    if args.dry_run:
        if args.skip_test:
            available = [p for p in paths if p.exists()]
            if len(available) != len(paths) and not args.partial:
                raise ValueError("missing result files")
            validate_merge_inputs(available)
            collect(
                available,
                batch_size=args.batch_size,
                estimator=args.estimator,
                partial=args.partial,
            )
        return 0
    root.mkdir(parents=True, exist_ok=True)
    if multi and not args.skip_test and args.test_mode == "run":
        # Freeze inputs for runs that may not have started when interrupted.
        frozen = root / "grids"
        frozen.mkdir()
        (root / "profile.snapshot.json").write_text(
            json.dumps(load_profile(args.profile), indent=2)
        )
        state.update(
            schema_version=2,
            inherited_environment=launches[0][1]["summary"]["environment"],
            profile_sha256=hashlib.sha256(
                (root / "profile.snapshot.json").read_bytes()
            ).hexdigest(),
            launch_options={
                k: getattr(args, k)
                for k in (
                    "runs",
                    "env",
                    "config",
                    "output_base",
                    "bazel",
                    "skip_reuse_validation",
                )
            },
        )
        for entry in runs:
            source = Path(entry["grid_json"])
            target = frozen / source.name
            target.write_bytes(source.read_bytes())
            entry["grid_json"] = str(target)
            entry["grid_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        state["runs"] = runs
        cli._write_manifest(listing, state)
    summary = dict(schema_version=2, status="running", stages={}, artifacts={})
    summary_path = root / "pipeline_summary.json"

    def finish(status, code):
        summary["status"] = status
        cli._write_manifest(summary_path, summary)
        return code

    cli._write_manifest(summary_path, summary)
    try:
        for entry, plan in launches:
            entry["status"] = "running"
            if multi:
                cli._write_manifest(listing, state)
            code = cli.execute_plan(plan)
            entry.update(status="failed" if code else "completed", returncode=code)
            if multi:
                cli._write_manifest(listing, state)
            if code:
                summary["stages"]["test"] = dict(returncode=code, runs=runs)
                return finish("failed", code)
        summary["stages"]["test"] = dict(skipped=args.skip_test, runs=runs)
        missing = [str(p) for p in paths if not p.exists()]
        if missing and not args.partial:
            raise ValueError(f"missing result files: {missing}")
        available = [p for p in paths if p.exists()]
        if not available:
            raise ValueError("no result files available")
        provenance, profile = validate_merge_inputs(available)
        rows, audit = collect(
            available,
            batch_size=args.batch_size,
            estimator=args.estimator,
            partial=args.partial,
        )
        audit["missing_sources"] = missing
        report = root / "report"
        report.mkdir(exist_ok=True)
        for name, value in (
            ("sources", provenance),
            ("observations", rows),
            ("audit", audit),
        ):
            cli._write_manifest(report / f"{name}.json", value)
        summary["artifacts"].update(
            report_dir=str(report), audit=str(report / "audit.json")
        )
        if not rows:
            raise ValueError("no valid observations; see report/audit.json")
        fit_code = 0
        if not args.skip_fit and not args.partial:
            from rtp_llm.test.perf_test.cache_grid.formula import (
                prefill_formula_fit as fitter,
            )

            formula = args.formula_output_dir or (
                report / "formula" if multi else root / "formula"
            )
            fit_args = [
                "fit",
                "--inputs",
                *map(str, available),
                "--output-dir",
                str(formula),
                "--estimator",
                args.estimator,
            ]
            if args.batch_size is not None:
                fit_args += ["--batch-size", str(args.batch_size)]
            snapshot = available[0].parent / "profile.snapshot.json"
            if snapshot.exists():
                fit_args += ["--profile", str(snapshot)]
            fit_code = fitter.run_fit(fitter.build_parser().parse_args(fit_args))
            summary["artifacts"]["formula_dir"] = str(formula)
            summary["stages"]["fit"] = dict(returncode=fit_code)
            if fit_code not in (0, 3):
                return finish("failed", fit_code)
        else:
            summary["stages"]["fit"] = dict(
                skipped=True, reason="partial" if args.partial else "skip-fit"
            )
        from rtp_llm.test.perf_test.cache_grid.plot.unified_report import (
            write_charts,
            write_scatter_svg,
        )

        cards = args.cards if args.cards is not None else profile_cards(profile)
        cards = cards or 1
        charts = write_charts(
            rows,
            report,
            cards=cards,
            partial=args.partial,
            model_label=resolve_label(profile, None, "Model"),
        )
        if args.html_output:
            args.html_output.parent.mkdir(parents=True, exist_ok=True)
            args.html_output.write_bytes(Path(charts["html"]).read_bytes())
            charts["html_output"] = str(args.html_output)
        svg = args.svg_output or root / "prefill_3d.svg"
        cold = args.cold_svg_output or root / "prefill_cold_miss.svg"
        write_scatter_svg(rows, svg)
        write_scatter_svg(rows, cold, cold=True)
        summary["artifacts"].update(charts, svg=str(svg), cold_svg=str(cold))
        summary["stages"]["charts"] = dict(
            returncode=0, observations=len(rows), cards=cards
        )
        final_status = (
            "partial" if args.partial else "fit_rejected" if fit_code else "completed"
        )
        finish(final_status, fit_code)
        if args.oss_destination:
            archive = create_archive(root)
            summary["artifacts"]["archive"] = archive
            summary["stages"]["upload"] = {
                "destination": args.oss_destination,
                "status": "running",
            }
            finish(final_status, fit_code)
            try:
                upload_archive(Path(archive["path"]), args.oss_destination)
            except BaseException:
                summary["stages"]["upload"]["status"] = "failed"
                raise
            summary["stages"]["upload"]["status"] = "completed"
            finish(final_status, fit_code)
        return fit_code
    except BaseException as error:
        summary["error"] = repr(error)
        finish("failed", 1)
        raise
