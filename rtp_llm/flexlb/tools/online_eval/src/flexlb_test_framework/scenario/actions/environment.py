"""Bounded same-lane environment replacement and actual Java startup probes."""

import copy
import json

from ..contracts import StageHandler, StageOutput

MUTATIONS = {"removed_auto_tpm", "fifo_default_priority", "removed_engine_cancellation"}
PARSER_MESSAGES = (
    "config validation failed",
    "unrecognized field",
    "invalid flexlb_config",
    "configvalidationexception",
    "is required when",
)


def mutate_config(config, mutation):
    raw = copy.deepcopy(config)
    ordering = raw["scheduler"]["ordering"]
    if mutation == "removed_auto_tpm":
        if ordering["type"] != "PRIORITY" or "autoTpmEnabled" in raw:
            raise ValueError("removed_auto_tpm requires a legal PRIORITY base")
        raw["autoTpmEnabled"] = True
    elif mutation == "fifo_default_priority":
        if ordering["type"] != "FIFO" or "defaultPriority" in ordering:
            raise ValueError("fifo_default_priority requires a legal FIFO base")
        ordering["defaultPriority"] = 50
    elif mutation == "removed_engine_cancellation":
        preemption = ordering.get("preemption", {})
        if ordering[
            "type"
        ] != "PRIORITY" or "DECODE_ENGINE_OWNED" not in preemption.get(
            "allowedVictimStages", []
        ):
            raise ValueError(
                "removed_engine_cancellation requires a valid owned-cancellation base"
            )
        preemption["engineCancellation"] = {
            "ackTimeoutMs": 50,
            "completionTimeoutMs": 1000,
        }
    else:
        raise ValueError("unknown startup mutation")
    return raw


def _validate(params, plan, probe=False):
    from ..compiler import environment

    required = {"config_overrides", "mutation"} if probe else {"config_overrides"}
    allowed = (
        required
        if probe
        else required
        | {
            "n_prefill",
            "n_decode",
            "metric_whitelist",
            "mock_auto_fetch",
            "mock_fetch_attach_timeout_ms",
        }
    )
    if (
        not isinstance(params, dict)
        or set(params) - allowed
        or not required <= set(params)
    ):
        raise ValueError(f"{plan.path}: expected {sorted(required)}")
    if probe and params["mutation"] not in MUTATIONS:
        raise ValueError(f"{plan.path}: unknown raw config mutation")
    compiled = {}
    for profile in plan.profiles:
        original = environment(plan.environment, plan.path, profile)
        replacement = copy.deepcopy(plan.environment)
        replacement["config_overrides"] = copy.deepcopy(params["config_overrides"])
        for role in (
            "n_prefill",
            "n_decode",
            "metric_whitelist",
            "mock_auto_fetch",
            "mock_fetch_attach_timeout_ms",
        ):
            if role in params:
                replacement[role] = params[role]
        target = environment(replacement, plan.path, profile)
        for axis in ("decision", "dispatcher"):
            if original["effective_axes"][axis] != target["effective_axes"][axis]:
                raise ValueError(
                    f"{plan.path}: environment replacement cannot change {axis}"
                )
        if probe:
            if target["master_layout"] != "single" or target["master_sync_log"]:
                raise ValueError(
                    "startup probes require a single Master and their own private logs"
                )
            mutate_config(target["resolved_config"], params["mutation"])
        compiled[profile] = target
    if not compiled:
        raise ValueError("environment replacement needs selected profiles")
    return dict(
        environments=compiled,
        environment=replacement,
        **({"mutation": params["mutation"]} if probe else {}),
    )


def _close(ctx, deadline):
    results = ctx.cleanup(min(30, deadline.remaining()), retain_failed=True)
    if any(row["status"] != "PASS" for row in results):
        raise RuntimeError(
            "environment replacement cleanup failed; next environment not started"
        )
    ctx.env = ctx.ops = None
    return results


def _begin(ctx):
    ctx.env_epoch += 1
    ctx.add_cleanup(
        f"environment_epoch_{ctx.env_epoch}", lambda d: ctx.backend.teardown(ctx, d)
    )


def reconfigure(ctx, params, deadline):
    _close(ctx, deadline)
    _begin(ctx)
    plan = params["environments"][ctx.instance["profile"]]
    ctx.env, ctx.ops = ctx.backend.setup(ctx, plan, deadline)
    return StageOutput({"environment": ctx.register_resource("environment", ctx.env)})


def startup_probe(ctx, params, deadline):
    _close(ctx, deadline)
    _begin(ctx)
    plan = params["environments"][ctx.instance["profile"]]
    raw_config = mutate_config(plan["resolved_config"], params["mutation"])
    observation = dict(
        env_epoch=ctx.env_epoch, mutation=params["mutation"], raw_config=raw_config
    )
    path = ctx.artifact_dir / f"startup-probe-{ctx.env_epoch}.json"
    try:
        observation.update(
            ctx.backend.probe_startup(ctx, plan, json.dumps(raw_config), deadline)
        )
    finally:
        # Persist even an incomplete attempt and retain final-cleanup callbacks
        # when intermediate teardown fails. Never start another epoch on failure.
        try:
            observation["cleanup"] = _close(ctx, deadline)
        finally:
            path.write_text(json.dumps(observation, indent=2) + "\n")
    text = "\n".join(log["tail"] for log in observation["logs"]).lower()
    matched = [message for message in PARSER_MESSAGES if message in text]
    rejected = (
        not observation["started"]
        and bool(observation["startup_error"])
        and bool(observation["master_returncodes"])
        and all(code is not None for code in observation["master_returncodes"])
        and bool(matched)
    )
    observation["matched_parser_messages"] = matched
    path.write_text(json.dumps(observation, indent=2) + "\n")
    return StageOutput(
        dict(
            rejected=rejected,
            parser_matched=bool(matched),
            environment_absent=observation["current_absent_before_cleanup"],
            evidence=ctx.register_resource(
                "environment_probe", observation, historical=True
            ),
        ),
        artifacts=[str(path)],
    )


HANDLERS = [
    StageHandler(
        "environment_reconfigure",
        _validate,
        reconfigure,
        {"environment": "environment"},
        max_environment_workers=lambda params, profile: (
            params["environments"][profile]["n_prefill"]
            + params["environments"][profile]["n_decode"]
        ),
        next_environment=lambda params: params["environment"],
    ),
    StageHandler(
        "environment_startup_probe",
        lambda params, plan: _validate(params, plan, probe=True),
        startup_probe,
        {
            "rejected": "boolean",
            "parser_matched": "boolean",
            "environment_absent": "boolean",
            "evidence": "environment_probe",
        },
    ),
]
