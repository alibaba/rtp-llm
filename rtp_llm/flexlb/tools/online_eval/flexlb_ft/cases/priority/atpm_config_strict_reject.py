from __future__ import annotations

import json

from ...context import CaseContext
from ...grade import GradeReport
from ...harness import render_env
from ...registry import case
from ...support.priority import _PREEMPT_DECODE, _prio_config, _spec


@case(
    "atpm_config_strict_reject",
    category="priority",
    profiles=["single-nonbatch"],
    source="design §2.4 #13 — AT1",
)
def atpm_config_strict_reject(ctx: CaseContext):
    """Strict FLEXLB_CONFIG rejection (AT1): three illegal config variants
    injected as RAW JSON strings (bypassing the generator's Python
    mirror validation on purpose — the Java strict parser is the system
    under test) must fail MASTER STARTUP.

    Variants:
      1. a legal priority base plus a top-level ``"autoTpmEnabled": true``
         — a removed field must not resurrect: the STRICT_MAPPER
         (FAIL_ON_UNKNOWN_PROPERTIES, ConfigService) rejects the
         unrecognized field (ConfigServiceTest.java:204-209 white-box
         precedent);
      2. ``ordering.type=FIFO`` with ``scheduler.ordering.defaultPriority``
         spliced in — a cross-field violation (FifoOrderingConfig has no
         such field; the Python mirror raises on the same shape, the raw
         JSON splice bypasses it to hit the Java parser);
      3. PRIORITY with ``allowedVictimStages`` containing
         DECODE_ENGINE_OWNED but the ``engineCancellation`` block DELETED
         — the validator's owned-cancellation cross-check.

    Assertion signal (design §2.4): the Spring context dies during config
    parsing → start_master's health check never sees the port → ensure()
    raises RuntimeError("master failed to start:\n<tail log>") — the
    black-box failure signal; the tail is grepped for the strict-parser
    message family ("Config validation failed" / "Unrecognized field" /
    "Invalid FLEXLB_CONFIG").  The harness's _build failure path already
    stops the half-started processes and resets current=None (verified),
    so each next variant builds cleanly.  Each variant costs a full
    wait_for_port timeout (~90s — the status poll does not early-exit on
    process death); the design explicitly accepts the runtime.

    Profile declaration (single-nonbatch + PRIORITY axis injected at the
    case layer) is semantic ownership + regression efficiency only (the
    G11b label-honesty precedent): config rejection is
    profile-independent behaviour.
    """
    report = GradeReport(run_grade=ctx.grade)

    cfg1 = json.loads(render_env(ctx.profile, _prio_config()))
    cfg1["autoTpmEnabled"] = True
    variants = [("removed_field_autoTpmEnabled", json.dumps(cfg1))]

    cfg2 = json.loads(render_env(ctx.profile, _prio_config(ordering="fifo")))
    cfg2["scheduler"]["ordering"]["defaultPriority"] = 50
    variants.append(("fifo_with_defaultPriority", json.dumps(cfg2)))

    cfg3 = json.loads(render_env(ctx.profile, _prio_config(preemption=_PREEMPT_DECODE)))
    del cfg3["scheduler"]["ordering"]["preemption"]["engineCancellation"]
    variants.append(("owned_without_engineCancellation", json.dumps(cfg3)))

    results = []
    try:
        for i, (label, raw_config) in enumerate(variants):
            spec = _spec(ctx, f"atpm_bad{i}", raw_config=raw_config)
            raised = None
            try:
                ctx.env_manager.ensure(spec)
            except Exception as exc:  # RuntimeError from start_master
                raised = exc
            tail_text = str(raised) if raised is not None else ""
            # The tail now includes the logback file appender's output
            # (harness start_master appends ~/ai-whale/logs/application.log
            # bytes written by THIS start — implementation-period fix for
            # the stdout-only tail that carried no parser message).  The
            # keyword family covers all three rejection shapes: Jackson
            # strict-mapper (Unrecognized field), the cross-field
            # validator (ConfigValidationException / "is required when"),
            # and the legacy raw-config gate.
            matched = [
                kw
                for kw in (
                    "config validation failed",
                    "unrecognized field",
                    "invalid flexlb_config",
                    "configvalidationexception",
                    "is required when",
                )
                if kw in tail_text.lower()
            ]
            ok = raised is not None and bool(matched)
            results.append(
                (
                    label,
                    ok,
                    (
                        "startup "
                        + ("failed" if raised is not None else "SUCCEEDED (UNEXPECTED)")
                        + (
                            f", matched={matched}"
                            if matched
                            else ", no strict-parser message in tail"
                        )
                        + (
                            f", exc_head={tail_text[:140]!r}"
                            if raised is not None
                            else ""
                        )
                    ),
                )
            )
        report.invariant(
            "AT1",
            all(ok for (_l, ok, _d) in results),
            context="strict_config_reject",
            detail="; ".join(
                f"{l}={'ok' if ok else 'FAIL(' + d + ')'}" for l, ok, d in results
            ),
        )
        # A2 (Ryan P2-1): was invariant("P6", True, ...) — a can't-fail
        # registration.  Real (failable) condition: every rejected variant
        # build leaves EnvManager.current at None (harness _build's
        # failure path stops the half-started processes and never
        # publishes the env), so a master surviving a supposedly-fatal
        # variant flips this.
        report.invariant(
            "P6",
            ctx.env_manager.current is None,
            detail=(
                f"no live env after {len(variants)} startup-failure "
                f"variants (env_manager.current is None="
                f"{ctx.env_manager.current is None})"
            ),
        )
        return report.finish(
            f"rejected={[l for l, ok, _d in results if ok]}, "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
