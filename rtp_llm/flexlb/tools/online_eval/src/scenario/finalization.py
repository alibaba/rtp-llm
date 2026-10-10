"""Durable run checkpoints and independently budgeted evidence/report delivery."""

import copy
from runtime.deadline import Deadline, interruptible
from runtime.outcome import apply_outcome


class RunFinalizer:
    def run(self, ctx, result, started, policy):
        instance = ctx.instance
        clock, sleeper = ctx.clock, ctx.sleeper
        enforce_deadlines = ctx.enforce_deadlines
        _policy = policy
        from artifacts.json_io import write_json
        result["execution_status"] = result["status"]
        result["finalization"] = []
        apply_outcome(result)
        result_path = ctx.artifact_dir / "result.json"
        def checkpoint():
            apply_outcome(result)
            # An interrupted finalizer must never leave a terminal-looking PASS.
            pending = dict(result, status="FINALIZING") if _policy is not None else result
            write_json(result_path, pending)
        checkpoint()
        if _policy is not None:
            analysis = None
            for phase, budget_key in (("evidence", "finalize_timeout_s"), ("report", "report_timeout_s")):
                phase_start = clock()
                budget = instance["execution"][budget_key]
                deadline = Deadline(phase_start + budget, clock, sleeper)
                completed_status = result["status"]
                completed_outcome = copy.deepcopy(result["outcome"])
                row = dict(phase=phase, status="RUNNING", budget_s=budget, error=None, duration_ms=0)
                result["finalization"].append(row)
                checkpoint()
                try:
                    with interruptible(deadline, enforce_deadlines):
                        if phase == "evidence":
                            analysis = _policy.finalize(ctx, result, deadline)
                        else:
                            analysis["status"] = completed_status
                            analysis["outcome"] = completed_outcome
                            _policy.render(ctx, result, analysis, deadline)
                        deadline.check()
                    row["status"] = "PASS"
                except Exception as exc:
                    row["status"] = "TIMEOUT" if isinstance(exc, TimeoutError) else "ERROR"
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    result["status"] = row["status"]
                    result["error"] = result["error"] or f"{phase} finalization failed: {row['error']}"
                    if phase == "evidence":
                        result.setdefault("workload", {}).update(runtime_validity="INVALID", report_status="BLOCKED")
                    else:
                        result["workload"]["report_status"] = row["status"]
                row["duration_ms"] = int((clock() - phase_start) * 1000)
                result["duration_ms"] = int((clock() - started) * 1000)
                if phase == "report" and row["status"] == "PASS":
                    result["workload"]["report_status"] = "PASS"
                checkpoint()
                if phase == "evidence" and row["status"] != "PASS":
                    result["finalization"].append(dict(phase="report", status="BLOCKED",
                        budget_s=instance["execution"]["report_timeout_s"],
                        error="evidence finalization failed", duration_ms=0))
                    break
        result["duration_ms"] = int((clock() - started) * 1000)
        apply_outcome(result)
        write_json(result_path, result)
        return result
