"""Durable checkpoints and independently budgeted evidence/report delivery."""

from runtime.deadline import Deadline, interruptible
from scenario.run_result import RunResult, FinalizationPhase, FinalizationResult, DeliveryStatus, WorkloadResult


class RunFinalizer:
    def run(self, ctx, result: RunResult, started, policy) -> RunResult:
        result.finalization_started = True
        result_path = ctx.artifact_dir / "result.json"

        def checkpoint(pending=False):
            from artifacts.json_io import write_json
            document = result.to_dict()
            if pending:
                document["status"] = "FINALIZING"
            write_json(result_path, document)

        checkpoint(pending=policy is not None)
        evidence = None
        if policy is not None:
            budgets = result.plan.execution
            evidence = self._phase(ctx, result, FinalizationPhase.EVIDENCE,
                                   budgets.finalize_timeout_s, started, checkpoint,
                                   lambda document, deadline: policy.finalize(ctx, document, deadline))
            if result.finalization[-1].status == DeliveryStatus.PASS:
                self._phase(ctx, result, FinalizationPhase.REPORT, budgets.report_timeout_s,
                            started, checkpoint,
                            lambda document, deadline: self._render(policy, ctx, document, evidence, deadline))
            else:
                result.finalization.append(FinalizationResult(FinalizationPhase.REPORT,
                    budgets.report_timeout_s, DeliveryStatus.BLOCKED, "evidence finalization failed"))
        self._phase(ctx, result, FinalizationPhase.ARCHIVE,
                    result.plan.execution.archive_timeout_s, started, checkpoint,
                    lambda document, deadline: self._archive(ctx, result, document, evidence, deadline))
        result.duration_ms = int((ctx.clock()-started)*1000)
        checkpoint()
        return result

    @staticmethod
    def _archive(ctx, result, document, analysis, deadline):
        from artifacts.experiment import publish_run
        # Archiving is delivery, not an experimental stage. The container records
        # settled execution/report results; its own successful commit is the index.
        document = dict(document)
        document['finalization'] = [row for row in document.get('finalization', [])
                                    if row['phase'] != FinalizationPhase.ARCHIVE.value]
        result.artifacts = publish_run(ctx.artifact_dir, ctx.experiment_definition, document,
                                      analysis=analysis, deadline=deadline)

    @staticmethod
    def _render(policy, ctx, document, evidence, deadline):
        if not isinstance(evidence, dict):
            raise TypeError("evidence finalization must return report analysis")
        evidence["status"] = document["status"]
        evidence["outcome"] = document["outcome"]
        return policy.render(ctx, document, evidence, deadline)

    def _phase(self, ctx, result, phase, budget, started, checkpoint, operation):
        phase_start = ctx.clock()
        deadline = Deadline(phase_start + budget, ctx.clock, ctx.sleeper)
        completed = result.adjudicate()
        row = FinalizationResult(phase, budget)
        result.finalization.append(row)
        checkpoint(pending=True)
        document = result.to_dict()
        # The policy receives execution's terminal outcome, not a transient delivery state.
        document["status"] = completed.status
        from dataclasses import asdict
        document["outcome"] = dict(asdict(completed))
        value = None
        try:
            with interruptible(deadline, ctx.enforce_deadlines):
                value = operation(document, deadline)
                deadline.check()
            row.status = DeliveryStatus.PASS
        except Exception as exc:
            row.status = DeliveryStatus.TIMEOUT if isinstance(exc, TimeoutError) else DeliveryStatus.ERROR
            row.error = f"{type(exc).__name__}: {exc}"
            result.error = result.error or f"{phase.value} finalization failed: {row.error}"
        finally:
            result.absorb_policy_result(document)
        if row.status != DeliveryStatus.PASS and phase != FinalizationPhase.ARCHIVE:
            if result.workload is None:
                result.workload = WorkloadResult()
            if phase == FinalizationPhase.EVIDENCE:
                result.workload.runtime_validity = "INVALID"
                result.workload.report_status = "BLOCKED"
            elif phase == FinalizationPhase.REPORT:
                result.workload.report_status = row.status.value
        elif phase == FinalizationPhase.REPORT and result.workload is not None:
            result.workload.report_status = "PASS"
        row.duration_ms = int((ctx.clock()-phase_start)*1000)
        result.duration_ms = int((ctx.clock()-started)*1000)
        checkpoint()
        return value
