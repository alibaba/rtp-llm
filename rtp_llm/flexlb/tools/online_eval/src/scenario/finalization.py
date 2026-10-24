"""Durable checkpoints and independently budgeted evidence/report delivery."""
from dataclasses import asdict

from runtime.deadline import Deadline, interruptible
from scenario.run_result import (
    RunResult, FinalizationPhase, FinalizationResult, DeliveryStatus,
    WorkloadResult, EvidenceResult, ReportResult,
)


class RunFinalizer:
    def run(self, ctx, result: RunResult, started, policy) -> RunResult:
        result.finalization_started = True
        result_path = ctx.artifact_dir / "result.json"

        def checkpoint(pending=False):
            from file_io import write_json
            document = result.to_dict()
            if pending:
                document["status"] = "FINALIZING"
            write_json(result_path, document)

        checkpoint(pending=policy is not None)
        analysis = None
        if policy is not None:
            budgets = result.plan.execution
            # The snapshot is input only: policy changes cannot replace execution state.
            snapshot = result.to_dict()
            evidence = self._phase(ctx, result, FinalizationPhase.EVIDENCE,
                budgets.finalize_timeout_s, started, checkpoint,
                lambda deadline: self._collect(policy, ctx, result, snapshot, deadline))
            if isinstance(evidence, EvidenceResult):
                analysis = evidence.analysis
            if result.finalization[-1].status == DeliveryStatus.PASS:
                outcome = result.adjudicate()
                self._phase(ctx, result, FinalizationPhase.REPORT,
                    budgets.report_timeout_s, started, checkpoint,
                    lambda deadline: self._render(policy, ctx, analysis, outcome, deadline))
            else:
                result.finalization.append(FinalizationResult(FinalizationPhase.REPORT,
                    budgets.report_timeout_s, DeliveryStatus.BLOCKED, "evidence finalization failed"))
        snapshot = result.to_dict()
        self._phase(ctx, result, FinalizationPhase.ARCHIVE,
            result.plan.execution.archive_timeout_s, started, checkpoint,
            lambda deadline: self._archive(ctx, result, snapshot, analysis, deadline))
        result.duration_ms = int((ctx.clock()-started)*1000)
        checkpoint()
        return result

    @staticmethod
    def _collect(policy, ctx, result, snapshot, deadline):
        # Own the typed evidence destination before I/O so failures retain partial metadata.
        result.workload = WorkloadResult()
        evidence = policy.finalize(ctx, snapshot, deadline, workload=result.workload)
        if not isinstance(evidence, EvidenceResult) or not isinstance(evidence.workload, WorkloadResult):
            raise TypeError("evidence finalization must return EvidenceResult")
        return evidence

    @staticmethod
    def _archive(ctx, result, document, analysis, deadline):
        from reporting.experiment import publish_run
        result.artifacts = publish_run(ctx.artifact_dir, ctx.experiment_definition, document,
                                      analysis=analysis, deadline=deadline)
        if (not getattr(ctx.backend, "retain_tsdb", True)
                and all(row["status"] == "PASS" for row in document["cleanup"])):
            from artifacts.retention import prune_tsdb
            prune_tsdb(ctx.artifact_dir, result.artifacts, deadline)

    @staticmethod
    def _render(policy, ctx, analysis, outcome, deadline):
        if not isinstance(analysis, dict):
            raise TypeError("evidence finalization must return report analysis")
        analysis["status"], analysis["outcome"] = outcome.status, asdict(outcome)
        reports = policy.render(ctx, analysis, deadline)
        if not isinstance(reports, ReportResult) or not reports.reports:
            raise TypeError("report delivery must return nonempty ReportResult")
        analysis.setdefault("workload", {}).update(reports.to_dict())
        return reports

    def _phase(self, ctx, result, phase, budget, started, checkpoint, operation):
        phase_start = ctx.clock()
        deadline = Deadline(phase_start + budget, ctx.clock, ctx.sleeper)
        row = FinalizationResult(phase, budget)
        result.finalization.append(row)
        checkpoint(pending=True)
        value = None
        try:
            with interruptible(deadline, ctx.enforce_deadlines):
                value = operation(deadline)
                if phase == FinalizationPhase.EVIDENCE:
                    result.workload = value.workload
                elif phase == FinalizationPhase.REPORT:
                    result.workload.details.update(value.to_dict())
                deadline.check()
            row.status = DeliveryStatus.PASS
        except Exception as exc:
            row.status = DeliveryStatus.TIMEOUT if isinstance(exc, TimeoutError) else DeliveryStatus.ERROR
            row.error = f"{type(exc).__name__}: {exc}"
            result.error = result.error or f"{phase.value} finalization failed: {row.error}"
        if row.status != DeliveryStatus.PASS and phase != FinalizationPhase.ARCHIVE:
            if result.workload is None:
                result.workload = WorkloadResult()
            workload = result.workload
            if phase == FinalizationPhase.EVIDENCE:
                workload.runtime_validity = "INVALID"
                workload.report_status = "BLOCKED"
            elif phase == FinalizationPhase.REPORT:
                workload.report_status = row.status.value
        elif phase == FinalizationPhase.REPORT and result.workload is not None:
            result.workload.report_status = "PASS"
        row.duration_ms = int((ctx.clock()-phase_start)*1000)
        result.duration_ms = int((ctx.clock()-started)*1000)
        checkpoint()
        return value
