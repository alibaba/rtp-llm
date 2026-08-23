package org.flexlb.balance.scheduler;

import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Logger;

import java.util.List;

/**
 * Queue decision policy that releases exactly one request at a time.
 *
 * <p>The request becomes a final decision only after all delivery capacity has
 * been reserved. A capacity-blocked head remains active in its original queue
 * position and the worker waits for the exact resource-change event.
 */
final class SingleRequestBatcherAlgorithm implements BatcherAlgorithm {

    private static final String DECISION_REASON = "single_request";

    @Override
    public BatcherCycleResult processQueue(BatcherContext ctx) {
        BatcherContext.ActiveQueueSnapshot snapshot = ctx.snapshotActiveQueue(1);
        BatchItem head = snapshot.head();
        if (head == null) {
            return BatcherCycleResult.Outcome.NO_ACTION;
        }

        long nowMs = ctx.now();
        if (head.ctx().requestExpired(nowMs)) {
            Logger.debug("flexlb_single_drop request_id={} reason=request_expired "
                            + "expires_at_ms={} now_ms={}",
                    head.requestId(), head.ctx().getRequestExpiresAtMs(), nowMs);
            ctx.dropHead(head);
            return BatcherCycleResult.Outcome.QUEUE_CHANGED;
        }

        BatchShape shape = BatchShape.empty().add(head);
        long tokenCapacity = ctx.batchTokenCapacity();
        if (!shape.fitsCompute(tokenCapacity)) {
            ctx.rejectForBatchTokenCapacity(head, tokenCapacity);
            return BatcherCycleResult.Outcome.QUEUE_CHANGED;
        }
        if (!shape.fitsKv(ctx.batchKvCapacity())) {
            return awaitPrefillKvCapacity(ctx, snapshot, head);
        }

        // WorkerStatus and delivery ledgers advance independently of the
        // active-queue version. Re-read every advisory gate immediately before
        // hard-capacity admission, matching the fixed-window final gate. A resource
        // drop leaves
        // the request active; if compute capacity has become permanently too
        // small, the next worker tick reaches the explicit rejection above.
        if (!shape.fitsCompute(ctx.batchTokenCapacity())) {
            return BatcherCycleResult.Outcome.NO_ACTION;
        }
        if (!shape.fitsKv(ctx.batchKvCapacity())) {
            return awaitPrefillKvCapacity(ctx, snapshot, head);
        }

        int queueBefore = ctx.size();
        DecisionGroupMetadata metadata = new DecisionGroupMetadata(
                DECISION_REASON, Math.max(0, queueBefore - 1));
        BatcherCycleResult result = ctx.admitAndDeliverCapacityFeasiblePrefix(
                List.of(head), metadata, null, 0L);
        if (!(result instanceof BatcherCycleResult.Admitted admitted)) {
            return result;
        }

        if (head.deliveryMode() == DeliveryMode.BATCH_ENQUEUE
                && ctx.prefillEp() != null && ctx.reporter() != null) {
            reportBatchDecision(ctx, admitted, head);
        }
        Logger.debug("flexlb_single_decision delivery_mode={} wait_ms={} "
                        + "queue_before={} worker={} request_id={}",
                head.deliveryMode(), Math.max(0L, ctx.now() - head.enqueuedAtMs()),
                queueBefore, ctx.key(), head.requestId());
        return result;
    }

    @Override
    public long queueWaitMs(BatcherContext ctx) {
        return 0L;
    }

    private static BatcherCycleResult awaitPrefillKvCapacity(
            BatcherContext ctx,
            BatcherContext.ActiveQueueSnapshot snapshot,
            BatchItem head) {
        return ctx.awaitingSchedulingChange(
                head, snapshot.queueVersion(), snapshot.schedulingInputVersion(),
                head.ctx().getRequestExpiresAtMs(),
                BatcherCycleResult.SchedulingWaitReason.PREFILL_KV_CAPACITY);
    }

    private static void reportBatchDecision(
            BatcherContext ctx,
            BatcherCycleResult.Admitted admitted,
            BatchItem head) {
        try {
            String role = RoleType.PREFILL.name();
            String worker = ctx.prefillEp().getIp();
            String reason = admitted.metadata().reason();
            ctx.reporter().reportDispatchReason(role, worker, reason);
            ctx.reporter().reportBatchSize(role, worker, reason, 1);
            ctx.reporter().reportBatchCacheHitMetrics(
                    role, worker, head.hitCache(), head.seqLen());
            ctx.reporter().reportBatchTotalTokens(
                    role, worker, reason, head.seqLen());
        } catch (Throwable telemetryFailure) {
            Logger.error("Single-request decision telemetry failed worker={} request_id={}",
                    ctx.key(), head.requestId(), telemetryFailure);
        }
    }
}
