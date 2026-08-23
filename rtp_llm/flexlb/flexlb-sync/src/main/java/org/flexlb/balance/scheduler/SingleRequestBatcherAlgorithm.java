package org.flexlb.balance.scheduler;

import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Logger;

import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Queue decision policy that releases exactly one request at a time.
 *
 * <p>This policy is independent of delivery ownership: the one request may be
 * returned to the frontend as a route decision or sent by the Master through a
 * singleton EnqueueBatch. If the selected worker has no delivery or resource
 * capacity, the request remains in the active ordered queue.
 */
final class SingleRequestBatcherAlgorithm implements BatcherAlgorithm {

    private static final String DECISION_REASON = "single_request";

    @Override
    public void processQueue(BatcherContext ctx) throws InterruptedException {
        BatcherContext.ActiveQueueSnapshot snapshot = ctx.snapshotActiveQueue(1);
        BatchItem head = snapshot.head();
        if (head == null) {
            return;
        }

        long nowMs = ctx.now();
        if (head.ctx().requestExpired(nowMs)) {
            Logger.debug("flexlb_single_drop request_id={} reason=request_expired "
                            + "expires_at_ms={} now_ms={}",
                    head.requestId(), head.ctx().getRequestExpiresAtMs(), nowMs);
            ctx.dropHead(head);
            return;
        }

        BatchShape shape = BatchShape.empty().add(head);
        long tokenCapacity = ctx.batchTokenCapacity();
        if (!shape.fitsCompute(tokenCapacity)) {
            ctx.rejectForBatchTokenCapacity(head, tokenCapacity);
            return;
        }
        if (!shape.fitsKv(ctx.batchKvCapacity()) || deliveryCapacityBlocked(ctx, head)) {
            TimeUnit.MILLISECONDS.sleep(1);
            return;
        }

        // WorkerStatus and delivery ledgers advance independently of the
        // active-queue version. Re-read every advisory gate immediately before
        // staging, matching the fixed-window final gate. A capacity drop leaves
        // the request active; if compute capacity has become permanently too
        // small, the next worker tick reaches the explicit rejection above.
        if (!shape.fitsCompute(ctx.batchTokenCapacity())
                || !shape.fitsKv(ctx.batchKvCapacity())
                || deliveryCapacityBlocked(ctx, head)) {
            TimeUnit.MILLISECONDS.sleep(1);
            return;
        }

        int queueBefore = ctx.size();
        DecisionGroupMetadata metadata = new DecisionGroupMetadata(
                DECISION_REASON, Math.max(0, queueBefore - 1));
        if (!ctx.stageDecisionGroupIfVersion(
                List.of(head), metadata, snapshot.queueVersion(), null, 0L)) {
            TimeUnit.MILLISECONDS.sleep(1);
            return;
        }

        if (head.deliveryMode() == DeliveryMode.BATCH_ENQUEUE
                && ctx.prefillEp() != null && ctx.reporter() != null) {
            String role = RoleType.PREFILL.name();
            String worker = ctx.prefillEp().getIp();
            ctx.reporter().reportDispatchReason(role, worker, DECISION_REASON);
            ctx.reporter().reportBatchSize(role, worker, DECISION_REASON, 1);
            ctx.reporter().reportBatchCacheHitMetrics(
                    role, worker, head.hitCache(), head.seqLen());
            ctx.reporter().reportBatchTotalTokens(
                    role, worker, DECISION_REASON, head.seqLen());
        }
        Logger.debug("flexlb_single_decision delivery_mode={} wait_ms={} "
                        + "queue_before={} worker={} request_id={}",
                head.deliveryMode(), Math.max(0L, ctx.now() - head.enqueuedAtMs()),
                queueBefore, ctx.key(), head.requestId());
    }

    @Override
    public long queueWaitMs(BatcherContext ctx) {
        return 0L;
    }

    private static boolean deliveryCapacityBlocked(BatcherContext ctx,
                                                    BatchItem head) {
        if (head.deliveryMode() == DeliveryMode.ROUTE_DECISION) {
            return ctx.availableDeliverySlots() == 0;
        }
        int maximum = ctx.maxInflightBatches();
        return maximum > 0 && ctx.prefillEp() != null
                && ctx.prefillEp().getInflightBatchCount() >= maximum;
    }
}
