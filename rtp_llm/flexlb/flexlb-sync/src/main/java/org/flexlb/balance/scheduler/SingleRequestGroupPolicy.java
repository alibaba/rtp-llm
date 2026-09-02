package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryMetadata;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.util.Logger;

import java.util.List;
import java.util.OptionalLong;

/**
 * Queue decision policy that releases exactly one request at a time.
 *
 * <p>The request becomes a final decision only after all delivery capacity has
 * been reserved. A capacity-blocked head remains active in its original queue
 * position and the worker waits for the exact resource-change event.
 */
final class SingleRequestGroupPolicy implements GroupPolicy {

    private static final String DECISION_REASON = "single_request";

    @Override
    public BatcherCycleResult processQueue(BatcherContext ctx) {
        BatcherContext.ActiveQueueSnapshot snapshot = ctx.snapshotActiveQueueHead();
        ScheduledRequest head = snapshot.head();
        if (head == null) {
            return BatcherCycleResult.NO_ACTION;
        }

        long nowMs = ctx.now();
        if (head.requestExpired(nowMs)) {
            Logger.debug("flexlb_single_drop request_id={} reason=request_expired "
                            + "expires_at_ms={} now_ms={}",
                    head.requestId(), head.expiresAtMs(), nowMs);
            ctx.dropHead(head);
            return BatcherCycleResult.QUEUE_CHANGED;
        }

        GroupPlanner.Shape shape =
                GroupPlanner.Shape.empty().add(head.seqLen());
        BatcherContext.BatchCapacitySnapshot capacity =
                ctx.batchCapacitySnapshot();
        if (!shape.fitsKv(capacity.batchKvCapacity())) {
            return awaitPrefillKvCapacity(ctx, snapshot, head);
        }

        // WorkerStatus and delivery ledgers advance independently of the
        // active-queue version. Re-read every advisory gate immediately before
        // hard-capacity admission, matching the fixed-window final gate. A resource
        // drop leaves the request active. Batch padded-token capacity does not
        // apply because this policy never combines requests.
        capacity = ctx.batchCapacitySnapshot();
        if (!shape.fitsKv(capacity.batchKvCapacity())) {
            return awaitPrefillKvCapacity(ctx, snapshot, head);
        }

        return ctx.runPredictionBound(head, () -> {
            int queueBefore = ctx.size();
            DeliveryMetadata metadata = new DeliveryMetadata(
                    DECISION_REASON, Math.max(0, queueBefore - 1));
            PrefillTimePredictor predictor = ctx.prefillEp().getPredictor();
            PrefillTimePredictor.Evaluator evaluator = predictor.evaluator();
            return ctx.admitAndDeliverCapacityFeasiblePrefix(
                    List.of(head), metadata, evaluator, OptionalLong.empty());
        });
    }

    private static BatcherCycleResult awaitPrefillKvCapacity(
            BatcherContext ctx,
            BatcherContext.ActiveQueueSnapshot snapshot,
            ScheduledRequest head) {
        return ctx.awaitingSchedulingChange(
                head, snapshot.queueVersion(), snapshot.schedulingInputVersion(),
                head.expiresAtMs(),
                BatcherCycleResult.SchedulingWaitReason.PREFILL_KV_CAPACITY);
    }

}
