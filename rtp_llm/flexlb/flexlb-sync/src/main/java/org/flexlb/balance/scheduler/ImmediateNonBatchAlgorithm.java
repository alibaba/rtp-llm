package org.flexlb.balance.scheduler;

import java.util.List;

/**
 * NON_BATCH dispatcher: every request becomes its own routing decision as
 * soon as it reaches the worker queue. There is no collection window, batch
 * size target, or predictor-based grouping.
 */
final class ImmediateNonBatchAlgorithm implements BatcherAlgorithm {

    @Override
    public void processQueue(BatcherContext ctx) {
        BatchItem head = ctx.peek();
        if (head == null) {
            return;
        }
        long nowMs = ctx.now();
        if (head.ctx().requestExpired(nowMs)) {
            ctx.dropHead(head);
            return;
        }
        long tokenCapacity = ctx.batchTokenCapacity();
        if (!BatchShape.empty().add(head).fitsCompute(tokenCapacity)) {
            ctx.rejectForBatchTokenCapacity(head, tokenCapacity);
            return;
        }
        ctx.stageDecisionGroup(List.of(head),
                new DecisionGroupMetadata("non_batch_immediate", ctx.size() - 1));
    }

    @Override
    public long queueWaitMs(BatcherContext ctx) {
        return 0;
    }
}
