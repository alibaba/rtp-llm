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
        // NON_BATCH returns an individual route decision. Batch token capacity
        // limits how requests may be combined; the Engine's max-sequence and KV
        // checks remain authoritative for this standalone request.
        ctx.stageDecisionGroup(List.of(head),
                new DecisionGroupMetadata("non_batch_immediate", ctx.size() - 1));
    }

    @Override
    public long queueWaitMs(BatcherContext ctx) {
        return 0;
    }
}
