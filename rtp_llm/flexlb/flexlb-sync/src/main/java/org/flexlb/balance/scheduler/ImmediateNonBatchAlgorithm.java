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
        // FlexLB only returns a route in NON_BATCH mode; the caller submits the
        // request and the Engine remains the admission authority. In particular,
        // do not apply the batch dispatcher's padded-token limit here. Older
        // route-only workers do not report that field and may use its wire number
        // for unrelated metadata.
        ctx.stageDecisionGroup(List.of(head),
                new DecisionGroupMetadata("non_batch_immediate", ctx.size() - 1));
    }

    @Override
    public long queueWaitMs(BatcherContext ctx) {
        return 0;
    }
}
