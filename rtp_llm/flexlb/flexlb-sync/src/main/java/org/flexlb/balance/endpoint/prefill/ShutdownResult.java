package org.flexlb.balance.endpoint.prefill;

import org.flexlb.balance.endpoint.prefill.batch.FinishedBatch;
import org.flexlb.balance.scheduler.ScheduledRequest;

import java.util.List;

public record ShutdownResult(List<ScheduledRequest> removedItems,
                             List<FinishedBatch> finishedBatches,
                             Throwable checkFailure) {
    public ShutdownResult {
        removedItems = List.copyOf(removedItems);
        finishedBatches = List.copyOf(finishedBatches);
    }
}
