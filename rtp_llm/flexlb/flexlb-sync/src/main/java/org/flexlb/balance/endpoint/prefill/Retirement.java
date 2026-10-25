package org.flexlb.balance.endpoint.prefill;

import org.flexlb.balance.endpoint.prefill.batch.FinishedBatch;
import org.flexlb.balance.scheduler.ScheduledRequest;

import java.util.List;

public record Retirement(List<ScheduledRequest> ownedItems,
                         List<FinishedBatch> finishedBatches,
                         Throwable invariantFailure) {
    public Retirement {
        ownedItems = List.copyOf(ownedItems);
        finishedBatches = List.copyOf(finishedBatches);
    }
}
