package org.flexlb.balance.endpoint.prefill;

import org.flexlb.balance.endpoint.prefill.batch.FinishedBatch;

import java.util.List;

public record StatusReconciliation(List<WorkerStatusFact> schedulerFacts,
                                   List<FinishedBatch> finishedBatches,
                                   Throwable publicationFailure) {
    public StatusReconciliation {
        schedulerFacts = List.copyOf(schedulerFacts);
        finishedBatches = List.copyOf(finishedBatches);
    }
}
