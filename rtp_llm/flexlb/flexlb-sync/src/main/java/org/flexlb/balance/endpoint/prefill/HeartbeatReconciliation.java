package org.flexlb.balance.endpoint.prefill;

import java.util.List;

public record HeartbeatReconciliation(List<WorkerStatusFact> schedulerFacts,
                                      boolean schedulingInputsChanged) {
    public HeartbeatReconciliation {
        schedulerFacts = List.copyOf(schedulerFacts);
    }
}
