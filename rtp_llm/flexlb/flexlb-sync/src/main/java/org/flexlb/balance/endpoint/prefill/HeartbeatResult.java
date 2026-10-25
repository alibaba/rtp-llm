package org.flexlb.balance.endpoint.prefill;

import java.util.List;

public record HeartbeatResult(List<WorkerStatusFact> schedulerFacts, boolean schedulingInputsChanged) {
    public HeartbeatResult {
        schedulerFacts = List.copyOf(schedulerFacts);
    }
}
