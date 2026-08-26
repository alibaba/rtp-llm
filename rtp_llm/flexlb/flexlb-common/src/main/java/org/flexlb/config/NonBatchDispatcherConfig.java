package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

/** Controls frontend route delivery, not decision-group formation. */
@Getter
@Setter
public final class NonBatchDispatcherConfig implements DispatcherConfig {

    private Integer maxInflightRequestsPerPrefillWorker;

    @Override
    public String typeName() {
        return "NON_BATCH";
    }

    @Override
    public boolean requiresGenerateInput() {
        return false;
    }

    @Override
    public Integer maxInflightDeliveriesPerPrefillWorker() {
        return maxInflightRequestsPerPrefillWorker;
    }

    @Override
    public void validateFor(SchedulerConfig scheduler) {
        if (maxInflightRequestsPerPrefillWorker == null) {
            return;
        }
        if (!(scheduler instanceof QueueSchedulerConfig)) {
            throw new ConfigValidationException(
                    "dispatcher.maxInflightRequestsPerPrefillWorker",
                    "is supported only with QUEUE");
        }
        if (maxInflightRequestsPerPrefillWorker <= 0) {
            throw new ConfigValidationException(
                    "dispatcher.maxInflightRequestsPerPrefillWorker",
                    "must be greater than zero");
        }
    }
}
