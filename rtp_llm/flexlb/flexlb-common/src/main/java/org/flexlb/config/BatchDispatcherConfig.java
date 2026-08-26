package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

/** Controls Master-side EnqueueBatch delivery, not decision-group formation. */
@Getter
@Setter
public final class BatchDispatcherConfig implements DispatcherConfig {

    private Integer maxInflightBatchesPerPrefillWorker;
    private long enqueueRpcTimeoutMs = 5000;

    @Override
    public String typeName() {
        return "BATCH";
    }

    @Override
    public boolean requiresGenerateInput() {
        return true;
    }

    @Override
    public Integer maxInflightDeliveriesPerPrefillWorker() {
        return maxInflightBatchesPerPrefillWorker;
    }

    @Override
    public void validateFor(SchedulerConfig scheduler) {
        if (scheduler instanceof DirectSchedulerConfig) {
            throw new ConfigValidationException(
                    "dispatcher.type", "DIRECT requires NON_BATCH");
        }
        if (enqueueRpcTimeoutMs <= 0L) {
            throw new ConfigValidationException(
                    "dispatcher.enqueueRpcTimeoutMs",
                    "must be greater than zero");
        }
        if (maxInflightBatchesPerPrefillWorker != null
                && maxInflightBatchesPerPrefillWorker <= 0) {
            throw new ConfigValidationException(
                    "dispatcher.maxInflightBatchesPerPrefillWorker",
                    "must be greater than zero");
        }
    }
}
