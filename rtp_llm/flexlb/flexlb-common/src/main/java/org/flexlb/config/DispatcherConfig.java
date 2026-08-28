package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

/** Delivery mode and its transport-owned limits. */
@Getter
@Setter
public final class DispatcherConfig {

    public enum Type {
        BATCH,
        NON_BATCH
    }

    private Type type = Type.BATCH;
    private Integer maxInflightBatchesPerPrefillWorker;
    private Integer maxInflightRequestsPerPrefillWorker;
    private long enqueueRpcTimeoutMs = 5000;

    public static DispatcherConfig nonBatch() {
        DispatcherConfig config = new DispatcherConfig();
        config.type = Type.NON_BATCH;
        return config;
    }

    public String typeName() {
        return type.name();
    }

    public boolean requiresGenerateInput() {
        return type == Type.BATCH;
    }

    public Integer maxInflightDeliveriesPerPrefillWorker() {
        return type == Type.BATCH
                ? maxInflightBatchesPerPrefillWorker
                : maxInflightRequestsPerPrefillWorker;
    }

    void validateFor(SchedulerConfig scheduler) {
        if (type == Type.BATCH) {
            if (scheduler.getType() == SchedulerConfig.Type.DIRECT) {
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
            if (maxInflightRequestsPerPrefillWorker != null) {
                throw new ConfigValidationException(
                        "dispatcher.maxInflightRequestsPerPrefillWorker",
                        "is supported only with NON_BATCH");
            }
            return;
        }

        if (maxInflightBatchesPerPrefillWorker != null) {
            throw new ConfigValidationException(
                    "dispatcher.maxInflightBatchesPerPrefillWorker",
                    "is supported only with BATCH");
        }
        if (maxInflightRequestsPerPrefillWorker != null) {
            if (scheduler.getType() != SchedulerConfig.Type.QUEUE) {
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
}
