package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

/** Per-Prefill concurrency: batches for BATCH, requests for NON_BATCH. */
@Getter
@Setter
public final class DispatcherConfig {

    public static final int DEFAULT_MAX_INFLIGHT_PER_PREFILL_WORKER = 2;

    public enum Type {
        BATCH,
        NON_BATCH
    }

    private Type type = Type.BATCH;
    private int maxInflightPerPrefillWorker = DEFAULT_MAX_INFLIGHT_PER_PREFILL_WORKER;

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

    void validateFor(SchedulerConfig scheduler) {
        if (type == null) {
            throw new ConfigValidationException("dispatcher.type", "is required");
        }
        if (type == Type.BATCH && scheduler.getType() == SchedulerConfig.Type.DIRECT) {
            throw new ConfigValidationException("dispatcher.type", "DIRECT requires NON_BATCH");
        }
        if (maxInflightPerPrefillWorker <= 0) {
            throw new ConfigValidationException("dispatcher.maxInflightPerPrefillWorker",
                    "must be greater than zero");
        }
    }
}
