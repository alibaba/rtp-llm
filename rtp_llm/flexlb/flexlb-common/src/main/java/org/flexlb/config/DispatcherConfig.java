package org.flexlb.config;

import com.fasterxml.jackson.annotation.JsonAlias;
import lombok.Getter;
import lombok.Setter;

/** Per-Prefill delivery concurrency: batches for BATCH, requests for NON_BATCH. Waiting queues do not consume it. */
@Getter
@Setter
public final class DispatcherConfig {

    public static final int DEFAULT_MAX_INFLIGHT_PER_PREFILL_WORKER = 2;
    public static final int DEFAULT_FETCH_ATTACH_TIMEOUT_MS = 60_000;

    public enum Type {
        BATCH,
        NON_BATCH
    }

    private Type type = Type.BATCH;
    /** Order requests waiting on the same Prefill by compute cost within business priority. */
    private boolean prefillCostOrder;
    private int maxInflightPerPrefillWorker = DEFAULT_MAX_INFLIGHT_PER_PREFILL_WORKER;
    /** Engine waiting requests plus local dispatches awaiting Engine acknowledgement. */
    @JsonAlias("maxRunningSizePrefillWorker")
    private int maxWaitingSizePrefillWorker = -1;
    /** BATCH only: engine wait for FetchResponse attachment, not generation duration. */
    private int fetchAttachTimeoutMs = DEFAULT_FETCH_ATTACH_TIMEOUT_MS;

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
        if (maxWaitingSizePrefillWorker != -1 && maxWaitingSizePrefillWorker <= 0) {
            throw new ConfigValidationException("dispatcher.maxWaitingSizePrefillWorker",
                    "must be -1 (disabled) or greater than zero");
        }
        if (fetchAttachTimeoutMs <= 0) {
            throw new ConfigValidationException("dispatcher.fetchAttachTimeoutMs",
                    "must be greater than zero");
        }
    }
}
