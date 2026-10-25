package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

/**
 * Per-worker concurrency limits for dispatch decisions.
 */
@Getter
@Setter
public final class DispatcherConfig {

    public static final int DEFAULT_MAX_INFLIGHT_PER_PREFILL_WORKER = 2;
    public static final int DEFAULT_PREFILL_SUBQUEUE_LIMIT_MULTIPLIER = 4;
    public static final int DEFAULT_MAX_INFLIGHT_PER_ENCODER_WORKER = Integer.MAX_VALUE;
    public static final int DEFAULT_FETCH_ATTACH_TIMEOUT_MS = 3_000;
    public static final long DEFAULT_DECODE_STARVATION_THRESHOLD_MS = 500L;

    public enum Type {
        BATCH, NON_BATCH,
        /**
         * 新的非 batch 链路：先选 Prefill，Prefill 出队后再选 Decode。
         */
        NON_BATCH_V2
    }

    private Type type = Type.BATCH;
    private int maxInflightPerPrefillWorker = DEFAULT_MAX_INFLIGHT_PER_PREFILL_WORKER;
    /** Shared waiting capacity = endpoint count * in-flight limit * this multiplier. */
    private int prefillSubqueueLimitMultiplier = DEFAULT_PREFILL_SUBQUEUE_LIMIT_MULTIPLIER;
    private int maxInflightPerEncoderWorker = DEFAULT_MAX_INFLIGHT_PER_ENCODER_WORKER;
    /** BATCH only: engine wait for FetchResponse attachment, not generation duration. */
    private int fetchAttachTimeoutMs = DEFAULT_FETCH_ATTACH_TIMEOUT_MS;
    /**
     * NON_BATCH_V2：同一个 group 里排在最前面的请求放不下时，后面放得下的请求可以先走；
     * 最前面的请求从选完 Prefill 起等了超过这个时间，这个 group 改回严格按顺序。0 表示一直严格按顺序。
     * 支持热更新，Decode 队列每一轮都读最新值。
     */
    private long decodeStarvationThresholdMs = DEFAULT_DECODE_STARVATION_THRESHOLD_MS;

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

    void validate() {
        if (type == null) {
            throw new ConfigValidationException("dispatcher.type", "is required");
        }
        if (maxInflightPerPrefillWorker <= 0) {
            throw new ConfigValidationException("dispatcher.maxInflightPerPrefillWorker",
                    "must be greater than zero");
        }
        if (prefillSubqueueLimitMultiplier <= 0) {
            throw new ConfigValidationException("dispatcher.prefillSubqueueLimitMultiplier",
                    "must be greater than zero");
        }
        if (maxInflightPerEncoderWorker <= 0) {
            throw new ConfigValidationException("dispatcher.maxInflightPerEncoderWorker",
                    "must be greater than zero");
        }
        if (fetchAttachTimeoutMs <= 0) {
            throw new ConfigValidationException("dispatcher.fetchAttachTimeoutMs",
                    "must be greater than zero");
        }
        if (decodeStarvationThresholdMs < 0L) {
            throw new ConfigValidationException("dispatcher.decodeStarvationThresholdMs", "must not be negative");
        }
    }
}
