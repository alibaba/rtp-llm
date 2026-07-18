package org.flexlb.mock;

import org.flexlb.engine.grpc.EngineRpcService;

/**
 * Behavior configuration for mock workers (Builder pattern).
 *
 * <p>Controls how the mock gRPC service responds to requests:
 * <ul>
 *   <li>{@code enqueueDelayMs} — delay before responding to EnqueueBatch</li>
 *   <li>{@code failOnEnqueue} — return error response for EnqueueBatch</li>
 *   <li>{@code ignoreCancel} — skip responding to Cancel (tests TTL fallback)</li>
 *   <li>{@code availableConcurrency} — WorkerStatusPB.available_concurrency</li>
 *   <li>{@code availableKvCache} — WorkerStatusPB.available_kv_cache</li>
 *   <li>{@code totalKvCache} — WorkerStatusPB.total_kv_cache</li>
 * </ul>
 *
 * <pre>{@code
 * MockWorkerBehavior.builder()
 *     .enqueueDelayMs(5000)
 *     .failOnEnqueue(false)
 *     .ignoreCancel(false)
 *     .availableConcurrency(10)
 *     .availableKvCache(1000000L)
 *     .build()
 * }</pre>
 */
public final class MockWorkerBehavior {

    private final long enqueueDelayMs;
    private final boolean failOnEnqueue;
    private final boolean ignoreCancel;
    private final int availableConcurrency;
    private final long availableKvCache;
    private final long totalKvCache;
    private final String enqueueErrorMessage;
    private final long enqueueErrorCode;
    private final EngineRpcService.RoleTypePB roleType;

    private MockWorkerBehavior(Builder b) {
        this.enqueueDelayMs = b.enqueueDelayMs;
        this.failOnEnqueue = b.failOnEnqueue;
        this.ignoreCancel = b.ignoreCancel;
        this.availableConcurrency = b.availableConcurrency;
        this.availableKvCache = b.availableKvCache;
        this.totalKvCache = b.totalKvCache;
        this.enqueueErrorMessage = b.enqueueErrorMessage;
        this.enqueueErrorCode = b.enqueueErrorCode;
        this.roleType = b.roleType;
    }

    public long getEnqueueDelayMs() {
        return enqueueDelayMs;
    }

    public boolean isFailOnEnqueue() {
        return failOnEnqueue;
    }

    public boolean isIgnoreCancel() {
        return ignoreCancel;
    }

    public int getAvailableConcurrency() {
        return availableConcurrency;
    }

    public long getAvailableKvCache() {
        return availableKvCache;
    }

    public long getTotalKvCache() {
        return totalKvCache;
    }

    public String getEnqueueErrorMessage() {
        return enqueueErrorMessage;
    }

    public long getEnqueueErrorCode() {
        return enqueueErrorCode;
    }

    public EngineRpcService.RoleTypePB getRoleType() {
        return roleType;
    }

    /**
     * Create a new mutable builder with sensible defaults.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Create a copy of this behavior with modifications.
     */
    public Builder toBuilder() {
        return new Builder()
                .enqueueDelayMs(enqueueDelayMs)
                .failOnEnqueue(failOnEnqueue)
                .ignoreCancel(ignoreCancel)
                .availableConcurrency(availableConcurrency)
                .availableKvCache(availableKvCache)
                .totalKvCache(totalKvCache)
                .enqueueErrorMessage(enqueueErrorMessage)
                .enqueueErrorCode(enqueueErrorCode)
                .roleType(roleType);
    }

    public static final class Builder {
        private long enqueueDelayMs = 0;
        private boolean failOnEnqueue = false;
        private boolean ignoreCancel = false;
        private int availableConcurrency = 10;
        private long availableKvCache = 1_000_000L;
        private long totalKvCache = 2_000_000L;
        private String enqueueErrorMessage = "mock enqueue failure";
        private long enqueueErrorCode = 13;
        private EngineRpcService.RoleTypePB roleType = EngineRpcService.RoleTypePB.ROLE_TYPE_PDFUSION;

        private Builder() {
        }

        public Builder enqueueDelayMs(long ms) {
            this.enqueueDelayMs = ms;
            return this;
        }

        public Builder failOnEnqueue(boolean fail) {
            this.failOnEnqueue = fail;
            return this;
        }

        public Builder ignoreCancel(boolean ignore) {
            this.ignoreCancel = ignore;
            return this;
        }

        public Builder availableConcurrency(int concurrency) {
            this.availableConcurrency = concurrency;
            return this;
        }

        public Builder availableKvCache(long kv) {
            this.availableKvCache = kv;
            return this;
        }

        public Builder totalKvCache(long kv) {
            this.totalKvCache = kv;
            return this;
        }

        public Builder enqueueErrorMessage(String msg) {
            this.enqueueErrorMessage = msg;
            return this;
        }

        public Builder enqueueErrorCode(long code) {
            this.enqueueErrorCode = code;
            return this;
        }

        public Builder roleType(EngineRpcService.RoleTypePB role) {
            this.roleType = role;
            return this;
        }

        public MockWorkerBehavior build() {
            return new MockWorkerBehavior(this);
        }
    }
}
