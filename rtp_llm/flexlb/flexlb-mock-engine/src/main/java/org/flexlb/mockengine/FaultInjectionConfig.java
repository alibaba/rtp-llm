package org.flexlb.mockengine;

/**
 * Immutable fault-injection configuration for the Java mock engine.
 *
 * <p>Controls how the mock gRPC service responds to requests in various failure
 * scenarios. Modeled after {@code MockWorkerBehavior} with a fluent builder.
 *
 * <pre>{@code
 * FaultInjectionConfig config = FaultInjectionConfig.builder()
 *     .failOnEnqueue(true)
 *     .enqueueDelayMs(500)
 *     .kvPressureTokens(500_000)
 *     .crashAfterNRequests(10)
 *     .build()
 * }</pre>
 */
public final class FaultInjectionConfig {

    private final boolean failOnEnqueue;
    private final long enqueueErrorCode;
    private final String enqueueErrorMessage;
    private final long enqueueDelayMs;
    private final long generateDelayMs;
    private final boolean generateError;
    private final boolean fetchError;
    private final boolean noRespond;
    private final long kvPressureTokens;
    private final int queueDepthLimit;
    private final int crashAfterNRequests;

    private FaultInjectionConfig(Builder b) {
        this.failOnEnqueue = b.failOnEnqueue;
        this.enqueueErrorCode = b.enqueueErrorCode;
        this.enqueueErrorMessage = b.enqueueErrorMessage;
        this.enqueueDelayMs = b.enqueueDelayMs;
        this.generateDelayMs = b.generateDelayMs;
        this.generateError = b.generateError;
        this.fetchError = b.fetchError;
        this.noRespond = b.noRespond;
        this.kvPressureTokens = b.kvPressureTokens;
        this.queueDepthLimit = b.queueDepthLimit;
        this.crashAfterNRequests = b.crashAfterNRequests;
    }

    public boolean isFailOnEnqueue() {
        return failOnEnqueue;
    }

    public long getEnqueueErrorCode() {
        return enqueueErrorCode;
    }

    public String getEnqueueErrorMessage() {
        return enqueueErrorMessage;
    }

    public long getEnqueueDelayMs() {
        return enqueueDelayMs;
    }

    public long getGenerateDelayMs() {
        return generateDelayMs;
    }

    public boolean isGenerateError() {
        return generateError;
    }

    public boolean isFetchError() {
        return fetchError;
    }

    public boolean isNoRespond() {
        return noRespond;
    }

    public long getKvPressureTokens() {
        return kvPressureTokens;
    }

    public int getQueueDepthLimit() {
        return queueDepthLimit;
    }

    public int getCrashAfterNRequests() {
        return crashAfterNRequests;
    }

    /**
     * Create a new mutable builder with sensible defaults.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Create a copy of this config with modifications.
     */
    public Builder toBuilder() {
        return new Builder()
                .failOnEnqueue(failOnEnqueue)
                .enqueueErrorCode(enqueueErrorCode)
                .enqueueErrorMessage(enqueueErrorMessage)
                .enqueueDelayMs(enqueueDelayMs)
                .generateDelayMs(generateDelayMs)
                .generateError(generateError)
                .fetchError(fetchError)
                .noRespond(noRespond)
                .kvPressureTokens(kvPressureTokens)
                .queueDepthLimit(queueDepthLimit)
                .crashAfterNRequests(crashAfterNRequests);
    }

    @Override
    public String toString() {
        return "FaultInjectionConfig{"
                + "failOnEnqueue=" + failOnEnqueue
                + ", enqueueErrorCode=" + enqueueErrorCode
                + ", enqueueDelayMs=" + enqueueDelayMs
                + ", generateDelayMs=" + generateDelayMs
                + ", generateError=" + generateError
                + ", fetchError=" + fetchError
                + ", noRespond=" + noRespond
                + ", kvPressureTokens=" + kvPressureTokens
                + ", queueDepthLimit=" + queueDepthLimit
                + ", crashAfterNRequests=" + crashAfterNRequests
                + '}';
    }

    public static final class Builder {
        private boolean failOnEnqueue = false;
        private long enqueueErrorCode = 13;
        private String enqueueErrorMessage = "mock enqueue failure";
        private long enqueueDelayMs = 0;
        private long generateDelayMs = 0;
        private boolean generateError = false;
        private boolean fetchError = false;
        private boolean noRespond = false;
        private long kvPressureTokens = 0;
        private int queueDepthLimit = 0;
        private int crashAfterNRequests = 0;

        private Builder() {
        }

        public Builder failOnEnqueue(boolean fail) {
            this.failOnEnqueue = fail;
            return this;
        }

        public Builder enqueueErrorCode(long code) {
            this.enqueueErrorCode = code;
            return this;
        }

        public Builder enqueueErrorMessage(String msg) {
            this.enqueueErrorMessage = msg;
            return this;
        }

        public Builder enqueueDelayMs(long ms) {
            this.enqueueDelayMs = ms;
            return this;
        }

        public Builder generateDelayMs(long ms) {
            this.generateDelayMs = ms;
            return this;
        }

        public Builder generateError(boolean generate) {
            this.generateError = generate;
            return this;
        }

        public Builder fetchError(boolean fetch) {
            this.fetchError = fetch;
            return this;
        }

        public Builder noRespond(boolean noRespond) {
            this.noRespond = noRespond;
            return this;
        }

        public Builder kvPressureTokens(long tokens) {
            this.kvPressureTokens = tokens;
            return this;
        }

        public Builder queueDepthLimit(int limit) {
            this.queueDepthLimit = limit;
            return this;
        }

        public Builder crashAfterNRequests(int n) {
            this.crashAfterNRequests = n;
            return this;
        }

        public FaultInjectionConfig build() {
            return new FaultInjectionConfig(this);
        }
    }
}
