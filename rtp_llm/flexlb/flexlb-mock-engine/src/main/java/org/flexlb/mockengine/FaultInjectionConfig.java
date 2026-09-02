package org.flexlb.mockengine;

import java.util.List;

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
    // ── Status-report fault family (getWorkerStatus output-layer filters;
    // the completion queue / version protocol core stays untouched) ──
    private final boolean statusSuppressFinished;
    private final boolean statusSuppressRunning;
    private final List<Long> statusSuppressRids;
    private final boolean statusNoRespond;
    private final List<StatusFakeTask> statusFakeTasks;
    private final boolean statusDuplicateFinished;
    private final int statusCursorRegress;
    private final boolean statusVersionRegress;
    private final boolean statusZombieRunning;
    // ── EnqueueBatch ack fault family (ack content corruption only;
    // engine-side processing is untouched) ──
    private final int enqueueAckPartialFail;
    private final long enqueueAckErrorCode;
    private final boolean enqueueAckDrop;

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
        this.statusSuppressFinished = b.statusSuppressFinished;
        this.statusSuppressRunning = b.statusSuppressRunning;
        this.statusSuppressRids = b.statusSuppressRids;
        this.statusNoRespond = b.statusNoRespond;
        this.statusFakeTasks = b.statusFakeTasks;
        this.statusDuplicateFinished = b.statusDuplicateFinished;
        this.statusCursorRegress = b.statusCursorRegress;
        this.statusVersionRegress = b.statusVersionRegress;
        this.statusZombieRunning = b.statusZombieRunning;
        this.enqueueAckPartialFail = b.enqueueAckPartialFail;
        this.enqueueAckErrorCode = b.enqueueAckErrorCode;
        this.enqueueAckDrop = b.enqueueAckDrop;
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

    public boolean isStatusSuppressFinished() {
        return statusSuppressFinished;
    }

    public boolean isStatusSuppressRunning() {
        return statusSuppressRunning;
    }

    public List<Long> getStatusSuppressRids() {
        return statusSuppressRids;
    }

    public boolean isStatusNoRespond() {
        return statusNoRespond;
    }

    public List<StatusFakeTask> getStatusFakeTasks() {
        return statusFakeTasks;
    }

    public boolean isStatusDuplicateFinished() {
        return statusDuplicateFinished;
    }

    public int getStatusCursorRegress() {
        return statusCursorRegress;
    }

    public boolean isStatusVersionRegress() {
        return statusVersionRegress;
    }

    public boolean isStatusZombieRunning() {
        return statusZombieRunning;
    }

    public int getEnqueueAckPartialFail() {
        return enqueueAckPartialFail;
    }

    public long getEnqueueAckErrorCode() {
        return enqueueAckErrorCode;
    }

    public boolean isEnqueueAckDrop() {
        return enqueueAckDrop;
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
                .crashAfterNRequests(crashAfterNRequests)
                .statusSuppressFinished(statusSuppressFinished)
                .statusSuppressRunning(statusSuppressRunning)
                .statusSuppressRids(statusSuppressRids)
                .statusNoRespond(statusNoRespond)
                .statusFakeTasks(statusFakeTasks)
                .statusDuplicateFinished(statusDuplicateFinished)
                .statusCursorRegress(statusCursorRegress)
                .statusVersionRegress(statusVersionRegress)
                .statusZombieRunning(statusZombieRunning)
                .enqueueAckPartialFail(enqueueAckPartialFail)
                .enqueueAckErrorCode(enqueueAckErrorCode)
                .enqueueAckDrop(enqueueAckDrop);
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
                + ", statusSuppressFinished=" + statusSuppressFinished
                + ", statusSuppressRunning=" + statusSuppressRunning
                + ", statusSuppressRids=" + statusSuppressRids
                + ", statusNoRespond=" + statusNoRespond
                + ", statusFakeTasks=" + statusFakeTasks
                + ", statusDuplicateFinished=" + statusDuplicateFinished
                + ", statusCursorRegress=" + statusCursorRegress
                + ", statusVersionRegress=" + statusVersionRegress
                + ", statusZombieRunning=" + statusZombieRunning
                + ", enqueueAckPartialFail=" + enqueueAckPartialFail
                + ", enqueueAckErrorCode=" + enqueueAckErrorCode
                + ", enqueueAckDrop=" + enqueueAckDrop
                + '}';
    }

    /**
     * One synthetic task for the {@code status_fake_task} injection: a task
     * that never existed, appended to the getWorkerStatus output on every
     * poll until the injection is cleared (multiple injects accumulate into
     * a continuously reported set).
     *
     * <p>{@code phase} is one of {@code RUNNING}, {@code KV_ALLOCATED},
     * {@code RECEIVED} (running-form: appended to runningTaskInfo) or
     * {@code finished} (finished-form: appended to finishedTaskList with an
     * optional {@code errorCode} carried in the task's error_info).
     */
    public record StatusFakeTask(long requestId,
                                 long batchId,
                                 String phase,
                                 long errorCode) {

        /** True for the finished-form synthetic completion. */
        public boolean isFinishedForm() {
            return "finished".equalsIgnoreCase(phase);
        }
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
        private boolean statusSuppressFinished = false;
        private boolean statusSuppressRunning = false;
        private List<Long> statusSuppressRids = List.of();
        private boolean statusNoRespond = false;
        private List<StatusFakeTask> statusFakeTasks = List.of();
        private boolean statusDuplicateFinished = false;
        private int statusCursorRegress = 0;
        private boolean statusVersionRegress = false;
        private boolean statusZombieRunning = false;
        private int enqueueAckPartialFail = 0;
        private long enqueueAckErrorCode = 0;
        private boolean enqueueAckDrop = false;

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

        public Builder statusSuppressFinished(boolean suppress) {
            this.statusSuppressFinished = suppress;
            return this;
        }

        public Builder statusSuppressRunning(boolean suppress) {
            this.statusSuppressRunning = suppress;
            return this;
        }

        public Builder statusSuppressRids(List<Long> rids) {
            this.statusSuppressRids = rids == null ? List.of() : List.copyOf(rids);
            return this;
        }

        public Builder statusNoRespond(boolean noRespond) {
            this.statusNoRespond = noRespond;
            return this;
        }

        public Builder statusFakeTasks(List<StatusFakeTask> fakeTasks) {
            this.statusFakeTasks = fakeTasks == null ? List.of() : List.copyOf(fakeTasks);
            return this;
        }

        public Builder statusDuplicateFinished(boolean duplicate) {
            this.statusDuplicateFinished = duplicate;
            return this;
        }

        public Builder statusCursorRegress(int regress) {
            this.statusCursorRegress = regress;
            return this;
        }

        public Builder statusVersionRegress(boolean regress) {
            this.statusVersionRegress = regress;
            return this;
        }

        public Builder statusZombieRunning(boolean zombie) {
            this.statusZombieRunning = zombie;
            return this;
        }

        public Builder enqueueAckPartialFail(int k) {
            this.enqueueAckPartialFail = k;
            return this;
        }

        public Builder enqueueAckErrorCode(long code) {
            this.enqueueAckErrorCode = code;
            return this;
        }

        public Builder enqueueAckDrop(boolean drop) {
            this.enqueueAckDrop = drop;
            return this;
        }

        public FaultInjectionConfig build() {
            return new FaultInjectionConfig(this);
        }
    }
}
