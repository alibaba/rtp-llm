package org.flexlb.mock;

import org.flexlb.engine.grpc.EngineRpcService;

import java.util.List;

/**
 * Behavior configuration for mock workers (Builder pattern).
 *
 * <p>Controls how the mock gRPC service responds to requests:
 * <ul>
 *   <li>{@code enqueueDelayMs} — delay before responding to EnqueueBatch</li>
 *   <li>{@code failOnEnqueue} — return error response for EnqueueBatch</li>
 *   <li>{@code availableConcurrency} — WorkerStatusPB.available_concurrency</li>
 *   <li>{@code availableKvCache} — WorkerStatusPB.available_kv_cache</li>
 *   <li>{@code totalKvCache} — WorkerStatusPB.total_kv_cache</li>
 *   <li>{@code runningTasks} — WorkerStatusPB.running_task_info entries</li>
 *   <li>{@code finishedTasks} — WorkerStatusPB.finished_task_list entries, replayed
 *       while the requester's latest_finished_version cursor is behind
 *       {@code latestFinishedVersion} (engine finished-window semantics)</li>
 *   <li>{@code runningDetailTruncated} — WorkerStatusPB.running_detail_truncated</li>
 * </ul>
 *
 * <pre>{@code
 * MockWorkerBehavior.builder()
 *     .enqueueDelayMs(5000)
 *     .failOnEnqueue(false)
 *     .availableConcurrency(10)
 *     .availableKvCache(1000000L)
 *     .build()
 * }</pre>
 */
public final class MockWorkerBehavior {

    private final long enqueueDelayMs;
    private final boolean failOnEnqueue;
    private final int availableConcurrency;
    private final long availableKvCache;
    private final long totalKvCache;
    private final String enqueueErrorMessage;
    private final long enqueueErrorCode;
    private final EngineRpcService.RoleTypePB roleType;
    private final List<EngineRpcService.TaskInfoPB> runningTasks;
    private final List<EngineRpcService.TaskInfoPB> finishedTasks;
    private final boolean runningDetailTruncated;
    private final Long latestFinishedVersion;

    private MockWorkerBehavior(Builder b) {
        this.enqueueDelayMs = b.enqueueDelayMs;
        this.failOnEnqueue = b.failOnEnqueue;
        this.availableConcurrency = b.availableConcurrency;
        this.availableKvCache = b.availableKvCache;
        this.totalKvCache = b.totalKvCache;
        this.enqueueErrorMessage = b.enqueueErrorMessage;
        this.enqueueErrorCode = b.enqueueErrorCode;
        this.roleType = b.roleType;
        this.runningTasks = List.copyOf(b.runningTasks);
        this.finishedTasks = List.copyOf(b.finishedTasks);
        this.runningDetailTruncated = b.runningDetailTruncated;
        this.latestFinishedVersion = b.latestFinishedVersion;
    }

    public long getEnqueueDelayMs() {
        return enqueueDelayMs;
    }

    public boolean isFailOnEnqueue() {
        return failOnEnqueue;
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

    /** Running-task detail entries echoed in every GetWorkerStatus response. */
    public List<EngineRpcService.TaskInfoPB> getRunningTasks() {
        return runningTasks;
    }

    /** Finished-task entries replayed while the requester cursor lags. */
    public List<EngineRpcService.TaskInfoPB> getFinishedTasks() {
        return finishedTasks;
    }

    public boolean isRunningDetailTruncated() {
        return runningDetailTruncated;
    }

    /**
     * Engine-reported latest_finished_version. {@code null} keeps the legacy
     * echo-the-request behavior (zero change for existing tests).
     */
    public Long getLatestFinishedVersion() {
        return latestFinishedVersion;
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
                .availableConcurrency(availableConcurrency)
                .availableKvCache(availableKvCache)
                .totalKvCache(totalKvCache)
                .enqueueErrorMessage(enqueueErrorMessage)
                .enqueueErrorCode(enqueueErrorCode)
                .roleType(roleType)
                .runningTasks(runningTasks)
                .finishedTasks(finishedTasks)
                .runningDetailTruncated(runningDetailTruncated)
                .latestFinishedVersion(latestFinishedVersion);
    }

    public static final class Builder {
        private long enqueueDelayMs = 0;
        private boolean failOnEnqueue = false;
        private int availableConcurrency = 10;
        private long availableKvCache = 1_000_000L;
        private long totalKvCache = 2_000_000L;
        private String enqueueErrorMessage = "mock enqueue failure";
        private long enqueueErrorCode = 13;
        private EngineRpcService.RoleTypePB roleType = EngineRpcService.RoleTypePB.ROLE_TYPE_PDFUSION;
        private List<EngineRpcService.TaskInfoPB> runningTasks = List.of();
        private List<EngineRpcService.TaskInfoPB> finishedTasks = List.of();
        private boolean runningDetailTruncated = false;
        private Long latestFinishedVersion = null;

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

        public Builder runningTasks(List<EngineRpcService.TaskInfoPB> tasks) {
            this.runningTasks = tasks;
            return this;
        }

        public Builder finishedTasks(List<EngineRpcService.TaskInfoPB> tasks) {
            this.finishedTasks = tasks;
            return this;
        }

        public Builder runningDetailTruncated(boolean truncated) {
            this.runningDetailTruncated = truncated;
            return this;
        }

        public Builder latestFinishedVersion(Long version) {
            this.latestFinishedVersion = version;
            return this;
        }

        public MockWorkerBehavior build() {
            return new MockWorkerBehavior(this);
        }
    }
}
