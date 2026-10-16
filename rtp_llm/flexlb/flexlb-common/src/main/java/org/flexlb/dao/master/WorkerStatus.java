package org.flexlb.dao.master;

import com.fasterxml.jackson.annotation.JsonIgnore;
import org.flexlb.dao.master.EndpointTrace.Stage;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;

/**
 * One service-discovery generation of an Engine worker.
 *
 * <p>The object identity is the generation identity. Its independently updated
 * concurrency domains are published explicitly:
 *
 * <ul>
 *   <li>{@link TopologySnapshot}: discovery-owned address and placement data;</li>
 *   <li>{@link CommittedWorkerStatus}: stable Engine fields and both cursors
 *       from one successfully reduced, strictly newer status response;</li>
 *   <li>{@link PollHealth}: transport health from status polling, independent
 *       from Engine state and endpoint routability;</li>
 *   <li>cache status and poll coordination, which have independent writers.</li>
 * </ul>
 *
 * <p>A status response is first converted into an opaque {@link PreparedStatus}.
 * Endpoint and scheduler reducers consume its frozen observation, and only
 * then does {@link #publishPreparedStatus(PreparedStatus)} replace the one
 * committed holder. Readers therefore cannot observe fields from a response
 * whose cursor is still uncommitted.</p>
 */
public class WorkerStatus {

    private static final AtomicLong NEXT_GENERATION_ID = new AtomicLong();

    /** Serializes lifecycle transitions for this generation. */
    public final transient ReentrantLock lock = new ReentrantLock();

    /** Serializes address-scoped cache writes and final cleanup; independent of the status lock. */
    public final transient ReentrantLock cacheMutationLock = new ReentrantLock();

    private final long generationId = NEXT_GENERATION_ID.incrementAndGet();

    private final EndpointTrace trace = new EndpointTrace();

    @JsonIgnore
    public EndpointTrace trace() {
        return trace;
    }

    /** Set under {@link #lock}; read lock-free by poll gates. */
    private volatile boolean retiring;

    public record TopologySnapshot(
            String group,
            String ip,
            int port,
            int grpcPort,
            String site) {
    }

    /** Deeply immutable copy of the task fields reported by one status RPC. */
    public record TaskObservation(
            long requestId,
            long prefixLength,
            long prefillTime,
            long inputLength,
            long waitingTime,
            long iterateCount,
            long endTimeMs,
            long dpRank,
            long errorCode,
            String errorMessage,
            long batchId,
            TaskPhase phase,
            long executionTimeMs,
            PriorityPreemptionProgress priorityPreemptionProgress) {

        /** A pending observation that reports only priority cancellation progress. */
        @JsonIgnore
        public boolean isPriorityCancelOverlayOnly() {
            return (priorityPreemptionProgress == PriorityPreemptionProgress.CANCELING
                    || priorityPreemptionProgress == PriorityPreemptionProgress.CANCELED)
                    && phase == TaskPhase.PENDING;
        }

        private static TaskObservation copyOf(TaskInfo task) {
            return new TaskObservation(
                    task.getRequestId(),
                    task.getPrefixLength(),
                    task.getPrefillTime(),
                    task.getInputLength(),
                    task.getWaitingTime(),
                    task.getIterateCount(),
                    task.getEndTimeMs(),
                    task.getDpRank(),
                    task.getErrorCode(),
                    task.getErrorMessage(),
                    task.getBatchId(),
                    task.getPhase(),
                    task.getExecutionTimeMs(),
                    task.getPriorityPreemptionProgress());
        }
    }

    /**
     * Immutable view of one atomically published Engine status observation.
     * {@code runningTaskList} is the exact immutable map carried by the
     * transaction observation that produced this value.
     */
    public record EngineObservation(
            RoleType role,
            Long availableConcurrency,
            long availableKvCacheTokens,
            long totalKvCacheTokens,
            Map<String, TaskObservation> runningTaskList,
            double stepLatencyMs,
            long iterateCount,
            long dpSize,
            long tpSize,
            long dpRank,
            long maxSeqLen,
            long maxBatchTokensSize,
            long runningQueryLen,
            long waitingQueryLen) {

        public EngineObservation {
            Objects.requireNonNull(role, "role");
            runningTaskList = Map.copyOf(runningTaskList);
        }
    }

    /**
     * The two cursors acknowledged by one successful status transaction.
     * Keeping them in one immutable value prevents readers from observing a
     * committed status version with an older finished-task cursor.
     */
    public record AppliedStatusCursor(
            long statusVersion,
            long latestFinishedTaskVersion) {
    }

    /** The single public value for one successfully committed status round. */
    public record CommittedWorkerStatus(
            EngineObservation fields,
            AppliedStatusCursor cursor) {
    }

    /**
     * Status-poll transport health. It is never consulted for routing: the
     * endpoint registry and the endpoint generation gate own routability.
     */
    public record PollHealth(
            long lastSuccessfulPollUs,
            long successfulPollIntervalUs,
            long consecutiveTransportFailures,
            boolean reportedAlive) {

        public PollHealth {
            if (lastSuccessfulPollUs < 0L
                    || successfulPollIntervalUs < 0L
                    || consecutiveTransportFailures < 0L) {
                throw new IllegalArgumentException(
                        "poll health counters must be non-negative");
            }
        }
    }

    public record CacheIndexSnapshot(
            CacheStatus cacheStatus,
            boolean indexInitialized,
            long indexedVersion) {
    }

    /** One deeply frozen RPC observation shared by every status reducer. */
    public static final class StatusObservation {
        private final WorkerStatus owner;
        private final EngineObservation engine;
        private final boolean reportedAlive;
        private final Long statusVersion;
        private final Long latestFinishedVersion;
        private final Map<String, TaskObservation> finishedTasks;

        private StatusObservation(
                WorkerStatus owner,
                EngineObservation engine,
                boolean reportedAlive,
                Long statusVersion,
                Long latestFinishedVersion,
                Map<String, TaskObservation> finishedTasks) {
            this.owner = owner;
            this.engine = engine;
            this.reportedAlive = reportedAlive;
            this.statusVersion = statusVersion;
            this.latestFinishedVersion = latestFinishedVersion;
            this.finishedTasks = finishedTasks;
        }

        /** Exact service-discovery generation which issued this observation. */
        public WorkerStatus owner() {
            return owner;
        }

        public EngineObservation engine() {
            return engine;
        }

        public RoleType role() {
            return engine.role();
        }

        public boolean alive() {
            return reportedAlive;
        }

        public Long statusVersion() {
            return statusVersion;
        }

        public Long latestFinishedVersion() {
            return latestFinishedVersion;
        }

        public Map<String, TaskObservation> runningTasks() {
            return engine.runningTaskList();
        }

        public Map<String, TaskObservation> finishedTasks() {
            return finishedTasks;
        }

        public long runningQueryLen() {
            return engine.runningQueryLen();
        }

        public long waitingQueryLen() {
            return engine.waitingQueryLen();
        }
    }

    /**
     * Worker-issued transaction token. Its private owner/base identities make
     * it impossible to forge or replay against another committed generation.
     */
    public static final class PreparedStatus {
        private final CommittedWorkerStatus baseCommitted;
        private final StatusObservation observation;
        private final CommittedWorkerStatus nextCommitted;
        private final EndpointTrace.Poll pollTrace;

        private PreparedStatus(
                CommittedWorkerStatus baseCommitted,
                StatusObservation observation,
                CommittedWorkerStatus nextCommitted,
                EndpointTrace.Poll pollTrace) {
            this.baseCommitted = baseCommitted;
            this.observation = observation;
            this.nextCommitted = nextCommitted;
            this.pollTrace = pollTrace;
        }

        public StatusObservation observation() {
            return observation;
        }
    }

    private final AtomicReference<TopologySnapshot> topology;

    private final AtomicReference<CommittedWorkerStatus> committedStatus;

    private final AtomicReference<PollHealth> pollHealth;

    /** Cache polling is independent from worker-status polling. */
    private final AtomicReference<CacheIndexSnapshot> cacheIndexSnapshot =
            new AtomicReference<>(new CacheIndexSnapshot(null, false, -1L));
    private final AtomicLong cacheLastUpdateTime = new AtomicLong(-1L);

    // Transport contact and applied scheduling state deliberately have separate clocks.
    private final AtomicLong lastStatusAppliedUs = new AtomicLong();
    private final AtomicReference<PollLease> statusApplicationLease = new AtomicReference<>();
    private volatile long statusApplicationReceivedUs;
    private volatile boolean statusFreshnessEnforced;
    private volatile long statusFreshnessUs = 10_000_000L;
    private volatile boolean receivedStatusRpc;
    // Never held across state reduction, endpoint withdrawal, logging or cleanup.
    private final Object transportHealthLock = new Object();
    private volatile boolean healthRetirementClaimed;

    /** Fence a stale-health decision against an RPC start/result which raced the scan. */
    public boolean tryClaimHealthRetirement(PollHealth expected) {
        requireGenerationLock();
        synchronized (transportHealthLock) {
            if (retiring || healthRetirementClaimed || pollHealth.get() != expected) { return false; }
            healthRetirementClaimed = true;
            return true;
        }
    }

    private void noteStatusRpcStarted() {
        synchronized (transportHealthLock) {
            if (retiring || healthRetirementClaimed) { return; }
            PollHealth current = pollHealth.get();
            // Invalidate an expiration scan's identity without refreshing success age.
            pollHealth.set(new PollHealth(current.lastSuccessfulPollUs(), current.successfulPollIntervalUs(),
                    current.consecutiveTransportFailures(), current.reportedAlive()));
        }
    }

    public void configureStatusFreshness(long maximumAgeUs) {
        if (maximumAgeUs <= 0L) { throw new IllegalArgumentException("status freshness must be positive"); }
        statusFreshnessUs = maximumAgeUs;
        statusFreshnessEnforced = true;
    }

    public boolean isStatusFreshnessEnforced() { return statusFreshnessEnforced; }

    public long lastStatusAppliedUs() { return lastStatusAppliedUs.get(); }

    public boolean hasReceivedStatusRpc() { return receivedStatusRpc; }

    /** Advisory admission guard; existing request cleanup remains allowed. */
    public boolean isSchedulingStatusFresh(long nowUs) {
        long applied = lastStatusAppliedUs.get();
        PollHealth health = pollHealth.get();
        return !retiring && !healthRetirementClaimed && health.reportedAlive() && health.consecutiveTransportFailures() < 3L
                && applied > 0L && nowUs >= applied && nowUs - applied <= statusFreshnessUs;
    }

    /** One retained response per generation. Busy responses are re-polled without advancing the finished cursor. */
    public boolean tryBeginStatusApplication(PollLease lease, long receivedUs) {
        if (retiring || healthRetirementClaimed || !statusApplicationLease.compareAndSet(null, lease)) { return false; }
        statusApplicationReceivedUs = receivedUs;
        return true;
    }

    public void finishStatusApplication(PollLease lease) {
        statusApplicationLease.compareAndSet(lease, null);
    }

    /** Runs at wire completion, never takes the state/application lock or changes capacity. */
    public void recordStatusRpcResult(PollLease lease, boolean validResponse, boolean alive, Throwable failure) {
        synchronized (transportHealthLock) {
            if (statusPollLease.get() != lease || retiring || healthRetirementClaimed) { return; }
            if (failure != null) {
                pollHealth.updateAndGet(current -> new PollHealth(current.lastSuccessfulPollUs(),
                        current.successfulPollIntervalUs(),
                        current.consecutiveTransportFailures() == Long.MAX_VALUE ? Long.MAX_VALUE
                                : current.consecutiveTransportFailures() + 1L, current.reportedAlive()));
                lease.pollTrace().record(Stage.TRANSPORT_FAILURE_RECORDED);
            } else if (validResponse) {
                long nowUs = System.nanoTime() / 1000;
                pollHealth.updateAndGet(current -> new PollHealth(nowUs,
                        Math.max(0L, nowUs-current.lastSuccessfulPollUs()), 0L, alive));
                receivedStatusRpc = true;
                lease.pollTrace().record(Stage.STATUS_HEALTH_UPDATED);
            }

        }
    }

    public void recordStatusApplied(EndpointTrace.Poll trace) {
        requireGenerationLock();
        requireActiveGeneration();
        long receivedUs = statusApplicationLease.get() == null ? System.nanoTime()/1000 : statusApplicationReceivedUs;
        lastStatusAppliedUs.set(receivedUs);
        if (trace != null) { trace.record(Stage.STATUS_APPLIED); }
    }

    private final AtomicReference<PollLease> statusPollLease =
            new AtomicReference<>();
    private final AtomicReference<PollLease> cachePollLease =
            new AtomicReference<>();

    private enum PollKind {
        STATUS,
        CACHE
    }

    /**
     * Exact ownership of one generation-local poll slot. Closing is
     * token-fenced and idempotent: a delayed or duplicate close cannot release
     * a later poll which acquired the same logical slot.
     */
    public static final class PollLease implements AutoCloseable {
        private final WorkerStatus owner;
        private final PollKind kind;
        private final AtomicReference<PollLease> slot;
        private final EndpointTrace.Poll pollTrace;

        private volatile StatusPollProgress statusProgress;

        /** Called when the RPC actually starts, never while merely queued. */
        public void statusRpcStarted(long timeoutMs) {
            owner.requireStatusPollLease(this);
            statusProgress = new StatusPollProgress(
                    System.nanoTime() / 1000, timeoutMs * 1000L, false);
            owner.noteStatusRpcStarted();
        }

        /** Transport completion does not depend on the callback executor or Worker lock. */
        public void statusRpcCompleted(boolean validAliveResponse) {
            statusProgress = validAliveResponse
                    ? new StatusPollProgress(System.nanoTime() / 1000, 0L, true)
                    : null;
        }

        private PollLease(
                WorkerStatus owner,
                PollKind kind,
                AtomicReference<PollLease> slot, long syncCycleId) {
            this.owner = owner;
            this.kind = kind;
            this.slot = slot;
            this.pollTrace = owner.trace.beginPoll(kind.name(), syncCycleId);
        }

        public EndpointTrace.Poll pollTrace() {
            return pollTrace;
        }

        @Override
        public void close() {
            if (slot.compareAndSet(this, null)) {
                pollTrace.record(Stage.LEASE_RELEASED);
            }
        }
    }

    private record StatusPollProgress(long sinceUs, long rpcTimeoutUs, boolean awaitingCommit) { }

    /** Bounded transport/local-commit grace; this never refreshes scheduling capacity. */
    public boolean isStatusPollProtected(long nowUs, long commitGraceUs) {
        PollLease lease = statusPollLease.get();
        StatusPollProgress progress = lease == null ? null : lease.statusProgress;
        if (progress == null || retiring) {
            return false;
        }
        long budgetUs = progress.awaitingCommit() ? commitGraceUs : progress.rpcTimeoutUs();
        long ageUs = nowUs - progress.sinceUs();
        return ageUs >= 0L && ageUs <= budgetUs;
    }

    private WorkerStatus(
            TopologySnapshot initialTopology,
            EngineObservation initialStatus) {
        topology = new AtomicReference<>(initialTopology);
        committedStatus = new AtomicReference<>(new CommittedWorkerStatus(
                initialStatus,
                new AppliedStatusCursor(-1L, -1L)));
        long discoveredAtUs = System.nanoTime() / 1000;
        pollHealth = new AtomicReference<>(new PollHealth(
                discoveredAtUs, 0L, 0L, false));
    }

    /** Create one fully initialized service-discovery generation. */
    public static WorkerStatus createDiscovered(
            RoleType role,
            String group,
            String ip,
            int port,
            int grpcPort,
            String site) {
        Objects.requireNonNull(role, "role");
        Objects.requireNonNull(ip, "ip");
        if (port <= 0 || grpcPort <= 0) {
            throw new IllegalArgumentException(
                    "worker ports must be positive");
        }
        return new WorkerStatus(
                new TopologySnapshot(group, ip, port, grpcPort, site),
                new EngineObservation(
                        role,
                        null,
                        0L,
                        0L,
                        Map.of(),
                        0.0,
                        0L,
                        0L,
                        0L,
                        0L,
                        0L,
                        0L,
                        0L,
                        0L));
    }

    @JsonIgnore
    public long getGenerationId() {
        return generationId;
    }

    public TopologySnapshot topologySnapshot() {
        return topology.get();
    }

    public CommittedWorkerStatus committedWorkerStatus() {
        return committedStatus.get();
    }

    public EngineObservation committedEngineObservation() {
        return committedStatus.get().fields();
    }

    public AppliedStatusCursor appliedStatusCursor() {
        return committedStatus.get().cursor();
    }

    /**
     * Deep-freeze one RPC response. Finished tasks remain response-local and
     * are never copied into the committed Engine observation.
     */
    public StatusObservation freezeStatusResponse(
            WorkerStatusResponse response) {
        Objects.requireNonNull(response, "response");
        Map<String, TaskObservation> runningTasks = freezeTaskMap(
                response.getRunningTaskInfo());
        Map<String, TaskObservation> finishedTasks = freezeTaskMap(
                response.getFinishedTaskInfo());
        EngineObservation engine = new EngineObservation(
                response.getRole(),
                response.getAvailableConcurrency(),
                response.getAvailableKvCacheTokens(),
                response.getTotalKvCacheTokens(),
                runningTasks,
                response.getStepLatencyMs(),
                response.getIterateCount(),
                response.getDpSize(),
                response.getTpSize(),
                response.getDpRank(),
                response.getMaxSeqLen(),
                response.getMaxBatchTokensSize(),
                response.getRunningQueryLen(),
                response.getWaitingQueryLen());
        return new StatusObservation(
                this,
                engine,
                response.isAlive(),
                response.getStatusVersion(),
                response.getLatestFinishedVersion(),
                finishedTasks);
    }

    /** Bind an already immutable protocol conversion to this exact generation. */
    public StatusObservation bindStatusObservation(
            EngineObservation engine,
            boolean reportedAlive,
            Long statusVersion,
            Long latestFinishedVersion,
            Map<String, TaskObservation> finishedTasks) {
        return new StatusObservation(
                this,
                Objects.requireNonNull(engine, "engine"),
                reportedAlive,
                statusVersion,
                latestFinishedVersion,
                finishedTasks == null || finishedTasks.isEmpty()
                        ? Map.of() : Map.copyOf(finishedTasks));
    }

    /** Carry poll correlation through the role reducer to the actual commit boundary. */
    public PreparedStatus prepareNewStatus(StatusObservation observation, EndpointTrace.Poll pollTrace) {
        requireGenerationLock();
        requireActiveGeneration();
        if (observation.owner != this) {
            throw new IllegalArgumentException(
                    "status observation belongs to another worker generation");
        }
        CommittedWorkerStatus current = committedStatus.get();
        Long responseVersion = observation.statusVersion();
        if (responseVersion == null || responseVersion <= 0L
                || responseVersion <= current.cursor().statusVersion()) {
            throw new IllegalArgumentException(
                    "new status version must strictly advance committed version: committed="
                            + current.cursor().statusVersion()
                            + ", response=" + responseVersion);
        }
        AppliedStatusCursor cursor = new AppliedStatusCursor(
                responseVersion,
                mergedFinishedVersion(current.cursor(),
                        observation.latestFinishedVersion()));
        return new PreparedStatus(
                current,
                observation,
                new CommittedWorkerStatus(observation.engine(), cursor),
                pollTrace);
    }

    /**
     * Publish fields and both cursors at one linearization point after every
     * response reducer has succeeded.
     */
    public void publishPreparedStatus(PreparedStatus prepared) {
        requireGenerationLock();
        requireActiveGeneration();
        if (prepared.observation.owner != this) {
            throw new IllegalArgumentException(
                    "prepared status belongs to another worker generation");
        }
        Long responseVersion = prepared.observation.statusVersion();
        if (responseVersion == null || responseVersion <= 0L
                || responseVersion
                        <= prepared.baseCommitted.cursor().statusVersion()) {
            throw new IllegalStateException(
                    "invalid prepared status version: committed="
                            + prepared.baseCommitted.cursor().statusVersion()
                            + ", response=" + responseVersion);
        }
        if (!committedStatus.compareAndSet(
                prepared.baseCommitted, prepared.nextCommitted)) {
            throw new IllegalStateException(
                    "prepared status is stale or has already been published");
        }
        recordStatusApplied(null);
        if (prepared.pollTrace != null) {
            prepared.pollTrace.record(Stage.STATUS_COMMITTED);
        }
    }

    /** Record one validated status response independently from Engine state. */
    public PollHealth recordSuccessfulPoll(boolean reportedAlive, EndpointTrace.Poll pollTrace) {
        requireGenerationLock();
        requireActiveGeneration();
        long nowUs = System.nanoTime() / 1000;
        lastStatusAppliedUs.set(nowUs);
        PollHealth updated = pollHealth.updateAndGet(current -> new PollHealth(
                nowUs,
                nowUs - current.lastSuccessfulPollUs(),
                0L,
                reportedAlive));
        if (pollTrace != null) {
            pollTrace.record(Stage.STATUS_HEALTH_UPDATED);
        }
        return updated;
    }

    /** Record a transport failure without rewriting reported Engine state. */
    public PollHealth recordTransportFailure() {
        requireGenerationLock();
        requireActiveGeneration();
        return pollHealth.updateAndGet(current -> new PollHealth(
                current.lastSuccessfulPollUs(),
                current.successfulPollIntervalUs(),
                current.consecutiveTransportFailures() == Long.MAX_VALUE
                        ? Long.MAX_VALUE
                        : current.consecutiveTransportFailures() + 1L,
                current.reportedAlive()));
    }

    public void requireGenerationLock() {
        if (!lock.isHeldByCurrentThread()) {
            throw new IllegalStateException(
                    "status transaction requires generation lock");
        }
    }

    /** Reject any new publication or ownership after retirement starts. */
    public void requireActiveGeneration() {
        requireGenerationLock();
        if (retiring) {
            throw new IllegalStateException(
                    "WorkerStatus generation is retiring: " + generationId);
        }
    }

    /** Require an operation to target the one retiring generation. */
    public void requireRetiringGeneration() {
        requireGenerationLock();
        if (!retiring) {
            throw new IllegalStateException(
                    "WorkerStatus generation is not retiring: " + generationId);
        }
    }

    /**
     * Publish this generation as RETIRING after its exact endpoint admission
     * gate has closed. The status-map identity remains until endpoint drain
     * succeeds.
     *
     * @return {@code true} only for the caller which starts retirement
     */
    public boolean beginRetirementAfterEndpointGateClosed() {
        requireGenerationLock();
        if (retiring) {
            return false;
        }
        retiring = true;
        trace.record(Stage.GENERATION_RETIRING);
        return true;
    }

    @JsonIgnore
    public boolean isActiveGeneration() {
        return !retiring;
    }

    private static long mergedFinishedVersion(
            AppliedStatusCursor current,
            Long responseFinishedVersion) {
        return responseFinishedVersion == null
                ? current.latestFinishedTaskVersion()
                : Math.max(current.latestFinishedTaskVersion(),
                        responseFinishedVersion);
    }

    /** Atomically refresh discovery-owned placement labels for this generation. */
    public void updateDiscoveryLabels(String site, String group) {
        requireGenerationLock();
        requireActiveGeneration();
        topology.updateAndGet(current -> new TopologySnapshot(
                group, current.ip(), current.port(), current.grpcPort(), site));
    }

    public RoleType getRole() {
        return committedStatus.get().fields().role();
    }

    public String getGroup() {
        return topology.get().group();
    }

    public String getIp() {
        return topology.get().ip();
    }

    public int getPort() {
        return topology.get().port();
    }

    public int getGrpcPort() {
        return topology.get().grpcPort();
    }

    public String getSite() {
        return topology.get().site();
    }

    /**
     * Compatibility-only mirror of WorkerStatusPB.available_concurrency.
     * LocalRpcServer currently leaves that protobuf field unset, so this value
     * must not participate in routing, admission control, or batch sizing.
     */
    public Long getAvailableConcurrency() {
        return committedStatus.get().fields().availableConcurrency();
    }

    public long getAvailableKvCacheTokens() {
        return committedStatus.get().fields().availableKvCacheTokens();
    }

    public long getTotalKvCacheTokens() {
        return committedStatus.get().fields().totalKvCacheTokens();
    }

    public CacheStatus getCacheStatus() {
        return cacheIndexSnapshot.get().cacheStatus();
    }

    public CacheIndexSnapshot cacheIndexSnapshot() {
        return cacheIndexSnapshot.get();
    }

    public void publishCacheStatus(CacheStatus cacheStatus) {
        requireCacheMutationLock();
        CacheStatus next = Objects.requireNonNull(cacheStatus, "cacheStatus");
        cacheIndexSnapshot.updateAndGet(current -> new CacheIndexSnapshot(
                next, current.indexInitialized(), current.indexedVersion()));
    }

    public long getCacheIndexedVersion() {
        return cacheIndexSnapshot.get().indexedVersion();
    }

    public void publishCacheIndexedVersion(long version) {
        requireCacheMutationLock();
        CacheIndexSnapshot current = cacheIndexSnapshot.get();
        if (current.cacheStatus() == null
                || current.cacheStatus().getVersion() != version) {
            throw new IllegalStateException(
                    "indexed cache version must match current cache status");
        }
        cacheIndexSnapshot.set(new CacheIndexSnapshot(
                current.cacheStatus(), true, version));
    }

    public long recordSuccessfulCachePoll() {
        long nowUs = System.nanoTime() / 1000;
        long previousUs = cacheLastUpdateTime.getAndSet(nowUs);
        return previousUs <= 0L ? 0L : Math.max(0L, nowUs - previousUs);
    }

    public double getStepLatencyMs() {
        return committedStatus.get().fields().stepLatencyMs();
    }

    public long getIterateCount() {
        return committedStatus.get().fields().iterateCount();
    }

    public long getDpSize() {
        return committedStatus.get().fields().dpSize();
    }

    public long getTpSize() {
        return committedStatus.get().fields().tpSize();
    }

    public long getDpRank() {
        return committedStatus.get().fields().dpRank();
    }

    /** Model-level maximum sequence length reported by the Engine. */
    public long getMaxSeqLen() {
        return committedStatus.get().fields().maxSeqLen();
    }

    /** Strict aggregate context-token limit for an Engine batch/group. */
    public long getMaxBatchTokensSize() {
        return committedStatus.get().fields().maxBatchTokensSize();
    }

    public PollHealth pollHealth() {
        return pollHealth.get();
    }

    /** Acquire the one in-flight status-poll slot for this generation. */
    public PollLease tryBeginStatusPoll(long syncCycleId) {
        if (retiring || healthRetirementClaimed) { return null; }
        PollLease active = statusPollLease.get();
        if (active != null) { active.pollTrace.skippedBusy(); return null; }
        PollLease lease = new PollLease(this, PollKind.STATUS, statusPollLease, syncCycleId);
        if (!statusPollLease.compareAndSet(null, lease)) { return null; }
        if (retiring || healthRetirementClaimed) { lease.close(); return null; }
        return lease;
    }

    public void requireStatusPollLease(PollLease lease) {
        requirePollLease(lease, PollKind.STATUS, statusPollLease);
    }

    /** Acquire the independent cache-poll slot for this generation. */
    public PollLease tryBeginCachePoll(long syncCycleId) {
        return tryBeginPoll(PollKind.CACHE, cachePollLease, syncCycleId);
    }

    public void requireCachePollLease(PollLease lease) {
        requirePollLease(lease, PollKind.CACHE, cachePollLease);
    }

    private PollLease tryBeginPoll(PollKind kind, AtomicReference<PollLease> slot,
                                   long syncCycleId) {
        if (retiring) { return null; }
        PollLease active = slot.get();
        if (active != null) { active.pollTrace.skippedBusy(); return null; }
        PollLease lease = new PollLease(this, kind, slot, syncCycleId);
        if (!slot.compareAndSet(null, lease)) { return null; }
        if (retiring) { lease.close(); return null; }
        return lease;
    }

    private void requireCacheMutationLock() {
        if (!cacheMutationLock.isHeldByCurrentThread()) {
            throw new IllegalStateException("cache publication requires cache mutation lock");
        }
    }

    private void requirePollLease(
            PollLease lease,
            PollKind expectedKind,
            AtomicReference<PollLease> slot) {
        if (lease == null
                || lease.owner != this
                || lease.kind != expectedKind
                || slot.get() != lease) {
            throw new IllegalArgumentException(
                    "poll lease is not the active "
                            + expectedKind.name().toLowerCase()
                            + " lease for this worker generation");
        }
    }

    /** Get the HTTP IP:PORT address. */
    public String getIpPort() {
        TopologySnapshot current = topology.get();
        if (current.ip() == null) {
            return null;
        }
        return current.ip() + ":" + current.port();
    }

    private static Map<String, TaskObservation> freezeTaskMap(
            Map<String, TaskInfo> tasks) {
        if (tasks == null || tasks.isEmpty()) {
            return Map.of();
        }
        Map<String, TaskObservation> frozen = new HashMap<>(tasks.size());
        for (Map.Entry<String, TaskInfo> entry : tasks.entrySet()) {
            if (entry.getKey() != null && entry.getValue() != null) {
                frozen.put(entry.getKey(), TaskObservation.copyOf(
                        entry.getValue()));
            }
        }
        return Map.copyOf(frozen);
    }

    @Override
    public String toString() {
        return "WorkerStatus{" +
                "generationId=" + generationId +
                ", retiring=" + retiring +
                ", topology=" + topology.get() +
                ", committedStatus=" + committedStatus.get() +
                ", pollHealth=" + pollHealth.get() +
                '}';
    }
}
