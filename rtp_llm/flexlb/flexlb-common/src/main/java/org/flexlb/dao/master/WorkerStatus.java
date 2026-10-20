package org.flexlb.dao.master;

import com.fasterxml.jackson.annotation.JsonIgnore;
import org.flexlb.dao.master.EndpointTrace.Stage;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.KvCacheGroupMode;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;

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

    public record TopologySnapshot(
            String group,
            String ip,
            int port,
            int grpcPort,
            String site,
            String deploymentName,
            int engineIndex,
            int multiEngineNum) {

        public TopologySnapshot(
                String group,
                String ip,
                int port,
                int grpcPort,
                String site) {
            this(group, ip, port, grpcPort, site, null, 0, 1);
        }
    }

    /** Deeply immutable copy of the task fields reported by one status RPC. */
    public record TaskObservation(String requestId,
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
                                  PriorityPreemptionProgress priorityPreemptionProgress,
                                  TaskTelemetry telemetry) {

        /** A pending observation that reports only priority cancellation progress. */
        @JsonIgnore
        public boolean isPriorityCancelOverlayOnly() {
            return (priorityPreemptionProgress == PriorityPreemptionProgress.CANCELING
                    || priorityPreemptionProgress == PriorityPreemptionProgress.CANCELED)
                    && phase == TaskPhase.PENDING;
        }
    }

    /**
     * Immutable request telemetry retained from the Engine status RPC.
     * Nullable prefill progress values preserve omitted protobuf fields.
     */
    public record TaskTelemetry(boolean prefixLengthValid,
                                long requestReceivedTimeMs,
                                long inputQueueEnqueueTimeMs,
                                long inputQueueDrainTimeMs,
                                long waitingEnteredTimeMs,
                                long runningEnteredTimeMs,
                                long remoteKvWaitMs,
                                long firstTokenTimeMs,
                                long hbmLocalMatchTokens,
                                long remoteKvAddedMatchTokens,
                                long firstPrefillStepId,
                                long lastPrefillStepId,
                                long prefillStepCount,
                                long prefillNonfinalChunkTokensMin,
                                long prefillNonfinalChunkTokensMax,
                                Long completedPrefillTokens,
                                Long remainingPrefillTokens,
                                Long lastCompletedPrefillStepId) {

        public TaskTelemetry(boolean prefixLengthValid,
                             long requestReceivedTimeMs,
                             long inputQueueEnqueueTimeMs,
                             long inputQueueDrainTimeMs,
                             long waitingEnteredTimeMs,
                             long runningEnteredTimeMs,
                             long remoteKvWaitMs,
                             long firstTokenTimeMs,
                             long hbmLocalMatchTokens,
                             long remoteKvAddedMatchTokens,
                             long firstPrefillStepId,
                             long lastPrefillStepId,
                             long prefillStepCount,
                             long prefillNonfinalChunkTokensMin,
                             long prefillNonfinalChunkTokensMax) {
            this(prefixLengthValid, requestReceivedTimeMs, inputQueueEnqueueTimeMs, inputQueueDrainTimeMs,
                    waitingEnteredTimeMs, runningEnteredTimeMs, remoteKvWaitMs, firstTokenTimeMs,
                    hbmLocalMatchTokens, remoteKvAddedMatchTokens, firstPrefillStepId, lastPrefillStepId,
                    prefillStepCount, prefillNonfinalChunkTokensMin, prefillNonfinalChunkTokensMax,
                    null, null, null);
        }
    }

    public record StepMetrics(long stepId, long completedTimeMs, long totalScheduledTokens,
                              long prefillRequestCount, long prefillTokens, long tokenBudget, double budgetFillRatio) {
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
            long blockSize,
            int blockHashLookaheadTokens,
            int cacheMatchRollbackBlocks,
            KvCacheGroupMode kvCacheGroupMode,
            long maxSeqLen,
            long maxBatchTokensSize,
            long runningQueryLen,
            long waitingQueryLen,
            StepMetrics lastStepMetrics) {

        public EngineObservation {
            Objects.requireNonNull(role, "role");
            kvCacheGroupMode = kvCacheGroupMode == null
                    ? KvCacheGroupMode.UNSPECIFIED : kvCacheGroupMode;
            runningTaskList = Map.copyOf(runningTaskList);
        }

        public EngineObservation(RoleType role,
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
            this(role,
                    availableConcurrency,
                    availableKvCacheTokens,
                    totalKvCacheTokens,
                    runningTaskList,
                    stepLatencyMs,
                    iterateCount,
                    dpSize,
                    tpSize,
                    dpRank,
                    0L,
                    0,
                    0,
                    KvCacheGroupMode.UNSPECIFIED,
                    maxSeqLen,
                    maxBatchTokensSize,
                    runningQueryLen,
                    waitingQueryLen,
                    null);
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
     * Status-poll transport health, updated independently of local status application.
     * Successful timestamps are used for diagnostics, never worker expiration.
     * A zero timestamp means no RPC has succeeded in this generation.
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

        /**
         * Returns all active task phases reported in running_task_info.
         */
        public Map<String, TaskObservation> activeTasks() {
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

    private final WorkerIdentity workerIdentity;

    private final AtomicReference<CommittedWorkerStatus> committedStatus;

    /** Cache polling is independent from worker-status polling. */
    private final AtomicReference<CacheIndexSnapshot> cacheIndexSnapshot =
            new AtomicReference<>(new CacheIndexSnapshot(null, false, -1L));
    private final AtomicLong cacheLastUpdateTime = new AtomicLong(-1L);

    private static final long MAX_CONSECUTIVE_STATUS_FAILURES = 3L;
    private final WorkerHealth workerHealth = new WorkerHealth();

    /** Only short state transitions use this monitor; never hold it while waiting for Worker locks. */
    private final class WorkerHealth {
        private volatile PollHealth health =
                new PollHealth(0L, 0L, 0L, false);
        private volatile RetirementReason unavailableReason;
        private volatile boolean retiring;

        private synchronized boolean beginRetirement() {
            if (retiring) { return false; }
            retiring = true;
            return true;
        }

        private synchronized Stage rpcResult(PollLease lease, boolean valid, boolean alive, Throwable failure) {
            if (statusPollLease.get() != lease || retiring || unavailableReason != null) { return null; }
            if (failure != null) {
                long failures = health.consecutiveTransportFailures() + 1L;
                health = new PollHealth(health.lastSuccessfulPollUs(), health.successfulPollIntervalUs(),
                        failures, health.reportedAlive());
                if (failures >= MAX_CONSECUTIVE_STATUS_FAILURES) {
                    unavailableReason = RetirementReason.CONSECUTIVE_STATUS_FAILURES;
                }
                return Stage.TRANSPORT_FAILURE_RECORDED;
            }
            long nowUs = System.nanoTime() / 1000;
            boolean reportedAlive = valid ? alive : health.reportedAlive();
            long intervalUs = health.lastSuccessfulPollUs() == 0L
                    ? 0L : Math.max(0L, nowUs - health.lastSuccessfulPollUs());
            health = new PollHealth(nowUs, intervalUs, 0L, reportedAlive);
            if (valid && !alive) {
                unavailableReason = RetirementReason.WORKER_REPORTED_NOT_ALIVE;
            }
            return Stage.STATUS_HEALTH_UPDATED;
        }
    }

    public enum RetirementReason {
        CONSECUTIVE_STATUS_FAILURES,
        WORKER_REPORTED_NOT_ALIVE,
        DISCOVERY_REMOVED
    }

    /** The first confirmed reason remains latched until this generation is removed. */
    public RetirementReason retirementReason() {
        return workerHealth.unavailableReason;
    }

    /** Only call after a successful discovery query confirms that this address is absent. */
    public void markRemovedFromDiscovery() {
        synchronized (workerHealth) {
            if (!workerHealth.retiring && workerHealth.unavailableReason == null) {
                workerHealth.unavailableReason = RetirementReason.DISCOVERY_REMOVED;
            }
        }
    }

    /** Keep using committed capacity while newer status is waiting for local application. */
    public boolean isAvailableForScheduling() {
        return !workerHealth.retiring && workerHealth.unavailableReason == null
                && workerHealth.health.reportedAlive();
    }

    /** Record transport outcome without taking the Worker lock or waiting for local application. */
    public void recordStatusRpcResult(PollLease lease, boolean validResponse, boolean alive, Throwable failure) {
        Stage stage = workerHealth.rpcResult(lease, validResponse, alive, failure);
        if (stage != null) { lease.pollTrace().record(stage); }
    }

    private final AtomicReference<PollLease> statusPollLease =
            new AtomicReference<>();
    private final AtomicReference<PollLease> statusApplication = new AtomicReference<>();
    private final AtomicReference<PollLease> cachePollLease =
            new AtomicReference<>();

    private enum PollKind {
        STATUS,
        CACHE
    }

    /**
     * Owns one generation-local RPC slot (cache polls also include application). Closing is
     * token-fenced and idempotent: a delayed or duplicate close cannot release
     * a later poll which acquired the same logical slot.
     */
    public static final class PollLease implements AutoCloseable {
        private final WorkerStatus owner;
        private final PollKind kind;
        private final AtomicReference<PollLease> slot;
        private final EndpointTrace.Poll pollTrace;

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

    private WorkerStatus(
            TopologySnapshot initialTopology,
            EngineObservation initialStatus) {
        topology = new AtomicReference<>(initialTopology);
        workerIdentity = new WorkerIdentity(
                initialTopology.ip(), initialTopology.port(), initialTopology.engineIndex());
        committedStatus = new AtomicReference<>(new CommittedWorkerStatus(
                initialStatus,
                new AppliedStatusCursor(-1L, -1L)));
    }

    /** Create one fully initialized service-discovery generation. */
    public static WorkerStatus createDiscovered(
            RoleType role,
            String group,
            String ip,
            int port,
            int grpcPort,
            String site) {
        return createDiscovered(
                role, group, ip, port, grpcPort, site, null);
    }

    public static WorkerStatus createDiscovered(
            RoleType role,
            String group,
            String ip,
            int port,
            int grpcPort,
            String site,
            String deploymentName) {
        return createDiscovered(
                role, group, ip, port, grpcPort, site, deploymentName, 0, 1);
    }

    public static WorkerStatus createDiscovered(
            RoleType role,
            String group,
            String ip,
            int port,
            int grpcPort,
            String site,
            String deploymentName,
            int engineIndex,
            int multiEngineNum) {
        Objects.requireNonNull(role, "role");
        Objects.requireNonNull(ip, "ip");
        if (port <= 0 || grpcPort <= 0) {
            throw new IllegalArgumentException(
                    "worker ports must be positive");
        }
        if (engineIndex < 0 || multiEngineNum <= 0 || engineIndex >= multiEngineNum) {
            throw new IllegalArgumentException("invalid logical worker index");
        }
        return new WorkerStatus(
                new TopologySnapshot(
                        group, ip, port, grpcPort, site, deploymentName,
                        engineIndex, multiEngineNum),
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
    public PreparedStatus prepareNewStatus(StatusObservation observation, PollLease lease) {
        requireActiveGeneration();
        if (lease == null || statusApplication.get() != lease) {
            throw new IllegalArgumentException("status application is not owned by this poll");
        }
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
                lease.pollTrace);
    }

    /**
     * Publish fields and both cursors at one linearization point after every
     * response reducer has succeeded.
     */
    public void publishPreparedStatus(PreparedStatus prepared) {
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
        prepared.pollTrace.record(Stage.STATUS_COMMITTED);
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
        if (!isActiveGeneration()) {
            throw new IllegalStateException(
                    "WorkerStatus generation is retiring: " + generationId);
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
        if (!workerHealth.beginRetirement()) { return false; }
        trace.record(Stage.GENERATION_RETIRING);
        return true;
    }

    @JsonIgnore
    public boolean isActiveGeneration() {
        return !workerHealth.retiring;
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
        updateDiscoveryLabels(site, group, topology.get().deploymentName());
    }

    public void updateDiscoveryLabels(
            String site, String group, String deploymentName) {
        requireGenerationLock();
        requireActiveGeneration();
        topology.updateAndGet(current -> new TopologySnapshot(
                group,
                current.ip(),
                current.port(),
                current.grpcPort(),
                site,
                deploymentName,
                current.engineIndex(),
                current.multiEngineNum()));
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

    public String getDeploymentName() {
        return topology.get().deploymentName();
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

    public int getBlockHashLookaheadTokens() {
        return committedStatus.get().fields().blockHashLookaheadTokens();
    }

    public int getCacheMatchRollbackBlocks() {
        return committedStatus.get().fields().cacheMatchRollbackBlocks();
    }

    public KvCacheGroupMode getKvCacheGroupMode() {
        return committedStatus.get().fields().kvCacheGroupMode();
    }

    /** Model-level maximum sequence length reported by the Engine. */
    public long getMaxSeqLen() {
        return committedStatus.get().fields().maxSeqLen();
    }

    /** Strict aggregate context-token limit for an Engine batch/group. */
    public long getMaxBatchTokensSize() {
        return committedStatus.get().fields().maxBatchTokensSize();
    }

    public boolean isAlive() { return pollHealth().reportedAlive(); }

    public PollHealth pollHealth() {
        return workerHealth.health;
    }

    /** Reserve at most one response for application before releasing its RPC slot. */
    public boolean tryBeginStatusApplication(PollLease lease) {
        synchronized (workerHealth) {
            requireStatusPollLease(lease);
            return !workerHealth.retiring && workerHealth.unavailableReason == null
                    && statusApplication.compareAndSet(null, lease);
        }
    }

    public void finishStatusApplication(PollLease lease) {
        statusApplication.compareAndSet(lease, null);
    }

    /** Acquire the status RPC slot; local application has a separate owner. */
    public PollLease tryBeginStatusPoll(long syncCycleId) {
        return tryBeginPoll(PollKind.STATUS, statusPollLease, syncCycleId);
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
        synchronized (workerHealth) {
            if (workerHealth.retiring || workerHealth.unavailableReason != null) { return null; }
            PollLease active = slot.get();
            if (active != null) { active.pollTrace.skippedBusy(); return null; }
            // Only the admitted poll may publish a new diagnostic snapshot.
            PollLease lease = new PollLease(this, kind, slot, syncCycleId);
            slot.set(lease);
            return lease;
        }
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
        return workerIdentity.getPhysicalIpPort();
    }

    public WorkerIdentity getWorkerIdentity() {
        return workerIdentity;
    }

    public String getPhysicalIpPort() {
        return workerIdentity.getPhysicalIpPort();
    }

    public String getLogicalIpPort() {
        return workerIdentity.getLogicalIpPort();
    }

    @JsonIgnore
    public String getMetricIpPort() {
        return workerIdentity.getMetricIpPort(getMultiEngineNum());
    }

    public int getEngineIndex() {
        return workerIdentity.getEngineIndex();
    }

    public int getMultiEngineNum() {
        return topology.get().multiEngineNum();
    }

    @Override
    public String toString() {
        return "WorkerStatus{" +
                "generationId=" + generationId +
                ", retiring=" + !isActiveGeneration() +
                ", topology=" + topology.get() +
                ", committedStatus=" + committedStatus.get() +
                ", pollHealth=" + pollHealth() +
                '}';
    }
}
