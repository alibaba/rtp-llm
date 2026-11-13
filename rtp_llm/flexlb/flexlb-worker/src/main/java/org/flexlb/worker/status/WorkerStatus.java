package org.flexlb.worker.status;

import com.fasterxml.jackson.annotation.JsonIgnore;
import org.flexlb.config.service.RoleType;
import org.flexlb.util.discovery.DiscoveredHost;
import org.flexlb.worker.WorkerIdentity;
import org.flexlb.worker.cache.CacheStatus;
import org.flexlb.worker.cache.KvCacheGroupMode;

import java.util.Map;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

/**
 * One service-discovery generation of an Engine worker.
 *
 * <p>The object identity is the generation identity. Its independently updated
 * concurrency domains are published explicitly:
 *
 * <ul>
 *   <li>{@link TopologySnapshot}: discovery-owned address and placement data;</li>
 *   <li>{@link CommittedWorkerStatus}: stable Engine fields and both cursors
 *       from one successfully reduced status response;</li>
 *   <li>{@link PollHealth}: health reported by status polling, independent
 *       from local status application;</li>
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

    private static final long MAX_CONSECUTIVE_WORKER_FAILURES = 3L;

    /**
     * Serializes lifecycle transitions for this generation.
     */
    public final transient ReentrantLock lock = new ReentrantLock();

    private final long generationId = NEXT_GENERATION_ID.incrementAndGet();

    /**
     * Set under {@link #lock}; read lock-free by poll gates.
     */
    private volatile boolean updatesStopped;

    private final Object healthGate = new Object();
    private final EndpointTrace trace = new EndpointTrace();

    private volatile CloseReason closeReason;

    /**
     * Serializes cache-index writes and exact-generation cleanup.
     */
    public final transient ReentrantLock cacheMutationLock = new ReentrantLock();

    private final AtomicReference<TopologySnapshot> topology;
    private final WorkerIdentity workerIdentity;
    private final AtomicReference<CommittedWorkerStatus> committedStatus;
    private final AtomicReference<PollHealth> pollHealth;

    /**
     * Cache polling is independent from worker-status polling.
     */
    private final AtomicReference<CacheIndexSnapshot> cacheIndexSnapshot =
            new AtomicReference<>(new CacheIndexSnapshot(null, false, -1L));

    private final AtomicLong cacheLastUpdateTime = new AtomicLong(-1L);
    private final AtomicLong lastCachePollCycle = new AtomicLong(Long.MIN_VALUE);
    private final AtomicReference<PollLease> statusPollLease = new AtomicReference<>();
    private final AtomicReference<PollLease> statusApplication = new AtomicReference<>();
    private final AtomicReference<PollLease> cachePollLease = new AtomicReference<>();

    private WorkerStatus(TopologySnapshot initialTopology, EngineObservation initialStatus) {
        topology = new AtomicReference<>(initialTopology);
        workerIdentity =
                new WorkerIdentity(
                        initialTopology.ip(),
                        initialTopology.port(),
                        initialTopology.engineIndex());
        committedStatus =
                new AtomicReference<>(
                        new CommittedWorkerStatus(
                                initialStatus, new AppliedStatusCursor(-1L, -1L)));
        long discoveredAtUs = TimeUnit.NANOSECONDS.toMicros(System.nanoTime());
        pollHealth = new AtomicReference<>(new PollHealth(discoveredAtUs, 0L, 0L, false));
    }

    @JsonIgnore
    public EndpointTrace trace() {
        return trace;
    }

    public EndpointTrace.Snapshot getSyncTrace() {
        return trace.snapshot();
    }

    public CloseReason closeReason() {
        return closeReason;
    }

    public void markRemovedFromDiscovery() {
        synchronized (healthGate) {
            if (!updatesStopped && closeReason == null) {
                closeReason = CloseReason.DISCOVERY_REMOVED;
                trace.recordUnavailable(
                        closeReason.name(),
                        pollHealth.get().lastSuccessfulPollUs(),
                        pollHealth.get().consecutiveWorkerFailures());
            }
        }
    }

    public boolean isAvailableForScheduling() {
        return !updatesStopped && closeReason == null;
    }

    /**
     * Create one fully initialized service-discovery generation.
     */
    public static WorkerStatus createDiscovered(
            RoleType role, String group, String ip, int port, int grpcPort, String site) {
        return createDiscovered(role, group, ip, port, grpcPort, site, null);
    }

    public static WorkerStatus createDiscovered(
            RoleType role,
            String group,
            String ip,
            int port,
            int grpcPort,
            String site,
            String deploymentName) {
        return createDiscovered(role, group, ip, port, grpcPort, site, deploymentName, 0, 1);
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
        Objects.requireNonNull(ip, "ip");
        if (port <= 0 || grpcPort <= 0) {
            throw new IllegalArgumentException("worker ports must be positive");
        }
        if (engineIndex < 0 || multiEngineNum <= 0 || engineIndex >= multiEngineNum) {
            throw new IllegalArgumentException("invalid logical worker index");
        }
        return new WorkerStatus(
                new TopologySnapshot(
                        group,
                        ip,
                        port,
                        grpcPort,
                        site,
                        deploymentName,
                        engineIndex,
                        multiEngineNum),
                new EngineObservation(
                        role, null, 0L, 0L, Map.of(), 0.0, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L));
    }

    public static WorkerStatus createDiscovered(RoleType role, DiscoveredHost host) {
        return new WorkerStatus(
                new TopologySnapshot(
                        host.getGroup(),
                        host.getIp(),
                        host.getHttpPort(),
                        host.getGrpcPort(),
                        host.getSite(),
                        host.getDeploymentName(),
                        host.getEngineIndex(),
                        host.getMultiEngineNum(),
                        host.getWorkerStatusPort()),
                new EngineObservation(
                        role, null, 0L, 0L, Map.of(), 0.0, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L));
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
     * Bind an already immutable protocol conversion to this exact generation.
     */
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
                        ? Map.of()
                        : Map.copyOf(finishedTasks));
    }

    /**
     * Binds a frozen observation to the current commit holder without allowing version regression.
     */
    public PreparedStatus prepareStatus(StatusObservation observation) {
        requireActiveGeneration();
        checkArgument(
                observation.owner == this,
                "status observation belongs to another worker generation");
        CommittedWorkerStatus current = committedStatus.get();
        Long responseVersion = observation.statusVersion();
        if (responseVersion == null
                || responseVersion <= 0L
                || responseVersion < current.cursor().statusVersion()) {
            throw new IllegalArgumentException(
                    "status version must not regress: committed="
                            + current.cursor().statusVersion()
                            + ", response="
                            + responseVersion);
        }
        AppliedStatusCursor cursor =
                new AppliedStatusCursor(
                        responseVersion,
                        mergedFinishedVersion(
                                current.cursor(), observation.latestFinishedVersion()));
        return new PreparedStatus(
                current, observation, new CommittedWorkerStatus(observation.engine(), cursor));
    }

    /**
     * Reject any new publication or ownership after cleanup starts.
     */
    public void requireActiveGeneration() {
        requireGenerationLock();
        checkState(
                !updatesStopped, "WorkerStatus generation updates are stopped: %s", generationId);
    }

    public void requireGenerationLock() {
        checkState(lock.isHeldByCurrentThread(), "status transaction requires generation lock");
    }

    /**
     * Publish fields and both cursors at one linearization point after every
     * response reducer has succeeded.
     */
    public void publishPreparedStatus(PreparedStatus prepared) {
        requireGenerationLock();
        publishReducedStatus(prepared);
    }

    /**
     * Commit a prepared observation after its DiscoveryEndpoint has reduced the request records.
     * The DiscoveryEndpoint serializes this commit with cleanup; the source cursor CAS
     * rejects a prepared update whose base has already been replaced.
     */
    public void publishReducedStatus(PreparedStatus prepared) {
        checkState(
                !updatesStopped, "WorkerStatus generation updates are stopped: %s", generationId);
        checkArgument(
                prepared.observation.owner == this,
                "prepared status belongs to another worker generation");
        Long responseVersion = prepared.observation.statusVersion();
        if (responseVersion == null
                || responseVersion <= 0L
                || responseVersion < prepared.baseCommitted.cursor().statusVersion()) {
            throw new IllegalStateException(
                    "invalid prepared status version: committed="
                            + prepared.baseCommitted.cursor().statusVersion()
                            + ", response="
                            + responseVersion);
        }
        checkState(
                committedStatus.compareAndSet(prepared.baseCommitted, prepared.nextCommitted),
                "prepared status is stale or has already been published");
    }

    /**
     * Record a validated response for a status transaction.
     */
    public PollHealth recordSuccessfulPoll(boolean reportedAlive) {
        requireActiveGeneration();
        synchronized (healthGate) {
            return updatePollHealth(reportedAlive);
        }
    }

    /**
     * The RPC owner records health before any local conversion or application.
     */
    public void recordStatusRpcResult(PollLease lease, boolean valid, boolean alive) {
        synchronized (healthGate) {
            if (lease == null
                    || statusPollLease.get() != lease
                    || lease.resultRecorded
                    || updatesStopped
                    || closeReason != null) {
                return;
            }
            lease.resultRecorded = true;
            if (valid) {
                updatePollHealth(alive);
                lease.trace.record(EndpointTrace.Stage.STATUS_HEALTH_UPDATED);
            }
        }
    }

    /**
     * One local application may outlive its RPC; later RPCs still update health.
     */
    public PollLease tryBeginStatusApplication(PollLease rpcLease) {
        synchronized (healthGate) {
            requireStatusPollLease(rpcLease);
            if (updatesStopped) {
                return null;
            }
            PollLease lease =
                    new PollLease(this, PollKind.APPLICATION, statusApplication, rpcLease.trace);
            return statusApplication.compareAndSet(null, lease) ? lease : null;
        }
    }

    public void requireStatusPollLease(PollLease lease) {
        requirePollLease(lease, PollKind.STATUS, statusPollLease);
    }

    /**
     * Publish this generation as RETIRING after its exact endpoint admission
     * gate has closed. The status-map identity remains until endpoint drain
     * succeeds.
     *
     * @return {@code true} only for the caller which starts cleanup
     */
    public boolean stopUpdatesAfterAdmissionClosed() {
        requireGenerationLock();
        synchronized (healthGate) {
            if (updatesStopped) {
                return false;
            }
            updatesStopped = true;
            trace.record(EndpointTrace.Stage.GENERATION_RETIRING);
            return true;
        }
    }

    @JsonIgnore
    public boolean isActiveGeneration() {
        return !updatesStopped;
    }

    /**
     * Atomically refresh discovery-owned placement labels for this generation.
     */
    public void updateDiscoveryLabels(String site, String group) {
        updateDiscoveryLabels(site, group, topology.get().deploymentName());
    }

    public void updateDiscoveryLabels(String site, String group, String deploymentName) {
        requireActiveGeneration();
        topology.updateAndGet(
                current ->
                        new TopologySnapshot(
                                group,
                                current.ip(),
                                current.port(),
                                current.grpcPort(),
                                site,
                                deploymentName,
                                current.engineIndex(),
                                current.multiEngineNum(),
                                current.workerStatusPort()));
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

    public int getWorkerStatusPort() {
        return topology.get().workerStatusPort();
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

    public CacheStatus getCacheStatus() {
        return cacheIndexSnapshot.get().cacheStatus();
    }

    public CacheIndexSnapshot cacheIndexSnapshot() {
        return cacheIndexSnapshot.get();
    }

    public void publishCacheStatus(CacheStatus cacheStatus) {
        requireCacheMutationLock();
        CacheStatus next = Objects.requireNonNull(cacheStatus, "cacheStatus");
        cacheIndexSnapshot.updateAndGet(
                current ->
                        new CacheIndexSnapshot(
                                next, current.indexInitialized(), current.indexedVersion()));
    }

    public void requireCacheMutationLock() {
        checkState(cacheMutationLock.isHeldByCurrentThread(), "cache mutation requires cache lock");
    }

    public void publishCacheIndexedVersion(long version) {
        requireCacheMutationLock();
        CacheIndexSnapshot current = cacheIndexSnapshot.get();
        checkState(
                current.cacheStatus() != null && current.cacheStatus().getVersion() == version,
                "indexed cache version must match current cache status");
        cacheIndexSnapshot.set(new CacheIndexSnapshot(current.cacheStatus(), true, version));
    }

    /**
     * Preserve cache cadence when a coalesced discovery cycle crosses a due tick.
     */
    public boolean shouldPollCache(long cycleId, long intervalCycles) {
        checkArgument(intervalCycles > 0L, "cache poll interval must be positive");
        long previous = lastCachePollCycle.getAndAccumulate(cycleId, Math::max);
        return cycleId >= previous
                && (cycleId % intervalCycles == 0L
                        || previous != Long.MIN_VALUE
                                && cycleId / intervalCycles > previous / intervalCycles);
    }

    public long recordSuccessfulCachePoll() {
        long nowUs = TimeUnit.NANOSECONDS.toMicros(System.nanoTime());
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

    /**
     * Model-level maximum sequence length reported by the Engine.
     */
    public long getMaxSeqLen() {
        return committedStatus.get().fields().maxSeqLen();
    }

    /**
     * Strict aggregate context-token limit for an Engine batch/group.
     */
    public long getMaxBatchTokensSize() {
        return committedStatus.get().fields().maxBatchTokensSize();
    }

    public PollHealth pollHealth() {
        return pollHealth.get();
    }

    /**
     * Acquire the one in-flight status-poll slot for this generation.
     */
    public PollLease tryBeginStatusPoll() {
        return tryBeginStatusPoll(0L);
    }

    public PollLease tryBeginStatusPoll(long cycle) {
        synchronized (healthGate) {
            if (updatesStopped || closeReason != null) {
                return null;
            }
            PollLease current = statusPollLease.get();
            if (current != null) {
                current.trace.skippedBusy();
                return null;
            }
            PollLease lease =
                    new PollLease(
                            this,
                            PollKind.STATUS,
                            statusPollLease,
                            trace.beginPoll("STATUS", cycle));
            statusPollLease.set(lease);
            return lease;
        }
    }

    /**
     * Acquire the independent cache-poll slot for this generation.
     */
    public PollLease tryBeginCachePoll() {
        return tryBeginCachePoll(0L);
    }

    public PollLease tryBeginCachePoll(long cycle) {
        synchronized (healthGate) {
            if (updatesStopped || closeReason != null) {
                return null;
            }
            PollLease current = cachePollLease.get();
            if (current != null) {
                current.trace.skippedBusy();
                return null;
            }
            PollLease lease =
                    new PollLease(
                            this, PollKind.CACHE, cachePollLease, trace.beginPoll("CACHE", cycle));
            cachePollLease.set(lease);
            return lease;
        }
    }

    public void requireCachePollLease(PollLease lease) {
        requirePollLease(lease, PollKind.CACHE, cachePollLease);
    }

    /**
     * Get the HTTP IP:PORT address.
     */
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

    public int getMultiEngineNum() {
        return topology.get().multiEngineNum();
    }

    public int getEngineIndex() {
        return workerIdentity.getEngineIndex();
    }

    public String getDeploymentName() {
        return topology.get().deploymentName();
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

    public boolean isAlive() {
        return pollHealth.get().reportedAlive();
    }

    @Override
    public String toString() {
        return "WorkerStatus{"
                + "generationId="
                + generationId
                + ", updatesStopped="
                + updatesStopped
                + ", topology="
                + topology.get()
                + ", committedStatus="
                + committedStatus.get()
                + ", pollHealth="
                + pollHealth.get()
                + '}';
    }

    private PollHealth updatePollHealth(boolean alive) {
        PollHealth previous = pollHealth.get();
        long failures =
                alive
                        ? 0L
                        : Math.min(
                                MAX_CONSECUTIVE_WORKER_FAILURES,
                                previous.consecutiveWorkerFailures() + 1L);
        long nowUs = TimeUnit.NANOSECONDS.toMicros(System.nanoTime());
        PollHealth next =
                new PollHealth(
                        nowUs,
                        Math.max(0L, nowUs - previous.lastSuccessfulPollUs()),
                        failures,
                        alive);
        pollHealth.set(next);
        if (failures == MAX_CONSECUTIVE_WORKER_FAILURES && closeReason == null) {
            closeReason = CloseReason.CONSECUTIVE_WORKER_FAILURES;
            trace.recordUnavailable(closeReason.name(), previous.lastSuccessfulPollUs(), failures);
        }
        return next;
    }

    private static long mergedFinishedVersion(
            AppliedStatusCursor current, Long responseFinishedVersion) {
        return responseFinishedVersion == null
                ? current.latestFinishedTaskVersion()
                : Math.max(current.latestFinishedTaskVersion(), responseFinishedVersion);
    }

    private void requirePollLease(
            PollLease lease, PollKind expectedKind, AtomicReference<PollLease> slot) {
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

    public enum CloseReason {
        DISCOVERY_REMOVED,
        CONSECUTIVE_WORKER_FAILURES
    }

    /**
     * Freezes a worker generation's discovery labels separately from its engine observations.
     */
    public record TopologySnapshot(
            String group,
            String ip,
            int port,
            int grpcPort,
            String site,
            String deploymentName,
            int engineIndex,
            int multiEngineNum,
            int workerStatusPort) {

        public TopologySnapshot(
                String group,
                String ip,
                int port,
                int grpcPort,
                String site,
                String deploymentName,
                int engineIndex,
                int multiEngineNum) {
            this(
                    group,
                    ip,
                    port,
                    grpcPort,
                    site,
                    deploymentName,
                    engineIndex,
                    multiEngineNum,
                    grpcPort);
        }

        public TopologySnapshot(String group, String ip, int port, int grpcPort, String site) {
            this(group, ip, port, grpcPort, site, null, 0, 1);
        }
    }

    /**
     * Deeply immutable copy of the task fields reported by one status RPC.
     */
    public record TaskObservation(
            String requestId,
            long prefixLength,
            long prefillTime,
            long inputLength,
            long waitingTime,
            long iterateCount,
            long endTimeMs,
            long dpRank,
            long errorCode,
            String errorMessage,
            TaskPhase phase,
            long executionTimeMs,
            PriorityPreemptionProgress priorityPreemptionProgress,
            TaskTelemetry telemetry) {

        /**
         * A pending observation that reports only priority cancellation progress.
         */
        @JsonIgnore
        public boolean isPriorityCancelOverlayOnly() {
            return (priorityPreemptionProgress == PriorityPreemptionProgress.CANCELING
                            || priorityPreemptionProgress == PriorityPreemptionProgress.CANCELED)
                    && phase == TaskPhase.PENDING;
        }
    }

    /**
     * Immutable view of one atomically published Engine status observation.
     * {@code runningTaskList} is the exact immutable map carried by the
     * transaction observation that produced this value.
     */
    public record TaskTelemetry(
            boolean prefixLengthValid,
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

        public TaskTelemetry(
                boolean prefixLengthValid,
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
            this(
                    prefixLengthValid,
                    requestReceivedTimeMs,
                    inputQueueEnqueueTimeMs,
                    inputQueueDrainTimeMs,
                    waitingEnteredTimeMs,
                    runningEnteredTimeMs,
                    remoteKvWaitMs,
                    firstTokenTimeMs,
                    hbmLocalMatchTokens,
                    remoteKvAddedMatchTokens,
                    firstPrefillStepId,
                    lastPrefillStepId,
                    prefillStepCount,
                    prefillNonfinalChunkTokensMin,
                    prefillNonfinalChunkTokensMax,
                    null,
                    null,
                    null);
        }
    }

    public record StepMetrics(
            long stepId,
            long completedTimeMs,
            long totalScheduledTokens,
            long prefillRequestCount,
            long prefillTokens,
            long tokenBudget,
            double budgetFillRatio) {}

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
            kvCacheGroupMode =
                    kvCacheGroupMode == null ? KvCacheGroupMode.UNSPECIFIED : kvCacheGroupMode;
            runningTaskList = Map.copyOf(runningTaskList);
        }

        public EngineObservation(
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
            this(
                    role,
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
    public record AppliedStatusCursor(long statusVersion, long latestFinishedTaskVersion) {}

    /**
     * The single public value for one successfully committed status round.
     */
    public record CommittedWorkerStatus(EngineObservation fields, AppliedStatusCursor cursor) {}

    /**
     * Valid worker-reported health. Transport and local processing failures
     * do not supply evidence that a worker is dead.
     */
    public record PollHealth(
            long lastSuccessfulPollUs,
            long successfulPollIntervalUs,
            long consecutiveWorkerFailures,
            boolean reportedAlive) {

        public PollHealth {
            checkArgument(
                    lastSuccessfulPollUs >= 0L
                            && successfulPollIntervalUs >= 0L
                            && consecutiveWorkerFailures >= 0L,
                    "poll health counters must be non-negative");
        }
    }

    public record CacheIndexSnapshot(
            CacheStatus cacheStatus, boolean indexInitialized, long indexedVersion) {}

    /**
     * One deeply frozen RPC observation shared by every status reducer.
     */
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

        /**
         * Exact service-discovery generation which issued this observation.
         */
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

        public Map<String, TaskObservation> activeTasks() {
            return engine.runningTaskList();
        }

        public Map<String, TaskObservation> finishedTasks() {
            return finishedTasks;
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

        private PreparedStatus(
                CommittedWorkerStatus baseCommitted,
                StatusObservation observation,
                CommittedWorkerStatus nextCommitted) {
            this.baseCommitted = baseCommitted;
            this.observation = observation;
            this.nextCommitted = nextCommitted;
        }

        public StatusObservation observation() {
            return observation;
        }
    }

    private enum PollKind {
        STATUS,
        APPLICATION,
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
        private final EndpointTrace.Poll trace;

        private boolean resultRecorded; // Guarded by owner.healthGate.

        private PollLease(
                WorkerStatus owner,
                PollKind kind,
                AtomicReference<PollLease> slot,
                EndpointTrace.Poll trace) {
            this.trace = trace;
            this.owner = owner;
            this.kind = kind;
            this.slot = slot;
        }

        public EndpointTrace.Poll trace() {
            return trace;
        }

        @Override
        public void close() {
            if (slot.compareAndSet(this, null)) {
                trace.record(
                        kind == PollKind.APPLICATION
                                ? EndpointTrace.Stage.CALLBACK_ENDED
                                : EndpointTrace.Stage.LEASE_RELEASED);
            }
        }
    }
}
